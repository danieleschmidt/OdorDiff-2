"""Tests for OdorDiff-2."""

import json
import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from odordiff.molecule import MoleculeRepresentation
from odordiff.text_conditioner import TextConditioner
from odordiff.synthesizability import SynthesizabilityScorer, RING_PENALTY, BRANCH_PENALTY
from odordiff.novelty import NoveltyChecker, REFERENCE_SET
from odordiff.vscode_plugin import VSCodePluginManifest
from odordiff.generator import OdorDiffGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_linear_molecule(n: int) -> MoleculeRepresentation:
    """Create a linear chain molecule with n carbon atoms."""
    atoms = [6] * n
    bonds = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        bonds[i, i + 1] = 1.0
        bonds[i + 1, i] = 1.0
    return MoleculeRepresentation(atoms=atoms, bonds=bonds)


def make_ring_molecule(n: int) -> MoleculeRepresentation:
    """Create a ring molecule with n carbon atoms."""
    atoms = [6] * n
    bonds = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = (i + 1) % n
        bonds[i, j] = 1.0
        bonds[j, i] = 1.0
    return MoleculeRepresentation(atoms=atoms, bonds=bonds)


def make_branched_molecule() -> MoleculeRepresentation:
    """Create a Y-shaped molecule: C-C(-C)-C (4 atoms, atom 1 has degree 3)."""
    #   0 - 1 - 2
    #       |
    #       3
    atoms = [6, 6, 6, 6]
    bonds = np.zeros((4, 4), dtype=float)
    bonds[0, 1] = bonds[1, 0] = 1.0
    bonds[1, 2] = bonds[2, 1] = 1.0
    bonds[1, 3] = bonds[3, 1] = 1.0
    return MoleculeRepresentation(atoms=atoms, bonds=bonds)


# ---------------------------------------------------------------------------
# Molecule tests
# ---------------------------------------------------------------------------

def test_molecule_rings():
    """Ring molecule should detect the correct number of rings."""
    mol = make_ring_molecule(6)
    assert mol.num_rings >= 1, f"Expected >=1 ring, got {mol.num_rings}"

    linear = make_linear_molecule(6)
    assert linear.num_rings == 0, f"Expected 0 rings in linear molecule, got {linear.num_rings}"


def test_molecule_branching():
    """Branching factor should reflect atoms with >2 bonds."""
    branched = make_branched_molecule()
    # Atom 1 has 3 bonds -> branching_factor = 1/4 = 0.25
    assert branched.branching_factor == pytest.approx(0.25, abs=1e-6)

    linear = make_linear_molecule(5)
    assert linear.branching_factor == pytest.approx(0.0, abs=1e-6)


def test_molecule_fingerprint_shape():
    """Fingerprint should be 128-dimensional."""
    mol = make_linear_molecule(5)
    fp = mol.to_fingerprint()
    assert fp.shape == (128,)
    assert fp.dtype in (np.float32, np.float64, float)


def test_molecule_num_bonds():
    """num_bonds should count edges, not directed pairs."""
    mol = make_linear_molecule(4)  # 3 bonds
    assert mol.num_bonds == 3


# ---------------------------------------------------------------------------
# Text conditioner tests
# ---------------------------------------------------------------------------

def test_text_conditioner_encode_shape():
    """encode() should return a 64-dim vector."""
    tc = TextConditioner()
    vec = tc.encode("fresh citrus")
    assert vec.shape == (64,)


def test_text_conditioner_multilabel():
    """multi_label_encode should return all descriptors with scores."""
    tc = TextConditioner()
    scores = tc.multi_label_encode("fresh citrus floral")
    assert isinstance(scores, dict)
    # All 10 descriptors should be present
    descriptors = {"fresh", "citrus", "floral", "woody", "musky",
                   "sweet", "spicy", "earthy", "marine", "green"}
    assert set(scores.keys()) == descriptors
    # Detected descriptors should have positive scores
    assert scores["fresh"] > 0
    assert scores["citrus"] > 0
    assert scores["floral"] > 0
    # Non-detected descriptor should be 0
    assert scores["musky"] == pytest.approx(0.0)


def test_text_conditioner_zero_for_unknown():
    """Unknown text should return zero vector."""
    tc = TextConditioner()
    vec = tc.encode("zzzzzzz unknown gibberish xyz")
    assert np.allclose(vec, 0.0)


# ---------------------------------------------------------------------------
# Synthesizability tests
# ---------------------------------------------------------------------------

def test_synthesizability_scorer_basic():
    """Simple linear molecule should score relatively high."""
    scorer = SynthesizabilityScorer()
    mol = make_linear_molecule(5)
    score = scorer.score(mol)
    assert 0.0 <= score <= 1.0
    assert score > 0.5, f"Expected score > 0.5 for simple linear molecule, got {score}"


def test_synthesizability_high_rings_penalized():
    """Molecules with many rings should be penalized."""
    scorer = SynthesizabilityScorer()
    simple = make_linear_molecule(5)
    ringy = make_ring_molecule(6)

    sa_simple = scorer.score(simple)
    sa_ringy = scorer.score(ringy)
    # The ringy molecule should score lower due to ring penalty
    # (a 6-membered ring has 1 ring = 0.15 penalty + degree effects)
    assert sa_ringy < sa_simple, (
        f"Ring molecule ({sa_ringy:.3f}) should score lower than "
        f"linear ({sa_simple:.3f})"
    )


def test_synthesizability_low_atoms_good():
    """Very small molecules should have minimal complexity penalty."""
    scorer = SynthesizabilityScorer()
    tiny = make_linear_molecule(2)
    score = scorer.score(tiny)
    assert score > 0.8, f"Expected score > 0.8 for 2-atom molecule, got {score}"


def test_synthesizability_branching_penalized():
    """Branched molecules should be penalized."""
    scorer = SynthesizabilityScorer()
    linear = make_linear_molecule(4)
    branched = make_branched_molecule()
    sa_linear = scorer.score(linear)
    sa_branched = scorer.score(branched)
    assert sa_branched < sa_linear, (
        f"Branched ({sa_branched:.3f}) should score lower than linear ({sa_linear:.3f})"
    )


# ---------------------------------------------------------------------------
# Novelty tests
# ---------------------------------------------------------------------------

def test_novelty_checker_similar():
    """A fingerprint identical to a reference should not be novel."""
    checker = NoveltyChecker(reference_set=REFERENCE_SET)
    # Create a molecule whose fingerprint is very similar to reference 0
    # by using the same binary pattern
    ref_fp = REFERENCE_SET[0]
    # Build a molecule whose to_fingerprint() would be close by using
    # a mock: directly test with tanimoto instead
    sim = checker.tanimoto_similarity(ref_fp, ref_fp)
    assert sim == pytest.approx(1.0), f"Self-similarity should be 1.0, got {sim}"


def test_novelty_checker_novel():
    """A molecule with all-zero fingerprint vs binary references should be novel."""
    checker = NoveltyChecker(reference_set=REFERENCE_SET)
    zero_fp = np.zeros(128, dtype=float)
    sim = checker.tanimoto_similarity(zero_fp, REFERENCE_SET[0])
    assert sim == pytest.approx(0.0), f"Zero vector similarity should be 0.0, got {sim}"

    # A molecule with a unique fingerprint pattern
    mol = make_linear_molecule(3)
    # max_similarity should be < 1.0 for a simple linear molecule
    max_sim = checker.max_similarity(mol)
    assert 0.0 <= max_sim <= 1.0


def test_tanimoto_similarity():
    """Tanimoto similarity should be symmetric and in [0, 1]."""
    checker = NoveltyChecker()
    fp1 = np.array([1, 1, 0, 0, 1, 0, 0, 1] * 16, dtype=float)
    fp2 = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 16, dtype=float)

    sim_12 = checker.tanimoto_similarity(fp1, fp2)
    sim_21 = checker.tanimoto_similarity(fp2, fp1)

    assert sim_12 == pytest.approx(sim_21)
    assert 0.0 <= sim_12 <= 1.0

    # Identical vectors
    sim_same = checker.tanimoto_similarity(fp1, fp1)
    assert sim_same == pytest.approx(1.0)

    # Known value: fp1 has bits {0,1,4,7,...}, fp2 has bits {0,2,4,6,...}
    # intersection (both 1) = positions where both=1 = {0,4} per 8-block -> 2/8 * 16 = 32
    # union = positions where at least one=1 = {0,1,2,4,6,7} per 8-block -> 6/8 * 16 = 96
    expected = 32 / 96
    assert sim_12 == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# VSCode plugin tests
# ---------------------------------------------------------------------------

def test_vscode_manifest_json():
    """generate_package_json should return a valid dict with expected keys."""
    manifest = VSCodePluginManifest()
    pkg = manifest.generate_package_json(
        name="test-ext",
        version="0.1.0",
        description="Test extension",
    )
    assert isinstance(pkg, dict)
    assert pkg["name"] == "test-ext"
    assert pkg["version"] == "0.1.0"
    assert pkg["description"] == "Test extension"
    assert "engines" in pkg
    assert "contributes" in pkg
    assert "commands" in pkg["contributes"]
    # Should have 3 commands
    commands = pkg["contributes"]["commands"]
    assert len(commands) == 3
    cmd_ids = [c["command"] for c in commands]
    assert "odordiff.generate" in cmd_ids
    assert "odordiff.score" in cmd_ids
    assert "odordiff.checkNovelty" in cmd_ids


def test_vscode_extension_stub():
    """generate_extension_stub should return TypeScript code as string."""
    manifest = VSCodePluginManifest()
    ts = manifest.generate_extension_stub()
    assert isinstance(ts, str)
    assert len(ts) > 100
    assert "activate" in ts
    assert "deactivate" in ts
    assert "vscode" in ts
    assert "odordiff.generate" in ts


def test_vscode_plugin_save():
    """save() should write package.json and extension.ts files."""
    manifest = VSCodePluginManifest()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "my-plugin")
        manifest.save(out)
        assert os.path.exists(os.path.join(out, "package.json"))
        assert os.path.exists(os.path.join(out, "extension.ts"))
        # Verify package.json is valid JSON
        with open(os.path.join(out, "package.json")) as f:
            pkg = json.load(f)
        assert pkg["name"] == "odordiff-vscode"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

def test_cli_generate(capsys):
    """CLI generate command should produce output."""
    from odordiff.cli import main

    with patch("sys.argv", ["odordiff", "generate", "fresh citrus", "--n", "3", "--seed", "42"]):
        main()

    captured = capsys.readouterr()
    assert "Generated" in captured.out or "Molecule" in captured.out


def test_cli_score(capsys):
    """CLI score command should output a float."""
    from odordiff.cli import main

    with patch("sys.argv", ["odordiff", "score", "CC(=O)O"]):
        main()

    captured = capsys.readouterr()
    score = float(captured.out.strip())
    assert 0.0 <= score <= 1.0


def test_cli_check_novelty(capsys):
    """CLI check-novelty command should output novelty info."""
    from odordiff.cli import main

    with patch("sys.argv", ["odordiff", "check-novelty", "CCCCC"]):
        main()

    captured = capsys.readouterr()
    assert "novel=" in captured.out


# ---------------------------------------------------------------------------
# Generator end-to-end test
# ---------------------------------------------------------------------------

def test_generator_end_to_end():
    """OdorDiffGenerator should produce molecules satisfying filters."""
    gen = OdorDiffGenerator(seed=7)
    molecules = gen.generate(
        text="fresh citrus",
        n_molecules=3,
        min_synthesizability=0.3,
        require_novel=False,  # relax novelty for determinism
        max_attempts=50,
    )
    assert len(molecules) >= 1, "Should generate at least 1 molecule"
    scorer = SynthesizabilityScorer()
    for mol in molecules:
        sa = scorer.score(mol)
        assert sa >= 0.3, f"SA score {sa} below minimum"
        assert mol.num_atoms > 0
        fp = mol.to_fingerprint()
        assert fp.shape == (128,)
