"""CLI entry point for OdorDiff-2."""

from __future__ import annotations

import argparse
import json
import sys

from .generator import OdorDiffGenerator
from .molecule import MoleculeRepresentation
from .synthesizability import SynthesizabilityScorer
from .novelty import NoveltyChecker
from .vscode_plugin import VSCodePluginManifest


def _parse_smiles_as_molecule(smiles: str) -> MoleculeRepresentation:
    """Parse a SMILES-like string into a MoleculeRepresentation.
    
    For demonstration purposes, creates a simple molecule based on
    the length and content of the SMILES string without RDKit.
    """
    import numpy as np

    # Map common element symbols to atomic numbers
    ELEMENT_MAP = {
        'C': 6, 'H': 1, 'O': 8, 'N': 7, 'S': 16,
        'P': 15, 'F': 9, 'Cl': 17, 'Br': 35, 'I': 53,
    }

    # Simple SMILES tokenizer (no full parser, demo-quality)
    atoms = []
    i = 0
    while i < len(smiles):
        # Two-char elements
        if i + 1 < len(smiles) and smiles[i:i+2] in ELEMENT_MAP:
            atoms.append(ELEMENT_MAP[smiles[i:i+2]])
            i += 2
        elif smiles[i] in ELEMENT_MAP:
            atoms.append(ELEMENT_MAP[smiles[i]])
            i += 1
        elif smiles[i].upper() in ELEMENT_MAP:
            atoms.append(ELEMENT_MAP[smiles[i].upper()])
            i += 1
        else:
            i += 1

    if not atoms:
        # Fallback: treat each char as carbon
        atoms = [6] * max(len(smiles), 1)

    n = len(atoms)
    bonds = np.zeros((n, n), dtype=float)
    # Chain connectivity
    for k in range(n - 1):
        bonds[k, k + 1] = 1.0
        bonds[k + 1, k] = 1.0

    return MoleculeRepresentation(atoms=atoms, bonds=bonds)


def cmd_generate(args: argparse.Namespace) -> None:
    """Handle `odordiff generate` command."""
    gen = OdorDiffGenerator(seed=args.seed)
    molecules = gen.generate(
        text=args.text,
        n_molecules=args.n,
        min_synthesizability=args.min_sa,
        require_novel=not args.allow_known,
    )
    if not molecules:
        print("No molecules generated matching the criteria.", file=sys.stderr)
        sys.exit(1)

    scorer = SynthesizabilityScorer()
    checker = NoveltyChecker()

    print(f"Generated {len(molecules)} molecules for: '{args.text}'")
    print("-" * 60)
    for i, mol in enumerate(molecules, 1):
        sa = scorer.score(mol)
        novel = checker.is_novel(mol)
        print(
            f"  Molecule {i}: "
            f"atoms={mol.num_atoms}, "
            f"bonds={mol.num_bonds}, "
            f"rings={mol.num_rings}, "
            f"SA={sa:.3f}, "
            f"novel={novel}"
        )


def cmd_score(args: argparse.Namespace) -> None:
    """Handle `odordiff score` command."""
    mol = _parse_smiles_as_molecule(args.smiles)
    scorer = SynthesizabilityScorer()
    sa = scorer.score(mol)
    print(f"{sa:.4f}")


def cmd_check_novelty(args: argparse.Namespace) -> None:
    """Handle `odordiff check-novelty` command."""
    mol = _parse_smiles_as_molecule(args.smiles)
    checker = NoveltyChecker()
    max_sim = checker.max_similarity(mol)
    novel = checker.is_novel(mol, threshold=args.threshold)
    print(f"novel={novel} (max_similarity={max_sim:.4f}, threshold={args.threshold})")


def cmd_vscode_plugin(args: argparse.Namespace) -> None:
    """Handle `odordiff vscode-plugin` command."""
    manifest = VSCodePluginManifest()
    output_dir = args.output
    manifest.save(output_dir)
    print(f"VSCode plugin files written to: {output_dir}")
    print("  - package.json")
    print("  - extension.ts")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="odordiff",
        description="OdorDiff-2: Advanced olfactory molecule generation",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")
    subparsers.required = True

    # generate subcommand
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate molecules from olfactory text description",
    )
    gen_parser.add_argument("text", help='Olfactory description (e.g., "fresh citrus")')
    gen_parser.add_argument("--n", type=int, default=5, help="Number of molecules")
    gen_parser.add_argument("--min-sa", type=float, default=0.5, help="Min synthesizability score")
    gen_parser.add_argument("--allow-known", action="store_true", help="Include non-novel molecules")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    gen_parser.set_defaults(func=cmd_generate)

    # score subcommand
    score_parser = subparsers.add_parser(
        "score",
        help="Score synthesizability of a molecule (SMILES)",
    )
    score_parser.add_argument("smiles", help="SMILES string")
    score_parser.set_defaults(func=cmd_score)

    # check-novelty subcommand
    novelty_parser = subparsers.add_parser(
        "check-novelty",
        help="Check novelty of a molecule (SMILES)",
    )
    novelty_parser.add_argument("smiles", help="SMILES string")
    novelty_parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="Tanimoto similarity threshold for novelty (default: 0.6)"
    )
    novelty_parser.set_defaults(func=cmd_check_novelty)

    # vscode-plugin subcommand
    vscode_parser = subparsers.add_parser(
        "vscode-plugin",
        help="Generate VSCode extension files",
    )
    vscode_parser.add_argument(
        "--output", default="./odordiff-vscode",
        help="Output directory for extension files"
    )
    vscode_parser.set_defaults(func=cmd_vscode_plugin)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
