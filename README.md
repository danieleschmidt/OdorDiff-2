# OdorDiff-2

Advanced olfactory molecule generation with synthesizability scoring and VSCode integration.

## Features

- Text-conditioned molecule generation with classifier-free guidance
- SA score proxy for synthesizability assessment
- Tanimoto-based novelty checking against reference molecules
- VSCode plugin manifest generator for IDE integration
- CLI: `odordiff generate "fresh citrus"`

## Installation

```bash
pip install -e .
```

## Usage

```python
from odordiff.generator import OdorDiffGenerator

gen = OdorDiffGenerator()
molecules = gen.generate("fresh citrus", min_synthesizability=0.5, require_novel=True)

for mol in molecules:
    print(f"Atoms: {mol.num_atoms}, Rings: {mol.num_rings}, Bonds: {mol.num_bonds}")
```

## CLI

```bash
odordiff generate "fresh citrus"
odordiff generate "woody musky" --n 10 --min-sa 0.6
odordiff score CC(=O)O
odordiff check-novelty CCCCC --threshold 0.5
odordiff vscode-plugin --output ./my-plugin
```

## Components

| Module | Description |
|--------|-------------|
| `odordiff.molecule` | `MoleculeRepresentation` with fingerprinting, ring detection, branching |
| `odordiff.text_conditioner` | Text → 64-dim olfactory embedding |
| `odordiff.diffusion` | DDPM sampler with classifier-free guidance |
| `odordiff.synthesizability` | SA score proxy (ring + branching + complexity penalties) |
| `odordiff.novelty` | Tanimoto-based novelty checker |
| `odordiff.vscode_plugin` | VSCode extension manifest generator |
| `odordiff.generator` | High-level `OdorDiffGenerator` orchestrator |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
