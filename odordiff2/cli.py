"""
Command-line interface for OdorDiff-2.
"""

import click
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from .core.diffusion import OdorDiffusion
from .core.async_diffusion import AsyncOdorDiffusion
from .safety.filter import SafetyFilter
from .models.molecule import Molecule
from .utils.logging import get_logger
from .utils.validation import InputValidator
from .data.cache import get_molecule_cache
from .visualization.viewer import MoleculeViewer

logger = get_logger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--device', default='cpu', help='Device to use (cpu/cuda)')
@click.pass_context
def cli(ctx, verbose, device):
    """OdorDiff-2: Safe Text-to-Scent Molecule Diffusion"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['device'] = device
    
    if verbose:
        logger.info(f"OdorDiff-2 CLI started with device: {device}")


@cli.command()
@click.argument('prompt', type=str)
@click.option('--num-molecules', '-n', default=5, help='Number of molecules to generate')
@click.option('--safety-threshold', '-s', default=0.1, help='Safety filtering threshold (0-1)')
@click.option('--synth-min', default=0.0, help='Minimum synthesizability score (0-1)')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--format', default='json', type=click.Choice(['json', 'csv', 'txt']), help='Output format')
@click.option('--visualize', is_flag=True, help='Generate visualizations')
@click.pass_context
def generate(ctx, prompt, num_molecules, safety_threshold, synth_min, output, format, visualize):
    """Generate scent molecules from text description."""
    device = ctx.obj['device']
    verbose = ctx.obj['verbose']
    
    try:
        click.echo(f"🧪 Generating {num_molecules} molecules for: '{prompt}'")
        
        # Initialize model
        model = OdorDiffusion(device=device)
        safety_filter = SafetyFilter(
            toxicity_threshold=safety_threshold,
            irritant_check=True
        )
        
        # Generate molecules
        start_time = time.time()
        molecules = model.generate(
            prompt=prompt,
            num_molecules=num_molecules,
            safety_filter=safety_filter,
            synthesizability_min=synth_min
        )
        generation_time = time.time() - start_time
        
        if not molecules:
            click.echo("❌ No molecules generated", err=True)
            sys.exit(1)
        
        click.echo(f"✅ Generated {len(molecules)} molecules in {generation_time:.2f}s")
        
        # Display results
        _display_molecules(molecules, verbose)
        
        # Save results
        if output:
            _save_results(molecules, output, format, prompt)
            click.echo(f"💾 Results saved to {output}")
        
        # Generate visualizations
        if visualize:
            _create_visualizations(molecules, prompt)
            click.echo("📊 Visualizations created")
            
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('prompts_file', type=click.Path(exists=True))
@click.option('--num-molecules', '-n', default=5, help='Number of molecules per prompt')
@click.option('--safety-threshold', '-s', default=0.1, help='Safety filtering threshold')
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.option('--workers', default=4, help='Number of parallel workers')
@click.pass_context
def batch(ctx, prompts_file, num_molecules, safety_threshold, output_dir, workers):
    """Process batch of prompts from file."""
    device = ctx.obj['device']
    
    try:
        # Load prompts
        with open(prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        if not prompts:
            click.echo("❌ No prompts found in file", err=True)
            sys.exit(1)
        
        click.echo(f"📋 Processing {len(prompts)} prompts with {workers} workers")
        
        # Run batch processing
        results = asyncio.run(_batch_process(
            prompts, num_molecules, safety_threshold, workers, device
        ))
        
        # Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            for i, (prompt, molecules) in enumerate(results):
                filename = f"batch_result_{i+1}.json"
                _save_results(molecules, output_path / filename, 'json', prompt)
            
            click.echo(f"💾 Batch results saved to {output_dir}")
        
        # Summary
        total_molecules = sum(len(molecules) for _, molecules in results)
        click.echo(f"✅ Batch processing completed: {total_molecules} total molecules")
        
    except Exception as e:
        click.echo(f"❌ Batch processing error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('base_notes', type=str)
@click.argument('heart_notes', type=str)
@click.argument('top_notes', type=str)
@click.argument('style', type=str)
@click.option('--output', '-o', type=click.Path(), help='Output file for formulation')
@click.option('--iterations', default=3, help='Optimization iterations')
@click.pass_context
def design_fragrance(ctx, base_notes, heart_notes, top_notes, style, output, iterations):
    """Design a complete fragrance formulation."""
    device = ctx.obj['device']
    
    try:
        click.echo(f"🌸 Designing fragrance with style: '{style}'")
        
        model = OdorDiffusion(device=device)
        
        # Design fragrance
        formulation = model.design_fragrance(
            base_notes=base_notes,
            heart_notes=heart_notes,
            top_notes=top_notes,
            style=style
        )
        
        # Display formulation
        click.echo("\n📋 Fragrance Formulation:")
        click.echo(f"Style: {formulation.style_descriptor}")
        
        click.echo(f"\n🎵 Base Notes ({len(formulation.base_accord)} molecules):")
        _display_molecules(formulation.base_accord, brief=True)
        
        click.echo(f"\n💖 Heart Notes ({len(formulation.heart_accord)} molecules):")
        _display_molecules(formulation.heart_accord, brief=True)
        
        click.echo(f"\n✨ Top Notes ({len(formulation.top_accord)} molecules):")
        _display_molecules(formulation.top_accord, brief=True)
        
        # Get perfume formula
        formula = formulation.to_perfume_formula()
        click.echo(f"\n🧪 Perfume Formula (Eau de Parfum):")
        click.echo(f"Total fragrance oil: {formula['fragrance_oil_percent']:.1f}%")
        click.echo(f"Top notes: {formula['top_notes']['actual_percent']:.2f}%")
        click.echo(f"Heart notes: {formula['heart_notes']['actual_percent']:.2f}%")
        click.echo(f"Base notes: {formula['base_notes']['actual_percent']:.2f}%")
        
        # Save formulation
        if output:
            with open(output, 'w') as f:
                json.dump({
                    'formulation': {
                        'style': formulation.style_descriptor,
                        'base_accord': [mol.to_dict() for mol in formulation.base_accord],
                        'heart_accord': [mol.to_dict() for mol in formulation.heart_accord],
                        'top_accord': [mol.to_dict() for mol in formulation.top_accord]
                    },
                    'formula': formula
                }, f, indent=2)
            click.echo(f"💾 Formulation saved to {output}")
            
    except Exception as e:
        click.echo(f"❌ Fragrance design error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('smiles', type=str)
@click.option('--detailed', is_flag=True, help='Show detailed assessment')
@click.pass_context
def assess_safety(ctx, smiles, detailed):
    """Assess safety of a molecule by SMILES."""
    try:
        # Validate SMILES
        validated_smiles = InputValidator.validate_smiles(smiles)
        
        click.echo(f"🛡️ Assessing safety of: {validated_smiles}")
        
        # Create molecule and assess
        molecule = Molecule(validated_smiles)
        if not molecule.is_valid:
            click.echo("❌ Invalid SMILES structure", err=True)
            sys.exit(1)
        
        safety_filter = SafetyFilter()
        report = safety_filter.assess_molecule(molecule)
        
        # Display results
        click.echo(f"\n📊 Safety Assessment Results:")
        click.echo(f"Toxicity Score: {report.toxicity:.3f} {'❌' if report.toxicity > 0.1 else '✅'}")
        click.echo(f"Skin Sensitizer: {'Yes ⚠️' if report.skin_sensitizer else 'No ✅'}")
        click.echo(f"IFRA Compliant: {'Yes ✅' if report.ifra_compliant else 'No ❌'}")
        click.echo(f"Environmental Score: {report.eco_score:.3f}")
        
        if report.regulatory_flags:
            click.echo(f"\n⚠️ Regulatory Flags:")
            for flag in report.regulatory_flags:
                click.echo(f"  - {flag['region']}: {flag['status']}")
        
        # Overall recommendation
        safe = (report.toxicity <= 0.1 and report.ifra_compliant and not report.skin_sensitizer)
        recommendation = "SAFE ✅" if safe else "CAUTION ⚠️"
        click.echo(f"\n🏷️ Overall: {recommendation}")
        
        if detailed:
            click.echo(f"\n🔬 Molecular Properties:")
            for prop, value in molecule._properties.items():
                click.echo(f"  {prop}: {value}")
                
    except Exception as e:
        click.echo(f"❌ Safety assessment error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--port', default=8000, help='Port to run API server')
@click.option('--host', default='0.0.0.0', help='Host to bind server')
@click.option('--workers', default=4, help='Number of worker processes')
@click.pass_context
def serve(ctx, port, host, workers):
    """Start the API server."""
    try:
        click.echo(f"🚀 Starting OdorDiff-2 API server on http://{host}:{port}")
        
        import uvicorn
        from .api.endpoints import app
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )
        
    except Exception as e:
        click.echo(f"❌ Server error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--port', default=8050, help='Port to run dashboard')
@click.option('--api-url', default='http://localhost:8000', help='API server URL')
@click.pass_context
def dashboard(ctx, port, api_url):
    """Start the interactive dashboard."""
    try:
        click.echo(f"📊 Starting OdorDiff-2 Dashboard on http://0.0.0.0:{port}")
        click.echo(f"API URL: {api_url}")
        
        from .visualization.dashboard import create_dashboard
        
        dash_app = create_dashboard(api_url=api_url)
        dash_app.run(port=port, debug=False)
        
    except Exception as e:
        click.echo(f"❌ Dashboard error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def cache_info(ctx):
    """Show cache information and statistics."""
    try:
        cache = get_molecule_cache()
        stats = cache.get_cache_stats()
        
        click.echo("💾 Cache Statistics:")
        for cache_name, cache_stats in stats.items():
            click.echo(f"\n{cache_name}:")
            for key, value in cache_stats.items():
                click.echo(f"  {key}: {value}")
        
        cleanup_stats = cache.cleanup_expired()
        if any(cleanup_stats.values()):
            click.echo(f"\n🧹 Cleanup Results:")
            for key, count in cleanup_stats.items():
                if count > 0:
                    click.echo(f"  {key}: {count} entries removed")
                    
    except Exception as e:
        click.echo(f"❌ Cache info error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def clear_cache(ctx):
    """Clear all cached data."""
    try:
        cache = get_molecule_cache()
        cache.clear_generation_cache()
        click.echo("🧹 Cache cleared successfully")
        
    except Exception as e:
        click.echo(f"❌ Cache clear error: {str(e)}", err=True)
        sys.exit(1)


def _display_molecules(molecules: List[Molecule], verbose: bool = False, brief: bool = False):
    """Display molecules in formatted output."""
    for i, mol in enumerate(molecules, 1):
        if not mol.is_valid:
            continue
            
        if brief:
            primary_notes = ", ".join(mol.odor_profile.primary_notes[:2])
            click.echo(f"  {i}. {mol.smiles[:30]}... | {primary_notes} | Safety: {mol.safety_score:.2f}")
        else:
            click.echo(f"\n🧬 Molecule {i}:")
            click.echo(f"  SMILES: {mol.smiles}")
            click.echo(f"  Confidence: {mol.confidence:.2f}")
            click.echo(f"  Safety Score: {mol.safety_score:.2f}")
            click.echo(f"  Synthesis Score: {mol.synth_score:.2f}")
            click.echo(f"  Estimated Cost: ${mol.estimated_cost:.2f}/kg")
            
            if mol.odor_profile.primary_notes:
                notes = ", ".join(mol.odor_profile.primary_notes)
                click.echo(f"  Odor: {notes}")
                click.echo(f"  Character: {mol.odor_profile.character}")
                
            if verbose and mol._properties:
                click.echo(f"  Properties: {mol._properties}")


async def _batch_process(
    prompts: List[str], 
    num_molecules: int, 
    safety_threshold: float, 
    workers: int, 
    device: str
) -> List[tuple]:
    """Process batch of prompts asynchronously."""
    async with AsyncOdorDiffusion(device=device, max_workers=workers) as model:
        safety_filter = SafetyFilter(toxicity_threshold=safety_threshold)
        
        results = []
        
        for i, prompt in enumerate(prompts, 1):
            click.echo(f"Processing {i}/{len(prompts)}: {prompt[:50]}...")
            
            result = await model.generate_async(
                prompt=prompt,
                num_molecules=num_molecules,
                safety_filter=safety_filter
            )
            
            results.append((prompt, result.molecules))
        
        return results


def _save_results(molecules: List[Molecule], output_path: Path, format: str, prompt: str):
    """Save results in specified format."""
    output_path = Path(output_path)
    
    if format == 'json':
        data = {
            'prompt': prompt,
            'timestamp': time.time(),
            'molecules': [mol.to_dict() for mol in molecules]
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    elif format == 'csv':
        rows = []
        for mol in molecules:
            if mol.is_valid:
                rows.append({
                    'smiles': mol.smiles,
                    'confidence': mol.confidence,
                    'safety_score': mol.safety_score,
                    'synth_score': mol.synth_score,
                    'estimated_cost': mol.estimated_cost,
                    'primary_notes': ', '.join(mol.odor_profile.primary_notes),
                    'character': mol.odor_profile.character
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
    elif format == 'txt':
        with open(output_path, 'w') as f:
            f.write(f"Prompt: {prompt}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, mol in enumerate(molecules, 1):
                if mol.is_valid:
                    f.write(f"Molecule {i}:\n")
                    f.write(f"  SMILES: {mol.smiles}\n")
                    f.write(f"  Safety: {mol.safety_score:.2f}\n")
                    f.write(f"  Synthesis: {mol.synth_score:.2f}\n")
                    f.write(f"  Cost: ${mol.estimated_cost:.2f}/kg\n")
                    f.write(f"  Odor: {', '.join(mol.odor_profile.primary_notes)}\n")
                    f.write("\n")


def _create_visualizations(molecules: List[Molecule], prompt: str):
    """Create and save visualizations."""
    viewer = MoleculeViewer()
    
    # Create plots
    safety_plot = viewer.create_safety_score_plot(molecules)
    odor_plot = viewer.create_odor_profile_plot(molecules)
    
    # Save plots
    safety_plot.write_html(f"safety_plot_{int(time.time())}.html")
    odor_plot.write_html(f"odor_plot_{int(time.time())}.html")


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()