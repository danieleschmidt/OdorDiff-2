"""
Advanced molecule visualization system.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdDepictor
import io
import base64

from ..models.molecule import Molecule
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MoleculeViewer:
    """Advanced molecule visualization and analysis tools."""
    
    def __init__(self):
        self.logger = logger
        
    def create_property_radar_chart(self, molecules: List[Molecule]) -> go.Figure:
        """Create radar chart of molecular properties."""
        if not molecules:
            return go.Figure()
        
        # Define properties to visualize
        properties = [
            'molecular_weight', 'logP', 'tpsa', 
            'hbd', 'hba', 'rotatable_bonds'
        ]
        property_labels = [
            'Mol Weight', 'LogP', 'TPSA',
            'H-Bond Donors', 'H-Bond Acceptors', 'Rotatable Bonds'
        ]
        
        fig = go.Figure()
        
        for i, mol in enumerate(molecules[:5]):  # Limit to 5 molecules
            if not mol.is_valid:
                continue
                
            values = []
            for prop in properties:
                val = mol.get_property(prop) or 0
                # Normalize values to 0-1 scale for radar chart
                if prop == 'molecular_weight':
                    val = min(val / 500, 1.0)
                elif prop == 'logP':
                    val = min(max(val + 5, 0) / 10, 1.0)  # Scale -5 to 5 -> 0 to 1
                elif prop == 'tpsa':
                    val = min(val / 200, 1.0)
                else:
                    val = min(val / 10, 1.0)  # Generic scaling
                values.append(val)
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=property_labels,
                fill='toself',
                name=f'Molecule {i+1}',
                line_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Molecular Property Comparison",
            height=500
        )
        
        return fig
    
    def create_safety_score_plot(self, molecules: List[Molecule]) -> go.Figure:
        """Create safety score visualization."""
        if not molecules:
            return go.Figure()
        
        data = []
        for i, mol in enumerate(molecules):
            if mol.is_valid:
                data.append({
                    'Molecule': f'Mol {i+1}',
                    'SMILES': mol.smiles[:20] + '...' if len(mol.smiles) > 20 else mol.smiles,
                    'Safety Score': mol.safety_score,
                    'Synth Score': mol.synth_score,
                    'Cost': mol.estimated_cost
                })
        
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Safety Scores', 'Synthesizability Scores', 'Cost Estimates', 'Safety vs Synth'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Safety scores
        fig.add_trace(
            go.Bar(x=df['Molecule'], y=df['Safety Score'], 
                   marker_color='green', name='Safety Score'),
            row=1, col=1
        )
        
        # Synthesizability scores
        fig.add_trace(
            go.Bar(x=df['Molecule'], y=df['Synth Score'], 
                   marker_color='blue', name='Synth Score'),
            row=1, col=2
        )
        
        # Cost estimates
        fig.add_trace(
            go.Bar(x=df['Molecule'], y=df['Cost'], 
                   marker_color='orange', name='Cost'),
            row=2, col=1
        )
        
        # Safety vs Synthesizability scatter
        fig.add_trace(
            go.Scatter(
                x=df['Safety Score'], y=df['Synth Score'],
                mode='markers+text',
                text=df['Molecule'],
                textposition='top center',
                marker=dict(size=10, color='purple'),
                name='Safety vs Synth'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Molecule Assessment Dashboard")
        return fig
    
    def create_odor_profile_plot(self, molecules: List[Molecule]) -> go.Figure:
        """Create odor profile visualization."""
        if not molecules:
            return go.Figure()
        
        # Collect all unique odor notes
        all_notes = set()
        for mol in molecules:
            if mol.is_valid:
                all_notes.update(mol.odor_profile.primary_notes)
                all_notes.update(mol.odor_profile.secondary_notes)
        
        all_notes = sorted(list(all_notes))
        
        if not all_notes:
            return go.Figure()
        
        # Create matrix of molecules vs odor notes
        matrix = []
        molecule_names = []
        
        for i, mol in enumerate(molecules):
            if not mol.is_valid:
                continue
                
            molecule_names.append(f'Mol {i+1}')
            row = []
            
            for note in all_notes:
                score = 0
                if note in mol.odor_profile.primary_notes:
                    score = 2  # Strong presence
                elif note in mol.odor_profile.secondary_notes:
                    score = 1  # Weak presence
                row.append(score)
            
            matrix.append(row)
        
        if not matrix:
            return go.Figure()
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=all_notes,
            y=molecule_names,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Note Intensity")
        ))
        
        fig.update_layout(
            title="Odor Profile Heatmap",
            xaxis_title="Odor Notes",
            yaxis_title="Molecules",
            height=max(400, len(molecule_names) * 40)
        )
        
        return fig
    
    def create_molecular_similarity_network(self, molecules: List[Molecule], threshold: float = 0.7) -> go.Figure:
        """Create molecular similarity network visualization."""
        if len(molecules) < 2:
            return go.Figure()
        
        valid_molecules = [mol for mol in molecules if mol.is_valid]
        if len(valid_molecules) < 2:
            return go.Figure()
        
        # Calculate similarity matrix
        n = len(valid_molecules)
        similarities = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                sim = valid_molecules[i].calculate_similarity(valid_molecules[j])
                similarities[i, j] = sim
                similarities[j, i] = sim
        
        # Create network layout
        pos = self._spring_layout(similarities, threshold)
        
        # Create edges
        edge_x = []
        edge_y = []
        edge_info = []
        
        for i in range(n):
            for j in range(i+1, n):
                if similarities[i, j] >= threshold:
                    x0, y0 = pos[i]
                    x1, y1 = pos[j]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_info.append(f'Similarity: {similarities[i, j]:.2f}')
        
        # Create nodes
        node_x = [pos[i][0] for i in range(n)]
        node_y = [pos[i][1] for i in range(n)]
        node_text = [f'Mol {i+1}<br>SMILES: {mol.smiles[:20]}...' 
                     for i, mol in enumerate(valid_molecules)]
        
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[f'M{i+1}' for i in range(n)],
            textposition='middle center',
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=20,
                color=[mol.safety_score for mol in valid_molecules],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Safety Score"),
                line=dict(width=2, color='black')
            ),
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"Molecular Similarity Network (threshold: {threshold})",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Node color = Safety Score<br>Edges = Molecular Similarity",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=10)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def _spring_layout(self, similarity_matrix: np.ndarray, threshold: float) -> List[Tuple[float, float]]:
        """Simple spring layout algorithm for network visualization."""
        n = similarity_matrix.shape[0]
        
        # Initialize random positions
        np.random.seed(42)  # For reproducibility
        pos = np.random.random((n, 2)) * 10
        
        # Simple force-directed layout
        for iteration in range(50):
            forces = np.zeros_like(pos)
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        diff = pos[i] - pos[j]
                        dist = np.linalg.norm(diff)
                        
                        if dist > 0:
                            # Repulsive force
                            forces[i] += diff / dist * 0.1
                            
                            # Attractive force if similar enough
                            if similarity_matrix[i, j] >= threshold:
                                forces[i] -= diff / dist * similarity_matrix[i, j] * 0.2
            
            # Update positions
            pos += forces * 0.1
            
            # Center the layout
            pos -= np.mean(pos, axis=0)
        
        return [(pos[i, 0], pos[i, 1]) for i in range(n)]
    
    def create_synthesis_route_diagram(self, synthesis_routes: List[Dict[str, Any]]) -> go.Figure:
        """Create synthesis route diagram."""
        if not synthesis_routes:
            return go.Figure()
        
        # Take the best route
        best_route = synthesis_routes[0]
        steps = best_route.get('steps', [])
        
        if not steps:
            return go.Figure()
        
        fig = go.Figure()
        
        # Create flowchart-like diagram
        y_positions = list(range(len(steps), 0, -1))
        x_position = 0
        
        for i, step in enumerate(steps):
            # Add step box
            fig.add_trace(go.Scatter(
                x=[x_position],
                y=[y_positions[i]],
                mode='markers+text',
                text=[f"Step {i+1}<br>{step.get('reaction_type', 'Unknown')}"],
                textposition='middle center',
                marker=dict(
                    size=80,
                    color='lightblue',
                    line=dict(width=2, color='darkblue')
                ),
                showlegend=False,
                hovertext=f"Yield: {step.get('yield_estimate', 0):.1%}<br>Difficulty: {step.get('difficulty', 0)}/5",
                hoverinfo='text'
            ))
            
            # Add arrow to next step
            if i < len(steps) - 1:
                fig.add_annotation(
                    x=x_position, y=y_positions[i] - 0.3,
                    ax=x_position, ay=y_positions[i] - 0.7,
                    arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="black"
                )
        
        fig.update_layout(
            title=f"Synthesis Route (Score: {best_route.get('score', 0):.2f})",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=max(300, len(steps) * 100),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def create_cost_breakdown_chart(self, molecules: List[Molecule]) -> go.Figure:
        """Create cost breakdown visualization."""
        if not molecules:
            return go.Figure()
        
        data = []
        for i, mol in enumerate(molecules):
            if mol.is_valid:
                # Simulate cost breakdown (in real implementation, this would be calculated)
                base_cost = mol.estimated_cost * 0.4  # Materials
                labor_cost = mol.estimated_cost * 0.3  # Labor
                equipment_cost = mol.estimated_cost * 0.2  # Equipment
                overhead_cost = mol.estimated_cost * 0.1  # Overhead
                
                data.extend([
                    {'Molecule': f'Mol {i+1}', 'Category': 'Materials', 'Cost': base_cost},
                    {'Molecule': f'Mol {i+1}', 'Category': 'Labor', 'Cost': labor_cost},
                    {'Molecule': f'Mol {i+1}', 'Category': 'Equipment', 'Cost': equipment_cost},
                    {'Molecule': f'Mol {i+1}', 'Category': 'Overhead', 'Cost': overhead_cost}
                ])
        
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        
        fig = px.bar(
            df, x='Molecule', y='Cost', color='Category',
            title='Cost Breakdown by Molecule',
            labels={'Cost': 'Cost ($/kg)'}
        )
        
        fig.update_layout(height=400)
        return fig
    
    def export_visualization_data(self, molecules: List[Molecule]) -> Dict[str, Any]:
        """Export visualization data for external use."""
        data = {
            'molecules': [],
            'properties': [],
            'odor_profiles': [],
            'safety_scores': []
        }
        
        for i, mol in enumerate(molecules):
            if not mol.is_valid:
                continue
                
            data['molecules'].append({
                'id': i,
                'smiles': mol.smiles,
                'confidence': mol.confidence
            })
            
            data['properties'].append({
                'molecule_id': i,
                'molecular_weight': mol.get_property('molecular_weight'),
                'logP': mol.get_property('logP'),
                'tpsa': mol.get_property('tpsa'),
                'safety_score': mol.safety_score,
                'synth_score': mol.synth_score,
                'estimated_cost': mol.estimated_cost
            })
            
            data['odor_profiles'].append({
                'molecule_id': i,
                'primary_notes': mol.odor_profile.primary_notes,
                'secondary_notes': mol.odor_profile.secondary_notes,
                'intensity': mol.odor_profile.intensity,
                'longevity_hours': mol.odor_profile.longevity_hours,
                'character': mol.odor_profile.character
            })
        
        return data