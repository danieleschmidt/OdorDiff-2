"""
Interactive dashboard for OdorDiff-2 molecule generation and analysis.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import requests
import json
from datetime import datetime, timedelta
import time

from ..models.molecule import Molecule
from ..utils.logging import get_logger

logger = get_logger(__name__)


class OdorDiffDashboard:
    """Interactive dashboard for OdorDiff-2."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.app = dash.Dash(__name__, external_stylesheets=[
            'https://codepen.io/chriddyp/pen/bWLwgP.css',
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'
        ])
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🌺 OdorDiff-2 Dashboard", className="text-center mb-4"),
                html.P("Safe Text-to-Scent Molecule Diffusion", className="text-center text-muted"),
                html.Hr()
            ], className="container-fluid py-3 bg-light"),
            
            # Main content
            dcc.Tabs(id="main-tabs", value="generation-tab", children=[
                # Generation Tab
                dcc.Tab(label="🧪 Molecule Generation", value="generation-tab", children=[
                    html.Div([
                        self._generation_layout()
                    ], className="container-fluid py-4")
                ]),
                
                # Analysis Tab
                dcc.Tab(label="📊 Analysis", value="analysis-tab", children=[
                    html.Div([
                        self._analysis_layout()
                    ], className="container-fluid py-4")
                ]),
                
                # Safety Tab
                dcc.Tab(label="🛡️ Safety Assessment", value="safety-tab", children=[
                    html.Div([
                        self._safety_layout()
                    ], className="container-fluid py-4")
                ]),
                
                # Performance Tab
                dcc.Tab(label="⚡ Performance", value="performance-tab", children=[
                    html.Div([
                        self._performance_layout()
                    ], className="container-fluid py-4")
                ])
            ])
        ])
    
    def _generation_layout(self):
        """Layout for molecule generation tab."""
        return html.Div([
            # Input section
            html.Div([
                html.H3("Generate Scent Molecules"),
                html.Div([
                    html.Div([
                        html.Label("Scent Description:", className="form-label"),
                        dcc.Textarea(
                            id="prompt-input",
                            placeholder="Enter scent description (e.g., 'fresh citrus with floral undertones')",
                            className="form-control",
                            style={"height": "100px"},
                            value="fresh lavender field in morning dew"
                        )
                    ], className="col-md-8"),
                    
                    html.Div([
                        html.Label("Number of Molecules:", className="form-label"),
                        dcc.Slider(
                            id="num-molecules-slider",
                            min=1, max=20, step=1, value=5,
                            marks={i: str(i) for i in [1, 5, 10, 15, 20]},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        
                        html.Label("Safety Threshold:", className="form-label mt-3"),
                        dcc.Slider(
                            id="safety-threshold-slider",
                            min=0, max=1, step=0.1, value=0.1,
                            marks={i/10: f"{i/10:.1f}" for i in range(0, 11, 2)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        
                        html.Label("Min Synthesizability:", className="form-label mt-3"),
                        dcc.Slider(
                            id="synth-threshold-slider",
                            min=0, max=1, step=0.1, value=0.0,
                            marks={i/10: f"{i/10:.1f}" for i in range(0, 11, 2)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className="col-md-4")
                ], className="row"),
                
                html.Div([
                    html.Button("🚀 Generate Molecules", id="generate-btn", 
                              className="btn btn-primary me-2", n_clicks=0),
                    html.Button("💾 Save Results", id="save-btn", 
                              className="btn btn-success me-2", n_clicks=0, disabled=True),
                    html.Button("🔄 Clear", id="clear-btn", 
                              className="btn btn-secondary", n_clicks=0)
                ], className="mt-3")
            ], className="card p-4 mb-4"),
            
            # Results section
            html.Div(id="generation-results", children=[
                html.Div([
                    html.H4("Generated Molecules"),
                    html.P("Click 'Generate Molecules' to see results here.")
                ], className="text-center text-muted py-5")
            ], className="card p-4"),
            
            # Loading indicator
            dcc.Loading(
                id="generation-loading",
                type="default",
                children=html.Div(id="loading-output")
            )
        ])
    
    def _analysis_layout(self):
        """Layout for analysis tab."""
        return html.Div([
            html.H3("Molecule Analysis & Comparison"),
            
            html.Div([
                html.Div([
                    html.H5("Property Distribution"),
                    dcc.Graph(id="property-distribution-plot")
                ], className="col-md-6"),
                
                html.Div([
                    html.H5("Odor Profile Comparison"),
                    dcc.Graph(id="odor-profile-plot")
                ], className="col-md-6")
            ], className="row mb-4"),
            
            html.Div([
                html.Div([
                    html.H5("Safety vs Synthesizability"),
                    dcc.Graph(id="safety-synth-plot")
                ], className="col-md-6"),
                
                html.Div([
                    html.H5("Molecular Similarity Network"),
                    dcc.Graph(id="similarity-network")
                ], className="col-md-6")
            ], className="row")
        ])
    
    def _safety_layout(self):
        """Layout for safety assessment tab."""
        return html.Div([
            html.H3("Safety Assessment Tools"),
            
            html.Div([
                html.H5("Individual Molecule Assessment"),
                html.Div([
                    dcc.Input(
                        id="safety-smiles-input",
                        type="text",
                        placeholder="Enter SMILES string",
                        className="form-control",
                        style={"width": "70%", "display": "inline-block"}
                    ),
                    html.Button("Assess Safety", id="assess-safety-btn", 
                              className="btn btn-warning ms-2", n_clicks=0)
                ], className="mb-3"),
                
                html.Div(id="safety-assessment-results")
            ], className="card p-4 mb-4"),
            
            html.Div([
                html.H5("Safety Statistics"),
                html.Div(id="safety-stats")
            ], className="card p-4")
        ])
    
    def _performance_layout(self):
        """Layout for performance monitoring tab."""
        return html.Div([
            html.H3("System Performance Monitoring"),
            
            # Real-time metrics
            html.Div([
                html.Div([
                    html.H5("Real-time Metrics"),
                    html.Div(id="realtime-metrics")
                ], className="col-md-4"),
                
                html.Div([
                    html.H5("Cache Performance"),
                    dcc.Graph(id="cache-performance-plot")
                ], className="col-md-4"),
                
                html.Div([
                    html.H5("Response Times"),
                    dcc.Graph(id="response-times-plot")
                ], className="col-md-4")
            ], className="row mb-4"),
            
            # System health
            html.Div([
                html.H5("System Health"),
                html.Div(id="system-health")
            ], className="card p-4"),
            
            # Auto-refresh interval
            dcc.Interval(
                id="performance-interval",
                interval=5000,  # Update every 5 seconds
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output("generation-results", "children"),
             Output("save-btn", "disabled"),
             Output("loading-output", "children")],
            [Input("generate-btn", "n_clicks")],
            [State("prompt-input", "value"),
             State("num-molecules-slider", "value"),
             State("safety-threshold-slider", "value"),
             State("synth-threshold-slider", "value")]
        )
        def generate_molecules(n_clicks, prompt, num_molecules, safety_threshold, synth_threshold):
            """Handle molecule generation."""
            if n_clicks == 0 or not prompt:
                return self._empty_results(), True, ""
            
            try:
                # Call API
                response = requests.post(f"{self.api_url}/generate", json={
                    "prompt": prompt,
                    "num_molecules": num_molecules,
                    "safety_threshold": safety_threshold,
                    "synthesizability_min": synth_threshold,
                    "use_cache": True
                })
                
                if response.status_code == 200:
                    data = response.json()
                    return self._format_results(data), False, ""
                else:
                    error_msg = response.json().get("detail", "Unknown error")
                    return self._error_results(error_msg), True, ""
                    
            except Exception as e:
                return self._error_results(str(e)), True, ""
        
        @self.app.callback(
            Output("safety-assessment-results", "children"),
            [Input("assess-safety-btn", "n_clicks")],
            [State("safety-smiles-input", "value")]
        )
        def assess_safety(n_clicks, smiles):
            """Handle safety assessment."""
            if n_clicks == 0 or not smiles:
                return html.Div("Enter a SMILES string and click 'Assess Safety'", 
                              className="text-muted")
            
            try:
                response = requests.post(f"{self.api_url}/assess/safety", json={
                    "smiles": smiles
                })
                
                if response.status_code == 200:
                    data = response.json()
                    return self._format_safety_results(data)
                else:
                    error_msg = response.json().get("detail", "Assessment failed")
                    return html.Div(f"Error: {error_msg}", className="alert alert-danger")
                    
            except Exception as e:
                return html.Div(f"Error: {str(e)}", className="alert alert-danger")
        
        @self.app.callback(
            [Output("realtime-metrics", "children"),
             Output("cache-performance-plot", "figure"),
             Output("response-times-plot", "figure"),
             Output("system-health", "children")],
            [Input("performance-interval", "n_intervals")]
        )
        def update_performance(n):
            """Update performance metrics."""
            try:
                # Get stats
                stats_response = requests.get(f"{self.api_url}/stats")
                health_response = requests.get(f"{self.api_url}/health")
                
                if stats_response.status_code == 200 and health_response.status_code == 200:
                    stats = stats_response.json()
                    health = health_response.json()
                    
                    metrics = self._format_metrics(stats, health)
                    cache_plot = self._create_cache_plot(stats)
                    response_plot = self._create_response_plot(stats)
                    health_status = self._format_health_status(health)
                    
                    return metrics, cache_plot, response_plot, health_status
                
            except Exception as e:
                logger.error(f"Error updating performance: {e}")
            
            # Return empty/error states
            return (
                html.Div("Performance data unavailable", className="text-muted"),
                {"data": [], "layout": {"title": "Cache Performance"}},
                {"data": [], "layout": {"title": "Response Times"}},
                html.Div("Health check unavailable", className="text-muted")
            )
    
    def _empty_results(self):
        """Return empty results placeholder."""
        return html.Div([
            html.H4("Generated Molecules"),
            html.P("Click 'Generate Molecules' to see results here.")
        ], className="text-center text-muted py-5")
    
    def _error_results(self, error_msg: str):
        """Return error results."""
        return html.Div([
            html.H4("Generation Error", className="text-danger"),
            html.P(f"Error: {error_msg}")
        ], className="text-center py-5")
    
    def _format_results(self, data: Dict[str, Any]):
        """Format generation results for display."""
        molecules = data.get("molecules", [])
        
        if not molecules:
            return html.Div("No molecules generated", className="text-muted text-center py-5")
        
        cards = []
        for i, mol in enumerate(molecules):
            card = html.Div([
                html.Div([
                    html.H6(f"Molecule {i+1}", className="card-title"),
                    html.P(f"SMILES: {mol['smiles']}", className="font-monospace small"),
                    
                    html.Div([
                        html.Span([
                            html.Strong("Safety: "),
                            f"{mol['safety_score']:.2f}",
                            self._get_score_badge(mol['safety_score'])
                        ], className="me-3"),
                        html.Span([
                            html.Strong("Synth: "),
                            f"{mol['synth_score']:.2f}",
                            self._get_score_badge(mol['synth_score'])
                        ], className="me-3"),
                        html.Span([
                            html.Strong("Cost: "),
                            f"${mol['estimated_cost']:.2f}/kg"
                        ])
                    ], className="mb-2"),
                    
                    html.Div([
                        html.Strong("Odor: "),
                        ", ".join(mol['odor_profile']['primary_notes'][:3])
                    ], className="text-muted small"),
                    
                    html.Div([
                        html.Strong("Character: "),
                        mol['odor_profile']['character']
                    ], className="text-muted small")
                    
                ], className="card-body")
            ], className="card mb-2", style={"height": "200px"})
            cards.append(card)
        
        # Arrange in grid
        rows = []
        for i in range(0, len(cards), 3):
            row_cards = cards[i:i+3]
            row = html.Div([
                html.Div(card, className="col-md-4") for card in row_cards
            ], className="row mb-3")
            rows.append(row)
        
        return html.Div([
            html.H4(f"Generated {len(molecules)} Molecules"),
            html.P(f"Prompt: {data.get('prompt', '')}", className="text-muted"),
            html.P(f"Processing time: {data.get('processing_time', 0):.2f}s | Cache hit: {data.get('cache_hit', False)}", 
                  className="small text-info"),
            html.Hr(),
            *rows
        ])
    
    def _get_score_badge(self, score: float) -> html.Span:
        """Get color-coded badge for score."""
        if score >= 0.8:
            color = "success"
        elif score >= 0.5:
            color = "warning"
        else:
            color = "danger"
        
        return html.Span("●", className=f"text-{color} ms-1")
    
    def _format_safety_results(self, data: Dict[str, Any]):
        """Format safety assessment results."""
        assessment = data.get("assessment", {})
        
        return html.Div([
            html.H6(f"Safety Assessment: {data.get('smiles')}"),
            html.Hr(),
            
            html.Div([
                html.Div([
                    html.Strong("Toxicity Score: "),
                    f"{assessment.get('toxicity_score', 0):.3f}",
                    self._get_score_badge(1 - assessment.get('toxicity_score', 1))
                ], className="mb-2"),
                
                html.Div([
                    html.Strong("Skin Sensitizer: "),
                    "Yes" if assessment.get('skin_sensitizer', False) else "No",
                    html.Span("⚠️" if assessment.get('skin_sensitizer', False) else "✅", className="ms-2")
                ], className="mb-2"),
                
                html.Div([
                    html.Strong("IFRA Compliant: "),
                    "Yes" if assessment.get('ifra_compliant', False) else "No",
                    html.Span("✅" if assessment.get('ifra_compliant', False) else "❌", className="ms-2")
                ], className="mb-2"),
                
                html.Div([
                    html.Strong("Environmental Score: "),
                    f"{assessment.get('eco_score', 0):.3f}",
                    self._get_score_badge(1 - assessment.get('eco_score', 1))
                ], className="mb-3"),
                
                html.Div([
                    html.Strong("Overall Recommendation: "),
                    html.Span(
                        data.get('recommendation', 'unknown').upper(),
                        className=f"badge bg-{'success' if data.get('recommendation') == 'safe' else 'warning'}"
                    )
                ])
            ])
        ], className="p-3 border rounded")
    
    def _format_metrics(self, stats: Dict[str, Any], health: Dict[str, Any]):
        """Format real-time metrics."""
        stats_data = stats.get("stats", {})
        
        metrics = [
            html.Div([
                html.H6("Memory Usage"),
                html.H4(f"{health.get('memory_usage_mb', 0):.1f} MB", className="text-primary")
            ], className="text-center"),
            
            html.Div([
                html.H6("Cache Hit Rate"),
                html.H4(f"{stats_data.get('generation_cache', {}).get('hit_rate', 0):.1%}", 
                        className="text-success")
            ], className="text-center"),
            
            html.Div([
                html.H6("Active Workers"),
                html.H4(f"{health.get('worker_count', 0)}", className="text-info")
            ], className="text-center")
        ]
        
        return html.Div(metrics)
    
    def _create_cache_plot(self, stats: Dict[str, Any]):
        """Create cache performance plot."""
        # Placeholder plot - would use real historical data
        x = list(range(10))
        y = np.random.uniform(0.7, 0.9, 10)  # Simulated cache hit rates
        
        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'))
        fig.update_layout(
            title="Cache Hit Rate Over Time",
            xaxis_title="Time",
            yaxis_title="Hit Rate",
            height=300
        )
        return fig
    
    def _create_response_plot(self, stats: Dict[str, Any]):
        """Create response time plot."""
        # Placeholder plot - would use real historical data
        x = list(range(10))
        y = np.random.uniform(0.1, 2.0, 10)  # Simulated response times
        
        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'))
        fig.update_layout(
            title="Response Times",
            xaxis_title="Time",
            yaxis_title="Response Time (s)",
            height=300
        )
        return fig
    
    def _format_health_status(self, health: Dict[str, Any]):
        """Format system health status."""
        status = health.get("status", "unknown")
        color = {
            "healthy": "success",
            "degraded": "warning", 
            "unhealthy": "danger"
        }.get(status, "secondary")
        
        return html.Div([
            html.H5([
                "System Status: ",
                html.Span(status.upper(), className=f"badge bg-{color}")
            ]),
            html.P(f"Response Time: {health.get('response_time', 0):.2f}s"),
            html.P(f"Cache Enabled: {'Yes' if health.get('cache_enabled') else 'No'}")
        ])
    
    def run(self, host: str = "0.0.0.0", port: int = 8050, debug: bool = False):
        """Run the dashboard server."""
        logger.info(f"Starting OdorDiff-2 Dashboard on http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


def create_dashboard(api_url: str = "http://localhost:8000") -> OdorDiffDashboard:
    """Create and return dashboard instance."""
    return OdorDiffDashboard(api_url=api_url)


if __name__ == "__main__":
    dashboard = create_dashboard()
    dashboard.run(debug=True)