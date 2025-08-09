#!/usr/bin/env python3
"""
Example integration of enhanced observability features with OdorDiff-2 API.

This example demonstrates:
1. Structured logging with correlation IDs
2. Distributed tracing for request flows
3. Performance monitoring and metrics
4. Error tracking and context
"""

import asyncio
import time
from typing import List, Dict, Any

# Import OdorDiff-2 components
from odordiff2.utils.logging import (
    configure_observability, 
    get_logger, 
    with_correlation_context,
    trace_function,
    log_function_call,
    CorrelationContext,
    TraceSpan
)
from odordiff2.api.endpoints import MoleculeGenerator
from odordiff2.safety.assessment import SafetyAssessment


class ObservabilityDemo:
    """Demonstration of observability features in OdorDiff-2."""
    
    def __init__(self):
        # Configure comprehensive observability
        self.logger = configure_observability(
            service_name="odordiff2-demo",
            log_level="DEBUG",
            structured_logging=True,
            enable_tracing=True,
            log_directory="logs"
        )
        
        # Initialize components
        self.generator = MoleculeGenerator()
        self.safety = SafetyAssessment()
    
    @trace_function(get_logger(), "molecule_generation_pipeline")
    async def generate_molecules_with_tracing(
        self, 
        prompt: str, 
        num_molecules: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate molecules with full observability."""
        
        # Create correlation context for this request
        with with_correlation_context() as ctx:
            self.logger.info(
                "Starting molecule generation pipeline",
                prompt=prompt,
                num_molecules=num_molecules,
                correlation_id=ctx.correlation_id
            )
            
            results = []
            
            # Step 1: Generation with tracing
            with self.logger.start_span("molecule_generation") as gen_span:
                gen_span.add_tag("prompt", prompt)
                gen_span.add_tag("num_molecules", num_molecules)
                
                self.logger.log_span_event("Starting molecule generation")
                
                try:
                    molecules = await self._generate_molecules(prompt, num_molecules)
                    gen_span.add_tag("generated_count", len(molecules))
                    self.logger.log_span_event(f"Generated {len(molecules)} molecules")
                    
                except Exception as e:
                    gen_span.set_error(e)
                    self.logger.error("Generation failed", error=str(e))
                    raise
            
            # Step 2: Safety assessment with tracing
            safe_molecules = []
            with self.logger.start_span("safety_assessment") as safety_span:
                safety_span.add_tag("input_count", len(molecules))
                
                for i, molecule in enumerate(molecules):
                    with self.logger.start_span(f"assess_molecule_{i}") as mol_span:
                        mol_span.add_tag("molecule_index", i)
                        mol_span.add_tag("smiles", molecule.get("smiles", ""))
                        
                        try:
                            safety_result = await self._assess_safety(molecule)
                            
                            if safety_result.get("safe", False):
                                safe_molecules.append({
                                    **molecule,
                                    "safety": safety_result
                                })
                                mol_span.add_tag("safety_result", "safe")
                            else:
                                mol_span.add_tag("safety_result", "unsafe")
                                
                        except Exception as e:
                            mol_span.set_error(e)
                            self.logger.error(f"Safety assessment failed for molecule {i}", error=str(e))
                
                safety_span.add_tag("safe_molecules", len(safe_molecules))
                self.logger.log_span_event(f"Safety assessment complete: {len(safe_molecules)} safe molecules")
            
            # Log final results
            self.logger.info(
                "Generation pipeline completed",
                total_generated=len(molecules),
                safe_molecules=len(safe_molecules),
                success_rate=len(safe_molecules) / len(molecules) if molecules else 0
            )
            
            return safe_molecules
    
    @log_function_call(get_logger(), enable_tracing=True)
    async def _generate_molecules(self, prompt: str, num_molecules: int) -> List[Dict[str, Any]]:
        """Generate molecules with function call logging."""
        self.logger.add_span_tag("operation", "generation")
        
        # Simulate generation with some processing time
        await asyncio.sleep(0.1 * num_molecules)  # Simulate work
        
        # Mock molecule generation
        molecules = []
        for i in range(num_molecules):
            molecule = {
                "smiles": f"C{i+1}C=CC=C", # Mock SMILES
                "id": f"mol_{i}",
                "properties": {
                    "molecular_weight": 100 + i * 10,
                    "logP": 2.5 + i * 0.1
                }
            }
            molecules.append(molecule)
            
            self.logger.debug(f"Generated molecule {i+1}", molecule_id=molecule["id"])
        
        return molecules
    
    @log_function_call(get_logger(), enable_tracing=True)
    async def _assess_safety(self, molecule: Dict[str, Any]) -> Dict[str, Any]:
        """Assess molecular safety with logging."""
        self.logger.add_span_tag("molecule_id", molecule.get("id", "unknown"))
        
        # Simulate safety assessment
        await asyncio.sleep(0.05)  # Simulate work
        
        # Mock safety assessment
        safety_score = 0.8  # Mock score
        is_safe = safety_score > 0.5
        
        result = {
            "safe": is_safe,
            "toxicity_score": 1.0 - safety_score,
            "ifra_compliant": is_safe,
            "regulatory_flags": [] if is_safe else ["potential_allergen"]
        }
        
        self.logger.debug(
            "Safety assessment completed",
            molecule_id=molecule.get("id"),
            safety_score=safety_score,
            is_safe=is_safe
        )
        
        return result
    
    async def demonstrate_error_handling(self):
        """Demonstrate error handling with observability."""
        with with_correlation_context() as ctx:
            try:
                with self.logger.start_span("error_demo") as span:
                    span.add_tag("demo_type", "error_handling")
                    
                    self.logger.info("Demonstrating error handling")
                    
                    # Simulate an error
                    raise ValueError("This is a demonstration error")
                    
            except Exception as e:
                self.logger.log_error_with_context(
                    e,
                    {
                        "demo": True,
                        "correlation_id": ctx.correlation_id,
                        "operation": "error_demo"
                    }
                )
    
    def demonstrate_metrics(self):
        """Demonstrate metrics recording."""
        # Record some example metrics
        self.logger.record_metric("api_requests_total", 1.0, {"endpoint": "/generate"})
        self.logger.record_metric("response_time_seconds", 0.5, {"endpoint": "/generate"})
        self.logger.record_metric("molecules_generated_total", 5.0)
        
        # Get metrics summary
        metrics_summary = self.logger.get_metrics_summary()
        self.logger.info("Metrics recorded", metrics=metrics_summary)
    
    def demonstrate_tracing_stats(self):
        """Demonstrate tracing statistics."""
        stats = self.logger.get_tracing_stats()
        self.logger.info("Tracing statistics", stats=stats)


async def main():
    """Run the observability demonstration."""
    demo = ObservabilityDemo()
    
    print("=== OdorDiff-2 Observability Demo ===")
    
    # 1. Demonstrate traced molecule generation
    print("\n1. Running traced molecule generation...")
    molecules = await demo.generate_molecules_with_tracing(
        prompt="vanilla fragrance",
        num_molecules=3
    )
    print(f"Generated {len(molecules)} safe molecules")
    
    # 2. Demonstrate error handling
    print("\n2. Demonstrating error handling...")
    await demo.demonstrate_error_handling()
    
    # 3. Demonstrate metrics
    print("\n3. Recording metrics...")
    demo.demonstrate_metrics()
    
    # 4. Show tracing statistics
    print("\n4. Tracing statistics...")
    demo.demonstrate_tracing_stats()
    
    # 5. Performance monitoring example
    print("\n5. Starting performance monitoring...")
    from odordiff2.utils.logging import PerformanceMonitor
    
    monitor = PerformanceMonitor(demo.logger)
    monitor.start_monitoring(interval=10)  # Monitor every 10 seconds
    
    print("Performance monitoring started. Check logs for system metrics.")
    
    print("\n=== Demo completed ===")
    print("Check the following log files:")
    print("- logs/odordiff2-demo.log (traditional logs)")
    print("- logs/odordiff2-demo_structured.jsonl (structured JSON logs)")
    print("- logs/odordiff2-demo_errors.log (error logs)")


if __name__ == "__main__":
    asyncio.run(main())