"""
Evaluation Module
Advanced evaluation with llama_index and RAGAS
"""

from .base_evaluator import (
    BaseCustomEvaluator,
    LlamaIndexEvaluator,
    RAGASEvaluator,
    HybridEvaluator,
    EvaluationManager,
    EvaluationConfig,
    EvaluationResult,
    BatchEvaluationResult,
    EvaluationMetric,
    EvaluationLevel,
    create_llama_index_evaluator,
    create_ragas_evaluator,
    create_hybrid_evaluator,
    RAGAS_AVAILABLE
)

__all__ = [
    # Base classes
    "BaseCustomEvaluator",
    
    # Evaluators
    "LlamaIndexEvaluator",
    "RAGASEvaluator",
    "HybridEvaluator",
    
    # Management
    "EvaluationManager",
    
    # Data types
    "EvaluationConfig",
    "EvaluationResult",
    "BatchEvaluationResult",
    "EvaluationMetric",
    "EvaluationLevel",
    
    # Utility functions
    "create_llama_index_evaluator",
    "create_ragas_evaluator",
    "create_hybrid_evaluator",
    
    # Constants
    "RAGAS_AVAILABLE"
]