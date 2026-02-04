"""
Pipelines module - Record Linkage e Entity Matching

Contiene:
- dedupe.py: Deduplication con algoritmi tradizionali
- record_linkage.py: Record Linkage con Logistic Regression
- ditto_pipeline.py: Entity Matching con Transformer (auto-ottimizzato)
"""

from .record_linkage import load_data, setup_comparator

# Ditto imports (opzionali - richiedono torch)
try:
    from .ditto_pipeline import (
        DittoConfig,
        DittoTrainer,
        DittoMatcher,
        DittoDataset,
        DittoDataAugmentor,
        VehicleDomainKnowledge,
        TextSummarizer,
        get_gpu_info
    )
    DITTO_AVAILABLE = True
except ImportError:
    DITTO_AVAILABLE = False

__all__ = [
    'load_data',
    'setup_comparator',
    'DITTO_AVAILABLE'
]

if DITTO_AVAILABLE:
    __all__.extend([
        'DittoConfig',
        'DittoTrainer', 
        'DittoMatcher',
        'DittoDataset',
        'DittoDataAugmentor',
        'VehicleDomainKnowledge',
        'TextSummarizer',
        'get_gpu_info'
    ])
