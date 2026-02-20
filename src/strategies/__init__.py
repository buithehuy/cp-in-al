"""Acquisition strategies package."""
from .base import AcquisitionStrategy
from .uncertainty import (
    RandomSampling,
    EntropySampling,
    LeastConfidenceSampling,
    MarginSampling
)
from .conformal import (
    CPSizeSampling,
    CPVShapedSampling,
    CPVShapedEntropySampling,
    CPAPSSampling,
    RMCPSampling,
    CombinedSampling,
    CombinedVShapedSampling,
    ConformalBoundaryUncertaintySampling,
    CPSetPartitionMISampling,
    CPDiversityAPSSampling,
    CPAPSSPMISampling,
    CPWiseSampling,
    CPRelativeMarginSampling,
)

# Strategy registry
STRATEGIES = {
    "random": RandomSampling,
    "entropy": EntropySampling,
    "least_confidence": LeastConfidenceSampling,
    "margin": MarginSampling,
    "cp_size": CPSizeSampling,
    "cp_v_shaped": CPVShapedSampling,
    "cp_v_shaped_entropy": CPVShapedEntropySampling,
    "cp_aps": CPAPSSampling,
    "cp_rmcp": RMCPSampling,
    "combined": CombinedSampling,
    "combined_v_shaped": CombinedVShapedSampling,
    "cp_boundary_uncertainty": ConformalBoundaryUncertaintySampling,
    "cp_spm_info": CPSetPartitionMISampling,
    "cp_diversity": CPDiversityAPSSampling,
    "cp_aps_spm": CPAPSSPMISampling,
    "cp_wise": CPWiseSampling,
    "cp_rel_margin": CPRelativeMarginSampling,
}


def get_strategy(name):
    """Get acquisition strategy by name.
    
    Args:
        name: Strategy name
        
    Returns:
        Instantiated strategy object
        
    Raises:
        ValueError: If strategy name is not recognized
    """
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]()


__all__ = [
    'AcquisitionStrategy',
    'RandomSampling',
    'EntropySampling',
    'LeastConfidenceSampling',
    'MarginSampling',
    'CPSizeSampling',
    'CPVShapedSampling',
    'CPVShapedEntropySampling',
    'CPAPSSampling',
    'RMCPSampling',
    'CombinedSampling',
    'CombinedVShapedSampling',
    'ConformalBoundaryUncertaintySampling',
    'CPSetPartitionMISampling',
    'CPDiversityAPSSampling',
    'CPAPSSPMISampling',
    'CPWiseSampling',
    'CPRelativeMarginSampling',
    'STRATEGIES',
    'get_strategy',
]
