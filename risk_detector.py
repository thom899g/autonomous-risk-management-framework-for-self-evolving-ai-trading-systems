"""
Real-time risk detection module for trading systems.
Detects volatility spikes, concentration risks, correlation breakdowns, and liquidity risks.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import datetime

class RiskType(Enum):
    VOLATILITY_SPIKE = "volatility_spike"
    CONCENTRATION = "concentration"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    MAX_DRAWDOWN = "max_drawdown"
    LEVERAGE_EXCEEDED = "leverage_exceeded"
    BLACK_SWAN = "black_swan"

@dataclass
class DetectedRisk:
    risk_type: RiskType
    severity: float  # 0-1 scale
    confidence: float  # 0-1 scale
    assets: List[str]
    indicators: Dict[str, float]
    timestamp: datetime.datetime
    description: str

class RiskDetector:
    """Main risk detection engine with multiple detection algorithms."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._default_config()
        
        # Initialize detection thresholds
        self.volatility_threshold = self.config.get('volatility_threshold', 0.05)
        self.concentration_limit = self.config.get('concentration_limit', 0.3)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        
        # Detection history for adaptive learning
        self.detection_history: List[DetectedRisk] = []
        
        self.logger.info("RiskDetector initialized with adaptive thresholds")
    
    def _default_config(self) -> Dict:
        """Provide safe default configuration."""
        return {
            'volatility_threshold': 0.05,
            'concentration_limit': 0.3,
            'correlation_threshold': 0.7,
            'lookback_period': 20,
            'min_confidence': 0.6
        }
    
    def detect_volatility_spike(self, price_series: pd.Series, 
                               window: int = 20) -> Optional[DetectedRisk]:
        """Detect abnormal volatility increases using