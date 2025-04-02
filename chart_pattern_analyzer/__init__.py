"""
Chart Pattern Analyzer
A comprehensive trading analysis application that uses computer vision 
and machine learning to detect optimal trading patterns in financial charts.
"""

from .chart_analyzer import ChartAnalyzer, ChartPatternDatabase, ChartImageProcessor, ChartPatternModel, AlertManager
from .visualization import ChartAnnotator, ChartVisualizer
from .binance_data_provider import BinanceDataProvider
from .openrouter_vision_model import OpenRouterVisionAnalyzer
from .vision_integration import VisionIntegration
from .live_market_analyzer import LiveMarketAnalyzer

__version__ = "0.2.0"
__author__ = "Chart Pattern Analyzer Team"