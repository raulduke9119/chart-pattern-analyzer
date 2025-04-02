# Chart Pattern Analyzer

A comprehensive trading analysis application that uses computer vision and machine learning to identify optimal trading patterns in financial charts.

## Features

- **Image-Based Pattern Detection**: Identify common chart patterns using machine learning
- **Live Market Monitoring**: Connect to Binance for real-time chart data
- **AI-Powered Analysis**: Integrate with OpenRouter Vision models for advanced chart insights
- **Multi-Timeframe Analysis**: Compare patterns across different timeframes
- **Pattern Visualization**: Annotate charts with detected patterns and entry/exit points
- **Historical Pattern Storage**: Keep track of detected patterns for future reference
- **Alert Notifications**: Get notified about newly detected patterns
- **Command-Line Interface**: Powerful CLI for automation and scripting
- **Web Interface**: User-friendly web UI for chart analysis

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/chart-pattern-analyzer.git
   cd chart-pattern-analyzer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up API keys (optional, for full functionality):
   ```
   export OPENROUTER_API_KEY="your_openrouter_api_key"
   export BINANCE_API_KEY="your_binance_api_key"
   export BINANCE_API_SECRET="your_binance_api_secret"
   ```

## Usage

### Command Line Interface

The application provides a powerful CLI for chart analysis:

```
python -m chart_pattern_analyzer.chart_cli analyze --image chart.png
python -m chart_pattern_analyzer.chart_cli monitor --interval 60
python -m chart_pattern_analyzer.chart_cli patterns --list
```

### Web Interface

To start the web interface:

```
python -m chart_pattern_analyzer.server
```

Then open your browser and navigate to `http://localhost:5000`.

### Live Market Analysis

To start monitoring live market data from Binance:

```python
from chart_pattern_analyzer import LiveMarketAnalyzer

# Initialize analyzer
analyzer = LiveMarketAnalyzer(
    openrouter_api_key="your_openrouter_key",
    binance_api_key="your_binance_key",
    binance_api_secret="your_binance_secret"
)

# Start monitoring specific symbols and timeframes
analyzer.start_monitoring(
    symbols=["BTCUSDT", "ETHUSDT"],
    timeframes=["1h", "4h", "1d"],
    update_interval=300  # Check every 5 minutes
)
```

### OpenRouter Vision Analysis

The OpenRouter Vision integration provides AI-powered chart analysis:

```python
from chart_pattern_analyzer import OpenRouterVisionAnalyzer

# Initialize analyzer
analyzer = OpenRouterVisionAnalyzer(api_key="your_openrouter_key")

# Analyze a chart image
result = analyzer.analyze_image("path/to/chart.png", timeframe="Daily")

# Generate a summary report
report = analyzer.generate_summary_report(result)
print(report)
```

### Combined Vision and Traditional Analysis

The Vision Integration module provides combined analysis using both OpenRouter Vision and traditional ML models:

```python
from chart_pattern_analyzer import VisionIntegration

# Initialize integration module
integrator = VisionIntegration(
    api_key="your_openrouter_key",
    model_path="models/pattern_detector.model"
)

# Analyze a chart using both methods
result = integrator.analyze_chart("path/to/chart.png", timeframe="Daily")

# Access the combined confidence scores
print(result["confidence_scores"])

# View the report path
print(f"Report generated at: {result['report_files']['markdown']}")

# Batch analyze a directory of charts
batch_results = integrator.batch_analyze_directory(
    image_dir="charts/",
    pattern_filter="Double Bottom"  # Optional filter
)
```

### Binance Data Provider

Get real-time and historical data from Binance:

```python
from chart_pattern_analyzer import BinanceDataProvider

# Initialize data provider
provider = BinanceDataProvider()

# Get historical candlestick data
df = provider.get_historical_klines("BTCUSDT", "1h", limit=100)

# Save a chart image
chart_path = provider.save_chart_image("BTCUSDT", "1h")

# Calculate technical indicators
df_with_indicators = provider.calculate_technical_indicators(df)

# Detect candlestick patterns
patterns = provider.detect_candlestick_patterns(df)
```

## Configuration

The application can be configured through various settings:

- Database settings in `chart_patterns.db`
- Model settings in `models/pattern_detector.model`
- Output directory for charts and analysis in `output/`
- Vision model settings in environment variables

## Requirements

- Python 3.7+
- TensorFlow 2.5+
- OpenCV 4.5+
- Pandas 1.3+
- Matplotlib 3.4+
- NumPy 1.20+
- Seaborn 0.11+
- OpenRouter API key (for AI vision analysis)
- Binance API key (for live market data)

## Project Structure

```
chart_pattern_analyzer/
├── __init__.py                # Package initialization
├── chart_analyzer.py          # Core analysis engine
├── visualization.py           # Chart visualization components
├── openrouter_vision_model.py # OpenRouter Vision integration
├── vision_integration.py      # Combined vision and traditional analysis
├── binance_data_provider.py   # Binance data integration
├── live_market_analyzer.py    # Real-time market monitoring
├── chart_cli.py               # Command-line interface
├── server.py                  # Web server
└── web/                       # Web interface files
    ├── index.html             # Main web page
    └── api.js                 # API integration
```

## Key Components

### Chart Analyzer
The core engine that processes chart images using traditional computer vision and ML techniques to detect patterns.

### OpenRouter Vision Analyzer
Leverages OpenRouter's Vision API to provide AI-powered analysis of chart images, including pattern recognition, trend analysis, and trading recommendations.

### Vision Integration
Combines results from both traditional ML models and AI vision models to provide comprehensive analysis with enhanced confidence scoring.

### Visualization Module
Tools for annotating charts, highlighting patterns, and creating visual representations of analysis results.

### Binance Data Provider
Connects to the Binance API to fetch real-time and historical market data for analysis.

### Web and CLI Interfaces
User-friendly interfaces for interacting with the analysis engine.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.