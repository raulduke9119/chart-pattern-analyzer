#!/usr/bin/env python3
"""
Binance Data Provider for Chart Pattern Analyzer
Provides real-time and historical candlestick data from Binance API
"""

import os
import logging
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("binance_data.log"),
        logging.StreamHandler()
    ]
)

class BinanceDataProvider:
    """
    Provides real-time and historical candlestick data from Binance API
    for chart pattern analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize the Binance Data Provider.
        
        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load API credentials from environment or parameters
        self.api_key = api_key or os.environ.get("BINANCE_API_KEY")
        self.api_secret = api_secret or os.environ.get("BINANCE_API_SECRET")
        
        # Base URLs for API requests
        self.base_url = "https://api.binance.com/api/v3"
        self.output_dir = "chart_data"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define timeframe mappings
        self.timeframes = {
            "1m": {"interval": "1m", "limit": 1000, "description": "1 minute"},
            "5m": {"interval": "5m", "limit": 1000, "description": "5 minutes"},
            "15m": {"interval": "15m", "limit": 1000, "description": "15 minutes"},
            "30m": {"interval": "30m", "limit": 1000, "description": "30 minutes"},
            "1h": {"interval": "1h", "limit": 1000, "description": "1 hour"},
            "2h": {"interval": "2h", "limit": 1000, "description": "2 hours"},
            "4h": {"interval": "4h", "limit": 1000, "description": "4 hours"},
            "1d": {"interval": "1d", "limit": 1000, "description": "1 day"},
            "1w": {"interval": "1w", "limit": 1000, "description": "1 week"},
            "1M": {"interval": "1M", "limit": 1000, "description": "1 month"}
        }
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a request to the Binance API.
        
        Args:
            endpoint: API endpoint to query
            params: Query parameters
            
        Returns:
            JSON response data
        """
        url = f"{self.base_url}/{endpoint}"
        headers = {}
        
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()  # Raise exception for 4XX/5XX status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            if hasattr(response, 'text'):
                self.logger.error(f"Response: {response.text}")
            raise
    
    def get_historical_klines(self, symbol: str, interval: str, 
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None,
                             limit: int = 1000) -> pd.DataFrame:
        """
        Get historical candlestick data from Binance.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Candlestick interval (e.g., "1h", "4h", "1d")
            start_time: Start time for historical data
            end_time: End time for historical data
            limit: Maximum number of candlesticks to retrieve (max 1000)
            
        Returns:
            DataFrame with candlestick data
        """
        # Validate interval
        if interval not in self.timeframes:
            valid_intervals = ", ".join(self.timeframes.keys())
            self.logger.error(f"Invalid interval: {interval}. Valid intervals: {valid_intervals}")
            raise ValueError(f"Invalid interval: {interval}. Valid intervals: {valid_intervals}")
        
        # Prepare parameters
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000)  # Binance limit is 1000
        }
        
        # Add start/end times if provided
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        
        try:
            # Make API request
            self.logger.info(f"Fetching {interval} candlestick data for {symbol}")
            data = self._make_request("klines", params)
            
            # Convert to DataFrame
            columns = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore"
            ]
            
            df = pd.DataFrame(data, columns=columns)
            
            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            
            for col in ["open", "high", "low", "close", "volume", 
                        "quote_asset_volume", "taker_buy_base_asset_volume", 
                        "taker_buy_quote_asset_volume"]:
                df[col] = df[col].astype(float)
            
            df["number_of_trades"] = df["number_of_trades"].astype(int)
            
            # Drop the "ignore" column
            df = df.drop("ignore", axis=1)
            
            # Set timestamp as index
            df = df.set_index("timestamp")
            
            self.logger.info(f"Retrieved {len(df)} candlesticks for {symbol} {interval}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving historical klines: {e}")
            raise
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            
        Returns:
            Latest price as float
        """
        try:
            params = {"symbol": symbol.upper()}
            data = self._make_request("ticker/price", params)
            return float(data["price"])
        except Exception as e:
            self.logger.error(f"Error retrieving latest price: {e}")
            raise
    
    def get_exchange_info(self) -> Dict:
        """
        Get exchange information (symbols, filters, rules, etc.).
        
        Returns:
            Dictionary with exchange information
        """
        try:
            return self._make_request("exchangeInfo")
        except Exception as e:
            self.logger.error(f"Error retrieving exchange info: {e}")
            raise
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading pairs.
        
        Returns:
            List of trading pair symbols
        """
        try:
            exchange_info = self.get_exchange_info()
            return [s["symbol"] for s in exchange_info["symbols"] if s["status"] == "TRADING"]
        except Exception as e:
            self.logger.error(f"Error retrieving available symbols: {e}")
            return []
    
    def save_chart_image(self, symbol: str, interval: str, 
                        lookback_periods: int = 100,
                        width: int = 1200, height: int = 800,
                        include_volume: bool = True,
                        output_path: Optional[str] = None) -> str:
        """
        Generate and save a chart image for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Candlestick interval (e.g., "1h", "4h", "1d")
            lookback_periods: Number of periods to include in the chart
            width: Image width in pixels
            height: Image height in pixels
            include_volume: Whether to include volume bars
            output_path: Path to save the image, or None for auto-generation
            
        Returns:
            Path to the saved image
        """
        # Calculate start time based on lookback periods
        end_time = datetime.now()
        
        # Rough estimation of start time based on interval
        interval_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "2h": timedelta(hours=2),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
            "1w": timedelta(weeks=1),
            "1M": timedelta(days=30)
        }
        
        start_time = end_time - (interval_map.get(interval, timedelta(days=1)) * lookback_periods * 1.1)
        
        # Get historical data
        df = self.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=lookback_periods
        )
        
        if df.empty:
            self.logger.error(f"No data retrieved for {symbol} {interval}")
            raise ValueError(f"No data retrieved for {symbol} {interval}")
        
        # Generate chart
        self.logger.info(f"Generating chart for {symbol} {interval}")
        
        # Set up figure and axes
        fig_height = height / 100  # Convert to inches
        fig_width = width / 100
        
        if include_volume:
            # Create figure with two subplots (price and volume)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height), 
                                         gridspec_kw={'height_ratios': [3, 1]},
                                         sharex=True)
            # Hide the space between subplots
            fig.subplots_adjust(hspace=0)
        else:
            # Create figure with just price plot
            fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Plot candlesticks
        df.reset_index(inplace=True)  # Temporarily reset index for plotting
        
        # Plot candlesticks
        up = df[df.close >= df.open]
        down = df[df.close < df.open]
        
        # Up candles (green)
        ax1.bar(up.timestamp, up.high - up.low, width=0.6, bottom=up.low, color='green', alpha=0.5)
        ax1.bar(up.timestamp, up.close - up.open, width=0.8, bottom=up.open, color='green')
        
        # Down candles (red)
        ax1.bar(down.timestamp, down.high - down.low, width=0.6, bottom=down.low, color='red', alpha=0.5)
        ax1.bar(down.timestamp, down.open - down.close, width=0.8, bottom=down.close, color='red')
        
        # Plot volume if requested
        if include_volume:
            # Volume bars
            ax2.bar(up.timestamp, up.volume, color='green', alpha=0.5, width=0.8)
            ax2.bar(down.timestamp, down.volume, color='red', alpha=0.5, width=0.8)
            ax2.set_ylabel('Volume')
            # Format y-axis with K, M suffixes
            ax2.yaxis.set_major_formatter(lambda x, pos: f'{x/1000:.0f}K' if x < 1000000 else f'{x/1000000:.1f}M')
        
        # Add grid to main chart
        ax1.grid(True, alpha=0.3)
        
        # Add title and labels
        title = f"{symbol} {interval} Chart"
        ax1.set_title(title, fontsize=14, color='black')
        ax1.set_ylabel('Price')
        
        # Rotate x-axis labels for better readability
        fig.autofmt_xdate()
        
        # Add latest price annotation
        last_price = df.iloc[-1].close
        ax1.axhline(y=last_price, color='blue', linestyle='--', alpha=0.7)
        ax1.text(df.iloc[-1].timestamp, last_price, f' {last_price:.2f}', 
                verticalalignment='bottom', horizontalalignment='left',
                color='blue', fontweight='bold')
        
        # Format y-axis with comma separators for thousands
        ax1.yaxis.set_major_formatter(lambda x, pos: f'{x:,.2f}')
        
        # Add timestamp
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        plt.figtext(0.01, 0.01, f"Generated: {timestamp_str}", fontsize=8)
        
        # Create output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"{symbol}_{interval}_{timestamp}.png")
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Chart saved to {output_path}")
        return output_path
    
    def start_continuous_monitoring(self, symbol: str, interval: str, 
                                  callback=None, interval_seconds: int = 60):
        """
        Start continuous monitoring of a symbol at specified interval.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Candlestick interval (e.g., "1h", "4h", "1d")
            callback: Function to call with new data
            interval_seconds: Seconds between updates
        """
        import threading
        
        def monitor_loop():
            self.logger.info(f"Starting continuous monitoring of {symbol} {interval}")
            
            # Track the last candle we've seen
            last_candle_time = None
            
            while True:
                try:
                    # Get the most recent candles
                    df = self.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=5  # Just get a few recent candles
                    )
                    
                    if not df.empty:
                        # Get the most recent complete candle
                        latest_candle = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
                        latest_time = latest_candle.name
                        
                        # Check if this is a new candle
                        if last_candle_time is None or latest_time > last_candle_time:
                            self.logger.info(f"New {interval} candle for {symbol}: {latest_time}")
                            
                            # Update the last candle time
                            last_candle_time = latest_time
                            
                            # Generate and save chart
                            chart_path = self.save_chart_image(symbol, interval)
                            
                            # Call the callback function if provided
                            if callback:
                                candle_data = {
                                    'symbol': symbol,
                                    'interval': interval,
                                    'time': latest_time,
                                    'open': latest_candle['open'],
                                    'high': latest_candle['high'],
                                    'low': latest_candle['low'],
                                    'close': latest_candle['close'],
                                    'volume': latest_candle['volume'],
                                    'chart_path': chart_path
                                }
                                callback(candle_data)
                    
                    # Sleep until next check
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    # Sleep and try again
                    time.sleep(interval_seconds)
        
        # Start the monitoring in a background thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        return monitor_thread
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate common technical indicators on the price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Simple Moving Averages
        result['SMA_20'] = result['close'].rolling(window=20).mean()
        result['SMA_50'] = result['close'].rolling(window=50).mean()
        result['SMA_200'] = result['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        result['EMA_12'] = result['close'].ewm(span=12, adjust=False).mean()
        result['EMA_26'] = result['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        result['MACD'] = result['EMA_12'] - result['EMA_26']
        result['MACD_Signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
        result['MACD_Hist'] = result['MACD'] - result['MACD_Signal']
        
        # Relative Strength Index (RSI)
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        result['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        result['BB_Middle'] = result['close'].rolling(window=20).mean()
        result['BB_StdDev'] = result['close'].rolling(window=20).std()
        result['BB_Upper'] = result['BB_Middle'] + (result['BB_StdDev'] * 2)
        result['BB_Lower'] = result['BB_Middle'] - (result['BB_StdDev'] * 2)
        
        # Average True Range (ATR)
        high_low = result['high'] - result['low']
        high_close = (result['high'] - result['close'].shift()).abs()
        low_close = (result['low'] - result['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        result['ATR'] = true_range.rolling(14).mean()
        
        return result

    def detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Detect common candlestick patterns in the price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping pattern names to lists of indices where they appear
        """
        patterns = {}
        
        # Make sure index is reset for easier indexing
        df = df.reset_index() if df.index.name is not None else df.copy()
        
        # Doji pattern (open and close are very close)
        doji_threshold = 0.001  # 0.1% difference between open and close
        doji_indices = []
        
        for i in range(len(df)):
            body_size = abs(df.loc[i, 'close'] - df.loc[i, 'open'])
            price = df.loc[i, 'close']
            
            if body_size / price < doji_threshold:
                doji_indices.append(i)
        
        if doji_indices:
            patterns['doji'] = doji_indices
        
        # Hammer pattern
        # Small real body near the top, long lower shadow, little or no upper shadow
        hammer_indices = []
        
        for i in range(len(df)):
            if i == 0:
                continue
                
            body_size = abs(df.loc[i, 'close'] - df.loc[i, 'open'])
            total_range = df.loc[i, 'high'] - df.loc[i, 'low']
            
            if total_range == 0:  # Avoid division by zero
                continue
                
            # For hammer, the body should be in the upper third
            body_position = (min(df.loc[i, 'open'], df.loc[i, 'close']) - df.loc[i, 'low']) / total_range
            
            # Lower shadow should be at least twice the body size
            lower_shadow = min(df.loc[i, 'open'], df.loc[i, 'close']) - df.loc[i, 'low']
            
            # Upper shadow should be small
            upper_shadow = df.loc[i, 'high'] - max(df.loc[i, 'open'], df.loc[i, 'close'])
            
            if (body_position > 0.6 and 
                lower_shadow > 2 * body_size and 
                upper_shadow < 0.1 * total_range):
                hammer_indices.append(i)
        
        if hammer_indices:
            patterns['hammer'] = hammer_indices
        
        # Engulfing patterns
        bullish_engulfing = []
        bearish_engulfing = []
        
        for i in range(1, len(df)):
            # Previous candle
            prev_open = df.loc[i-1, 'open']
            prev_close = df.loc[i-1, 'close']
            prev_body_size = abs(prev_close - prev_open)
            
            # Current candle
            curr_open = df.loc[i, 'open']
            curr_close = df.loc[i, 'close']
            curr_body_size = abs(curr_close - curr_open)
            
            # Bullish engulfing
            if (prev_close < prev_open and  # Previous candle is bearish
                curr_close > curr_open and  # Current candle is bullish
                curr_open < prev_close and  # Current open is below previous close
                curr_close > prev_open and  # Current close is above previous open
                curr_body_size > prev_body_size):  # Current body engulfs previous
                bullish_engulfing.append(i)
            
            # Bearish engulfing
            if (prev_close > prev_open and  # Previous candle is bullish
                curr_close < curr_open and  # Current candle is bearish
                curr_open > prev_close and  # Current open is above previous close
                curr_close < prev_open and  # Current close is below previous open
                curr_body_size > prev_body_size):  # Current body engulfs previous
                bearish_engulfing.append(i)
        
        if bullish_engulfing:
            patterns['bullish_engulfing'] = bullish_engulfing
        
        if bearish_engulfing:
            patterns['bearish_engulfing'] = bearish_engulfing
        
        # More patterns could be added here...
        
        return patterns


if __name__ == "__main__":
    # Example usage
    provider = BinanceDataProvider()
    
    # Example: Get and save a chart
    try:
        symbol = "BTCUSDT"
        interval = "1d"
        
        # Check if the symbol is available
        available_symbols = provider.get_available_symbols()
        if symbol in available_symbols:
            print(f"Getting data for {symbol} at {interval} interval")
            
            # Generate and save chart
            chart_path = provider.save_chart_image(symbol, interval)
            print(f"Chart saved to: {chart_path}")
            
            # Get and analyze data
            df = provider.get_historical_klines(symbol, interval, limit=200)
            df_with_indicators = provider.calculate_technical_indicators(df)
            
            # Detect candlestick patterns
            patterns = provider.detect_candlestick_patterns(df_with_indicators)
            
            print(f"Detected patterns: {patterns.keys()}")
            print(f"Latest price: {provider.get_latest_price(symbol)}")
        else:
            print(f"Symbol {symbol} not available. Available symbols include: {available_symbols[:5]}...")
    
    except Exception as e:
        print(f"Error in example: {e}")