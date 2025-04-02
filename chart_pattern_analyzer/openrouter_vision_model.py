#!/usr/bin/env python3
"""
OpenRouter Vision Model Analyzer for Chart Pattern Recognition
Integrates with OpenRouter's Vision API to analyze chart images
and generate structured reports on patterns, trends and predictions.
"""

import os
import requests
import json
import logging
import base64
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openrouter_vision.log"),
        logging.StreamHandler()
    ]
)

class OpenRouterVisionAnalyzer:
    """
    Provides integration with OpenRouter's Vision models for analyzing chart patterns
    and generating structured reports with insights, predictions and trend analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "anthropic/claude-3-opus-20240229-vision"):
        """
        Initialize the OpenRouter Vision Model Analyzer.
        
        Args:
            api_key: OpenRouter API key, can be loaded from environment var if None
            model: Vision model identifier to use for analysis
        """
        self.logger = logging.getLogger(__name__)
        
        # Load API key from environment or parameter
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            self.logger.warning("No OpenRouter API key provided. Set OPENROUTER_API_KEY environment variable or pass api_key")
        
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
        # Analysis categories
        self.analysis_categories = [
            "Chart Pattern Recognition",
            "Trend Analysis",
            "Support and Resistance Levels",
            "Volume Analysis",
            "Momentum Indicators",
            "Price Action",
            "Trading Recommendations",
            "Risk Assessment"
        ]
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64 for API transmission.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error encoding image: {e}")
            raise
    
    def _prepare_prompt(self, image_data: str, 
                       timeframe: str = "Daily", 
                       previous_context: Optional[str] = None) -> List[Dict]:
        """
        Prepare the prompt for the vision model with the image and context.
        
        Args:
            image_data: Base64 encoded image data
            timeframe: Chart timeframe (e.g., "Daily", "4h", "1h")
            previous_context: Any previous analysis context to consider
            
        Returns:
            List of message dictionaries for the API request
        """
        system_prompt = f"""You are an expert financial chart analyst.
You'll be given a financial chart image to analyze. 
Perform a comprehensive analysis of this chart and provide structured insights.

Your analysis must include:

1. Chart Pattern Recognition: Identify any technical patterns (e.g., Head and Shoulders, Double Bottom)
2. Trend Analysis: Determine the primary trend and any potential reversals
3. Support and Resistance Levels: Identify key price levels
4. Volume Analysis: If volume is shown, analyze its significance
5. Momentum Indicators: Comment on any visible momentum indicators
6. Price Action: Analyze recent price movements and candle patterns
7. Trading Recommendations: Provide actionable insights with:
   - Entry points
   - Stop loss levels
   - Take profit targets
8. Risk Assessment: Evaluate the risk/reward ratio for potential trades

For each section of your analysis, include a confidence score (1-10).
This chart is on a {timeframe} timeframe.

Format your response as a JSON object with the structure matching the analysis categories above.
Each category should have:
1. "findings": The detailed analysis
2. "confidence": A numeric score from 1-10
3. "summary": A brief 1-sentence summary of findings

Your JSON must be valid and properly escaped."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": f"Please analyze this {timeframe} chart and provide structured insights:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]}
        ]
        
        if previous_context:
            messages.append({
                "role": "user", 
                "content": f"Additionally, consider this previous analysis context: {previous_context}"
            })
            
        return messages
    
    def analyze_image(self, image_path: str, 
                     timeframe: str = "Daily",
                     previous_context: Optional[str] = None,
                     temperature: float = 0.2) -> Dict:
        """
        Submit a chart image to the OpenRouter Vision API and get structured analysis.
        
        Args:
            image_path: Path to the chart image
            timeframe: Chart timeframe
            previous_context: Previous analysis for context
            temperature: Model temperature (lower for more factual responses)
            
        Returns:
            Structured analysis from the vision model
        """
        if not self.api_key:
            return {"error": "No API key provided"}
        
        try:
            # Encode image to base64
            image_data = self._encode_image(image_path)
            
            # Prepare messages for API call
            messages = self._prepare_prompt(image_data, timeframe, previous_context)
            
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4000,
                "response_format": {"type": "json_object"}
            }
            
            # Attempt request with retries
            response_json = None
            for attempt in range(self.max_retries):
                try:
                    self.logger.info(f"Sending request to OpenRouter Vision API (attempt {attempt+1})")
                    response = requests.post(self.api_url, headers=headers, json=payload)
                    response.raise_for_status()
                    response_json = response.json()
                    break
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"API request failed (attempt {attempt+1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        return {"error": f"API request failed after {self.max_retries} attempts: {str(e)}"}
            
            # Process and structure the response
            if response_json and "choices" in response_json:
                try:
                    # Extract the content from the response
                    content = response_json["choices"][0]["message"]["content"]
                    
                    # Parse the JSON content
                    analysis_data = json.loads(content)
                    
                    # Add metadata to the response
                    result = {
                        "analysis": analysis_data,
                        "metadata": {
                            "image_path": image_path,
                            "timeframe": timeframe,
                            "timestamp": datetime.now().isoformat(),
                            "model": self.model
                        }
                    }
                    
                    self.logger.info(f"Successfully analyzed chart image: {image_path}")
                    return result
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse response JSON: {e}")
                    return {
                        "error": "Failed to parse analysis response",
                        "raw_response": content if 'content' in locals() else "No content"
                    }
            else:
                self.logger.error(f"Unexpected API response format: {response_json}")
                return {"error": "Unexpected API response format", "response": response_json}
                
        except Exception as e:
            self.logger.error(f"Error analyzing image: {e}")
            return {"error": str(e)}
    
    def generate_summary_report(self, analysis_result: Dict) -> str:
        """
        Generate a human-readable summary report from the structured analysis.
        
        Args:
            analysis_result: The structured analysis from analyze_image()
            
        Returns:
            A formatted markdown string with the analysis summary
        """
        if "error" in analysis_result:
            return f"## Error in Analysis\n\n{analysis_result['error']}"
        
        if "analysis" not in analysis_result:
            return "## Invalid Analysis Result\n\nNo analysis data found."
        
        analysis = analysis_result["analysis"]
        metadata = analysis_result.get("metadata", {})
        
        # Start building the markdown report
        report = [
            f"# Chart Analysis Report",
            f"\nTimeframe: {metadata.get('timeframe', 'Unknown')}",
            f"Generated: {metadata.get('timestamp', datetime.now().isoformat())}",
            f"\n## Summary of Findings\n"
        ]
        
        # Add summary points from each category
        for category in self.analysis_categories:
            if category in analysis:
                confidence = analysis[category].get("confidence", "N/A")
                summary = analysis[category].get("summary", "No summary available")
                report.append(f"- **{category}** (Confidence: {confidence}/10): {summary}")
        
        # Add detailed analysis for each category
        report.append("\n## Detailed Analysis\n")
        
        for category in self.analysis_categories:
            if category in analysis:
                confidence = analysis[category].get("confidence", "N/A")
                findings = analysis[category].get("findings", "No detailed analysis available")
                report.append(f"### {category} (Confidence: {confidence}/10)\n")
                report.append(f"{findings}\n")
        
        # Add trading recommendation if available
        if "Trading Recommendations" in analysis:
            rec = analysis["Trading Recommendations"]
            report.append("## Trading Action Plan\n")
            report.append(f"{rec.get('findings', 'No recommendations available')}\n")
        
        # Add risk assessment
        if "Risk Assessment" in analysis:
            risk = analysis["Risk Assessment"]
            report.append("## Risk Assessment\n")
            report.append(f"{risk.get('findings', 'No risk assessment available')}\n")
        
        return "\n".join(report)
    
    def compare_timeframes(self, image_paths: Dict[str, str]) -> Dict:
        """
        Compare analysis across multiple timeframes for the same instrument.
        
        Args:
            image_paths: Dictionary mapping timeframes to image paths
            
        Returns:
            Comparative analysis across timeframes
        """
        if not image_paths:
            return {"error": "No images provided for timeframe comparison"}
        
        results = {}
        consolidated_analysis = {
            "aligned_signals": [],
            "conflicting_signals": [],
            "overall_trend": "Neutral",
            "confidence": 0,
            "timeframes_analyzed": list(image_paths.keys())
        }
        
        # Analyze each timeframe
        for timeframe, image_path in image_paths.items():
            self.logger.info(f"Analyzing {timeframe} timeframe chart")
            results[timeframe] = self.analyze_image(image_path, timeframe)
        
        # Process results to find alignments and conflicts
        trend_signals = {"Bullish": 0, "Bearish": 0, "Neutral": 0}
        pattern_counts = {}
        support_resistance_levels = []
        
        # Extract key findings across timeframes
        for timeframe, result in results.items():
            if "error" in result or "analysis" not in result:
                continue
                
            analysis = result["analysis"]
            
            # Process trend analysis
            if "Trend Analysis" in analysis:
                trend_data = analysis["Trend Analysis"]
                trend_text = trend_data.get("findings", "").lower()
                
                # Simple sentiment analysis for trend
                if "bullish" in trend_text:
                    trend_signals["Bullish"] += 1
                elif "bearish" in trend_text:
                    trend_signals["Bearish"] += 1
                else:
                    trend_signals["Neutral"] += 1
            
            # Process pattern recognition
            if "Chart Pattern Recognition" in analysis:
                pattern_data = analysis["Chart Pattern Recognition"]
                findings = pattern_data.get("findings", "")
                
                # Extract patterns (this is simplified)
                for pattern in ["Head and Shoulders", "Double Top", "Double Bottom", 
                               "Triangle", "Flag", "Wedge", "Channel"]:
                    if pattern.lower() in findings.lower():
                        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Process support/resistance
            if "Support and Resistance Levels" in analysis:
                levels_data = analysis["Support and Resistance Levels"]
                support_resistance_levels.append({
                    "timeframe": timeframe,
                    "levels": levels_data.get("findings", "No levels identified")
                })
        
        # Determine overall trend
        max_trend = max(trend_signals, key=trend_signals.get)
        consolidated_analysis["overall_trend"] = max_trend
        consolidated_analysis["confidence"] = trend_signals[max_trend] / len(results) * 10
        
        # Find aligned signals
        for pattern, count in pattern_counts.items():
            if count >= 2:  # Pattern appears in at least 2 timeframes
                consolidated_analysis["aligned_signals"].append(
                    f"Pattern '{pattern}' identified across {count} timeframes"
                )
        
        # Find potential conflicts
        if trend_signals["Bullish"] > 0 and trend_signals["Bearish"] > 0:
            consolidated_analysis["conflicting_signals"].append(
                f"Mixed trend signals: {trend_signals['Bullish']} timeframes bullish, " +
                f"{trend_signals['Bearish']} timeframes bearish"
            )
        
        # Add support/resistance findings
        consolidated_analysis["support_resistance"] = support_resistance_levels
        
        # Include detailed results for each timeframe
        consolidated_analysis["timeframe_details"] = results
        
        return consolidated_analysis
    
    def track_pattern_evolution(self, image_series: List[Dict[str, str]]) -> Dict:
        """
        Track how a pattern evolves over time using a series of chart images.
        
        Args:
            image_series: List of dictionaries with timestamp and image_path keys
            
        Returns:
            Analysis of pattern evolution over time
        """
        if not image_series:
            return {"error": "No image series provided"}
        
        # Sort images by timestamp
        sorted_series = sorted(image_series, key=lambda x: x.get("timestamp", ""))
        
        evolution_analysis = {
            "pattern_evolution": [],
            "key_developments": [],
            "timestamps_analyzed": [item.get("timestamp") for item in sorted_series],
            "detailed_analysis": []
        }
        
        previous_context = None
        
        # Analyze each image in sequence
        for i, item in enumerate(sorted_series):
            timestamp = item.get("timestamp", f"Point {i+1}")
            image_path = item.get("image_path")
            timeframe = item.get("timeframe", "Daily")
            
            if not image_path or not os.path.exists(image_path):
                evolution_analysis["detailed_analysis"].append({
                    "timestamp": timestamp,
                    "error": f"Invalid or missing image path: {image_path}"
                })
                continue
            
            # Analyze this chart with context from previous analysis
            self.logger.info(f"Analyzing chart at timestamp: {timestamp}")
            result = self.analyze_image(
                image_path, 
                timeframe=timeframe,
                previous_context=previous_context
            )
            
            # Update previous context for next iteration
            if "analysis" in result:
                # Extract key findings for context
                trend = result["analysis"].get("Trend Analysis", {}).get("summary", "")
                patterns = result["analysis"].get("Chart Pattern Recognition", {}).get("summary", "")
                previous_context = f"Previous chart ({timestamp}): {trend}. {patterns}"
                
                # Record key developments
                if i > 0:  # Not the first image
                    prev_analysis = evolution_analysis["detailed_analysis"][-1].get("analysis", {})
                    curr_analysis = result["analysis"]
                    
                    # Compare with previous analysis to find developments
                    developments = self._detect_developments(prev_analysis, curr_analysis)
                    if developments:
                        evolution_analysis["key_developments"].append({
                            "timestamp": timestamp,
                            "developments": developments
                        })
                
                # Track pattern evolution
                pattern_data = result["analysis"].get("Chart Pattern Recognition", {})
                evolution_analysis["pattern_evolution"].append({
                    "timestamp": timestamp,
                    "patterns": pattern_data.get("findings", "No patterns detected"),
                    "confidence": pattern_data.get("confidence", 0)
                })
            
            # Add to detailed analysis
            evolution_analysis["detailed_analysis"].append({
                "timestamp": timestamp,
                "timeframe": timeframe,
                "analysis": result.get("analysis", {}),
                "image_path": image_path
            })
        
        return evolution_analysis
    
    def _detect_developments(self, prev_analysis: Dict, curr_analysis: Dict) -> List[str]:
        """
        Compare two analysis results to detect significant developments.
        
        Args:
            prev_analysis: Previous analysis result
            curr_analysis: Current analysis result
            
        Returns:
            List of detected developments
        """
        developments = []
        
        # Check for trend changes
        prev_trend = prev_analysis.get("Trend Analysis", {}).get("summary", "").lower()
        curr_trend = curr_analysis.get("Trend Analysis", {}).get("summary", "").lower()
        
        if "bullish" in prev_trend and "bearish" in curr_trend:
            developments.append("Trend reversal: Changed from bullish to bearish")
        elif "bearish" in prev_trend and "bullish" in curr_trend:
            developments.append("Trend reversal: Changed from bearish to bullish")
        
        # Check for new pattern formation
        prev_patterns = prev_analysis.get("Chart Pattern Recognition", {}).get("findings", "").lower()
        curr_patterns = curr_analysis.get("Chart Pattern Recognition", {}).get("findings", "").lower()
        
        pattern_types = ["head and shoulders", "double top", "double bottom", "flag", 
                        "wedge", "triangle", "channel", "cup and handle"]
        
        for pattern in pattern_types:
            if pattern not in prev_patterns and pattern in curr_patterns:
                developments.append(f"New pattern formed: {pattern.title()}")
        
        # Check for breakouts
        prev_price = prev_analysis.get("Price Action", {}).get("findings", "").lower()
        curr_price = curr_analysis.get("Price Action", {}).get("findings", "").lower()
        
        if ("resistance" in prev_price and "breakout" in curr_price) or \
           ("resistance" in prev_price and "broken" in curr_price):
            developments.append("Resistance breakout detected")
        
        if ("support" in prev_price and "breakdown" in curr_price) or \
           ("support" in prev_price and "broken" in curr_price):
            developments.append("Support breakdown detected")
        
        # Check for changes in trading recommendations
        prev_rec = prev_analysis.get("Trading Recommendations", {}).get("summary", "").lower()
        curr_rec = curr_analysis.get("Trading Recommendations", {}).get("summary", "").lower()
        
        if "buy" in curr_rec and "buy" not in prev_rec:
            developments.append("New buy signal generated")
        elif "sell" in curr_rec and "sell" not in prev_rec:
            developments.append("New sell signal generated")
        
        return developments
    
    def batch_analyze(self, image_directory: str, pattern_filter: Optional[str] = None) -> Dict:
        """
        Analyze multiple chart images in a directory, optionally filtering for specific patterns.
        
        Args:
            image_directory: Directory containing chart images
            pattern_filter: Only include results with this pattern type
            
        Returns:
            Dictionary with analysis results for each image
        """
        if not os.path.isdir(image_directory):
            return {"error": f"Directory not found: {image_directory}"}
        
        results = {
            "analyses": [],
            "summary": {
                "total_images": 0,
                "successful_analyses": 0,
                "pattern_counts": {},
                "highest_confidence": {"pattern": None, "confidence": 0, "image_path": None}
            }
        }
        
        # Get list of image files
        image_files = [f for f in os.listdir(image_directory) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        results["summary"]["total_images"] = len(image_files)
        
        # Process each image
        for image_file in image_files:
            image_path = os.path.join(image_directory, image_file)
            
            # Try to extract timeframe from filename (e.g., "btc_usd_daily.png")
            timeframe = "Daily"  # Default
            for tf in ["1m", "5m", "15m", "30m", "1h", "4h", "daily", "weekly", "monthly"]:
                if tf in image_file.lower():
                    timeframe = tf.title()
                    break
            
            # Analyze the image
            self.logger.info(f"Analyzing image: {image_file}")
            analysis = self.analyze_image(image_path, timeframe=timeframe)
            
            # Check if analysis was successful
            if "error" not in analysis and "analysis" in analysis:
                results["summary"]["successful_analyses"] += 1
                
                # Extract pattern information
                if "Chart Pattern Recognition" in analysis["analysis"]:
                    pattern_data = analysis["analysis"]["Chart Pattern Recognition"]
                    patterns_text = pattern_data.get("findings", "").lower()
                    confidence = pattern_data.get("confidence", 0)
                    
                    # Count patterns found
                    for pattern in ["head and shoulders", "double top", "double bottom", 
                                   "triangle", "flag", "wedge", "channel"]:
                        if pattern in patterns_text:
                            results["summary"]["pattern_counts"][pattern] = \
                                results["summary"]["pattern_counts"].get(pattern, 0) + 1
                            
                            # Check if this is the highest confidence pattern
                            if confidence > results["summary"]["highest_confidence"]["confidence"]:
                                results["summary"]["highest_confidence"] = {
                                    "pattern": pattern,
                                    "confidence": confidence,
                                    "image_path": image_path
                                }
                
                # Check pattern filter
                include_result = True
                if pattern_filter:
                    pattern_found = False
                    if "Chart Pattern Recognition" in analysis["analysis"]:
                        patterns_text = analysis["analysis"]["Chart Pattern Recognition"].get("findings", "").lower()
                        if pattern_filter.lower() in patterns_text:
                            pattern_found = True
                    
                    include_result = pattern_found
                
                if include_result:
                    results["analyses"].append({
                        "image_path": image_path,
                        "timeframe": timeframe,
                        "analysis": analysis
                    })
            else:
                self.logger.warning(f"Analysis failed for {image_file}: {analysis.get('error', 'Unknown error')}")
                results["analyses"].append({
                    "image_path": image_path,
                    "error": analysis.get('error', 'Analysis failed')
                })
        
        return results


if __name__ == "__main__":
    # Example usage
    api_key = os.environ.get("OPENROUTER_API_KEY")
    analyzer = OpenRouterVisionAnalyzer(api_key)
    
    # Example: Analyze a single chart
    test_image = "path/to/chart.png"
    if os.path.exists(test_image):
        print("Analyzing test chart...")
        result = analyzer.analyze_image(test_image, timeframe="Daily")
        print(json.dumps(result, indent=2))
        
        # Generate summary report
        report = analyzer.generate_summary_report(result)
        print("\nSummary Report:")
        print(report)
    else:
        print(f"Test image not found at {test_image}")