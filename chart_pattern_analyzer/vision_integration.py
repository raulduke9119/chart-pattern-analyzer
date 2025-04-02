#!/usr/bin/env python3
"""
Vision Integration Module
Connects OpenRouter Vision Model with the Chart Pattern Analyzer
to provide enhanced analysis and reporting capabilities.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from chart_pattern_analyzer.openrouter_vision_model import OpenRouterVisionAnalyzer
from chart_pattern_analyzer.chart_analyzer import ChartAnalyzer, ChartPatternModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vision_integration.log"),
        logging.StreamHandler()
    ]
)

class VisionIntegration:
    """
    Integrates the OpenRouter Vision Model with the existing Chart Analyzer
    to provide combined analysis capabilities and enhanced reports.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None,
                model_path: str = "models/pattern_detector.model",
                output_dir: str = "output/vision_analysis",
                confidence_threshold: float = 0.6):
        """
        Initialize the Vision Integration module.
        
        Args:
            api_key: OpenRouter API key
            model_path: Path to the local chart pattern model
            output_dir: Directory to store analysis outputs
            confidence_threshold: Minimum confidence threshold for pattern detection
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.vision_analyzer = OpenRouterVisionAnalyzer(api_key)
        self.chart_analyzer = ChartAnalyzer(model_path=model_path)
        
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info("Vision Integration module initialized")
    
    def analyze_chart(self, 
                     image_path: str, 
                     timeframe: str = "Daily",
                     return_annotated: bool = True) -> Dict:
        """
        Perform combined analysis using both vision model and traditional chart analyzer.
        
        Args:
            image_path: Path to the chart image
            timeframe: Chart timeframe
            return_annotated: Whether to return the annotated image path
            
        Returns:
            Combined analysis results from both systems
        """
        start_time = time.time()
        self.logger.info(f"Starting combined analysis for: {image_path}")
        
        # Prepare result structure
        result = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "timeframe": timeframe,
            "analysis_sources": [],
            "combined_patterns": [],
            "support_resistance": [],
            "trading_recommendations": {},
            "risk_assessment": {},
            "confidence_scores": {}
        }
        
        # 1. Run traditional pattern detector
        try:
            traditional_result = self.chart_analyzer.analyze_image(
                image_path, 
                confidence_threshold=self.confidence_threshold
            )
            result["analysis_sources"].append("traditional")
            result["traditional_analysis"] = traditional_result
            
            # Extract patterns from traditional analysis
            if "patterns" in traditional_result:
                result["combined_patterns"].extend([
                    {
                        "pattern": p["name"],
                        "confidence": p["confidence"],
                        "source": "traditional",
                        "location": p.get("box", [0, 0, 0, 0])
                    } for p in traditional_result["patterns"]
                ])
            
            # Get annotated image if available
            if return_annotated and "annotated_image" in traditional_result:
                result["annotated_image"] = traditional_result["annotated_image"]
                
        except Exception as e:
            self.logger.error(f"Error in traditional analysis: {e}")
            result["traditional_error"] = str(e)
        
        # 2. Run vision model analysis
        try:
            vision_result = self.vision_analyzer.analyze_image(
                image_path, 
                timeframe=timeframe
            )
            result["analysis_sources"].append("vision")
            result["vision_analysis"] = vision_result
            
            # Process vision analysis results
            if "analysis" in vision_result:
                analysis = vision_result["analysis"]
                
                # Extract patterns from vision analysis
                if "Chart Pattern Recognition" in analysis:
                    pattern_data = analysis["Chart Pattern Recognition"]
                    patterns_text = pattern_data.get("findings", "")
                    
                    # Parse patterns from text (simplified approach)
                    pattern_keywords = [
                        "Head and Shoulders", "Double Top", "Double Bottom", "Ascending Triangle", 
                        "Descending Triangle", "Symmetrical Triangle", "Flag", "Pennant", 
                        "Channel", "Cup and Handle", "Wedge"
                    ]
                    
                    for pattern in pattern_keywords:
                        if pattern.lower() in patterns_text.lower():
                            result["combined_patterns"].append({
                                "pattern": pattern,
                                "confidence": pattern_data.get("confidence", 5) / 10,  # Convert 1-10 to 0-1
                                "source": "vision",
                                "description": self._extract_pattern_context(patterns_text, pattern)
                            })
                
                # Extract support/resistance levels
                if "Support and Resistance Levels" in analysis:
                    levels_data = analysis["Support and Resistance Levels"]
                    result["support_resistance"] = {
                        "findings": levels_data.get("findings", "No levels identified"),
                        "confidence": levels_data.get("confidence", 5) / 10
                    }
                
                # Extract trading recommendations
                if "Trading Recommendations" in analysis:
                    result["trading_recommendations"] = analysis["Trading Recommendations"]
                
                # Extract risk assessment
                if "Risk Assessment" in analysis:
                    result["risk_assessment"] = analysis["Risk Assessment"]
                
        except Exception as e:
            self.logger.error(f"Error in vision analysis: {e}")
            result["vision_error"] = str(e)
        
        # 3. Generate combined confidence scores
        self._calculate_combined_confidence(result)
        
        # 4. Generate output reports
        try:
            self._generate_combined_report(result)
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            result["report_error"] = str(e)
        
        # Add processing time
        result["processing_time"] = time.time() - start_time
        self.logger.info(f"Completed combined analysis in {result['processing_time']:.2f} seconds")
        
        return result
    
    def _extract_pattern_context(self, text: str, pattern: str) -> str:
        """
        Extract the context around a pattern mention in analysis text.
        
        Args:
            text: The full analysis text
            pattern: The pattern to find context for
            
        Returns:
            Contextual description of the pattern
        """
        lower_text = text.lower()
        lower_pattern = pattern.lower()
        
        # Find the pattern in the text
        pattern_index = lower_text.find(lower_pattern)
        if pattern_index == -1:
            return f"Pattern: {pattern}"
        
        # Get context (sentence containing the pattern)
        start = max(0, lower_text.rfind('. ', 0, pattern_index) + 2)
        if start == 1:  # No period found
            start = 0
            
        end = lower_text.find('. ', pattern_index)
        if end == -1:  # No period found
            end = len(lower_text)
        
        context = text[start:end] + '.'
        return context.strip()
    
    def _calculate_combined_confidence(self, result: Dict) -> None:
        """
        Calculate combined confidence scores based on both analysis methods.
        
        Args:
            result: The analysis result dictionary to update
        """
        # Group patterns by type
        pattern_confidence = {}
        
        for pattern_item in result["combined_patterns"]:
            pattern_name = pattern_item["pattern"]
            confidence = pattern_item["confidence"]
            source = pattern_item["source"]
            
            if pattern_name not in pattern_confidence:
                pattern_confidence[pattern_name] = {
                    "traditional": [],
                    "vision": [],
                    "combined": None
                }
            
            pattern_confidence[pattern_name][source].append(confidence)
        
        # Calculate combined confidence scores
        for pattern, scores in pattern_confidence.items():
            # Average confidence for each source
            trad_avg = sum(scores["traditional"]) / len(scores["traditional"]) if scores["traditional"] else 0
            vision_avg = sum(scores["vision"]) / len(scores["vision"]) if scores["vision"] else 0
            
            # If both sources detected the pattern, weight them
            if scores["traditional"] and scores["vision"]:
                # Weight traditional slightly higher for concrete patterns
                combined = (trad_avg * 0.6) + (vision_avg * 0.4)
                source = "both"
            elif scores["traditional"]:
                combined = trad_avg
                source = "traditional"
            else:
                combined = vision_avg
                source = "vision"
            
            pattern_confidence[pattern]["combined"] = combined
            pattern_confidence[pattern]["source"] = source
        
        result["confidence_scores"] = pattern_confidence
    
    def _generate_combined_report(self, result: Dict) -> None:
        """
        Generate combined report documents and visualizations.
        
        Args:
            result: The analysis result
        """
        # Create unique output prefix for this analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = os.path.basename(result["image_path"])
        output_prefix = f"{self.output_dir}/{timestamp}_{image_name.split('.')[0]}"
        
        # 1. Save JSON result
        json_path = f"{output_prefix}_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        result["report_files"] = {"json": json_path}
        
        # 2. Generate markdown report
        md_path = f"{output_prefix}_report.md"
        with open(md_path, 'w') as f:
            f.write(self._generate_markdown_report(result))
        
        result["report_files"]["markdown"] = md_path
        
        # 3. Generate confidence visualization
        if result["combined_patterns"]:
            chart_path = f"{output_prefix}_confidence.png"
            self._create_confidence_chart(result, chart_path)
            result["report_files"]["confidence_chart"] = chart_path
    
    def _generate_markdown_report(self, result: Dict) -> str:
        """
        Generate a comprehensive markdown report from the analysis results.
        
        Args:
            result: The analysis result
            
        Returns:
            Markdown formatted report
        """
        image_name = os.path.basename(result["image_path"])
        
        # Build the report
        report = [
            f"# Combined Chart Analysis Report",
            f"\n## Chart Information",
            f"- **Image**: {image_name}",
            f"- **Timeframe**: {result['timeframe']}",
            f"- **Analysis Date**: {result['timestamp']}",
            f"- **Analysis Methods**: {', '.join(result['analysis_sources'])}",
            
            f"\n## Pattern Detection Results"
        ]
        
        # Add pattern detection results
        if result["combined_patterns"]:
            report.append("| Pattern | Confidence | Source | Description |")
            report.append("|---------|------------|--------|-------------|")
            
            for pattern in sorted(result["combined_patterns"], 
                                 key=lambda x: x["confidence"], reverse=True):
                pattern_name = pattern["pattern"]
                confidence = f"{pattern['confidence']:.2f}"
                source = pattern["source"].title()
                description = pattern.get("description", "No description available")
                
                report.append(f"| {pattern_name} | {confidence} | {source} | {description} |")
        else:
            report.append("No patterns detected with confidence above threshold.")
        
        # Add support and resistance levels
        report.append(f"\n## Support and Resistance Levels")
        if result["support_resistance"]:
            sr_data = result["support_resistance"]
            report.append(f"**Confidence**: {sr_data.get('confidence', 'N/A')}")
            report.append(f"\n{sr_data.get('findings', 'No support/resistance levels identified.')}")
        else:
            report.append("No support/resistance levels identified.")
        
        # Add trading recommendations
        report.append(f"\n## Trading Recommendations")
        if "trading_recommendations" in result and result["trading_recommendations"]:
            tr_data = result["trading_recommendations"]
            report.append(f"**Confidence**: {tr_data.get('confidence', 'N/A')}/10")
            report.append(f"\n{tr_data.get('findings', 'No trading recommendations available.')}")
        else:
            report.append("No trading recommendations available.")
        
        # Add risk assessment
        report.append(f"\n## Risk Assessment")
        if "risk_assessment" in result and result["risk_assessment"]:
            risk_data = result["risk_assessment"]
            report.append(f"**Confidence**: {risk_data.get('confidence', 'N/A')}/10")
            report.append(f"\n{risk_data.get('findings', 'No risk assessment available.')}")
        else:
            report.append("No risk assessment available.")
        
        # Add technical details
        report.append(f"\n## Technical Details")
        report.append(f"- **Processing Time**: {result.get('processing_time', 'N/A'):.2f} seconds")
        report.append(f"- **Confidence Threshold**: {self.confidence_threshold}")
        
        if "annotated_image" in result:
            report.append(f"- **Annotated Image**: {os.path.basename(result['annotated_image'])}")
        
        return "\n".join(report)
    
    def _create_confidence_chart(self, result: Dict, output_path: str) -> None:
        """
        Create a visualization of pattern confidence scores.
        
        Args:
            result: Analysis result
            output_path: Path to save the chart
        """
        if not result["confidence_scores"]:
            return
        
        # Prepare data for visualization
        patterns = []
        traditional_scores = []
        vision_scores = []
        combined_scores = []
        
        for pattern, scores in result["confidence_scores"].items():
            patterns.append(pattern)
            
            # Get average scores for each method
            trad_avg = sum(scores["traditional"]) / len(scores["traditional"]) if scores["traditional"] else 0
            vision_avg = sum(scores["vision"]) / len(scores["vision"]) if scores["vision"] else 0
            
            traditional_scores.append(trad_avg)
            vision_scores.append(vision_avg)
            combined_scores.append(scores["combined"])
        
        # Create DataFrame
        df = pd.DataFrame({
            'Pattern': patterns,
            'Traditional': traditional_scores,
            'Vision': vision_scores,
            'Combined': combined_scores
        })
        
        # Sort by combined confidence
        df = df.sort_values('Combined', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Set up the plot
        ax = sns.barplot(x='Pattern', y='value', hue='Method', 
                         data=pd.melt(df, id_vars=['Pattern'], 
                                     value_vars=['Traditional', 'Vision', 'Combined'],
                                     var_name='Method', value_name='value'),
                         palette="viridis")
        
        # Add titles and labels
        plt.title('Pattern Confidence by Analysis Method', fontsize=16)
        plt.xlabel('Pattern Type', fontsize=12)
        plt.ylabel('Confidence Score', fontsize=12)
        plt.ylim(0, 1.0)
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        
        # Add confidence threshold line
        plt.axhline(y=self.confidence_threshold, color='r', linestyle='--', 
                   label=f'Confidence Threshold ({self.confidence_threshold})')
        
        # Add legend
        plt.legend(title='Analysis Method')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def batch_analyze_directory(self, 
                              image_dir: str, 
                              timeframe: Optional[str] = None,
                              pattern_filter: Optional[str] = None) -> Dict:
        """
        Analyze all chart images in a directory using combined analysis.
        
        Args:
            image_dir: Directory containing chart images
            timeframe: Chart timeframe (if None, will try to detect from filenames)
            pattern_filter: Only include results with this pattern type
            
        Returns:
            Summary of batch analysis results
        """
        if not os.path.isdir(image_dir):
            return {"error": f"Directory not found: {image_dir}"}
        
        results = {
            "directory": image_dir,
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "pattern_filter": pattern_filter,
            "total_images": 0,
            "successful_analyses": 0,
            "pattern_counts": {},
            "reports_dir": self.output_dir,
            "analysis_results": []
        }
        
        # Get list of image files
        image_files = [f for f in os.listdir(image_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        results["total_images"] = len(image_files)
        
        # Process each image
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            
            # Try to extract timeframe from filename if not provided
            img_timeframe = timeframe
            if not img_timeframe:
                for tf in ["1m", "5m", "15m", "30m", "1h", "4h", "daily", "weekly", "monthly"]:
                    if tf in image_file.lower():
                        img_timeframe = tf.title()
                        break
                if not img_timeframe:
                    img_timeframe = "Daily"  # Default
            
            # Analyze the image
            self.logger.info(f"Analyzing image: {image_file}")
            try:
                analysis = self.analyze_chart(image_path, timeframe=img_timeframe)
                
                # Apply pattern filter if specified
                include_result = True
                if pattern_filter:
                    pattern_found = False
                    for pattern in analysis.get("combined_patterns", []):
                        if pattern_filter.lower() in pattern["pattern"].lower():
                            pattern_found = True
                            break
                    include_result = pattern_found
                
                if include_result:
                    results["successful_analyses"] += 1
                    
                    # Count detected patterns
                    for pattern in analysis.get("combined_patterns", []):
                        pattern_name = pattern["pattern"]
                        results["pattern_counts"][pattern_name] = \
                            results["pattern_counts"].get(pattern_name, 0) + 1
                    
                    # Add to results
                    results["analysis_results"].append({
                        "image_path": image_path,
                        "timeframe": img_timeframe,
                        "reports": analysis.get("report_files", {}),
                        "patterns": analysis.get("combined_patterns", []),
                        "confidence_scores": analysis.get("confidence_scores", {})
                    })
                
            except Exception as e:
                self.logger.error(f"Analysis failed for {image_file}: {e}")
                results["analysis_results"].append({
                    "image_path": image_path,
                    "error": str(e)
                })
        
        # Generate batch summary report
        self._generate_batch_summary(results)
        
        return results
    
    def _generate_batch_summary(self, results: Dict) -> None:
        """
        Generate a summary report for batch analysis.
        
        Args:
            results: Batch analysis results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = f"{self.output_dir}/{timestamp}_batch_summary.md"
        
        # Calculate statistics
        success_rate = (results["successful_analyses"] / results["total_images"]) * 100 if results["total_images"] > 0 else 0
        
        # Build the summary report
        report = [
            f"# Batch Analysis Summary Report",
            f"\n## Overview",
            f"- **Directory**: {results['directory']}",
            f"- **Analysis Date**: {results['timestamp']}",
            f"- **Timeframe**: {results.get('timeframe', 'Multiple')}",
            f"- **Pattern Filter**: {results.get('pattern_filter', 'None')}",
            f"- **Total Images**: {results['total_images']}",
            f"- **Successful Analyses**: {results['successful_analyses']} ({success_rate:.1f}%)",
            
            f"\n## Detected Patterns"
        ]
        
        # Add pattern statistics
        if results["pattern_counts"]:
            report.append("| Pattern | Count | Percentage |")
            report.append("|---------|-------|------------|")
            
            for pattern, count in sorted(results["pattern_counts"].items(), 
                                        key=lambda x: x[1], reverse=True):
                percentage = (count / results["successful_analyses"]) * 100 if results["successful_analyses"] > 0 else 0
                report.append(f"| {pattern} | {count} | {percentage:.1f}% |")
        else:
            report.append("No patterns detected in any images.")
        
        # Add list of analyzed images
        report.append(f"\n## Analysis Results")
        report.append("| Image | Patterns | Reports |")
        report.append("|-------|----------|---------|")
        
        for result in results["analysis_results"]:
            image_name = os.path.basename(result["image_path"])
            
            # Get pattern list
            patterns = ", ".join([p["pattern"] for p in result.get("patterns", [])]) if "patterns" in result else "None"
            
            # Get report links
            reports = []
            if "reports" in result:
                for report_type, report_path in result["reports"].items():
                    report_name = os.path.basename(report_path)
                    reports.append(f"[{report_type}]({report_name})")
            
            reports_str = ", ".join(reports) if reports else "None"
            
            report.append(f"| {image_name} | {patterns} | {reports_str} |")
        
        # Write the report
        with open(summary_path, 'w') as f:
            f.write("\n".join(report))
        
        results["summary_report"] = summary_path
    
    def analyze_with_feedback(self, 
                            image_path: str, 
                            timeframe: str = "Daily",
                            feedback: Optional[Dict] = None) -> Dict:
        """
        Analyze a chart with optional feedback from previous analysis.
        
        Args:
            image_path: Path to the chart image
            timeframe: Chart timeframe
            feedback: Optional dictionary with feedback on previous analysis
            
        Returns:
            Analysis results incorporating feedback
        """
        # First, run standard analysis
        result = self.analyze_chart(image_path, timeframe)
        
        # If feedback is provided, incorporate it
        if feedback:
            self.logger.info("Incorporating user feedback into analysis")
            
            # Apply feedback adjustments
            if "pattern_adjustments" in feedback:
                self._apply_pattern_feedback(result, feedback["pattern_adjustments"])
            
            # Apply support/resistance adjustments
            if "support_resistance_adjustments" in feedback:
                self._apply_sr_feedback(result, feedback["support_resistance_adjustments"])
            
            # Re-generate reports with feedback incorporated
            self._generate_combined_report(result)
            
            # Note that feedback was incorporated
            result["feedback_incorporated"] = True
        
        return result
    
    def _apply_pattern_feedback(self, result: Dict, pattern_feedback: List[Dict]) -> None:
        """
        Apply user feedback to pattern detection results.
        
        Args:
            result: Analysis result to update
            pattern_feedback: List of pattern adjustments
        """
        # Create mapping of existing patterns by name
        pattern_map = {p["pattern"]: p for p in result["combined_patterns"]}
        
        # Apply each feedback adjustment
        for feedback_item in pattern_feedback:
            pattern_name = feedback_item.get("pattern")
            if not pattern_name:
                continue
                
            action = feedback_item.get("action")
            
            if action == "remove" and pattern_name in pattern_map:
                # Remove pattern from results
                result["combined_patterns"] = [
                    p for p in result["combined_patterns"] 
                    if p["pattern"] != pattern_name
                ]
                # Remove from confidence scores
                if pattern_name in result["confidence_scores"]:
                    del result["confidence_scores"][pattern_name]
                    
            elif action == "add" and pattern_name not in pattern_map:
                # Add new pattern
                new_pattern = {
                    "pattern": pattern_name,
                    "confidence": feedback_item.get("confidence", 0.75),
                    "source": "feedback",
                    "description": feedback_item.get("description", "Added through user feedback")
                }
                result["combined_patterns"].append(new_pattern)
                
                # Add to confidence scores
                result["confidence_scores"][pattern_name] = {
                    "traditional": [],
                    "vision": [],
                    "combined": new_pattern["confidence"],
                    "source": "feedback"
                }
                
            elif action == "adjust" and pattern_name in pattern_map:
                # Adjust existing pattern
                for p in result["combined_patterns"]:
                    if p["pattern"] == pattern_name:
                        if "confidence" in feedback_item:
                            p["confidence"] = feedback_item["confidence"]
                        if "description" in feedback_item:
                            p["description"] = feedback_item["description"]
                
                # Update confidence scores
                if pattern_name in result["confidence_scores"] and "confidence" in feedback_item:
                    result["confidence_scores"][pattern_name]["combined"] = feedback_item["confidence"]
                    result["confidence_scores"][pattern_name]["source"] = "feedback_adjusted"
    
    def _apply_sr_feedback(self, result: Dict, sr_feedback: Dict) -> None:
        """
        Apply user feedback to support/resistance levels.
        
        Args:
            result: Analysis result to update
            sr_feedback: Support/resistance adjustments
        """
        # Create or update support/resistance section
        if not result.get("support_resistance"):
            result["support_resistance"] = {
                "findings": "",
                "confidence": 0.7
            }
        
        # Apply each adjustment
        if "levels" in sr_feedback:
            levels_text = result["support_resistance"].get("findings", "")
            new_levels = sr_feedback["levels"]
            
            # Simple approach: replace with user-provided text
            result["support_resistance"]["findings"] = new_levels
            
        if "confidence" in sr_feedback:
            result["support_resistance"]["confidence"] = sr_feedback["confidence"]


if __name__ == "__main__":
    # Example usage
    api_key = os.environ.get("OPENROUTER_API_KEY")
    integrator = VisionIntegration(api_key)
    
    # Example: Analyze a single chart
    test_image = "path/to/chart.png"
    if os.path.exists(test_image):
        print("Analyzing test chart...")
        result = integrator.analyze_chart(test_image, timeframe="Daily")
        print(json.dumps(result["confidence_scores"], indent=2))
        
        # Print report path
        if "report_files" in result:
            print(f"\nReport generated at: {result['report_files'].get('markdown')}")
    else:
        print(f"Test image not found at {test_image}")