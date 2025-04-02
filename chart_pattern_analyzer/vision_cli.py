#!/usr/bin/env python3
"""
Vision CLI Module
Command-line interface extension for OpenRouter Vision Model
and combined analysis capabilities.
"""

import os
import sys
import argparse
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from chart_pattern_analyzer.openrouter_vision_model import OpenRouterVisionAnalyzer
from chart_pattern_analyzer.vision_integration import VisionIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vision_cli.log"),
        logging.StreamHandler()
    ]
)

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str) -> None:
    """Print a formatted header text"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 50}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(50)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 50}{Colors.ENDC}\n")

def print_success(text: str) -> None:
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_error(text: str) -> None:
    """Print an error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_warning(text: str) -> None:
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def print_info(text: str) -> None:
    """Print an informational message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")

def print_progress(text: str, progress: int, total: int) -> None:
    """Print a progress bar with percentage"""
    percent = int(progress / total * 100)
    bar_length = 30
    filled_length = int(bar_length * progress // total)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    print(f"\r{Colors.BLUE}{text}: [{bar}] {percent}%{Colors.ENDC}", end='', flush=True)
    if progress == total:
        print()

def print_table(headers: List[str], rows: List[List[Any]]) -> None:
    """Print a formatted table"""
    # Determine column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print header
    header_row = " | ".join(f"{h.ljust(col_widths[i])}" for i, h in enumerate(headers))
    print(f"{Colors.BOLD}{header_row}{Colors.ENDC}")
    print("-" * len(header_row))
    
    # Print rows
    for row in rows:
        print(" | ".join(f"{str(cell).ljust(col_widths[i])}" for i, cell in enumerate(row)))

class ChartVisionCLI:
    """
    Command-line interface for the Chart Pattern Vision Analysis.
    Provides commands for analyzing charts and managing vision model settings.
    """
    
    def __init__(self):
        """Initialize the CLI with default settings"""
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.config = {
            "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "model_path": "models/pattern_detector.model",
            "output_dir": "output/vision_analysis",
            "confidence_threshold": 0.6,
            "model": "anthropic/claude-3-opus-20240229-vision",
            "temperature": 0.2,
            "batch_size": 10
        }
        
        # Initialize components (will be lazy-loaded)
        self.vision_analyzer = None
        self.vision_integrator = None
        
        # Load config from file if exists
        self.config_path = "vision_config.json"
        self.load_config()
        
        self.logger.info("Vision CLI initialized")
    
    def load_config(self) -> None:
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
                self.logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                self.logger.error(f"Error loading configuration: {e}")
    
    def save_config(self) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def _get_vision_analyzer(self) -> OpenRouterVisionAnalyzer:
        """Lazy load the vision analyzer"""
        if not self.vision_analyzer:
            self.vision_analyzer = OpenRouterVisionAnalyzer(
                api_key=self.config["openrouter_api_key"],
                model=self.config["model"]
            )
        return self.vision_analyzer
    
    def _get_vision_integrator(self) -> VisionIntegration:
        """Lazy load the vision integrator"""
        if not self.vision_integrator:
            self.vision_integrator = VisionIntegration(
                api_key=self.config["openrouter_api_key"],
                model_path=self.config["model_path"],
                output_dir=self.config["output_dir"],
                confidence_threshold=self.config["confidence_threshold"]
            )
        return self.vision_integrator
    
    def parse_arguments(self) -> None:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description="Chart Pattern Vision Analysis CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter)
        
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # Analyze image with vision model
        analyze_parser = subparsers.add_parser("analyze", help="Analyze a chart image with vision model")
        analyze_parser.add_argument("--image", required=True, help="Path to chart image")
        analyze_parser.add_argument("--timeframe", default="Daily", help="Chart timeframe")
        analyze_parser.add_argument("--output", help="Path to save analysis output")
        analyze_parser.add_argument("--combined", action="store_true", help="Use combined analysis (vision + traditional)")
        
        # Batch analyze images
        batch_parser = subparsers.add_parser("batch", help="Analyze multiple chart images")
        batch_parser.add_argument("--dir", required=True, help="Directory containing chart images")
        batch_parser.add_argument("--pattern", help="Filter for specific pattern")
        batch_parser.add_argument("--timeframe", help="Chart timeframe (will attempt to detect from filenames if not specified)")
        batch_parser.add_argument("--combined", action="store_true", help="Use combined analysis (vision + traditional)")
        
        # Compare timeframes
        compare_parser = subparsers.add_parser("compare", help="Compare analysis across timeframes")
        compare_parser.add_argument("--images", required=True, nargs="+", help="Paths to chart images for different timeframes")
        compare_parser.add_argument("--timeframes", required=True, nargs="+", help="Corresponding timeframes for each image")
        
        # Track pattern evolution
        track_parser = subparsers.add_parser("track", help="Track pattern evolution over time")
        track_parser.add_argument("--images", required=True, nargs="+", help="Paths to chart images in chronological order")
        track_parser.add_argument("--timestamps", required=True, nargs="+", help="Timestamps for each image")
        track_parser.add_argument("--timeframe", default="Daily", help="Chart timeframe")
        
        # View settings
        settings_parser = subparsers.add_parser("settings", help="View or update settings")
        settings_parser.add_argument("--view", action="store_true", help="View current settings")
        settings_parser.add_argument("--api-key", help="Set OpenRouter API key")
        settings_parser.add_argument("--model", help="Set vision model")
        settings_parser.add_argument("--output-dir", help="Set output directory")
        settings_parser.add_argument("--confidence", type=float, help="Set confidence threshold")
        settings_parser.add_argument("--temperature", type=float, help="Set model temperature")
        
        args = parser.parse_args()
        
        # Execute command
        if args.command is None:
            parser.print_help()
        elif args.command == "analyze":
            self.handle_analyze(args)
        elif args.command == "batch":
            self.handle_batch(args)
        elif args.command == "compare":
            self.handle_compare(args)
        elif args.command == "track":
            self.handle_track(args)
        elif args.command == "settings":
            self.handle_settings(args)
    
    def handle_analyze(self, args) -> None:
        """Handle the analyze command"""
        if not os.path.exists(args.image):
            print_error(f"Image not found: {args.image}")
            return
        
        print_header("Chart Vision Analysis")
        print_info(f"Analyzing image: {args.image}")
        print_info(f"Timeframe: {args.timeframe}")
        
        try:
            start_time = time.time()
            
            if args.combined:
                # Use combined analysis
                integrator = self._get_vision_integrator()
                result = integrator.analyze_chart(args.image, args.timeframe)
                
                # Print summary of findings
                print_success(f"Analysis completed in {time.time() - start_time:.2f} seconds")
                
                if "combined_patterns" in result and result["combined_patterns"]:
                    print_header("Detected Patterns")
                    
                    # Prepare table data
                    headers = ["Pattern", "Confidence", "Source"]
                    rows = []
                    
                    for pattern in sorted(result["combined_patterns"], 
                                         key=lambda x: x["confidence"], reverse=True):
                        rows.append([
                            pattern["pattern"],
                            f"{pattern['confidence']:.2f}",
                            pattern["source"].title()
                        ])
                    
                    print_table(headers, rows)
                else:
                    print_warning("No patterns detected with sufficient confidence")
                
                # Print report location
                if "report_files" in result and "markdown" in result["report_files"]:
                    print_success(f"Report generated: {result['report_files']['markdown']}")
                
            else:
                # Use vision-only analysis
                analyzer = self._get_vision_analyzer()
                result = analyzer.analyze_image(args.image, args.timeframe, 
                                              temperature=self.config["temperature"])
                
                # Print summary of findings
                print_success(f"Analysis completed in {time.time() - start_time:.2f} seconds")
                
                if "analysis" in result:
                    analysis = result["analysis"]
                    
                    print_header("Analysis Summary")
                    
                    for category in analyzer.analysis_categories:
                        if category in analysis:
                            confidence = analysis[category].get("confidence", "N/A")
                            summary = analysis[category].get("summary", "No summary available")
                            print(f"{Colors.BOLD}{category}{Colors.ENDC} (Confidence: {confidence}/10)")
                            print(f"  {summary}")
                            print()
                    
                    # Generate and save report
                    report = analyzer.generate_summary_report(result)
                    output_path = args.output
                    
                    if not output_path:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = f"{self.config['output_dir']}/{timestamp}_analysis.md"
                        
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    with open(output_path, 'w') as f:
                        f.write(report)
                    
                    print_success(f"Report saved to: {output_path}")
                else:
                    if "error" in result:
                        print_error(f"Analysis failed: {result['error']}")
                    else:
                        print_error("Analysis failed: Unknown error")
            
        except Exception as e:
            print_error(f"Error during analysis: {e}")
            self.logger.exception("Error during analysis")
    
    def handle_batch(self, args) -> None:
        """Handle the batch analysis command"""
        if not os.path.exists(args.dir):
            print_error(f"Directory not found: {args.dir}")
            return
        
        print_header("Batch Chart Analysis")
        print_info(f"Analyzing images in: {args.dir}")
        if args.pattern:
            print_info(f"Filtering for pattern: {args.pattern}")
        if args.timeframe:
            print_info(f"Using timeframe: {args.timeframe}")
        
        try:
            start_time = time.time()
            
            if args.combined:
                # Use combined batch analysis
                integrator = self._get_vision_integrator()
                results = integrator.batch_analyze_directory(
                    image_dir=args.dir,
                    timeframe=args.timeframe,
                    pattern_filter=args.pattern
                )
                
                print_success(f"Batch analysis completed in {time.time() - start_time:.2f} seconds")
                
                # Print summary
                if "total_images" in results:
                    print_info(f"Processed {results['total_images']} images")
                    print_info(f"Found patterns in {results['successful_analyses']} images")
                    
                    # Print pattern statistics
                    if results.get("pattern_counts"):
                        print_header("Pattern Statistics")
                        
                        headers = ["Pattern", "Count", "Percentage"]
                        rows = []
                        
                        for pattern, count in sorted(results["pattern_counts"].items(), 
                                                  key=lambda x: x[1], reverse=True):
                            percentage = (count / results["successful_analyses"]) * 100 if results["successful_analyses"] > 0 else 0
                            rows.append([pattern, count, f"{percentage:.1f}%"])
                        
                        print_table(headers, rows)
                    
                    # Print summary report location
                    if "summary_report" in results:
                        print_success(f"Summary report: {results['summary_report']}")
                
            else:
                # Use vision-only batch analysis
                analyzer = self._get_vision_analyzer()
                results = analyzer.batch_analyze(
                    image_directory=args.dir,
                    pattern_filter=args.pattern
                )
                
                print_success(f"Batch analysis completed in {time.time() - start_time:.2f} seconds")
                
                # Print summary
                if "summary" in results:
                    summary = results["summary"]
                    print_info(f"Processed {summary['total_images']} images")
                    print_info(f"Successfully analyzed {summary['successful_analyses']} images")
                    
                    # Print pattern statistics
                    if summary.get("pattern_counts"):
                        print_header("Pattern Statistics")
                        
                        headers = ["Pattern", "Count"]
                        rows = []
                        
                        for pattern, count in sorted(summary["pattern_counts"].items(), 
                                                  key=lambda x: x[1], reverse=True):
                            rows.append([pattern, count])
                        
                        print_table(headers, rows)
                    
                    # Print highest confidence pattern
                    if summary.get("highest_confidence") and summary["highest_confidence"]["pattern"]:
                        hc = summary["highest_confidence"]
                        print_info(f"Highest confidence pattern: {hc['pattern']} ({hc['confidence']}/10)")
                        print_info(f"Found in: {os.path.basename(hc['image_path'])}")
                
                # Generate and save detailed report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{self.config['output_dir']}/{timestamp}_batch_analysis.json"
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print_success(f"Detailed results saved to: {output_path}")
            
        except Exception as e:
            print_error(f"Error during batch analysis: {e}")
            self.logger.exception("Error during batch analysis")
    
    def handle_compare(self, args) -> None:
        """Handle the timeframe comparison command"""
        # Validate inputs
        if len(args.images) != len(args.timeframes):
            print_error("Number of images must match number of timeframes")
            return
        
        for image_path in args.images:
            if not os.path.exists(image_path):
                print_error(f"Image not found: {image_path}")
                return
        
        print_header("Timeframe Comparison Analysis")
        
        # Map timeframes to image paths
        image_paths = {tf: img for tf, img in zip(args.timeframes, args.images)}
        
        for tf, img in image_paths.items():
            print_info(f"Timeframe {tf}: {img}")
        
        try:
            start_time = time.time()
            
            # Use vision analyzer for comparison
            analyzer = self._get_vision_analyzer()
            result = analyzer.compare_timeframes(image_paths)
            
            print_success(f"Comparison completed in {time.time() - start_time:.2f} seconds")
            
            # Print overall trend
            if "overall_trend" in result:
                trend = result["overall_trend"]
                confidence = result.get("confidence", 0)
                print_info(f"Overall trend: {trend} (Confidence: {confidence:.1f}/10)")
            
            # Print aligned signals
            if "aligned_signals" in result and result["aligned_signals"]:
                print_header("Aligned Signals")
                for signal in result["aligned_signals"]:
                    print_success(f"- {signal}")
            else:
                print_warning("No aligned signals found across timeframes")
            
            # Print conflicting signals
            if "conflicting_signals" in result and result["conflicting_signals"]:
                print_header("Conflicting Signals")
                for signal in result["conflicting_signals"]:
                    print_warning(f"- {signal}")
            
            # Generate and save detailed report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.config['output_dir']}/{timestamp}_timeframe_comparison.json"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            print_success(f"Detailed results saved to: {output_path}")
            
        except Exception as e:
            print_error(f"Error during timeframe comparison: {e}")
            self.logger.exception("Error during timeframe comparison")
    
    def handle_track(self, args) -> None:
        """Handle the pattern evolution tracking command"""
        # Validate inputs
        if len(args.images) != len(args.timestamps):
            print_error("Number of images must match number of timestamps")
            return
        
        for image_path in args.images:
            if not os.path.exists(image_path):
                print_error(f"Image not found: {image_path}")
                return
        
        print_header("Pattern Evolution Tracking")
        print_info(f"Tracking {len(args.images)} images over time")
        print_info(f"Timeframe: {args.timeframe}")
        
        try:
            start_time = time.time()
            
            # Prepare image series
            image_series = [
                {"timestamp": ts, "image_path": img, "timeframe": args.timeframe}
                for ts, img in zip(args.timestamps, args.images)
            ]
            
            # Use vision analyzer for tracking
            analyzer = self._get_vision_analyzer()
            result = analyzer.track_pattern_evolution(image_series)
            
            print_success(f"Tracking completed in {time.time() - start_time:.2f} seconds")
            
            # Print key developments
            if "key_developments" in result and result["key_developments"]:
                print_header("Key Developments")
                
                for dev in result["key_developments"]:
                    timestamp = dev["timestamp"]
                    developments = dev["developments"]
                    
                    print(f"{Colors.BOLD}{timestamp}{Colors.ENDC}")
                    for item in developments:
                        print(f"  - {item}")
                    print()
            else:
                print_warning("No significant developments detected")
            
            # Print pattern evolution
            if "pattern_evolution" in result and result["pattern_evolution"]:
                print_header("Pattern Evolution")
                
                headers = ["Timestamp", "Patterns", "Confidence"]
                rows = []
                
                for entry in result["pattern_evolution"]:
                    # Truncate patterns text if too long
                    patterns_text = entry["patterns"]
                    if len(patterns_text) > 50:
                        patterns_text = patterns_text[:47] + "..."
                    
                    rows.append([
                        entry["timestamp"],
                        patterns_text,
                        f"{entry['confidence']}/10"
                    ])
                
                print_table(headers, rows)
            
            # Generate and save detailed report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.config['output_dir']}/{timestamp}_pattern_evolution.json"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            print_success(f"Detailed results saved to: {output_path}")
            
        except Exception as e:
            print_error(f"Error during pattern evolution tracking: {e}")
            self.logger.exception("Error during pattern evolution tracking")
    
    def handle_settings(self, args) -> None:
        """Handle the settings command"""
        if args.view or (not args.api_key and not args.model and 
                       not args.output_dir and args.confidence is None and 
                       args.temperature is None):
            # View current settings
            print_header("Current Settings")
            
            # Hide full API key for security
            api_key = self.config["openrouter_api_key"]
            if api_key:
                masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
            else:
                masked_key = "Not set"
            
            settings = [
                ["OpenRouter API Key", masked_key],
                ["Vision Model", self.config["model"]],
                ["ML Model Path", self.config["model_path"]],
                ["Output Directory", self.config["output_dir"]],
                ["Confidence Threshold", self.config["confidence_threshold"]],
                ["Temperature", self.config["temperature"]],
                ["Batch Size", self.config["batch_size"]]
            ]
            
            for setting in settings:
                print(f"{Colors.BOLD}{setting[0]}{Colors.ENDC}: {setting[1]}")
            
        else:
            # Update settings
            updated = False
            
            if args.api_key:
                self.config["openrouter_api_key"] = args.api_key
                print_success("API key updated")
                updated = True
            
            if args.model:
                self.config["model"] = args.model
                print_success(f"Vision model updated to: {args.model}")
                updated = True
            
            if args.output_dir:
                self.config["output_dir"] = args.output_dir
                os.makedirs(args.output_dir, exist_ok=True)
                print_success(f"Output directory updated to: {args.output_dir}")
                updated = True
            
            if args.confidence is not None:
                if 0 <= args.confidence <= 1:
                    self.config["confidence_threshold"] = args.confidence
                    print_success(f"Confidence threshold updated to: {args.confidence}")
                    updated = True
                else:
                    print_error("Confidence threshold must be between 0 and 1")
            
            if args.temperature is not None:
                if 0 <= args.temperature <= 1:
                    self.config["temperature"] = args.temperature
                    print_success(f"Temperature updated to: {args.temperature}")
                    updated = True
                else:
                    print_error("Temperature must be between 0 and 1")
            
            if updated:
                # Reset components to ensure they use new settings next time
                self.vision_analyzer = None
                self.vision_integrator = None
                
                # Save updated config
                self.save_config()
                print_success("Settings saved to config file")


def main():
    """Main entry point for the CLI"""
    cli = ChartVisionCLI()
    cli.parse_arguments()


if __name__ == "__main__":
    main()