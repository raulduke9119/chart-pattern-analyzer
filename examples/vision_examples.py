#!/usr/bin/env python3
"""
OpenRouter Vision Model Examples
Demonstrates how to use the OpenRouter Vision integration for chart analysis
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path so we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chart_pattern_analyzer.openrouter_vision_model import OpenRouterVisionAnalyzer
from chart_pattern_analyzer.vision_integration import VisionIntegration


def example_1_basic_vision_analysis():
    """
    Basic example of analyzing a single chart with the OpenRouter Vision Model
    """
    print("\n=== Example 1: Basic Vision Analysis ===\n")
    
    # Get API key from environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key: export OPENROUTER_API_KEY='your_key'")
        return
    
    # Initialize the vision analyzer
    analyzer = OpenRouterVisionAnalyzer(api_key=api_key)
    
    # Path to a sample chart image
    image_path = "examples/data/btc_daily.png"
    if not os.path.exists(image_path):
        print(f"Chart image not found: {image_path}")
        print("Please provide a valid chart image path")
        return
    
    print(f"Analyzing chart: {image_path}")
    print("Timeframe: Daily")
    print("Please wait, this may take 30-60 seconds...")
    
    # Analyze the chart
    result = analyzer.analyze_image(image_path, timeframe="Daily")
    
    # Check if analysis was successful
    if "error" in result:
        print(f"Analysis failed: {result['error']}")
        return
    
    # Print analysis summary
    if "analysis" in result:
        analysis = result["analysis"]
        
        print("\n--- Analysis Summary ---\n")
        
        for category in analyzer.analysis_categories:
            if category in analysis:
                confidence = analysis[category].get("confidence", "N/A")
                summary = analysis[category].get("summary", "No summary available")
                print(f"{category} (Confidence: {confidence}/10)")
                print(f"  {summary}")
                print()
        
        # Generate and print the summary report
        report = analyzer.generate_summary_report(result)
        print("\n--- Full Analysis Report ---\n")
        print(report)
        
        # Save the report to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"examples/output/{timestamp}_vision_analysis.md"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {report_path}")


def example_2_combined_analysis():
    """
    Example of using combined analysis with both traditional ML and vision models
    """
    print("\n=== Example 2: Combined Analysis ===\n")
    
    # Get API key from environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key: export OPENROUTER_API_KEY='your_key'")
        return
    
    # Initialize the vision integration
    integrator = VisionIntegration(
        api_key=api_key,
        model_path="models/pattern_detector.model",
        output_dir="examples/output"
    )
    
    # Path to a sample chart image
    image_path = "examples/data/eth_daily.png"
    if not os.path.exists(image_path):
        print(f"Chart image not found: {image_path}")
        print("Please provide a valid chart image path")
        return
    
    print(f"Analyzing chart with combined approach: {image_path}")
    print("Timeframe: Daily")
    print("Please wait, this may take 30-60 seconds...")
    
    # Analyze the chart with both methods
    result = integrator.analyze_chart(image_path, timeframe="Daily")
    
    # Print detected patterns
    if "combined_patterns" in result and result["combined_patterns"]:
        print("\n--- Detected Patterns ---\n")
        
        for pattern in sorted(result["combined_patterns"], 
                             key=lambda x: x["confidence"], reverse=True):
            print(f"Pattern: {pattern['pattern']}")
            print(f"Confidence: {pattern['confidence']:.2f}")
            print(f"Source: {pattern['source'].title()}")
            if "description" in pattern:
                print(f"Description: {pattern['description']}")
            print()
    else:
        print("No patterns detected with sufficient confidence")
    
    # Print confidence scores
    if "confidence_scores" in result:
        print("\n--- Confidence Scores ---\n")
        print(json.dumps(result["confidence_scores"], indent=2))
    
    # Print report location
    if "report_files" in result and "markdown" in result["report_files"]:
        print(f"\nReport generated: {result['report_files']['markdown']}")


def example_3_timeframe_comparison():
    """
    Example of comparing analysis across multiple timeframes
    """
    print("\n=== Example 3: Timeframe Comparison ===\n")
    
    # Get API key from environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key: export OPENROUTER_API_KEY='your_key'")
        return
    
    # Initialize the vision analyzer
    analyzer = OpenRouterVisionAnalyzer(api_key=api_key)
    
    # Paths to sample chart images for different timeframes
    image_paths = {
        "Daily": "examples/data/btc_daily.png",
        "4h": "examples/data/btc_4h.png",
        "1h": "examples/data/btc_1h.png"
    }
    
    # Verify all images exist
    all_exist = True
    for tf, path in image_paths.items():
        if not os.path.exists(path):
            print(f"Chart image not found: {path}")
            all_exist = False
    
    if not all_exist:
        print("Please provide valid chart image paths")
        return
    
    print("Comparing analysis across timeframes:")
    for tf, path in image_paths.items():
        print(f"- {tf}: {path}")
    
    print("Please wait, this may take 1-2 minutes...")
    
    # Compare timeframes
    result = analyzer.compare_timeframes(image_paths)
    
    # Print overall trend
    if "overall_trend" in result:
        trend = result["overall_trend"]
        confidence = result.get("confidence", 0)
        print(f"\nOverall trend: {trend} (Confidence: {confidence:.1f}/10)")
    
    # Print aligned signals
    if "aligned_signals" in result and result["aligned_signals"]:
        print("\n--- Aligned Signals ---\n")
        for signal in result["aligned_signals"]:
            print(f"- {signal}")
    else:
        print("\nNo aligned signals found across timeframes")
    
    # Print conflicting signals
    if "conflicting_signals" in result and result["conflicting_signals"]:
        print("\n--- Conflicting Signals ---\n")
        for signal in result["conflicting_signals"]:
            print(f"- {signal}")
    
    # Save the results to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"examples/output/{timestamp}_timeframe_comparison.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")


def example_4_pattern_evolution():
    """
    Example of tracking pattern evolution over time
    """
    print("\n=== Example 4: Pattern Evolution Tracking ===\n")
    
    # Get API key from environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key: export OPENROUTER_API_KEY='your_key'")
        return
    
    # Initialize the vision analyzer
    analyzer = OpenRouterVisionAnalyzer(api_key=api_key)
    
    # Paths to sample chart images in chronological order
    image_series = [
        {
            "timestamp": "2023-01-01",
            "image_path": "examples/data/btc_day1.png",
            "timeframe": "Daily"
        },
        {
            "timestamp": "2023-01-08",
            "image_path": "examples/data/btc_day2.png",
            "timeframe": "Daily"
        },
        {
            "timestamp": "2023-01-15",
            "image_path": "examples/data/btc_day3.png",
            "timeframe": "Daily"
        }
    ]
    
    # Verify all images exist
    all_exist = True
    for item in image_series:
        if not os.path.exists(item["image_path"]):
            print(f"Chart image not found: {item['image_path']}")
            all_exist = False
    
    if not all_exist:
        print("Please provide valid chart image paths")
        return
    
    print(f"Tracking pattern evolution over {len(image_series)} time points:")
    for item in image_series:
        print(f"- {item['timestamp']}: {item['image_path']}")
    
    print("Please wait, this may take 1-2 minutes...")
    
    # Track pattern evolution
    result = analyzer.track_pattern_evolution(image_series)
    
    # Print key developments
    if "key_developments" in result and result["key_developments"]:
        print("\n--- Key Developments ---\n")
        
        for dev in result["key_developments"]:
            timestamp = dev["timestamp"]
            developments = dev["developments"]
            
            print(f"{timestamp}")
            for item in developments:
                print(f"  - {item}")
            print()
    else:
        print("\nNo significant developments detected")
    
    # Print pattern evolution
    if "pattern_evolution" in result and result["pattern_evolution"]:
        print("\n--- Pattern Evolution ---\n")
        
        for entry in result["pattern_evolution"]:
            print(f"Timestamp: {entry['timestamp']}")
            print(f"Confidence: {entry['confidence']}/10")
            print(f"Patterns: {entry['patterns'][:100]}..." if len(entry['patterns']) > 100 else entry['patterns'])
            print()
    
    # Save the results to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"examples/output/{timestamp}_pattern_evolution.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")


def example_5_batch_analysis():
    """
    Example of batch analyzing multiple charts
    """
    print("\n=== Example 5: Batch Analysis ===\n")
    
    # Get API key from environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key: export OPENROUTER_API_KEY='your_key'")
        return
    
    # Initialize the vision integration
    integrator = VisionIntegration(
        api_key=api_key,
        model_path="models/pattern_detector.model",
        output_dir="examples/output"
    )
    
    # Directory containing chart images
    image_dir = "examples/data"
    if not os.path.exists(image_dir):
        print(f"Directory not found: {image_dir}")
        return
    
    print(f"Batch analyzing images in: {image_dir}")
    print("Please wait, this may take several minutes depending on the number of images...")
    
    # Batch analyze with combined approach
    results = integrator.batch_analyze_directory(
        image_dir=image_dir,
        pattern_filter=None  # Optional: filter for specific pattern
    )
    
    # Print summary
    if "total_images" in results:
        print(f"\nProcessed {results['total_images']} images")
        print(f"Found patterns in {results['successful_analyses']} images")
        
        # Print pattern statistics
        if results.get("pattern_counts"):
            print("\n--- Pattern Statistics ---\n")
            
            for pattern, count in sorted(results["pattern_counts"].items(), 
                                      key=lambda x: x[1], reverse=True):
                percentage = (count / results["successful_analyses"]) * 100 if results["successful_analyses"] > 0 else 0
                print(f"{pattern}: {count} ({percentage:.1f}%)")
    
    # Print summary report location
    if "summary_report" in results:
        print(f"\nSummary report: {results['summary_report']}")


if __name__ == "__main__":
    # Create examples/data directory if it doesn't exist
    os.makedirs("examples/data", exist_ok=True)
    os.makedirs("examples/output", exist_ok=True)
    
    # Placeholder message for example data
    if not any(os.path.exists(f"examples/data/{f}") for f in ["btc_daily.png", "eth_daily.png"]):
        print("Note: This example requires chart images in the examples/data directory.")
        print("Please add some chart images with filenames like:")
        print("  - examples/data/btc_daily.png")
        print("  - examples/data/eth_daily.png")
        print("  - examples/data/btc_4h.png")
        print("  - examples/data/btc_1h.png")
        print("\nFor example 4, you need chronological images named:")
        print("  - examples/data/btc_day1.png")
        print("  - examples/data/btc_day2.png")
        print("  - examples/data/btc_day3.png")
        print("\nYou can rename your own chart images to match these patterns.")
        print()
    
    # Run all examples
    try:
        example_1_basic_vision_analysis()
        example_2_combined_analysis()
        example_3_timeframe_comparison()
        example_4_pattern_evolution()
        example_5_batch_analysis()
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nError running examples: {e}")