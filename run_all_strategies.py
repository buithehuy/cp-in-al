"""Script to run multiple acquisition strategies."""
import subprocess
import sys
import os

# Add src to path for visualization
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# All available strategies
ALL_STRATEGIES = [
    "random",
    "entropy", 
    "least_confidence",
    "margin",
    "cp_size",
    "cp_v_shaped",
    "combined",
    "combined_v_shaped"
]

def run_strategies(strategies, num_rounds=21, plot_results=True, **kwargs):
    """Run training for specified strategies.
    
    Args:
        strategies: List of strategy names to run
        num_rounds: Number of AL rounds
        plot_results: Whether to generate plots after completion
        **kwargs: Additional arguments to pass to train.py
    """
    results = {}
    output_dirs = {}
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"Running strategy: {strategy.upper()}")
        print(f"{'='*80}\n")
        
        # Build command
        cmd = [sys.executable, "src/train.py", f"strategy={strategy}"]
        
        # Add num_rounds
        if num_rounds != 21:
            cmd.append(f"num_rounds={num_rounds}")
        
        # Add other kwargs
        for key, value in kwargs.items():
            cmd.append(f"{key}={value}")
        
        # Run command
        try:
            result = subprocess.run(cmd, check=True, cwd=".", capture_output=True, text=True)
            results[strategy] = "SUCCESS"
            
            # Extract output directory from output
            for line in result.stdout.split('\n'):
                if 'Results saved to:' in line:
                    output_path = line.split('Results saved to:')[1].strip()
                    output_dirs[strategy] = os.path.dirname(output_path)
                    break
                    
        except subprocess.CalledProcessError as e:
            print(f"ERROR running {strategy}: {e}")
            results[strategy] = "FAILED"
    
    # Print summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    for strategy, status in results.items():
        print(f"{strategy:20s}: {status}")
    print("="*80)
    
    # Generate plots if requested and we have results
    if plot_results and output_dirs:
        try:
            print("\n" + "="*80)
            print("GENERATING PLOTS")
            print("="*80)
            
            from utils.visualization import (
                load_results, 
                plot_all_metrics,
                print_summary_table
            )
            
            # Find common parent directory
            if output_dirs:
                parent_dir = os.path.dirname(list(output_dirs.values())[0])
                # Go up one more level to get outputs/
                parent_dir = os.path.dirname(parent_dir)
                
                print(f"Loading results from: {parent_dir}")
                results_dict = load_results(parent_dir)
                
                if results_dict:
                    # Print summary
                    print_summary_table(results_dict)
                    
                    # Generate plots
                    plot_dir = os.path.join(parent_dir, 'plots')
                    print(f"\nGenerating comprehensive plots...")
                    plot_all_metrics(results_dict, output_dir=plot_dir, show=True)
                    print(f"Plots saved to: {plot_dir}")
                else:
                    print("No results found to plot")
                    
        except Exception as e:
            print(f"Error generating plots: {e}")
            print("You can manually generate plots using: python plot_results.py outputs/")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run acquisition strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all strategies
  python run_all_strategies.py --all
  
  # Run specific strategies
  python run_all_strategies.py -s entropy combined
  
  # Quick test with specific strategies
  python run_all_strategies.py -s entropy cp_v_shaped --quick
  
  # Run without auto-plotting
  python run_all_strategies.py --all --no-plot
  
  # Available strategies:
  - random, entropy, least_confidence, margin
  - cp_size, cp_v_shaped
  - combined, combined_v_shaped
        """
    )
    
    parser.add_argument(
        "-s", "--strategies", 
        nargs='+', 
        choices=ALL_STRATEGIES,
        help="Specific strategies to run (space-separated)"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all strategies"
    )
    parser.add_argument(
        "--num_rounds", 
        type=int, 
        default=21, 
        help="Number of AL rounds (default: 21)"
    )
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Quick test mode (2 rounds, small data)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable automatic plot generation"
    )
    
    args = parser.parse_args()
    
    # Determine which strategies to run
    if args.all:
        strategies_to_run = ALL_STRATEGIES
    elif args.strategies:
        strategies_to_run = args.strategies
    else:
        parser.error("Please specify either --all or -s/--strategies")
    
    print(f"Running strategies: {', '.join(strategies_to_run)}")
    
    if args.quick:
        # Quick test: 2 rounds, small data
        run_strategies(
            strategies_to_run,
            num_rounds=2,
            plot_results=not args.no_plot,
            initial_labeled=100,
            budget_per_round=50,
            calibration_size=100,
            **{"trainer.epochs_per_round": 1}
        )
    else:
        # Full experiment
        run_strategies(
            strategies_to_run, 
            num_rounds=args.num_rounds,
            plot_results=not args.no_plot
        )
