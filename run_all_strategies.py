"""Script to run multiple acquisition strategies."""
import subprocess
import sys

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

def run_strategies(strategies, num_rounds=21, **kwargs):
    """Run training for specified strategies.
    
    Args:
        strategies: List of strategy names to run
        num_rounds: Number of AL rounds
        **kwargs: Additional arguments to pass to train.py
    """
    results = {}
    
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
            result = subprocess.run(cmd, check=True, cwd=".")
            results[strategy] = "SUCCESS"
        except subprocess.CalledProcessError as e:
            print(f"ERROR running {strategy}: {e}")
            results[strategy] = "FAILED"
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for strategy, status in results.items():
        print(f"{strategy:20s}: {status}")
    

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
            initial_labeled=100,
            budget_per_round=50,
            calibration_size=100,
            **{"trainer.epochs_per_round": 1}
        )
    else:
        # Full experiment
        run_strategies(strategies_to_run, num_rounds=args.num_rounds)
