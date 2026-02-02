"""Script to plot results from saved experiments."""
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.visualization import (
    load_results,
    plot_accuracy_vs_samples,
    plot_cp_coverage,
    plot_cp_set_size,
    plot_all_metrics,
    print_summary_table,
    export_accuracy_table
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot results from saved experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all metrics from outputs directory
  python plot_results.py outputs/
  
  # Plot only accuracy
  python plot_results.py outputs/ --plot accuracy
  
  # Save plots without showing
  python plot_results.py outputs/ --save-dir plots/ --no-show
  
  # Print summary table only
  python plot_results.py outputs/ --table-only
  
  # Export accuracy table to CSV
  python plot_results.py outputs/ --export-csv accuracy_results.csv
        """
    )
    
    parser.add_argument(
        'results_dir',
        help='Directory containing results (e.g., outputs/)'
    )
    parser.add_argument(
        '--plot',
        choices=['all', 'accuracy', 'coverage', 'set_size'],
        default='all',
        help='Which plots to generate (default: all)'
    )
    parser.add_argument(
        '--save-dir',
        help='Directory to save plots (optional)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots (only save)'
    )
    parser.add_argument(
        '--table-only',
        action='store_true',
        help='Only print summary table, no plots'
    )
    parser.add_argument(
        '--export-csv',
        help='Export accuracy table to CSV file (e.g., --export-csv results.csv)'
    )
    parser.add_argument(
        '--ylim',
        nargs=2,
        type=float,
        metavar=('MIN', 'MAX'),
        help='Y-axis limits for accuracy plot (e.g., --ylim 80 100)'
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results_dict = load_results(args.results_dir)
    
    if not results_dict:
        print("No results found!")
        return
    
    print(f"Found {len(results_dict)} strategies: {', '.join(results_dict.keys())}")
    
    # Print summary table
    print_summary_table(results_dict)
    
    # Detect dataset from results directory path
    dataset = None
    for ds_name in ['svhn', 'cifar10', 'cifar100', 'stl10']:
        if ds_name in args.results_dir.lower():
            dataset = ds_name
            print(f"Detected dataset: {dataset}")
            break
    
    # Export CSV if requested
    if args.export_csv:
        export_accuracy_table(results_dict, args.export_csv)
    
    if args.table_only:
        return
    
    # Generate plots
    show = not args.no_show
    
    if args.plot == 'all':
        print("\nGenerating comprehensive plots...")
        ylim = tuple(args.ylim) if args.ylim else None
        plot_all_metrics(results_dict, output_dir=args.save_dir, show=show, dataset=dataset, ylim=ylim)
        
    elif args.plot == 'accuracy':
        print("\nGenerating accuracy plot...")
        save_path = os.path.join(args.save_dir, 'accuracy.png') if args.save_dir else None
        ylim = tuple(args.ylim) if args.ylim else None
        plot_accuracy_vs_samples(results_dict, save_path=save_path, dataset=dataset, ylim=ylim)
        
    elif args.plot == 'coverage':
        print("\nGenerating coverage plot...")
        save_path = os.path.join(args.save_dir, 'coverage.png') if args.save_dir else None
        plot_cp_coverage(results_dict, save_path=save_path)
        
    elif args.plot == 'set_size':
        print("\nGenerating set size plot...")
        save_path = os.path.join(args.save_dir, 'set_size.png') if args.save_dir else None
        plot_cp_set_size(results_dict, save_path=save_path)


if __name__ == '__main__':
    main()
