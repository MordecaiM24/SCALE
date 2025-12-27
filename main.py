import argparse
import json
import os
import pandas as pd

from utils.config_loader import load_config
from utils.logger import Logger
from simulation.content_analysis_simulation import ContentAnalysisSimulation
from evaluator import Evaluator, load_ground_truth, calc_stats, evaluate_results_file


def run_single(config, run_id=0):
    """Run a single simulation."""
    logger = Logger(
        dataset_name=config['dataset_name'],
        model_name=config['settings']['model'],
        seed=config['settings']['seed'] + run_id
    )
    logger.log(f"Configuration for dataset '{config['dataset_name']}' loaded successfully.\n")
    logger.log(f"Run ID: {run_id}\n")
    
    sim = ContentAnalysisSimulation(config, logger, run_id=run_id)
    return sim.run(), logger.log_dir


def run_multiple(config, num_runs):
    """Run multiple simulations and aggregate statistics."""
    print(f"\nRunning {num_runs} simulation(s)...\n")
    
    all_coding_acc = []
    all_disc_acc = []
    
    for run_id in range(num_runs):
        print(f"--- Run {run_id + 1}/{num_runs} ---")
        result, log_dir = run_single(config, run_id)
        
        coding_acc = result.get('coding', {}).get('accuracy', 0)
        all_coding_acc.append(coding_acc)
        
        if result.get('discussion'):
            all_disc_acc.append(result['discussion']['accuracy'])
        
        print(f"   Coding: {coding_acc:.2%}")
    
    # Aggregate stats
    print("========= AGGREGATE STATISTICS =========")
    
    stats = calc_stats(all_coding_acc)
    print(f"\nCoding Accuracy (n={num_runs}):")
    print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    print(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
    
    if all_disc_acc:
        disc_stats = calc_stats(all_disc_acc)
        print(f"\nDiscussion Accuracy:")
        print(f"  Mean: {disc_stats['mean']:.4f}, Std: {disc_stats['std']:.4f}")
    
    # Save aggregate results
    results_dir = os.path.join(config['paths']['result_path'], config['settings']['model'])
    os.makedirs(results_dir, exist_ok=True)
    
    from datetime import datetime
    agg_file = os.path.join(results_dir, f"aggregate_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{num_runs}runs.json")
    
    with open(agg_file, 'w') as f:
        json.dump({
            "num_runs": num_runs,
            "coding_accuracy": stats,
            "discussion_accuracy": calc_stats(all_disc_acc) if all_disc_acc else None
        }, f, indent=2)
    
    print(f"\nResults saved to: {agg_file}")


def main():
    parser = argparse.ArgumentParser(description="SCALE: Content Analysis Simulation")
    parser.add_argument('--path', type=str, default='./configs/config.json', help="Config file path")
    parser.add_argument('--runs', type=int, default=1, help="Number of runs for statistics")
    parser.add_argument('--evaluate', type=str, help="Evaluate existing results file")
    args = parser.parse_args()

    # 1. Load configuration
    config = load_config(args.path)
    
    if args.evaluate:
        data_path = os.path.join(config['paths']['data_path'], config['dataset_name'], 'data.xlsx')
        df = pd.read_excel(data_path)
        ground_truth = load_ground_truth(df)
        evaluate_results_file(args.evaluate, ground_truth)
        return
    
    if args.runs > 1:
        run_multiple(config, args.runs)
    else:
        result, _ = run_single(config)
        print(f"\nCoding Accuracy: {result['coding']['accuracy']:.2%}")


if __name__ == "__main__":
    main()