import argparse
import json
import os
import pandas as pd

from utils.config_loader import load_config
from utils.logger import Logger
from utils.data_loader import load_dataset
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
    all_coding_agreement = []
    all_disc_agreement = []
    has_ground_truth = None
    
    for run_id in range(num_runs):
        print(f"--- Run {run_id + 1}/{num_runs} ---")
        result, log_dir = run_single(config, run_id)
        
        coding_result = result.get('coding', {})
        has_ground_truth = coding_result.get('has_ground_truth', False)
        
        if has_ground_truth:
            coding_acc = coding_result.get('accuracy', 0)
            all_coding_acc.append(coding_acc)
            print(f"   Coding: {coding_acc:.2%}")
        
        if 'agreement_rate' in coding_result:
            all_coding_agreement.append(coding_result['agreement_rate'])
            if not has_ground_truth:
                print(f"   Coding Agreement: {coding_result['agreement_rate']:.2%}")
        
        if result.get('discussion'):
            disc_result = result['discussion']
            if has_ground_truth and 'accuracy' in disc_result:
                all_disc_acc.append(disc_result['accuracy'])
            if 'agreement_rate' in disc_result:
                all_disc_agreement.append(disc_result['agreement_rate'])
    
    # Aggregate stats
    print("========= AGGREGATE STATISTICS =========")
    
    agg_data = {"num_runs": num_runs, "has_ground_truth": has_ground_truth}
    
    if has_ground_truth and all_coding_acc:
        stats = calc_stats(all_coding_acc)
        print(f"\nCoding Accuracy (n={num_runs}):")
        print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
        agg_data["coding_accuracy"] = stats
        
        if all_disc_acc:
            disc_stats = calc_stats(all_disc_acc)
            print(f"\nDiscussion Accuracy:")
            print(f"  Mean: {disc_stats['mean']:.4f}, Std: {disc_stats['std']:.4f}")
            agg_data["discussion_accuracy"] = disc_stats
    
    if all_coding_agreement:
        agreement_stats = calc_stats(all_coding_agreement)
        print(f"\nCoding Agreement Rate (n={num_runs}):")
        print(f"  Mean: {agreement_stats['mean']:.4f}, Std: {agreement_stats['std']:.4f}")
        agg_data["coding_agreement_rate"] = agreement_stats
    
    if all_disc_agreement:
        disc_agreement_stats = calc_stats(all_disc_agreement)
        print(f"\nDiscussion Agreement Rate:")
        print(f"  Mean: {disc_agreement_stats['mean']:.4f}, Std: {disc_agreement_stats['std']:.4f}")
        agg_data["discussion_agreement_rate"] = disc_agreement_stats
    
    # Save aggregate results
    results_dir = os.path.join(config['paths']['result_path'], config['settings']['model'])
    os.makedirs(results_dir, exist_ok=True)
    
    from datetime import datetime
    agg_file = os.path.join(results_dir, f"aggregate_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{num_runs}runs.json")
    
    with open(agg_file, 'w') as f:
        json.dump(agg_data, f, indent=2)
    
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
        dataset_info = load_dataset(data_path, task_type_override=config.get('task_type'))
        
        # Find ground truth column from first task
        gt_column = None
        for task_info in dataset_info.tasks.values():
            if task_info.ground_truth_column:
                gt_column = task_info.ground_truth_column
                break
        
        ground_truth = load_ground_truth(dataset_info.df, ground_truth_column=gt_column)
        evaluate_results_file(args.evaluate, ground_truth)
        return
    
    if args.runs > 1:
        run_multiple(config, args.runs)
    else:
        result, _ = run_single(config)
        coding_result = result.get('coding', {})
        if coding_result.get('has_ground_truth'):
            print(f"\nCoding Accuracy: {coding_result['accuracy']:.2%}")
        elif 'agreement_rate' in coding_result:
            print(f"\nCoding Agreement Rate: {coding_result['agreement_rate']:.2%}")
        else:
            print(f"\nCoding Phase: {coding_result.get('total', 0)} texts coded")


if __name__ == "__main__":
    main()