import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from TIG_experiment_system import EnhancedExperimentSystem, ExperimentConfig
from config.azure_config import setup_lab_environment

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run 24-point / 67-problem experiments')
    parser.add_argument('--runs', type=int, default=3, help='Runs per problem')
    parser.add_argument('--parallel', type=int, default=2, help='Parallelism')
    parser.add_argument('--dataset', choices=['24point','comprehensive_24point_67_dataset'], default='24point')
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args()

def main():
    args = parse_arguments()
    setup_lab_environment()
    
    config = ExperimentConfig(
        datasets=[args.dataset],
        methods=['cot','tot','got','aot','egot'],
        runs_per_experiment=args.runs,
        parallel_experiments=args.parallel,
        output_dir=f"results/{args.dataset}_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save_intermediate_results=True,
        enable_rgwl_analysis=True,
        save_visualizations=False
    )

    if args.quiet:
        os.environ.setdefault('BENCHMARK_QUIET','1')

    system = EnhancedExperimentSystem(config)
    t0 = time.time()
    system.run_comprehensive_experiments()
    print(f"Done, elapsed {time.time()-t0:.2f}s")

if __name__ == '__main__':
    sys.exit(main()) 