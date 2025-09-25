import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import os
import logging

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from TIG_experiment_system import EnhancedExperimentSystem, ExperimentConfig
from config.azure_config import setup_lab_environment


def _apply_quiet_logging():
    try:
        os.environ.setdefault('AZURE_LOG_LEVEL', 'WARNING')
    except Exception:
        pass
    for name in [
        'azure', 'azure.core', 'azure.identity', 'azure.ai', 'openai',
        'httpx', 'urllib3', 'requests'
    ]:
        try:
            logging.getLogger(name).setLevel(logging.WARNING)
        except Exception:
            pass
    try:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    except Exception:
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run LeetCode dataset experiments')
    parser.add_argument('--test', action='store_true', help='Test mode: run only 5 problems')
    parser.add_argument('--methods', nargs='+', default=['cot', 'tot', 'got', 'aot', 'egot'], 
                       help='Specify reasoning methods')
    parser.add_argument('--runs', type=int, default=3, help='Runs per problem')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel experiments (default 1 for stability)')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode: suppress third-party HTTP logs')
    return parser.parse_args()


def load_leetcode_config():
    """Load optional LeetCode custom config; fallback to empty when missing."""
    try:
        config_path = project_root / "config/leetcode_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def validate_leetcode_setup():
    print("🔍 Validating LeetCode experiment setup...")
    dataset_path = project_root / "data/leetcode_dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"LeetCode dataset file not found: {dataset_path}")
    print("  ✅ LeetCode dataset exists")
    print("  ✅ LeetCode experiment setup validated")
    return True


def create_leetcode_experiment_config(args):
    custom_cfg = load_leetcode_config()
    config = ExperimentConfig(
        datasets=["leetcode"],
        methods=args.methods,
        runs_per_experiment=args.runs,
        max_nodes_per_experiment=35,
        parallel_experiments=args.parallel,
        output_dir=f"results/leetcode_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save_intermediate_results=True,
        enable_rgwl_analysis=True,
        save_visualizations=False,
        custom_config=custom_cfg
    )
    return config


def main():
    args = parse_arguments()
    print("🚀 Starting LeetCode dataset experiments")
    print("=" * 60)

    try:
        os.chdir(str(project_root))
    except Exception:
        pass

    validate_leetcode_setup()

    print("\n🔧 Setting up experiment environment...")
    if args.quiet:
        _apply_quiet_logging()
        os.environ.setdefault('BENCHMARK_QUIET', '1')
    setup_lab_environment()

    print("\n📋 Creating experiment configuration...")
    config = create_leetcode_experiment_config(args)

    print("\n🏗️ Creating experiment system...")
    experiment_system = EnhancedExperimentSystem(config)

    if args.test:
        print("🧪 Test mode: limit to 5 problems")
        experiment_system.set_dataset_limit(5)

    print(f"\n📊 Experiment configuration:")
    print(f"  - Datasets: {config.datasets}")
    print(f"  - Methods: {config.methods}")
    print(f"  - Runs per problem: {config.runs_per_experiment}")
    print(f"  - Parallelism: {config.parallel_experiments}")
    print(f"  - Output directory: {config.output_dir}")

    print("\n🎯 Starting LeetCode experiments...")
    start_time = time.time()

    try:
        experiment_system.run_comprehensive_experiments()
        duration = time.time() - start_time
        print(f"\n✅ LeetCode experiments completed! ⏱️ {duration:.2f} s")
        print(f"📁 Results saved: {config.output_dir}")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 