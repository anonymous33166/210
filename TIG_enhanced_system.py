import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

sys.path.append(str(Path(__file__).parent))

def setup_environment():
    try:
        from config.azure_config import setup_lab_environment
        setup_lab_environment()
        print("[✅] Lab Azure OpenAI environment variables set")
    except ImportError:
        print("[⚠️] Failed to import lab config; using manual environment variables")
    
    azure_env_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT", 
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION"
    ]
    
    lab_env_vars = [
        "ENDPOINT_URL",
        "DEPLOYMENT_NAME"
    ]
    
    has_azure_config = (
        os.getenv("AZURE_OPENAI_API_KEY") and 
        (os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("ENDPOINT_URL")) and
        (os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("DEPLOYMENT_NAME"))
    )
    
    if has_azure_config:
        print("[✅] Azure OpenAI configuration is set")
    else:
        print("[⚠️] Azure OpenAI configuration incomplete; will try alternative models")
        
        alternative_keys = ["OPENAI_API_KEY", "DEEPSEEK_API_KEY", "QWEN_API_KEY", "CLAUDE_API_KEY"]
        available_alternatives = [key for key in alternative_keys if os.getenv(key)]
        
        if available_alternatives:
            print(f"[✅] Detected available API keys: {available_alternatives}")
        else:
            print("[❌] No usable API keys detected")
            print("[💡] Please set one of the following:")
            print("   Azure OpenAI: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT")
            print("   or another provider's API key")
            return False
    
    directories = ["results", "logs", "cache", "visualizations"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    return True

def create_optimized_experiment_config() -> 'ExperimentConfig':

    from TIG_experiment_system import ExperimentConfig
    
    config = ExperimentConfig(
        methods=["cot", "tot", "got", "aot", "egot"],
        datasets=["24point", "creative_writing", "sorting", "mathematics", "coding", "legal"],
        runs_per_experiment=5, 
        max_nodes_per_experiment=25, 
        parallel_experiments=3, 
        save_intermediate_results=True,
        enable_rgwl_analysis=True,
        output_dir=f"results/enhanced_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    print("[⚙️] Experiment config optimized for highest-accuracy mode")
    return config

def run_single_method_test(method: str = "cot", problem: str = None):
    print(f"🧪 Running single-method test: {method.upper()}")
    print("=" * 60)
    
    try:
        from TIG_reasoning_framework import AdvancedReasoningFramework
        from TIG_multi_model_backend import create_enhanced_multi_model_backend
        from TIG_rgwl_kernel import create_enhanced_rgwl_kernel
        
        llm_manager = create_enhanced_multi_model_backend()
        framework = AdvancedReasoningFramework({}, llm_manager)
        rgwl_kernel = create_enhanced_rgwl_kernel()
        
        framework.configure_for_method(method)
        
        framework.set_judgment_criteria(
            "If a complete expression using all four numbers evaluates to 24, accept it as the answer.",
            "If all four numbers are used and the result is not 24, terminate the attempt."
        )
        
        if not problem:
            problem = "Using 3, 8, 3, and 8, compute 24"
        
        print(f"[📝] Test problem: {problem}")
        print(f"[⚙️] Reasoning method: {method.upper()}")
        
        start_time = time.time()
        result = framework.start_reasoning(problem, max_total_nodes=20)
        execution_time = time.time() - start_time
        
        print(f"\n📊 Test result:")
        print(f"   Success: {'✅' if result['success'] else '❌'}")
        print(f"   Answer: {result.get('final_answer', 'N/A')}")
        print(f"   Execution time: {execution_time:.2f}s")
        print(f"   Total nodes: {result.get('total_nodes', 0)}")
        print(f"   Total rounds: {result.get('total_rounds', 0)}")
        
        if result.get('metrics'):
            print(f"   Metrics:")
            for key, value in result['metrics'].items():
                print(f"     {key}: {value}")
    
        output_file = f"visualizations/single_test_{method}_{datetime.now().strftime('%H%M%S')}.png"
        framework.save_graph_visualization(output_file)
        
        return result
        
    except Exception as e:
        print(f"❌ Single-method test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_comprehensive_benchmark():
    print("🚀 Starting comprehensive benchmark system")
    print("=" * 80)
    
    try:
        config = create_optimized_experiment_config()
        
        from TIG_experiment_system import EnhancedExperimentSystem
        experiment_system = EnhancedExperimentSystem(config)
        
        print(f"[📊] Running comprehensive experiments...")
        print(f"   Methods: {config.methods}")
        print(f"   Datasets: {config.datasets}")
        print(f"   Runs per problem: {config.runs_per_experiment}")
        
        start_time = time.time()
        results = experiment_system.run_comprehensive_experiments()
        total_time = time.time() - start_time
        
        print(f"\n🎉 Benchmark completed!")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Output dir: {config.output_dir}")
        
        overall = results["overall_stats"]
        print(f"\n📊 Key metrics:")
        print(f"   Total experiments: {overall['total_experiments']}")
        print(f"   Success rate: {overall['overall_success_rate']:.2%}")
        print(f"   Avg execution time: {overall['avg_execution_time']:.2f}s")
        
        best_method = None
        best_rate = 0
        for method, stats in results["results_by_method"].items():
            if stats["success_rate"] > best_rate:
                best_rate = stats["success_rate"]
                best_method = method
        
        if best_method:
            print(f"   Best method: {best_method.upper()} (success rate: {best_rate:.2%})")
        
        return results
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_rgwl_analysis_demo():
    print("🧮 RGWL kernel analysis demo")
    print("=" * 50)
    
    try:
        from TIG_rgwl_kernel import create_enhanced_rgwl_kernel
        
        kernel = create_enhanced_rgwl_kernel()
        
        graph1 = {
            "nodes": {
                "root": {"content": "compute 24-point problem", "cluster_label": 0},
                "node1": {"content": "try addition", "cluster_label": 1},
                "node2": {"content": "8+8+8=24", "cluster_label": 2},
                "node3": {"content": "verify result", "cluster_label": 3}
            },
            "edges": [("root", "node1"), ("node1", "node2"), ("node2", "node3")],
            "metadata": {
                "lambda_ratio": 0.85,
                "answer_paths": [["root", "node1", "node2", "node3"]],
                "answer_nodes": ["node3"],
                "total_nodes": 4,
                "effective_nodes": 4
            }
        }
        
        graph2 = {
            "nodes": {
                "root": {"content": "compute 24-point problem", "cluster_label": 0},
                "node1": {"content": "try multiplication", "cluster_label": 1},
                "node2": {"content": "6×4=24", "cluster_label": 2},
                "node3": {"content": "answer confirmed", "cluster_label": 3}
            },
            "edges": [("root", "node1"), ("node1", "node2"), ("node2", "node3")],
            "metadata": {
                "lambda_ratio": 0.90,
                "answer_paths": [["root", "node1", "node2", "node3"]],
                "answer_nodes": ["node3"],
                "total_nodes": 4,
                "effective_nodes": 4
            }
        }
        
        print("[🧮] Computing RGWL similarity for two reasoning graphs...")
        result = kernel.compute_kernel(graph1, graph2, h=3, save_intermediate=True)
        
        print(f"\n📊 RGWL analysis result:")
        print(f"   Kernel value: {result['kernel_value']:.6f}")
        print(f"   Answer term sum: {result['answer_term_sum']:.6f}")
        print(f"   Full term sum: {result['full_term_sum']:.6f}")
        print(f"   Lambda product: {result['lambda_product']:.6f}")
        
        output_file = f"visualizations/rgwl_demo_{datetime.now().strftime('%H%M%S')}.png"
        kernel.visualize_kernel_computation(result, output_file)
        
        print(f"[💾] RGWL visualization saved: {output_file}")
        
        return result
        
    except Exception as e:
        print(f"❌ RGWL demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_backend_test():
    print("🔧 Multi-model backend test")
    print("=" * 40)
    
    try:
        from TIG_multi_model_backend import create_enhanced_multi_model_backend
        
        manager = create_enhanced_multi_model_backend()
        
        available_backends = manager.get_available_backends()
        print(f"[📋] Available backends: {available_backends}")
        
        if not available_backends:
            print("[❌] No available backend")
            return None
        
        test_prompt = "Briefly explain what artificial intelligence is."
        print(f"[📝] Test prompt: {test_prompt}")
        
        result = manager.generate(test_prompt)
        
        print(f"\n📊 Generation result:")
        print(f"   Success: {'✅' if result['success'] else '❌'}")
        print(f"   Backend: {result.get('backend', 'None')}")
        print(f"   Generation time: {result.get('generation_time', 0):.2f}s")
        print(f"   Result: {result['result'][:100]}...")
        
        stats = manager.get_all_stats()
        print(f"\n📈 Backend stats:")
        for backend_name, stat in stats.items():
            print(f"   {backend_name}:")
            print(f"     requests: {stat['request_count']}")
            print(f"     success rate: {stat['success_rate']:.2%}")
            print(f"     total tokens: {stat['total_tokens']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    
    parser = argparse.ArgumentParser(description="Refactored Prompt Engineering Benchmark System")
    parser.add_argument("--mode", choices=["test", "single", "benchmark", "rgwl", "backend"], 
                       default="test", help="Run mode")
    parser.add_argument("--method", choices=["cot", "tot", "got", "aot", "egot"], 
                       default="cot", help="Method to use in single-method test")
    parser.add_argument("--problem", type=str, help="Custom test problem")
    
    args = parser.parse_args()
    
    print("🎯 Refactored Prompt Engineering Benchmark System")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run mode: {args.mode}")
    print("=" * 80)
    
    if not setup_environment():
        print("❌ Environment setup failed")
        return 1
    
    try:
        if args.mode == "test":
            print("🧪 Running basic system tests...")
            
            print("\n1️⃣ Multi-model backend test")
            backend_result = run_backend_test()
            
            print("\n2️⃣ Single-method test")
            single_result = run_single_method_test("cot", "Using 1, 1, 8, and 8, compute 24")
            
            print("\n3️⃣ RGWL kernel demo")
            rgwl_result = run_rgwl_analysis_demo()
            
            if backend_result and single_result and rgwl_result:
                print("\n✅ All basic tests passed!")
            else:
                print("\n⚠️ Some tests failed; please check configuration")
        
        elif args.mode == "single":
            run_single_method_test(args.method, args.problem)
        
        elif args.mode == "benchmark":
            run_comprehensive_benchmark()
        
        elif args.mode == "rgwl":
            run_rgwl_analysis_demo()
        
        elif args.mode == "backend":
            run_backend_test()
        
        print(f"\n🎉 Run complete: mode = {args.mode}")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚡ Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Run failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 