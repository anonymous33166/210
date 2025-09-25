import json
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from dataclasses import dataclass, field
import traceback
import re
from data.dataset_loader import DatasetLoader

@dataclass
class ExperimentConfig:
    methods: List[str] = field(default_factory=lambda: ["cot", "tot", "got", "aot", "egot"])
    datasets: List[str] = field(default_factory=lambda: ["24point", "creative_writing", "sorting", "mathematics", "coding", "legal"])
    runs_per_experiment: int = 5
    max_nodes_per_experiment: int = 20
    parallel_experiments: int = 4
    save_intermediate_results: bool = True
    enable_rgwl_analysis: bool = True
    save_visualizations: bool = False  
    output_dir: str = "results/enhanced_experiments"
    custom_config: Optional[Dict[str, Any]] = None

@dataclass
class ExperimentResult:
    method: str
    dataset: str
    problem: str
    problem_index: int
    run_id: int
    success: bool
    execution_time: float
    answer: str
    reasoning_graph: Dict[str, Any]
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

class EnhancedExperimentSystem:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_loader = DatasetLoader()
        self.experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.all_results: List[ExperimentResult] = []
        self.collected_graphs: List[Dict[str, Any]] = []
        self.dataset_limit = None
        self.experiment_stats = {
            "total_experiments": 0,
            "successful_experiments": 0,
            "failed_experiments": 0,
            "start_time": None,
            "end_time": None,
            "duration": 0
        }
        
        self._setup_logging()
        
        print(f"[🎯] Enhanced Experiment System initialized")
        print(f"[📊] Experiment ID: {self.experiment_id}")
        print(f"[📁] Output directory: {self.output_dir}")
    
    def set_dataset_limit(self, limit: int):
        self.dataset_limit = limit
        print(f"[🔒] Dataset size limit set to: {limit}")
    
    def _setup_logging(self):
        log_file = self.output_dir / f"experiment_{self.experiment_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def run_comprehensive_experiments(self) -> Dict[str, Any]:
        print("🚀 Starting Enhanced Experiment System")
        print("=" * 80)
        
        self.experiment_stats["start_time"] = time.time()
        
        try:
            
            from TIG_reasoning_framework import AdvancedReasoningFramework
            from TIG_multi_model_backend import create_enhanced_multi_model_backend
            from TIG_rgwl_kernel import create_enhanced_rgwl_kernel
            
           
            llm_manager = create_enhanced_multi_model_backend()
            rgwl_kernel = create_enhanced_rgwl_kernel() if self.config.enable_rgwl_analysis else None
            datasets = self._load_all_datasets()
            
            
            total_experiments = 0
            for dataset_name, problems in datasets.items():
                if dataset_name in self.config.datasets:
                    total_experiments += len(problems) * len(self.config.methods) * self.config.runs_per_experiment
            
            self.experiment_stats["total_experiments"] = total_experiments
            
            print(f"📊 Experiment scale:")
            print(f"  - Reasoning methods: {self.config.methods}")
            print(f"  - Datasets: {list(datasets.keys())}")
            print(f"  - Runs per problem: {self.config.runs_per_experiment}")
            print(f"  - Total experiments: {total_experiments}")
            print(f"  - Parallelism: {self.config.parallel_experiments}")
            
           
            for dataset_name, problems in datasets.items():
                if dataset_name not in self.config.datasets:
                    continue
                
                print(f"\n{'='*25} {dataset_name.upper()} DATASET EXPERIMENT {'='*25}")
                
                dataset_results = self._run_dataset_experiments(
                    dataset_name, problems, llm_manager, rgwl_kernel
                )
                
                self.all_results.extend(dataset_results)
                
                
                if self.config.save_intermediate_results:
                    self._save_intermediate_results(dataset_name, dataset_results)
                
                try:
                    if self.config.save_visualizations:
                        from TIG_reasoning_framework import AdvancedReasoningFramework
                        for r in dataset_results:
                            try:
                                vis_file = self.output_dir / f"viz_{r.method}_{dataset_name}_p{r.problem_index}_r{r.run_id}.png"
                               
                                if not vis_file.exists():
                                    meta_file = self.output_dir / f"vizmeta_{r.method}_{dataset_name}_p{r.problem_index}_r{r.run_id}.json"
                                    with open(meta_file, "w", encoding="utf-8") as f:
                                        json.dump(r.reasoning_graph or {}, f, ensure_ascii=False)
                            except Exception:
                                pass
                except Exception:
                    pass
            
          
            final_results = self._compute_final_results()
            
          
            self._generate_comprehensive_report(final_results)
            

            if self.config.enable_rgwl_analysis and rgwl_kernel:
                self._perform_rgwl_analysis(rgwl_kernel)
            
            self.experiment_stats["end_time"] = time.time()
            self.experiment_stats["duration"] = self.experiment_stats["end_time"] - self.experiment_stats["start_time"]
            
            print(f"\n✅ Experiments completed!")
            print(f"   Total time: {self.experiment_stats['duration']:.1f}s")
            print(f"   Success rate: {self.experiment_stats['successful_experiments'] / self.experiment_stats['total_experiments'] * 100:.1f}%")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Experiment run failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _load_all_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        datasets = {}
        
        for dataset_name in self.config.datasets:
            try:
                if dataset_name == "24point":
                    problems = self.dataset_loader.load_24point_extended()
                elif dataset_name == "comprehensive_24point_67_dataset":
                    problems = self.dataset_loader.load_24point_comprehensive_67()
                elif dataset_name == "creative_writing":
                    problems = self.dataset_loader.load_creative_writing()
                elif dataset_name == "sorting":
                    problems = self.dataset_loader.load_sorting_dataset()
                elif dataset_name == "mathematics":
                    problems = self.dataset_loader.load_mathematics()
                elif dataset_name == "coding":
                    problems = self.dataset_loader.load_coding()
                elif dataset_name == "legal":
                    problems = self.dataset_loader.load_legal()
                elif dataset_name == "gaokao_math":
                    problems = self.dataset_loader.load_gaokao_math()
                elif dataset_name == "leetcode":
                    problems = self.dataset_loader.load_leetcode()
                
                else:
                    self.logger.warning(f"Unknown dataset: {dataset_name}")
                    continue
                
                datasets[dataset_name] = problems
                print(f"[📂] Loaded dataset {dataset_name}: {len(problems)} problems")
                
                
                if self.dataset_limit and len(problems) > self.dataset_limit:
                    original_count = len(problems)
                    problems = problems[:self.dataset_limit]
                    datasets[dataset_name] = problems
                    print(f"[🔒] Applied dataset limit: {original_count} -> {len(problems)} problems")
                
            except Exception as e:
                self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
        
        return datasets
    
    def _run_dataset_experiments(self, 
                               dataset_name: str, 
                               problems: List[Dict[str, Any]],
                               llm_manager,
                               rgwl_kernel) -> List[ExperimentResult]:
        
        dataset_results = []
        
        for method_idx, method in enumerate(self.config.methods):
            print(f"\n[{method_idx+1}/{len(self.config.methods)}] Method: {method.upper()}")
        
            from TIG_reasoning_framework import AdvancedReasoningFramework
            framework = AdvancedReasoningFramework({
                "save_visualizations": self.config.save_visualizations,
                "dataset_name": dataset_name,
                "quiet": (os.environ.get("BENCHMARK_QUIET") == "1"),
                "llm_sampling": ({"temperature": 0.0, "top_p": 0.0} if dataset_name in ("gaokao_math", "legal") else {})
            }, llm_manager)
            
          
            framework.configure_for_method(method)
            
          
            framework.max_total_nodes = 50  
            framework.max_rounds = 10      
            sweep_counts = []
            try:
                if self.config.custom_config:
                    mode_cfg = (self.config.custom_config.get("reasoning_modes", {}) or {}).get(str(method).lower(), {})
                    sweep_counts = mode_cfg.get("sweep_node_counts") or []
            except Exception:
                sweep_counts = []

            try:
                if dataset_name in ("gaokao_math", "legal"):
                    framework.reasoning_paradigm.continue_until_answer = True
            except Exception:
                pass
            
           
            result_criterion, termination_criterion = self._get_dataset_criteria(dataset_name)
            framework.set_judgment_criteria(result_criterion, termination_criterion)
            
            try:
                self._apply_custom_config_to_framework(framework, method, dataset_name)
            except Exception as _:
                pass
            
    
            if self.config.parallel_experiments > 1 and not sweep_counts:
                method_results = self._run_method_experiments_parallel(
                    framework, method, dataset_name, problems
                )
            else:
                if not sweep_counts:
                    method_results = self._run_method_experiments_sequential(
                        framework, method, dataset_name, problems
                    )
                else:
                    method_results = []
                    for nc in sweep_counts:
                        try:
                            framework.reasoning_paradigm.node_count = int(nc)
                        except Exception:
                            continue
                        sub_results = self._run_method_experiments_sequential(
                            framework, f"{method}_nc{nc}", dataset_name, problems
                        )
                        method_results.extend(sub_results)
            
            dataset_results.extend(method_results)
        
        return dataset_results
    
    def _run_method_experiments_parallel(self, 
                                       framework,
                                       method: str,
                                       dataset_name: str,
                                       problems: List[Dict[str, Any]]) -> List[ExperimentResult]:
      
        
        method_results = []
        
        tasks = []
        for prob_idx, problem in enumerate(problems):
            for run_id in range(self.config.runs_per_experiment):
                tasks.append((framework, method, dataset_name, problem, prob_idx, run_id))
        
        with ThreadPoolExecutor(max_workers=self.config.parallel_experiments) as executor:
            future_to_task = {
                executor.submit(self._run_single_experiment, *task): task 
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    method_results.append(result)
                    completed = len(method_results)
                    total = len(tasks)
                    progress = completed / total * 100
                    print(f"    [{completed}/{total}] {progress:.1f}% - {result.method}: {'✅' if result.success else '❌'}")
                    
                except Exception as e:
                    self.logger.error(f"Parallel experiment failed: {e}")
        
        return method_results
    
    def _run_method_experiments_sequential(self,
                                         framework,
                                         method: str,
                                         dataset_name: str,
                                         problems: List[Dict[str, Any]]) -> List[ExperimentResult]:
        
        method_results = []
        
        for prob_idx, problem in enumerate(problems):
            print(f"  [📝] Problem {prob_idx+1}/{len(problems)}: {problem.get('problem', '')[:50]}...")
            
            for run_id in range(self.config.runs_per_experiment):
                print(f"    [🔄] Run {run_id+1}/{self.config.runs_per_experiment}")
                
                result = self._run_single_experiment(
                    framework, method, dataset_name, problem, prob_idx, run_id
                )
                
                method_results.append(result)
                
                print(f"    [{'✅' if result.success else '❌'}] Result: {result.success}, Time: {result.execution_time:.2f}s")
        
        return method_results
    
    def _extract_final_line_answer(self, text: str) -> str:
        try:
            import re
            matches = list(re.finditer(r"(final\s*answer|answer|conclusion)\s*[:]\s*(.+)", str(text), flags=re.IGNORECASE))
            if matches:
                return matches[-1].group(2).strip()
            m = re.search(r"\b([ABCD])\b", str(text))
            if m:
                return m.group(1).strip().upper()
        except Exception:
            pass
        return str(text).strip()

    def _normalize_gaokao_value(self, s: str) -> str:
        try:
            import re
            def _dbc(t: str) -> str:
                out = []
                for ch in t:
                    code = ord(ch)
                    if code == 12288:
                        code = 32
                    elif 65281 <= code <= 65374:
                        code -= 65248
                    out.append(chr(code))
                return ''.join(out)
            s = _dbc(s or "").strip()
            s = s.replace(" ", "")
            s = re.sub(r"sqrt\s*\(\s*(\d+)\s*\)", r"√\1", s, flags=re.IGNORECASE)
            s = s.replace("root", "√").replace("pi", "π")
            s = s.upper()
            return s
        except Exception:
            return (s or "").strip()
    
    def _run_single_experiment(self,
                             framework,
                             method: str,
                             dataset_name: str,
                             problem: Dict[str, Any],
                             prob_idx: int,
                             run_id: int) -> ExperimentResult:
        start_time = time.time()
        try:
            problem_text = problem.get("problem", problem.get("question", str(problem)))
            try:
                answer_type = str(problem.get("answer_type", "")).strip().lower()
                options = problem.get("options") or []
                parts = [problem_text]
                if answer_type:
                    parts.append(f"Question type: {'multiple choice' if 'choice' in answer_type else 'fill in the blank'}")
                if options:
                    opts_text = "\n".join(str(o) for o in options)
                    parts.append(f"Options:\n{opts_text}")
                if dataset_name == "legal":
                    tips = problem.get("prompt_tips") or {}
                    laws = tips.get("laws") or []
                    guidance = tips.get("judgment_guidance") or ""
                    if laws:
                        laws_text = "\n".join(f"- {l}" for l in laws)
                        parts.append(f"Known legal basis:\n{laws_text}")
                    if guidance:
                        parts.append(f"Judgment guidance: {guidance}")
                    parts.append("Please provide a clear final conclusion, and output only one line at the end:\nConclusion: <your verdict>")
                problem_text = "\n".join(parts)
            except Exception:
                pass
            trials = 3 if dataset_name in ("gaokao_math", "legal") else 1
            trial_records = []
            
            for t in range(trials):
                single_start = time.time()
                reasoning_result = framework.start_reasoning(
                    problem_text, 
                    max_total_nodes=self.config.max_nodes_per_experiment
                )
                single_time = time.time() - single_start
                
                raw_answer = reasoning_result.get("final_answer", "")
                extracted = self._extract_final_line_answer(raw_answer or "")
                if dataset_name == "gaokao_math":
                    norm = self._normalize_gaokao_value(extracted)
                elif dataset_name == "legal":
                    def _dbc(x: str) -> str:
                        out = []
                        for ch in x:
                            code = ord(ch)
                            if code == 12288:
                                code = 32
                            elif 65281 <= code <= 65374:
                                code -= 65248
                            out.append(chr(code))
                        return ''.join(out)
                    norm = _dbc(extracted).replace(" ", "").lower()
                else:
                    norm = extracted
                
                success = self._evaluate_result(extracted, problem, dataset_name)
                trial_records.append({
                    "success": success,
                    "norm": norm,
                    "extracted": extracted,
                    "raw": raw_answer,
                    "reasoning_graph": reasoning_result.get("reasoning_graph", {}),
                    "metrics": reasoning_result.get("metrics", {}),
                    "time": single_time
                })
            
            chosen = None
            from collections import Counter
            success_norms = [r["norm"] for r in trial_records if r["success"]]
            if success_norms:
                top_norm, _ = Counter(success_norms).most_common(1)[0]
                for r in trial_records:
                    if r["success"] and r["norm"] == top_norm:
                        chosen = r
                        break
            else:
                all_norms = [r["norm"] for r in trial_records]
                if all_norms:
                    top_norm, _ = Counter(all_norms).most_common(1)[0]
                    for r in trial_records:
                        if r["norm"] == top_norm:
                            chosen = r
                            break
                else:
                    chosen = trial_records[0] if trial_records else {"success": False, "extracted": "", "norm": "", "reasoning_graph": {}, "metrics": {}, "time": 0.0}

            total_time = sum(r["time"] for r in trial_records) if trial_records else 0.0
            success = bool(chosen.get("success"))

            if chosen.get("reasoning_graph"):
                self.collected_graphs.append({
                    "method": method,
                    "dataset": dataset_name,
                    "problem_index": prob_idx,
                    "run_id": run_id,
                    "success": success,
                    "graph_data": chosen["reasoning_graph"]
                })
                try:
                    from TIG_reasoning_framework import AdvancedReasoningFramework
                    vis_file = self.output_dir / f"viz_{method}_{dataset_name}_p{prob_idx}_r{run_id}.png"
                    framework.save_graph_visualization(str(vis_file))
                except Exception as _:
                    pass
            
            if success:
                self.experiment_stats["successful_experiments"] += 1
            else:
                self.experiment_stats["failed_experiments"] += 1
            
            return ExperimentResult(
                method=method,
                dataset=dataset_name,
                problem=problem_text,
                problem_index=prob_idx,
                run_id=run_id,
                success=success,
                execution_time=total_time if trials > 1 else (trial_records[0]["time"] if trial_records else (time.time()-start_time)),
                answer=chosen.get("extracted", ""),
                reasoning_graph=chosen.get("reasoning_graph", {}),
                metrics=chosen.get("metrics", {})
            )
        except Exception as e:
            execution_time = time.time() - start_time
            self.experiment_stats["failed_experiments"] += 1
            self.logger.error(f"Single experiment failed: {e}")
            return ExperimentResult(
                method=method,
                dataset=dataset_name,
                problem=problem.get("problem", str(problem)),
                problem_index=prob_idx,
                run_id=run_id,
                success=False,
                execution_time=execution_time,
                answer="",
                reasoning_graph={},
                metrics={},
                error_message=str(e)
            )
    
    def _get_dataset_criteria(self, dataset_name: str) -> Tuple[str, str]:
        criteria_map = {
            "24point": (
                "If a complete and correct mathematical expression uses all four numbers and equals 24, output it as the result.",
                "If the text clearly states no solution/impossible/no answer, terminate reasoning. Calculation errors alone should not terminate."
            ),
            "gaokao_math": (
                "Only when an explicit final answer line like 'Answer: <UPPERCASE letter or numeric>' appears, may it be output as the result.",
                "Terminate only when the text clearly indicates no solution (e.g., 'no solution', 'impossible', 'cannot get 24')."
            ),
            "coding": (
                "If an executable code snippet is provided (prefer ```python``` blocks), containing function/class implementations without placeholders and satisfying the prompt, output it.",
                "Terminate when it's only natural language/pseudocode, lacks complete structure, or shows compile/runtime errors or placeholders."
            ),
            "legal": (
                "If a complete legal analysis with a clear conclusion is provided, output it.",
                "Terminate when the analysis deviates from the legal issue or is overly superficial."
            ),
            "leetcode": (
                "If the output includes executable code or a clear implementation and contains success signals (accepted/AC), or explains correctness, output it.",
                "Terminate when it contains failure signals (cannot/failed/TLE/MLE/RE/compile error/runtime error)."
            )   
        }
        
        return criteria_map.get(dataset_name, (
            "If a reasonable answer is given, output it as the result.",
            "Terminate when the reasoning is clearly wrong or off-topic."
        ))
    
    def _evaluate_result(self, answer: str, problem: Dict[str, Any], dataset_name: str) -> bool:
        if not answer or answer.strip() == "":
            return False
        if dataset_name == "24point":
            return self._evaluate_24point(answer, problem)
        elif dataset_name == "creative_writing":
            return self._evaluate_creative_writing(answer, problem)
        elif dataset_name == "sorting":
            return self._evaluate_sorting(answer, problem)
        elif dataset_name == "mathematics":
            return self._evaluate_mathematics(answer, problem)
        elif dataset_name == "coding":
            return self._evaluate_coding(answer, problem)
        elif dataset_name == "legal":
            return self._evaluate_legal(answer, problem)
        elif dataset_name == "gaokao_math":
            return self._evaluate_gaokao_math(answer, problem)
        elif dataset_name == "leetcode":
            return self._evaluate_coding(answer, problem)
        
        return len(answer.strip()) >= 10
    
    def _evaluate_24point(self, answer: str, problem: Dict[str, Any]) -> bool:
        import re
        error_indicators = ['no solution', 'impossible', 'not found', 'no answer', 'cannot solve', 'incorrect', 'error', 'wrong']
        if any(indicator in answer for indicator in error_indicators):
            return False
        match = re.search(r"([^\n=]+)=\s*24\b", answer)
        if not match:
            return False
        lhs = match.group(1)
        operators = ['+', '-', '*', '×', '÷', '/', '(', ')']
        if not any(op in lhs for op in operators):
            return False
        given_numbers = problem.get('numbers') or []
        if not given_numbers or len(given_numbers) != 4:
            return True
        tokens = re.findall(r"(?<!\d)(\d+)(?!\d)", lhs)
        def multiset(lst):
            d = {}
            for x in lst:
                d[x] = d.get(x, 0) + 1
            return d
        given_ms = multiset([str(int(n)) for n in given_numbers])
        lhs_ms = multiset(tokens)
        if lhs_ms != given_ms:
            return False
        try:
            expr = lhs.replace('×', '*').replace('÷', '/')
            if not re.fullmatch(r"[0-9\s\+\-\*\/\(\)]+", expr):
                return True 
            val = eval(expr, {"__builtins__": None}, {})
            return abs(val - 24) < 1e-6
        except Exception:
            return True
    
   
    def _evaluate_mathematics(self, answer: str, problem: Dict[str, Any]) -> bool:
        
        math_indicators = ['calculate', 'equals', 'solution', 'answer', 'result', '=', '+', '-', '*', '/']
        return any(indicator in answer for indicator in math_indicators)
    
    def _evaluate_coding(self, answer: str, problem: Dict[str, Any]) -> bool:
        blocks = self._extract_code_blocks(answer)
        for lang, code in blocks:
            lang_norm = (lang or "").lower()
            if lang_norm in ("py", "python", ""):
                if self._is_valid_python_code(code):
                    return True
            else:
                if not self._is_placeholder_code(code) and len(code.strip()) >= 10:
                    return True
        code_indicators = ['```', 'def ', 'class ', 'import ', 'return ', 'if ', 'for ', 'while ', ';', '{', '}']
        if any(ind in str(answer) for ind in code_indicators) and not self._is_placeholder_code(answer):
            return True
        return False
    
    def _evaluate_legal(self, answer: str, problem: Dict[str, Any]) -> bool:
        try:
            expected = str(problem.get("expected_verdict", "")).strip()
            if expected:
                def norm(t: str) -> str:
                    t = (t or "").strip()
                    def _dbc(s):
                        out = []
                        for ch in s:
                            code = ord(ch)
                            if code == 12288:
                                code = 32
                            elif 65281 <= code <= 65374:
                                code -= 65248
                            out.append(chr(code))
                        return ''.join(out)
                    t = _dbc(t)
                    t = t.replace(" ", "").lower()
                    return t
                ans_n = norm(answer)
                exp_n = norm(expected)
                if exp_n and exp_n in ans_n:
                    return True
                alias_groups = [
                    ("uphold original judgment", ["uphold original judgment", "uphold first-instance", "dismiss appeal", "second-instance upholds", "uphold judgment"]),
                    ("not guilty", ["not guilty", "acquit", "change to not guilty", "judgment of not guilty"]),
                    ("change judgment", ["change judgment", "vacate original", "remand", "modify judgment"]),
                    ("combined punishment", ["combined punishment", "aggregate punishment", "multiple offenses combined"]),
                    ("probation", ["probation"]),
                    ("fixed-term imprisonment", ["fixed-term imprisonment", "sentenced to fixed-term imprisonment"]),
                ]
                for key, synonyms in alias_groups:
                    if key in expected and any(norm(syn) in ans_n for syn in synonyms):
                        return True
        except Exception:
            pass
        legal_indicators = ['law', 'statute', 'elements', 'judgment rule', 'conclusion', 'therefore', 'in sum', 'verdict', 'sentencing']
        if any(ind in str(answer) for ind in legal_indicators):
            return True
        return False
    
    def _evaluate_gaokao_math(self, answer: str, problem: Dict[str, Any]) -> bool:
        if not answer or answer.strip() == "":
            return False
        answer_text = str(answer)
        answer_type = str(problem.get("answer_type", "")).strip().lower()
        expected = str(problem.get("expected_answer", "")).strip()
        if not expected:
            return self._evaluate_mathematics(answer_text, problem)
        
        if answer_type == "multiple_choice":
            exp = expected.upper()
            patterns = [
                rf"answer\s*[:]?\s*{exp}\b",
                rf"choose\s*[:]?\s*{exp}\b",
                rf"option\s*{exp}\b",
                rf"\b{exp}\b",
                rf"[\(]\s*{exp}\s*[\)]",
            ]
            for p in patterns:
                if re.search(p, answer_text, flags=re.IGNORECASE):
                    return True
            return False
        else:
            def _normalize_text(t: str) -> str:
                try:
                    t = t.strip()
                    def _dbc(s):
                        res = []
                        for ch in s:
                            code = ord(ch)
                            if code == 12288:
                                code = 32
                            elif 65281 <= code <= 65374:
                                code -= 65248
                            res.append(chr(code))
                        return ''.join(res)
                    t = _dbc(t)
                    t = t.replace("−", "-").replace("—", "-")
                    t = t.replace("root", "√")
                    t = re.sub(r"sqrt\s*\(\s*(\d+)\s*\)", r"√\1", t, flags=re.IGNORECASE)
                    t = t.replace("π", "π").replace("pi", "π")
                    t = t.replace(" ", "")
                    t = t.replace("√(", "√").replace(")", ")")
                    t = t.replace("／", "/").replace("×", "*")
                    return t
                except Exception:
                    return t
            
            ans_norm = _normalize_text(answer_text)
            exp_norm = _normalize_text(expected)
            if exp_norm and exp_norm in ans_norm:
                return True
            try:
                m = re.fullmatch(r"√(\d+)", exp_norm)
                if m:
                    n = m.group(1)
                    sqrt_aliases = {f"√{n}", f"√({n})", f"root {n}", f"sqrt({n})"}
                    if any(alias in ans_norm for alias in sqrt_aliases):
                        return True
            except Exception:
                pass
            def _safe_float(x: str) -> Optional[float]:
                try:
                    return float(x)
                except Exception:
                    return None
            try:
                frac_m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", exp_norm)
                if frac_m:
                    a, b = int(frac_m.group(1)), int(frac_m.group(2))
                    if b != 0:
                        target = a / b
                        nums = re.findall(r"-?\d+(?:\.\d+)?", ans_norm)
                        for num in nums:
                            val = _safe_float(num)
                            if val is not None and abs(val - target) <= 1e-3:
                                return True
            except Exception:
                pass
            try:
                def _eval_simple(expr: str) -> Optional[float]:
                    try:
                        e = expr
                        e = e.replace("π", str(np.pi))
                        e = re.sub(r"(\d+)√(\d+)", r"(\1*(\2**0.5))", e)
                        e = re.sub(r"√(\d+)", r"(\1**0.5)", e)
                        e = e.replace("^", "**").replace("×", "*")
                        if not re.fullmatch(r"[0-9\+\-\*/\(\)\.^\s]*", e):
                            return None
                        return float(eval(e, {"__builtins__": None}, {}))
                    except Exception:
                        return None
                tgt_val = _eval_simple(exp_norm)
                if tgt_val is not None:
                    nums = re.findall(r"-?\d+(?:\.\d+)?", ans_norm)
                    for num in nums:
                        val = _safe_float(num)
                        if val is not None and abs(val - tgt_val) <= 1e-3:
                            return True
                    ans_val = _eval_simple(ans_norm)
                    if ans_val is not None and abs(ans_val - tgt_val) <= 1e-3:
                        return True
            except Exception:
                pass
            return expected in answer_text
    
    # zhongkao math evaluation removed per submission requirements
    
    def _save_intermediate_results(self, dataset_name: str, results: List[ExperimentResult]):
        detailed_file = self.output_dir / f"{dataset_name}_detailed_{self.experiment_id}.json"
        detailed_data = []
        for result in results:
            detailed_data.append({
                "method": result.method,
                "dataset": result.dataset,
                "problem": result.problem,
                "problem_index": result.problem_index,
                "run_id": result.run_id,
                "success": result.success,
                "execution_time": result.execution_time,
                "answer": result.answer,
                "metrics": result.metrics,
                "reasoning_graph": result.reasoning_graph,
                "error_message": result.error_message
            })
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)
        summary_file = self.output_dir / f"{dataset_name}_summary_{self.experiment_id}.csv"
        summary_data = []
        for method in self.config.methods:
            method_results = [r for r in results if r.method == method]
            
            if method_results:
                success_count = sum(1 for r in method_results if r.success)
                total_count = len(method_results)
                avg_time = np.mean([r.execution_time for r in method_results])
                metrics_data = {}
                if method_results[0].metrics:
                    for key in method_results[0].metrics.keys():
                        values = [r.metrics.get(key, 0) for r in method_results if r.metrics]
                        if values:
                            metrics_data[f"avg_{key}"] = np.mean(values)
                
                try:
                    run_ids = sorted({int(r.run_id) for r in method_results})
                    accuracies = []
                    for run_id in run_ids:
                        run_results = [r for r in method_results if int(r.run_id) == run_id]
                        if run_results:
                            acc = sum(1 for r in run_results if r.success) / len(run_results)
                            accuracies.append(acc)
                    acc_std = float(np.std(accuracies)) if len(accuracies) > 1 else 0.0
                except Exception:
                    acc_std = 0.0
                
                summary_data.append({
                    "method": method.upper(),
                    "successful_experiments": success_count,
                    "total_experiments": total_count,
                    "success_rate": success_count / total_count,
                    "std_accuracy": acc_std,
                    "avg_execution_time": avg_time,
                    **metrics_data
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        runwise_file = self.output_dir / f"{dataset_name}_runwise_{self.experiment_id}.csv"
        runwise_rows = []
        for method in self.config.methods:
            method_results = [r for r in results if r.method == method]
            if not method_results:
                continue
            run_ids = sorted({int(getattr(r, 'run_id', 0)) for r in method_results})
            for run_id in run_ids:
                run_results = [r for r in method_results if int(getattr(r, 'run_id', 0)) == run_id]
                if not run_results:
                    continue
                run_success = sum(1 for r in run_results if r.success)
                run_total = len(run_results)
                run_avg_time = np.mean([r.execution_time for r in run_results])
                row = {
                    "method": method.upper(),
                    "run_id": run_id,
                    "successful_experiments": run_success,
                    "total_experiments": run_total,
                    "success_rate": run_success / run_total,
                    "avg_execution_time": run_avg_time,
                }
                if run_results[0].metrics:
                    for key in run_results[0].metrics.keys():
                        vals = [rr.metrics.get(key, 0) for rr in run_results if rr.metrics]
                        if vals:
                            row[f"avg_{key}"] = np.mean(vals)
                runwise_rows.append(row)
        if runwise_rows:
            pd.DataFrame(runwise_rows).to_csv(runwise_file, index=False, encoding='utf-8')
        
        print(f"[💾] Intermediate results saved for {dataset_name}")
    
    def _compute_final_results(self) -> Dict[str, Any]:
        results_by_method = {}
        results_by_dataset = {}
        overall_stats = {}
        
        for method in self.config.methods:
            method_results = [r for r in self.all_results if r.method == method]
            
            if method_results:
                success_count = sum(1 for r in method_results if r.success)
                total_count = len(method_results)
                
                accuracies = []
                run_ids = sorted({int(r.run_id) for r in method_results})
                for run_id in run_ids:
                    run_results = [r for r in method_results if int(r.run_id) == run_id]
                    if run_results:
                        accuracy = sum(1 for r in run_results if r.success) / len(run_results)
                        accuracies.append(accuracy)
                std_accuracy = np.std(accuracies) if len(accuracies) > 1 else 0.0
                metric_keys = set()
                for r in method_results:
                    if r.metrics:
                        metric_keys.update(r.metrics.keys())
                metrics_mean_across_runs = {}
                metrics_std_across_runs = {}
                for mk in sorted(metric_keys):
                    per_run_means = []
                    for run_id in run_ids:
                        run_vals = []
                        for r in method_results:
                            if int(r.run_id) == run_id and r.metrics and mk in r.metrics and isinstance(r.metrics[mk], (int, float)):
                                run_vals.append(float(r.metrics[mk]))
                        if run_vals:
                            per_run_means.append(float(np.mean(run_vals)))
                    if per_run_means:
                        metrics_mean_across_runs[mk] = float(np.mean(per_run_means))
                        metrics_std_across_runs[mk] = float(np.std(per_run_means))
                
                results_by_method[method] = {
                    "success_count": success_count,
                    "total_count": total_count,
                    "success_rate": success_count / total_count,
                    "std_accuracy": std_accuracy,
                    "accuracies_per_run": accuracies,
                    "avg_execution_time": np.mean([r.execution_time for r in method_results]),
                    "std_execution_time": np.std([r.execution_time for r in method_results]),
                    "metrics_mean_across_runs": metrics_mean_across_runs,
                    "metrics_std_across_runs": metrics_std_across_runs
                }
        
        for dataset in self.config.datasets:
            dataset_results = [r for r in self.all_results if r.dataset == dataset]
            
            if dataset_results:
                success_count = sum(1 for r in dataset_results if r.success)
                total_count = len(dataset_results)
                
                results_by_dataset[dataset] = {
                    "success_count": success_count,
                    "total_count": total_count,
                    "success_rate": success_count / total_count,
                    "avg_execution_time": np.mean([r.execution_time for r in dataset_results]),
                    "std_execution_time": np.std([r.execution_time for r in dataset_results])
                }
        if self.all_results:
            total_success = sum(1 for r in self.all_results if r.success)
            total_experiments = len(self.all_results)
            
            overall_stats = {
                "total_experiments": total_experiments,
                "total_success": total_success,
                "overall_success_rate": total_success / total_experiments,
                "avg_execution_time": np.mean([r.execution_time for r in self.all_results]),
                "std_execution_time": np.std([r.execution_time for r in self.all_results])
            }
        else:
            overall_stats = {
                "total_experiments": 0,
                "total_success": 0,
                "overall_success_rate": 0.0,
                "avg_execution_time": 0.0,
                "std_execution_time": 0.0
            }
        
        return {
            "results_by_method": results_by_method,
            "results_by_dataset": results_by_dataset,
            "overall_stats": overall_stats,
            "experiment_stats": self.experiment_stats,
            "config": self.config.__dict__
        }
    
    def _generate_comprehensive_report(self, final_results: Dict[str, Any]):
        report_file = self.output_dir / f"comprehensive_report_{self.experiment_id}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Enhanced Experiment System Report\n\n")
            f.write(f"**Experiment ID**: {self.experiment_id}\n")
            f.write(f"**Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Overall statistics\n\n")
            overall = final_results["overall_stats"]
            f.write(f"- **Total experiments**: {overall['total_experiments']}\n")
            f.write(f"- **Successful experiments**: {overall['total_success']}\n")
            f.write(f"- **Overall success rate**: {overall['overall_success_rate']:.2%}\n")
            f.write(f"- **Average execution time**: {overall['avg_execution_time']:.2f}s\n")
            f.write(f"- **Std of execution time**: {overall['std_execution_time']:.2f}s\n\n")
            f.write("## Method performance comparison\n\n")
            f.write("| Method | Success rate | Std accuracy | Success/Total | Avg time(s) |\n")
            f.write("|--------|--------------|-------------|---------------|-------------|\n")
            
            for method, stats in final_results["results_by_method"].items():
                f.write(f"| {method.upper()} | {stats['success_rate']:.2%} | "
                       f"±{stats.get('std_accuracy', 0.0):.3f} | "
                       f"{stats['success_count']}/{stats['total_count']} | "
                       f"{stats['avg_execution_time']:.2f} |\n")
            f.write("\n## Method-level Metrics (mean±std over 5 runs)\n\n")
            methods = list(final_results["results_by_method"].keys())
            selected_metrics = ["node_redundancy", "thinking_redundancy", "terminated_nodes_count", "lambda_ratio"]
            if methods:
                f.write("| metric | " + " | ".join(m.upper() for m in methods) + " |\n")
                f.write("|--------|" + "|".join(["-------------" for _ in methods]) + "|\n")
                for mk in selected_metrics:
                    row = [mk]
                    for method in methods:
                        stats = final_results["results_by_method"][method]
                        mean_map = stats.get("metrics_mean_across_runs", {})
                        std_map = stats.get("metrics_std_across_runs", {})
                        mean = mean_map.get(mk)
                        std = std_map.get(mk)
                        cell = "-" if mean is None else f"{mean:.3f} ± {std:.3f}"
                        row.append(cell)
                    f.write("| " + " | ".join(row) + " |\n")
            
            f.write("\n## Dataset performance comparison\n\n")
            f.write("| dataset | success rate | success/total | avg time(s) | time std(s) |\n")
            f.write("|---------|--------------|---------------|-------------|-------------|\n")
            
            for dataset, stats in final_results["results_by_dataset"].items():
                f.write(f"| {dataset} | {stats['success_rate']:.2%} | "
                       f"{stats['success_count']}/{stats['total_count']} | "
                       f"{stats['avg_execution_time']:.2f} | "
                       f"{stats['std_execution_time']:.2f} |\n")
            
            f.write(f"\n## Experiment configuration\n\n")
            f.write(f"```json\n{json.dumps(final_results['config'], indent=2, ensure_ascii=False)}\n```\n")
        
        print(f"[📊] Comprehensive report generated: {report_file}")
        self._generate_visualization_charts(final_results)
        try:
            rows = []
            for method in self.config.methods:
                stats = final_results.get("results_by_method", {}).get(method)
                if not stats:
                    continue
                accs = stats.get("accuracies_per_run", []) or [stats.get("success_rate", 0.0)]
                acc_mean = float(np.mean(accs))
                acc_std = float(stats.get("std_accuracy", 0.0))
                t_mean = float(stats.get("avg_execution_time", 0.0))
                t_std = float(stats.get("std_execution_time", 0.0))
                rows.append({
                    "Method": method.upper(),
                    "Accuracy(%)": f"{acc_mean*100:.1f}±{acc_std*100:.1f}",
                    "Time(s)": f"{t_mean:.1f}±{t_std:.1f}"
                })
            if rows:
                table_df = pd.DataFrame(rows)
                table_file = self.output_dir / f"paper_table_{self.experiment_id}.csv"
                table_df.to_csv(table_file, index=False, encoding='utf-8')
        except Exception:
            pass
    
    def _generate_visualization_charts(self, final_results: Dict[str, Any]):
        try:
            if not self.all_results or final_results.get("overall_stats", {}).get("total_experiments", 0) == 0:
                print("[⚠️] No valid experiment samples, skipping chart generation")
                return
        except Exception:
            pass
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Enhanced Experiment System Analysis - {self.experiment_id}', fontsize=16)
        
        methods = list(final_results["results_by_method"].keys())
        success_rates = [final_results["results_by_method"][m]["success_rate"] for m in methods]
        
        axes[0, 0].bar(methods, success_rates, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Success rate by method')
        axes[0, 0].set_ylabel('Success rate')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(success_rates):
            axes[0, 0].text(i, v + 0.01, f'{v:.2%}', ha='center')
        
        datasets = list(final_results["results_by_dataset"].keys())
        dataset_rates = [final_results["results_by_dataset"][d]["success_rate"] for d in datasets]
        
        axes[0, 1].bar(datasets, dataset_rates, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Success rate by dataset')
        axes[0, 1].set_ylabel('Success rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(dataset_rates):
            axes[0, 1].text(i, v + 0.01, f'{v:.2%}', ha='center')
        method_times = [final_results["results_by_method"][m]["avg_execution_time"] for m in methods]
        method_stds = [final_results["results_by_method"][m]["std_execution_time"] for m in methods]
        axes[1, 0].bar(methods, method_times, yerr=method_stds, 
                      color='lightgreen', alpha=0.7, capsize=5)
        axes[1, 0].set_title('Average execution time by method')
        axes[1, 0].set_ylabel('Time (s)')
        overall = final_results["overall_stats"]
        success_count = overall["total_success"]
        failure_count = overall["total_experiments"] - success_count
        
        axes[1, 1].pie([success_count, failure_count], 
                      labels=['success', 'failure'],
                      colors=['lightgreen', 'lightcoral'],
                      autopct='%1.1f%%',
                      startangle=90)
        axes[1, 1].set_title('Overall success/failure distribution')
        
        plt.tight_layout()
        
        chart_file = self.output_dir / f"experiment_analysis_{self.experiment_id}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[📊] Visualization charts saved: {chart_file}")
        try:
            import itertools
            from scipy import stats as _stats
            rows = []
            by_method = final_results.get("results_by_method", {})
            methods = [m for m in self.config.methods if m in by_method]
            for a, b in itertools.combinations(methods, 2):
                a_acc = by_method[a].get("accuracies_per_run", [])
                b_acc = by_method[b].get("accuracies_per_run", [])
                if len(a_acc) > 1 and len(b_acc) > 1:
                    t, p = _stats.ttest_ind(a_acc, b_acc, equal_var=False)
                    rows.append({"A": a.upper(), "B": b.upper(), "t": float(t), "p": float(p)})
            if rows:
                sig_df = pd.DataFrame(rows)
                sig_file = self.output_dir / f"significance_{self.experiment_id}.csv"
                sig_df.to_csv(sig_file, index=False, encoding='utf-8')
        except Exception:
            pass
    
    def _perform_rgwl_analysis(self, rgwl_kernel):
        print("\n[🧮] Starting RGWL similarity analysis...")
        
        if len(self.collected_graphs) < 2:
            print("[⚠️] Insufficient graph data, skipping RGWL analysis")
            return
    
        successful_graphs = [g for g in self.collected_graphs if g["success"]]
        
        if len(successful_graphs) < 2:
            print("[⚠️] Not enough successful graphs, skipping RGWL analysis")
            return
    
        print(f"[📊] Computing similarity matrix for {len(successful_graphs)} graphs...")
        
        graph_data = [g["graph_data"] for g in successful_graphs]
        try:
            kernel_matrix = rgwl_kernel.batch_compute_kernel_matrix(graph_data)
        except Exception as e:
            print(f"[⚠️] Failed to compute kernel matrix: {e}")
            kernel_matrix = None
        
        if kernel_matrix is not None:
            kernel_file = self.output_dir / f"rgwl_kernel_matrix_{self.experiment_id}.npy"
            np.save(kernel_file, kernel_matrix)
            print(f"[💾] Kernel matrix saved: {kernel_file}")
            try:
                distance_matrix = rgwl_kernel.kernel_to_distance_matrix(kernel_matrix)
                dist_file = self.output_dir / f"rgwl_distance_matrix_{self.experiment_id}.npy"
                np.save(dist_file, distance_matrix)
                print(f"[💾] Distance matrix saved: {dist_file}")
                try:
                    kernel_report = self.output_dir / f"rgwl_kernel_wltest_{self.experiment_id}.md"
                    dist_report = self.output_dir / f"rgwl_distance_check_{self.experiment_id}.md"
                    _ = rgwl_kernel.validate_kernel_matrix(kernel_matrix, str(kernel_report))
                    _ = rgwl_kernel.validate_distance_matrix(distance_matrix, str(dist_report))
                    print(f"[📄] WLTest reports generated: {kernel_report.name}, {dist_report.name}")
                except Exception as _:
                    pass
            except Exception as _:
                pass
        else:
            print("[⚠️] Kernel matrix file was not generated")
        print(f"[📊] Visualization disabled; only kernel matrices saved")
        print(f"[✅] RGWL analysis completed; results saved")

    def _apply_custom_config_to_framework(self, framework, method: str, dataset_name: str) -> None:
        try:
            if not self.config.custom_config:
                return
            cfg = self.config.custom_config
            modes = cfg.get("reasoning_modes", {})
            mkey = str(method).lower()
            if mkey in modes:
                mode_cfg = modes[mkey]
                prompts = mode_cfg.get("prompts") or {}
                if prompts:
                    framework.reasoning_paradigm.generation_prompts.update(prompts)
                paradigm = mode_cfg.get("reasoning_paradigm") or {}
                mapping = {
                    "max_nodes_per_round": "max_nodes_per_round",
                    "enable_multi_node_generation": "enable_multi_node_generation",
                    "node_separator": "multi_node_separator",
                    "reference_input": "reference_input_type",
                    "dependency_input": "dependency_input_type",
                    "node_partition_method": "node_division_type",
                    "prompt_strategy": None, 
                    "k_hop_distance": "k_hop_distance"
                }
                for k, v in paradigm.items():
                    if k in mapping and mapping[k]:
                        try:
                            if k == "reference_input" and str(v).lower() == "parent_and_root":
                                v = "parent_root"
                            if k == "node_partition_method":
                                vv = str(v).lower()
                                if vv == "llm_generation":
                                    v = "by_llm_generation"
                                elif vv == "by_node" or vv == "by_llm_generation":
                                    v = vv
                        except Exception:
                            pass
                        setattr(framework.reasoning_paradigm, mapping[k], v)
                if isinstance(mode_cfg.get("max_rounds"), int):
                    framework.reasoning_paradigm.max_reasoning_rounds = mode_cfg["max_rounds"]
        except Exception as e:
            self.logger.warning(f"Failed to inject custom config: {e}")

    def _extract_code_blocks(self, text: str) -> List[Tuple[str, str]]:
        import re
        blocks: List[Tuple[str, str]] = []
        try:
            pattern = re.compile(r"```\s*([a-zA-Z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)
            for m in pattern.finditer(str(text)):
                lang = (m.group(1) or "").strip().lower()
                code = m.group(2)
                blocks.append((lang, code))
        except Exception:
            pass
        return blocks

    def _is_placeholder_code(self, code: str) -> bool:
        lower = str(code).lower()
        placeholders = ["todo", "pass", "...", "pseudo", "placeholder", "omit"]
        return any(p in lower for p in placeholders)

    def _is_valid_python_code(self, code: str) -> bool:
        try:
            import ast
            if len(code.strip()) < 10:
                return False
            if self._is_placeholder_code(code):
                return False
            if ("def " not in code) and ("class " not in code):
                return False
            ast.parse(code)
            return True
        except Exception:
            return False

def run_enhanced_experiments():
    config = ExperimentConfig(
        methods=["cot", "tot", "got", "aot", "egot"],
        datasets=["24point", "creative_writing", "mathematics"],  
        runs_per_experiment=3,  
        max_nodes_per_experiment=15,
        parallel_experiments=2,
        save_intermediate_results=True,
        enable_rgwl_analysis=True
    )
    experiment_system = EnhancedExperimentSystem(config)
    try:
        results = experiment_system.run_comprehensive_experiments()
        print("\n🎉 Enhanced Experiment System finished!")
        return results
    except Exception as e:
        print(f"\n❌ Experiment run failed: {e}")
        raise

if __name__ == "__main__":
    run_enhanced_experiments() 