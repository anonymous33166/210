import json
import os
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from copy import deepcopy
import re
import logging
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

@dataclass
class ReasoningNode:
    id: str
    content: str
    round: int
    parent_ids: List[str]
    children_ids: List[str] = field(default_factory=list)
    node_type: str = "reasoning"  # reasoning, answer, terminated
    word_count: int = 0
    is_effective: bool = True
    is_inner_node: bool = False  
    inner_group_id: Optional[str] = None  
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)  
    generation_prompt: str = "" 
    auto_label: str = "" 

@dataclass 
class ReasoningParadigm:
    node_count: int = 1
    enable_multi_node_generation: bool = False
    multi_node_separator: str = "\n---\n"  
    multi_node_connection_rule: str = "sequential"  
    reference_input_type: str = "parent_only" 
    k_hop_distance: int = 2  
    node_division_type: str = "by_node"  
    dependency_input_type: str = "current_only" 
    dependency_selection_rule: str = ""  
    generation_prompts: Dict[str, str] = field(default_factory=dict)  
    few_shot_examples: Dict[str, List[str]] = field(default_factory=dict)  
    max_reasoning_rounds: int = 10     
    continue_until_answer: bool = False 
    enable_branch_scoring: bool = False
    beam_width: int = 2

class AdvancedReasoningFramework:
    def __init__(self, config: Dict[str, Any], llm_backend=None):
        self.config = config
        self.llm_backend = llm_backend
        self.reasoning_graph = nx.DiGraph()
        self.nodes_data = {}
        self.node_counter = 0
        self.inner_group_counter = 0
        self.current_round = 0
        self.max_rounds = config.get("max_rounds", 10)
        self.max_total_nodes = config.get("max_total_nodes", 50)
        self.max_nodes_per_round = config.get("max_nodes_per_round", 5)
        self.reasoning_paradigm = ReasoningParadigm()
        self.llm_sampling: Dict[str, Any] = config.get("llm_sampling", {}) or {}
        self.result_criterion = None
        self.termination_criterion = None
        self.graph_metadata = {
            "root_node": None,
            "answer_nodes": [],
            "terminated_nodes": [],
            "answer_paths": [],
            "lambda_ratio": 0.0,
            "total_nodes": 0,
            "effective_nodes": 0,
            "total_words": 0,
            "effective_words": 0,
            "round_nodes": {},
            "inner_node_groups": {}
        }
        self.save_visualizations = config.get("save_visualizations", False)
        self.dataset_name = config.get("dataset_name")
        self._quiet = bool(config.get("quiet")) or (os.environ.get("BENCHMARK_QUIET") == "1")
        def _qprint(*args, **kwargs):
            if not self._quiet:
                print(*args, **kwargs)
        self._qprint = _qprint
        
        self._qprint("[🎯] Advanced Reasoning Framework Initialization Completed")
    
    def configure_reasoning_paradigm(self, 
                                   node_count: int = 1,
                                   enable_multi_node: bool = False,
                                   reference_type: str = "parent_only",
                                   dependency_type: str = "current_only",
                                   prompts: Dict[str, str] = None,
                                   **kwargs):
        self.reasoning_paradigm.node_count = node_count
        self.reasoning_paradigm.enable_multi_node_generation = enable_multi_node
        self.reasoning_paradigm.reference_input_type = reference_type
        self.reasoning_paradigm.dependency_input_type = dependency_type
        
        if prompts:
            self.reasoning_paradigm.generation_prompts.update(prompts)
        for key, value in kwargs.items():
            if hasattr(self.reasoning_paradigm, key):
                setattr(self.reasoning_paradigm, key, value)
        
        self._qprint(f"[⚙️] Reasoning paradigm configured: nodes={node_count}, multi-node={enable_multi_node}")
    
    def configure_for_method(self, method: str):
        method = method.upper()
        
        if method == "COT":
            self.configure_reasoning_paradigm(
                node_count=1,                   
                enable_multi_node=False,         
                reference_type="parent_only",   
                dependency_type="current_only",  
                max_reasoning_rounds=10,         
                continue_until_answer=False,    
                prompts={"default": """Based on the previous step’s reasoning, continue with the next step of analysis:

[Chain-of-Thought Steps]:
1. State Analysis: Analyze the information known so far and the steps already attempted.
2. Problem Identification: Identify the specific problem or challenge currently faced.
3. Strategy Selection: Choose one concrete action to try.
4. Step Execution: Execute the selected strategy and provide the specific reasoning step.
5. Result Evaluation: Assess the effectiveness of this step and the next direction.

[Current State]:
{current_content}

[Please provide the next reasoning step]:
- Perform only a small reasoning step; do not give the full answer at once.
- First analyze the current state, then choose a specific action.
- If this is the first step, analyze the problem and propose a concrete direction to try.
- If previous attempts had issues, analyze the causes and propose a new attempt.
- Build the solution step by step; avoid skipping steps.
- If you have already found the answer, output only: Answer = [explicit expression]"""}
            )
        
        elif method == "TOT":
            self.configure_reasoning_paradigm(
                node_count=3,
                enable_multi_node=True,
                reference_type="parent_root",
                dependency_type="current_only",
                multi_node_separator="\n---Idea---\n",
                multi_node_connection_rule="none",
                enable_branch_scoring=True,
                beam_width=3,
                prompts={"default": """Explore a tree-of-thoughts and produce 3 different directions:

[Tree Search Strategy]:
1. Breadth-first: explore multiple distinct solution ideas in parallel.
2. Diversity: ensure the 3 ideas cover different solving strategies.
3. Specificity: each idea must state a concrete next step.
4. Feasibility: assess each idea’s feasibility based on the current state.

[Current State]:
{current_content}

[Please generate 3 different directions]:
- Separate each idea with '---Idea---'.
- For each idea, provide the concrete next action and the rationale.
- Ensure diversity among ideas; avoid repetition.
- If the answer has been found, output only: Answer = [explicit expression]"""}
            )
        
        elif method == "GOT":
            self.configure_reasoning_paradigm(
                node_count=2,
                enable_multi_node=True,
                reference_type="all_ancestors",
                dependency_type="current_plus_others",
                dependency_selection_rule="Top 2 most relevant nodes",
                multi_node_separator="\n---Idea---\n",
                multi_node_connection_rule="none",
                enable_branch_scoring=True,
                beam_width=2,
                prompts={"default": """Combine all historical ideas and related node information to perform graph-based reasoning:

[Graph Reasoning Strategy]:
1. Information Fusion: Integrate information from all relevant nodes to form a global view.
2. Path Analysis: Analyze the pros/cons and relationships among different reasoning paths.
3. Branch Generation: Generate 2 complementary reasoning branches based on the graph structure.
4. Constraint Check: Ensure each branch satisfies all problem constraints.

[Current State Analysis]:
{combined_content}

[Please generate 2 different reasoning branches]:
- Separate each branch with '---Idea---'.
- Each branch must be based on different graph node information.
- Ensure the branches are complementary and cover different solution ideas.
- If the final answer is determined, output only one line at the end: Answer: <UPPERCASE option letter or numeric value>"""}
            )
        
        elif method == "AOT":
            self.configure_reasoning_paradigm(
                node_count=1,
                enable_multi_node=False,
                reference_type="k_hop_ancestors",
                k_hop_distance=3,
                dependency_type="current_plus_others",
                dependency_selection_rule="All related nodes",
                prompts={"default": """Adopt an algorithmic mindset and analyze systematically using the following steps:

[Algorithmic Thinking Steps]:
1. Pattern Recognition: Identify the core patterns of the current problem and attempted paths.
2. Success Analysis: Analyze which attempts were successful and what patterns led to success.
3. Failure Analysis: Analyze which attempts failed and why.
4. Strategy Design: Design the optimal next step based on the successful patterns.
5. Constraint Check: Ensure the new strategy meets all constraints.
6. Answer Output: If an effective solution is found, output the answer directly.

[Current State Analysis]:
{aggregated_content}

[Please provide the next step with algorithmic thinking]:
- First perform pattern recognition and success/failure analysis.
- Then design a specific next-step strategy.
- If the answer has been found, output only: Answer = [explicit expression]"""}
            )
        
        elif method == "EGOT":
            self.configure_reasoning_paradigm(
                node_count=4,
                enable_multi_node=True,
                reference_type="all_ancestors",
                dependency_type="current_plus_others",
                dependency_selection_rule="Evaluate all feasible paths",
                multi_node_separator="\n---Idea---\n",
                multi_node_connection_rule="none",
                enable_branch_scoring=True,
                beam_width=4,
                prompts={"default": """Based on all historical information, perform an intelligent evaluation and generate 4 optimized reasoning directions:

[Intelligent Evaluation Criteria]:
1. Path Feasibility: Estimate success probability based on historical success patterns.
2. Novelty: Whether new, unexplored ideas are attempted.
3. Efficiency: Whether it directly targets the goal and reduces unnecessary computation.
4. Completeness: Whether all constraints and edge cases are considered.
5. Robustness: Whether the strategy is adaptable across different scenarios.

[Current State Analysis]:
{evaluated_content}

[Please generate 4 intelligently evaluated directions]:
- For each direction, clearly explain why this strategy is chosen.
- Separate each direction with '---Idea---'.
- If the answer has been found, output only: Answer = [explicit expression].
- Ensure the 4 directions are diverse and cover different solution ideas."""}
            )
        
        self._qprint(f"[🔧] Reasoning mode set to: {method}")
    
    def set_judgment_criteria(self, result_criterion: str, termination_criterion: str):
        self.result_criterion = result_criterion
        self.termination_criterion = termination_criterion
        self._qprint(f"[⚖️] Judgment criteria set")
    
    def start_reasoning(self, problem: str, max_total_nodes: int = 20) -> Dict[str, Any]:
        self._qprint(f"[🚀] Reasoning started: {problem}")
        self._reset_reasoning_state()
        self.max_total_nodes = max_total_nodes
        root_node = self._create_root_node(problem)
        self.graph_metadata["root_node"] = root_node.id
        self.graph_metadata["round_nodes"][0] = [root_node.id]
        round_t = 1
        reasoning_success = False
        final_answer = None
        max_rounds = getattr(self.reasoning_paradigm, 'max_reasoning_rounds', 10)
        continue_until_answer = getattr(self.reasoning_paradigm, 'continue_until_answer', False)
        
        while (len(self.nodes_data) < self.max_total_nodes and 
               round_t <= max_rounds):
            self._qprint(f"[🔄] Reasoning round {round_t}/{max_rounds}")
            pending_nodes = self._select_pending_nodes(round_t - 1)
            
            if not pending_nodes:
                self._qprint("[⚠️] No pending nodes; reasoning terminated")
                break
            
            round_new_nodes = []
            
    
            for node_id in pending_nodes:
                node = self.nodes_data[node_id]
                self._qprint(f"[📝] Processing node: {node_id} - {node.content[:50]}...")
                
                if self._check_result_criterion(node):
                    self._qprint(f"[✅] Answer found: {node.content}")
                    self.graph_metadata["answer_nodes"].append(node_id)
                    node.node_type = "answer" 
                    reasoning_success = True
                    final_answer = node.content
                    
                    if not continue_until_answer:
                        break
                
                
                if self._check_termination_criterion(node):
                    self._qprint(f"[🛑] Node meets termination condition: {node_id}")
                    node.node_type = "terminated"
                    self.graph_metadata["terminated_nodes"].append(node_id)
                    continue 

                new_nodes = self._generate_successor_nodes(node)
                round_new_nodes.extend(new_nodes)
            
            
            if reasoning_success and not continue_until_answer:
                break
            
            self.graph_metadata["round_nodes"][round_t] = [n.id for n in round_new_nodes]
            for node in round_new_nodes:
                node.round = round_t
            
            round_t += 1
            
            if not round_new_nodes:
                self._qprint("[⚠️] No new nodes generated this round; stopping reasoning")
                break
        return self._compute_final_results(reasoning_success, final_answer, round_t - 1)
    
    def _create_root_node(self, problem: str) -> ReasoningNode:
        node_id = f"root_0"
        node = ReasoningNode(
            id=node_id,
            content=problem,
            round=0,
            parent_ids=[],
            node_type="root",
            word_count=len(problem.split()),
            metadata={"created_time": datetime.now().isoformat()}
        )
        
        self.nodes_data[node_id] = node
        self.reasoning_graph.add_node(node_id, **node.__dict__)
        self.node_counter += 1
        
        self._qprint(f"[🌱] Created root node: {node_id}")
        return node
    
    def _select_pending_nodes(self, round_num: int) -> List[str]:
        if round_num not in self.graph_metadata["round_nodes"]:
            return []
        
        return self.graph_metadata["round_nodes"][round_num]
    
    def _check_result_criterion(self, node: ReasoningNode) -> bool:
        if not self.result_criterion:
            return False
        
        try:
            try:
                if node.node_type == "root" or node.id == self.graph_metadata.get("root_node"):
                    return False
            except Exception:
                pass
            
            if str(self.dataset_name) in ("24point", "comprehensive_24point_67_dataset"):
                return self._evaluate_24point_deterministic(node.content)
            
            try:
                if str(self.dataset_name) == "gaokao_math":
                    _text = str(node.content)
                    if re.search(r"(final answer|answer)\s*[:]?", _text, flags=re.I):
                        return True
                    if re.search(r"\b[ABCD]\b", _text):
                        return True
            except Exception:
                pass
            
            try:
                import re as _re
                if _re.search(r"(final\s*answer|answer)\s*[:]", node.content, flags=_re.I):
                    return True
            except Exception:
                pass
            
            return self._check_result_criterion_llm(node)
        except Exception as e:
            self._qprint(f"[❌] Result criterion exception: {e}")
            return False
    
    def _evaluate_24point_deterministic(self, content: str) -> bool:
        import re
        problem_numbers = self._extract_numbers_from_problem()
        if not problem_numbers:
            return self._evaluate_24point_loose(content)
        expressions = []
        patterns = [
            r'([^=\n]+)\s*=\s*24',    
            r'24\s*=\s*([^=\n]+)',   
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            expressions.extend(matches)
        
        if not expressions:
            return False
        
    
        for expr in expressions:
            if self._validate_expression_strict(expr.strip(), problem_numbers):
                self._qprint(f"[✅] Strict 24-point check passed")
                self._qprint(f"    Expression: {expr.strip()}")
                self._qprint(f"    Required numbers: {problem_numbers}")
                return True
        
        self._qprint(f"[❌] 24-point check failed - incomplete number usage")
        self._qprint(f"    Required numbers: {problem_numbers}")
        return False
    
    def _extract_numbers_from_problem(self):
        try:
            root_content = self.nodes_data.get("root", {}).content if "root" in self.nodes_data else ""
            
            import re
            patterns = [
                r'numbers\s*([0-9,\s,]+)', 
                r'using.*?([0-9,\s,]+).*?through',  
                r'\[([0-9,\s,]+)\]',    
            ]
            
            for pattern in patterns:
                match = re.search(pattern, root_content)
                if match:
                    numbers_str = match.group(1).replace('\uFF0C', ',').replace(' ', '')
                    numbers = [int(x.strip()) for x in numbers_str.split(',') if x.strip().isdigit()]
                    if numbers:
                        return sorted(numbers)
            
            return None
        except Exception as e:
            self._qprint(f"[⚠️] Failed to extract problem numbers: {e}")
            return None
    
    def _validate_expression_strict(self, expression: str, required_numbers: list) -> bool:
        import re
        
        try:
            left_part = expression.split('=')[0] if '=' in expression else expression
            found_numbers = []
            number_matches = re.findall(r'\b\d+\b', left_part)
            for num_str in number_matches:
                found_numbers.append(int(num_str))
            
    
            if sorted(found_numbers) != sorted(required_numbers):
                self._qprint(f"[⚠️] Number mismatch: found {sorted(found_numbers)}, required {sorted(required_numbers)}")
                return False
            
            clean_expr = left_part.replace('\u00D7', '*').replace('\u00F7', '/').replace('\uFF08', '(').replace('\uFF09', ')')
            
            clean_expr = re.sub(r'[^\d+\-*/().\s]', '', clean_expr)
            
            if not clean_expr.strip():
                self._qprint(f"[⚠️] Empty expression after cleaning")
                return False
            
            try:
                result = eval(clean_expr.strip())
                is_correct = abs(result - 24) < 0.001
                if not is_correct:
                    self._qprint(f"[⚠️] Incorrect calculation result: {result} ≠ 24")
                return is_correct
            except Exception as calc_e:
                self._qprint(f"[⚠️] Calculation failed: {calc_e}")
                return False
                
        except Exception as e:
            self._qprint(f"[⚠️] Expression validation failed: {e}")
            return False
    
    def _evaluate_24point_loose(self, content: str) -> bool:
        import re
        patterns = [
            r'[\)\d]\s*=\s*24\s*$',
            r'=\s*24\s*(?:$|[^0-9])'
        ]
        
        has_result_24 = False
        for pattern in patterns:
            if re.search(pattern, content, re.MULTILINE):
                has_result_24 = True
                break
        
        if not has_result_24:
            return False
        
        operators = ['+', '-', '*', '\u00D7', '\u00F7', '/', '(', ')']
        has_operators = any(op in content for op in operators)
        has_numbers = bool(re.search(r'\d+', content))
        
        return has_result_24 and has_operators and has_numbers
    
    def _check_result_criterion_llm(self, node: ReasoningNode) -> bool:
        judgment_prompt = f"""
Please judge, according to the following criteria, whether the given content can be output as the answer:

[Result Criteria]: {self.result_criterion}

[Content to Judge]: {node.content}

Please answer only "yes" or "no", without any explanation.
"""
        
        try:
            llm_result = self.llm_backend.generate(judgment_prompt, **(self.llm_sampling or {}))
            if isinstance(llm_result, dict):
                response = llm_result.get("result", "").strip()
            else:
                response = str(llm_result).strip()
            is_answer = "yes" in response.lower()
            
            if is_answer:
                self._qprint(f"[✅] Node {node.id} meets result criterion")
            
            return is_answer
        except Exception as e:
            self._qprint(f"[❌] Result judgment failed: {e}")
            return False
    
    def _check_termination_criterion(self, node: ReasoningNode) -> bool:
        if not self.termination_criterion:
            return False
        
        try:
            judgment_prompt = f"""
Please judge, according to the following criteria, whether the given content should terminate reasoning:

[Termination Criteria]: {self.termination_criterion}

[Content to Judge]: {node.content}

Please answer only "yes" or "no", without any explanation.
"""
            
            neg_keywords = ["no solution", "impossible", "no answer", "cannot continue", "contradiction", "invalid"]
            if str(self.dataset_name) in ("24point", "comprehensive_24point_67_dataset"):
                neg_keywords.extend(["not 24", "not equal to 24", "!= 24", "cannot reach 24"])
            if str(self.dataset_name) == "leetcode":
                neg_keywords.extend(["compile error", "runtime error", "failed", "timeout", "memory limit exceeded", "WA", "TLE", "MLE", "RE"])
            lowered = str(node.content)
            if any(k in lowered for k in neg_keywords):
                return True
            
            llm_result = self.llm_backend.generate(judgment_prompt, **(self.llm_sampling or {}))
            if isinstance(llm_result, dict):
                response = llm_result.get("result", "").strip()
            else:
                response = str(llm_result).strip()
            should_terminate = "yes" in response.lower()
            return should_terminate
        except Exception as e:
            self._qprint(f"[❌] Termination judgment failed: {e}")
            return False
    
    def _generate_successor_nodes(self, current_node: ReasoningNode) -> List[ReasoningNode]:
        paradigm = self.reasoning_paradigm
        reference_content = self._collect_reference_input(current_node)
        dependency_content = self._collect_dependency_input(current_node)
        node_tags = current_node.tags or ["default"]
        generation_prompt = self._get_generation_prompt(node_tags[0], reference_content, dependency_content)
        
        
        if paradigm.enable_multi_node_generation:
            return self._generate_multi_nodes(current_node, generation_prompt)
        else:
            return self._generate_single_node(current_node, generation_prompt)
    
    def _collect_reference_input(self, current_node: ReasoningNode) -> str:
        reference_type = self.reasoning_paradigm.reference_input_type
        
        if reference_type == "parent_only":
            if not current_node.parent_ids:
                return current_node.content
            parent_contents = [self.nodes_data[pid].content for pid in current_node.parent_ids]
            return " | ".join(parent_contents)
        
        elif reference_type == "parent_root":
            root_id = self.graph_metadata["root_node"]
            contents = [self.nodes_data[root_id].content]
            if current_node.parent_ids:
                contents.extend([self.nodes_data[pid].content for pid in current_node.parent_ids])
            return " | ".join(contents)
        
        elif reference_type == "all_ancestors":
            ancestors = self._get_all_ancestors(current_node.id)
            contents = [self.nodes_data[aid].content for aid in ancestors]
            return " | ".join(contents)
        
        elif reference_type == "k_hop_ancestors":
            k_ancestors = self._get_k_hop_ancestors(current_node.id, self.reasoning_paradigm.k_hop_distance)
            contents = [self.nodes_data[aid].content for aid in k_ancestors]
            return " | ".join(contents)
        
        return current_node.content
    
    def _collect_dependency_input(self, current_node: ReasoningNode) -> str:
        dependency_type = self.reasoning_paradigm.dependency_input_type
        
        if dependency_type == "current_only":
            return current_node.content
        
        elif dependency_type == "current_plus_others":
            rule = self.reasoning_paradigm.dependency_selection_rule
            other_nodes = self._select_nodes_by_rule(current_node, rule)
            try:
                if not other_nodes:
                    from sklearn.feature_extraction.text import TfidfVectorizer as _V
                    node_ids = list(self.nodes_data.keys())
                    texts = [self.nodes_data[nid].content for nid in node_ids]
                    vec = _V(max_features=256)
                    X = vec.fit_transform(texts)
                    idx = node_ids.index(current_node.id)
                    sims_row = (X @ X.T).toarray()[idx]
                    scores = []
                    for i, nid in enumerate(node_ids):
                        if nid == current_node.id:
                            scores.append(-1e9)
                            continue
                        try:
                            dist = nx.shortest_path_length(self.reasoning_graph, nid, current_node.id)
                        except Exception:
                            dist = 3
                        score = float(sims_row[i]) / (1.0 + float(dist))
                        scores.append(score)
                    order = sorted(range(len(node_ids)), key=lambda k: scores[k], reverse=True)
                    other_nodes = [node_ids[i] for i in order[:2]]
            except Exception:
                pass
            
            contents = [current_node.content]
            contents.extend([self.nodes_data[nid].content for nid in other_nodes])
            return " | ".join(contents)
        
        return current_node.content
    
    def _get_generation_prompt(self, tag: str, reference_content: str, dependency_content: str) -> str:

        prompts = self.reasoning_paradigm.generation_prompts
        
        if tag in prompts:
            template = prompts[tag]
        else:
            template = prompts.get("default", "Please continue reasoning: {combined_content}")
        
        try:
            root_id = self.graph_metadata.get("root_node")
            root_content = self.nodes_data[root_id].content if root_id in self.nodes_data else dependency_content
        except Exception:
            root_content = dependency_content
        
        if str(tag).lower() != "root":
            current_for_prompt = f"Problem: {root_content}\nContext: {dependency_content}"
            combined_for_prompt = f"Problem: {root_content} | Reference: {reference_content} | Context: {dependency_content}"
            aggregated_for_prompt = f"Problem: {root_content}\n{dependency_content}"
        else:
            current_for_prompt = dependency_content
            combined_for_prompt = f"{reference_content} | {dependency_content}"
            aggregated_for_prompt = dependency_content
        
        return template.format(
            current_content=current_for_prompt,
            reference_content=reference_content,
            combined_content=combined_for_prompt,
            aggregated_content=aggregated_for_prompt,
            evaluated_content=dependency_content
        )
    
    def _score_branch(self, text: str) -> float:
        try:
            length = max(1, len(text))
            ops = sum(1 for ch in text if ch in ['+', '-', '*', '\u00D7', '\u00F7', '/', '(', ')'])
            digits = sum(1 for ch in text if ch.isdigit())
            eq24 = 0.0
            try:
                if str(self.dataset_name) in ("24point", "comprehensive_24point_67_dataset") and re.search(r"=\s*24\b", text):
                    eq24 = 3.0
            except Exception:
                eq24 = 0.0
            density = (ops + digits) / length
            return density * 5.0 + eq24
        except Exception:
            return 0.0
    
    def _generate_multi_nodes(self, parent_node: ReasoningNode, prompt: str) -> List[ReasoningNode]:

        try:
            llm_result = self.llm_backend.generate(prompt, **(self.llm_sampling or {}))

            if isinstance(llm_result, dict):
                response = llm_result.get("result", "")
            else:
                response = str(llm_result)
            separator = self.reasoning_paradigm.multi_node_separator
            node_contents = [content.strip() for content in response.split(separator) if content.strip()]
            
            
            if not node_contents:
                import re
                candidates = [separator, "***", "---thinking---", "###", ">>>"]
                pattern = r"\n?(?:(?:\*{3,})|(?:---thinking---)|(?:###)|(?:>>>))\n?"
                if any(c in response for c in candidates):
                    node_contents = [c.strip() for c in re.split(pattern, response) if c.strip()]
            
            if not node_contents:
                return []
        
            if getattr(self.reasoning_paradigm, 'enable_branch_scoring', False):
                scored = [(self._score_branch(c), c) for c in node_contents]
                scored.sort(key=lambda x: x[0], reverse=True)
                beam = max(1, int(getattr(self.reasoning_paradigm, 'beam_width', 2)))
                node_contents = [c for _, c in scored[:beam]]
        
            group_id = f"group_{self.inner_group_counter}"
            self.inner_group_counter += 1
            
            new_nodes = []
            for i, content in enumerate(node_contents):
                node_id = f"node_{self.node_counter}"
                node = ReasoningNode(
                    id=node_id,
                    content=content,
                    round=parent_node.round + 1,
                    parent_ids=[parent_node.id],
                    is_inner_node=True,
                    inner_group_id=group_id,
                    word_count=len(content.split()),
                    generation_prompt=prompt
                )
                
                self.nodes_data[node_id] = node
                self.reasoning_graph.add_node(node_id, **node.__dict__)
                self.reasoning_graph.add_edge(parent_node.id, node_id)
                
                parent_node.children_ids.append(node_id)
                new_nodes.append(node)
                self.node_counter += 1
            
            rule = self.reasoning_paradigm.multi_node_connection_rule
            if rule == "sequential":
                for i in range(len(new_nodes) - 1):
                    self.reasoning_graph.add_edge(new_nodes[i].id, new_nodes[i+1].id)
            elif rule == "llm_defined":
                try:
                    pairs = []
                    try:
                        summaries = [f"[{i}] {n.content[:60]}" for i, n in enumerate(new_nodes)]
                        example = "Example: If segment [0] is the premise of [1], output [[0,1]]; if all are parallel with no dependencies, output []."
                        rules = (
                            "Rules: Only add edges when definite precedence exists; no cycles; when unsure, prefer fewer edges; at most 5 edges.\n"
                            "Output: Only output a JSON array, with no explanations or extra text."
                        )
                        edge_prompt = (
                            "You are a graph planner. Given several consecutive reasoning segments, determine which segments have a 'prior->posterior' dependency.\n"
                            + example + "\n" + rules + "\n"
                            f"Segment list:\n" + "\n".join(summaries)
                        )
                    except Exception:
                        edge_prompt = "Connect sequentially from 0..n-1, output JSON like [[0,1],[1,2]]."
                    llm_result = self.llm_backend.generate(edge_prompt, **(self.llm_sampling or {}))
                    text = llm_result.get("result", "") if isinstance(llm_result, dict) else str(llm_result)
                    import json as _json
                    edges_json = None
                    try:
                        import re as _re
                        m = _re.search(r"\[(?:\s*\[.*?\]\s*,?\s*)+\]", text, flags=_re.DOTALL)
                        if m:
                            edges_json = _json.loads(m.group(0))
                        else:
                            edges_json = _json.loads(text)
                    except Exception:
                        edges_json = []
                    used = set()
                    for e in (edges_json or [])[:5]:
                        try:
                            u_idx, v_idx = int(e[0]), int(e[1])
                            key = (u_idx, v_idx)
                            if (
                                0 <= u_idx < len(new_nodes)
                                and 0 <= v_idx < len(new_nodes)
                                and u_idx != v_idx
                                and key not in used
                            ):
                                self.reasoning_graph.add_edge(new_nodes[u_idx].id, new_nodes[v_idx].id)
                                used.add(key)
                        except Exception:
                            continue
                except Exception as _:
                    pass
          
            
            self.graph_metadata["inner_node_groups"][group_id] = [n.id for n in new_nodes]
            self._qprint(f"[🔗] Created inner node group {group_id}: {len(new_nodes)} nodes")
            
            return new_nodes
            
        except Exception as e:
            self._qprint(f"[❌] Multi-node generation failed: {e}")
            return []
    
    def _generate_single_node(self, parent_node: ReasoningNode, prompt: str) -> List[ReasoningNode]:

        try:
            llm_result = self.llm_backend.generate(prompt, **(self.llm_sampling or {}))
            if isinstance(llm_result, dict):
                response = llm_result.get("result", "")
            else:
                response = str(llm_result)
            
            node_id = f"node_{self.node_counter}"
            content_preview = (response or "").lower()
            if any(k in content_preview for k in ["def ", "class ", "public ", "#include", "```"]):
                auto_label = "code"
            elif any(k in content_preview for k in ["=24", "sqrt", "\u221A", "+", "-", "*", "/"]):
                auto_label = "math"
            elif any(k in content_preview for k in ["according to", "statute", "judgment", "ruling", "law"]):
                auto_label = "legal"
            elif any(k in content_preview for k in ["option", "answer:", "abcd", "a.", "b.", "answer="]):
                auto_label = "choice"
            else:
                auto_label = "text"

            node = ReasoningNode(
                id=node_id,
                content=response.strip(),
                round=parent_node.round + 1,
                parent_ids=[parent_node.id],
                word_count=len(response.split()),
                generation_prompt=prompt,
                auto_label=auto_label
            )
            
            self.nodes_data[node_id] = node
            self.reasoning_graph.add_node(node_id, **node.__dict__)
            self.reasoning_graph.add_edge(parent_node.id, node_id)
            
            parent_node.children_ids.append(node_id)
            self.node_counter += 1
            
            self._qprint(f"[🌿] Generated node {node_id}")
            return [node]
            
        except Exception as e:
            self._qprint(f"[❌] Single-node generation failed: {e}")
            return []
    
    def _get_all_ancestors(self, node_id: str) -> List[str]:
        ancestors = []
        try:
            ancestors = list(nx.ancestors(self.reasoning_graph, node_id))
        except:
            pass
        return ancestors
    
    def _get_k_hop_ancestors(self, node_id: str, k: int) -> List[str]:
        k_ancestors = []
        try:
            for ancestor in nx.ancestors(self.reasoning_graph, node_id):
                try:
                    distance = nx.shortest_path_length(self.reasoning_graph, ancestor, node_id)
                    if distance <= k:
                        k_ancestors.append(ancestor)
                except:
                    continue
        except:
            pass
        return k_ancestors
    
    def _select_nodes_by_rule(self, current_node: ReasoningNode, rule: str) -> List[str]:
        if not rule:
            return []
        
        if "most_related" in rule:
            # Select the nearest nodes
            all_nodes = list(self.nodes_data.keys())
            return all_nodes[-2:] if len(all_nodes) > 2 else []
        elif "all_related" in rule:
            neighbors = []
            for pred in self.reasoning_graph.predecessors(current_node.id):
                neighbors.append(pred)
            for succ in self.reasoning_graph.successors(current_node.id):
                neighbors.append(succ)
            return neighbors
        
        return []
    
    def _compute_final_results(self, success: bool, answer: str, total_rounds: int) -> Dict[str, Any]:
        self._compute_answer_paths()
        self._compute_lambda_ratio()
        
        try:
            self._mark_effective_nodes()
        except Exception as _:
            pass
        
        metrics = self._compute_comprehensive_metrics()
        graph_visualization = self._generate_graph_visualization()
        
        result = {
            "success": success,
            "final_answer": answer,
            "total_rounds": total_rounds,
            "total_nodes": len(self.nodes_data),
            "reasoning_graph": self._export_graph_data(),
            "metrics": metrics,
            "visualization": graph_visualization,
            "metadata": self.graph_metadata
        }
        
        self._qprint(f"[📊] Reasoning completed: success={success}, nodes={len(self.nodes_data)}, rounds={total_rounds}")
        return result
    
    def _compute_answer_paths(self):
        root_id = self.graph_metadata["root_node"]
        answer_nodes = self.graph_metadata["answer_nodes"]
        
        all_paths = []
        for answer_id in answer_nodes:
            try:
                paths = list(nx.all_simple_paths(self.reasoning_graph, root_id, answer_id))
                all_paths.extend(paths)
            except:
                continue
        
        self.graph_metadata["answer_paths"] = all_paths
    
    def _compute_lambda_ratio(self):
        answer_paths = self.graph_metadata["answer_paths"]
        
        if not answer_paths:
            self.graph_metadata["lambda_ratio"] = 0.0
            return
    
        path_nodes = set()
        for path in answer_paths:
            path_nodes.update(path)
        
        lambda_ratio = len(path_nodes) / len(self.nodes_data) if self.nodes_data else 0.0
        self.graph_metadata["lambda_ratio"] = lambda_ratio
        
        self._qprint(f"[📊] Lambda ratio: {lambda_ratio:.3f} ({len(path_nodes)}/{len(self.nodes_data)})")
    
   
    def _mark_effective_nodes(self) -> None:
        try:
            for node in self.nodes_data.values():
                node.is_effective = False
            effective_ids = set()
            for path in self.graph_metadata.get("answer_paths", []) or []:
                for nid in path:
                    if nid in self.nodes_data:
                        self.nodes_data[nid].is_effective = True
                        effective_ids.add(nid)
            self.graph_metadata["effective_node_ids"] = list(effective_ids)
        except Exception as _:
            pass
    
    def _compute_comprehensive_metrics(self) -> Dict[str, Any]:
        total_nodes = len(self.nodes_data)
        effective_nodes = len([n for n in self.nodes_data.values() if n.is_effective])
        def _token_count(text: str) -> int:
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(text or ""))
            except Exception:
                return len((text or "").split())
        total_words = sum(_token_count(n.content) for n in self.nodes_data.values())
        effective_words = sum(_token_count(n.content) for n in self.nodes_data.values() if n.is_effective)
        
        self.graph_metadata.update({
            "total_nodes": total_nodes,
            "effective_nodes": effective_nodes,
            "total_words": total_words,
            "effective_words": effective_words
        })
        
        return {
            "node_redundancy": 1 - (effective_nodes / total_nodes) if total_nodes > 0 else 0,
            "thinking_redundancy": 1 - (effective_words / total_words) if total_words > 0 else 0,
            "terminated_nodes_count": len(self.graph_metadata["terminated_nodes"]),
            "answer_nodes_count": len(self.graph_metadata["answer_nodes"]),
            "lambda_ratio": self.graph_metadata["lambda_ratio"],
            "total_nodes": total_nodes,
            "total_rounds": len(self.graph_metadata["round_nodes"]) - 1,
            "inner_group_count": len(self.graph_metadata["inner_node_groups"])
        }
    
    def _generate_graph_visualization(self) -> Dict[str, Any]:
        try:
            pos = nx.spring_layout(self.reasoning_graph, k=2, iterations=50)
            node_colors = {}
            node_sizes = {}
            
            for node_id, node in self.nodes_data.items():
                if node.node_type == "root":
                    node_colors[node_id] = "red"
                    node_sizes[node_id] = 500
                elif node.node_type == "answer":
                    node_colors[node_id] = "green"
                    node_sizes[node_id] = 400
                elif node.node_type == "terminated":
                    node_colors[node_id] = "gray"
                    node_sizes[node_id] = 200
                elif node.is_inner_node:
                    node_colors[node_id] = "lightblue"
                    node_sizes[node_id] = 250
                else:
                    node_colors[node_id] = "lightcoral"
                    node_sizes[node_id] = 300
            
            merged_groups = {}
            try:
                inner_groups = self.graph_metadata.get("inner_node_groups", {}) or {}
                for group_id, member_ids in inner_groups.items():
                    xs, ys, valid_ids = [], [], []
                    for mid in member_ids:
                        if mid in pos:
                            x, y = pos[mid]
                            xs.append(float(x))
                            ys.append(float(y))
                            valid_ids.append(mid)
                    if not valid_ids:
                        continue
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)
                    max_r = 0.0
                    for x, y in zip(xs, ys):
                        dx = x - cx
                        dy = y - cy
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist > max_r:
                            max_r = dist
                    sample_texts = []
                    for mid in valid_ids[:3]: 
                        content = str(self.nodes_data.get(mid).content if mid in self.nodes_data else "")
                        words = content.strip().split()
                        sample_texts.append(" ".join(words[:10]))
                    summary = " | ".join(sample_texts)
                    merged_groups[group_id] = {
                        "node_ids": valid_ids,
                        "centroid": [cx, cy],
                        "radius": float(max_r * 1.2 + 0.05),
                        "size": len(valid_ids),
                        "summary": summary
                    }
            except Exception as _:
                merged_groups = {}
            
            return {
                "positions": {k: [float(v[0]), float(v[1])] for k, v in pos.items()},
                "node_colors": node_colors,
                "node_sizes": node_sizes,
                "edges": list(self.reasoning_graph.edges()),
                "merged_groups": merged_groups,
                "graph_stats": {
                    "nodes": len(self.reasoning_graph.nodes()),
                    "edges": len(self.reasoning_graph.edges()),
                    "components": nx.number_weakly_connected_components(self.reasoning_graph)
                }
            }
        except Exception as e:
            self._qprint(f"[❌] Graph visualization generation failed: {e}")
            return {}
    
    def _export_graph_data(self) -> Dict[str, Any]:
        return {
            "nodes": {nid: node.__dict__ for nid, node in self.nodes_data.items()},
            "edges": list(self.reasoning_graph.edges()),
            "metadata": self.graph_metadata,
            "paradigm": self.reasoning_paradigm.__dict__
        }
    
    def _reset_reasoning_state(self):
        self.reasoning_graph.clear()
        self.nodes_data.clear()
        self.node_counter = 0
        self.inner_group_counter = 0
        
        self.graph_metadata = {
            "root_node": None,
            "answer_nodes": [],
            "terminated_nodes": [],
            "answer_paths": [],
            "lambda_ratio": 0.0,
            "total_nodes": 0,
            "effective_nodes": 0,
            "total_words": 0,
            "effective_words": 0,
            "round_nodes": {},
            "inner_node_groups": {}
        }
    
    def save_graph_visualization(self, output_path: str):
        if not self.save_visualizations:
            self._qprint(f"[📊] Visualization image generation disabled, skipping: {output_path}")
            return
            
        try:
            plt.figure(figsize=(15, 10))
            pos = nx.spring_layout(self.reasoning_graph, k=2, iterations=50)
            for node_id, node in self.nodes_data.items():
                x, y = pos[node_id]
                
                if node.node_type == "root":
                    color, size = "red", 500
                elif node.node_type == "answer":
                    color, size = "green", 400
                elif node.node_type == "terminated":
                    color, size = "gray", 200
                elif node.is_inner_node:
                    color, size = "lightblue", 250
                else:
                    color, size = "lightcoral", 300
                
                plt.scatter(x, y, c=color, s=size, alpha=0.7)
                plt.annotate(node_id, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
            nx.draw_networkx_edges(self.reasoning_graph, pos, alpha=0.5, arrows=True)
            
            try:
                inner_groups = self.graph_metadata.get("inner_node_groups", {}) or {}
                for group_id, member_ids in inner_groups.items():
                    xs, ys = [], []
                    for mid in member_ids:
                        if mid in pos:
                            x, y = pos[mid]
                            xs.append(float(x))
                            ys.append(float(y))
                    if not xs:
                        continue
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)
                    max_r = 0.0
                    for x, y in zip(xs, ys):
                        dx = x - cx
                        dy = y - cy
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist > max_r:
                            max_r = dist
                    radius = float(max_r * 1.2 + 0.05)
                    circle = plt.Circle((cx, cy), radius, color='dodgerblue', fill=False, linestyle='--', linewidth=1.5, alpha=0.7)
                    plt.gca().add_patch(circle)
                    plt.annotate(f"{group_id} ({len(member_ids)})", (cx, cy), color='dodgerblue', fontsize=9,
                                 xytext=(0, -10), textcoords='offset_points', ha='center')
            except Exception as _:
                pass
            
            plt.title(f"Reasoning Graph (nodes: {len(self.nodes_data)}, Lambda: {self.graph_metadata['lambda_ratio']:.3f})")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._qprint(f"[💾] Graph visualization saved: {output_path}")
            
        except Exception as e:
            self._qprint(f"[❌] Failed to save graph visualization: {e}")