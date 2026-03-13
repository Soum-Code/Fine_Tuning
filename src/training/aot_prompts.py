# Atom of Thoughts Prompts
# Based on Research Methodology

DECOMPOSE_PROMPT = """You are an expert logical decomposer. 
Given the following question and its reasoning trajectory, decompose it into a Directed Acyclic Graph (DAG) of atomic sub-questions.

Question: {question}
Trajectory: {trajectory}

Format the output as a JSON list of nodes:
[
  {{"id": 0, "description": "sub-question description", "dependencies": []}},
  ...
]
Remember: A node can only depend on nodes with a smaller ID."""

CONTRACT_PROMPT = """You are a mathematical and logical contractor. 
Your goal is to simplify a complex state by incorporating already solved sub-problems into a new, self-contained question.

Original Question: {original_question}
Known Answers from Sub-Problems:
{known_answers}

Remaining Dependent Nodes:
{remaining_nodes}

Generate a new, self-contained question that is simpler than the original but leads to the same final answer."""

JUDGE_PROMPT = """You are a logical judge. 
Given the original question and several candidate reasoning paths/answers, select the objectively best and most correct solution.

Original Question: {question}

Candidate Solutions:
1. {candidate_1}
2. {candidate_2}
3. {candidate_3}

Output ONLY the number of the best candidate and a brief justification."""

DIRECT_SOLVE_PROMPT = """Solve the following problem step-by-step using atomic reasoning.
Question: {question}
"""
