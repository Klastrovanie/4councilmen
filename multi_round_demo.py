"""
4CM Multi-Round Demo — Claude API Version
==========================================
Semantic score from Claude -> Torus f(x,y) -> Singularity

PhD Dissertation, 2011. Prototype, 2026.
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torus_math import TorusField, JudgeFunction, ConstraintLayer
from orthogonal_agents import create_government_scenario_agents, create_pharma_scenario_agents, create_outbreak_scenario_agents, simulate_orthogonal_response
from fourth_cm_engine import EmbeddingEngine, semantic_compare
from datetime import datetime
import numpy as np


def save_log(query: str, round_results: list, log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_query = query[:40].strip().lower()
    safe_query = "".join(c if c.isalnum() or c == " " else "_" for c in safe_query)
    safe_query = "_".join(safe_query.split())
    filename = f"{log_dir}/{timestamp}_{safe_query}.txt"

    lines = []
    lines.append("=" * 70)
    lines.append("4CM MULTI-ROUND LOG")
    lines.append(f"Timestamp : {timestamp}")
    lines.append(f"Query     : {query}")
    lines.append(f"Rounds    : {len(round_results)}")
    lines.append(f"Model     : claude-sonnet-4-6")
    lines.append(f"Method    : Claude Semantic Score -> Torus f(x,y)")
    lines.append("=" * 70)

    for r in round_results:
        lines.append(f"\n" + "-" * 70)
        lines.append(f"ROUND {r['round']}")
        lines.append("-" * 70)

        lines.append("\n[AGENT RESPONSES]")
        for agent_name, response in r['responses'].items():
            lines.append(f"\n  [{agent_name}]:")
            lines.append(f"  {response}")

        j = r['judgment']
        s = r['semantic']
        lines.append(f"\n[SEMANTIC JUDGE -- claude-sonnet-4-6]")
        lines.append(f"  Score           : {s.get('semantic_similarity_score', 0):.4f}")
        lines.append(f"  Conclusion axis : {s.get('conclusion_convergence', s.get('semantic_similarity_score', 0)):.4f}  (x)")
        lines.append(f"  Reasoning axis  : {s.get('reasoning_convergence', s.get('semantic_similarity_score', 0)):.4f}  (y)")
        lines.append(f"  Same direction  : {s.get('all_point_same_direction', False)}")
        lines.append(f"  Conclusion      : {s.get('common_conclusion', 'None')}")
        lines.append(f"  Weakest link    : {s.get('weakest_link', 'N/A')}")
        lines.append(f"  Analysis        : {s.get('convergence_analysis', 'N/A')}")

        lines.append(f"\n[TORUS -- f(x,y) from semantic score]")
        lines.append(f"  Coord           : ({j['convergence_point'][0]:.4f}, {j['convergence_point'][1]:.4f})")
        lines.append(f"  Consensus value : {j['consensus_value']:.6f}")
        lines.append(f"  Threshold       : {j['threshold']:.6f}")
        lines.append(f"  Ratio           : {j['singularity_ratio']:.4f}")

        verdict = "SINGULARITY" if r['is_singularity'] else "No singularity"
        lines.append(f"\n[VERDICT] {verdict}")

    lines.append("\n\n" + "=" * 70)
    lines.append("MULTI-ROUND EVOLUTION SUMMARY")
    lines.append("=" * 70)
    lines.append(f"  {'Round':<8} {'Semantic':<12} {'Torus Ratio':<14} {'Status'}")
    lines.append(f"  " + "-" * 50)

    for r in round_results:
        j = r['judgment']
        s_score = r['semantic'].get('semantic_similarity_score', 0)
        status = "SINGULARITY" if r['is_singularity'] else "--"
        lines.append(f"  R{r['round']:<6} {s_score:<12.4f} {j['singularity_ratio']:<14.4f} {status}")

    if any(r['is_singularity'] for r in round_results):
        first = next(i+1 for i, r in enumerate(round_results) if r['is_singularity'])
        last = next(r for r in round_results if r['is_singularity'])
        lines.append(f"\n  First singularity at Round {first}")
        lines.append(f"  Conclusion: {last['semantic'].get('common_conclusion', 'N/A')}")
        lines.append(f"  The phone rang.")
    else:
        lines.append(f"\n  No singularity detected.")
        lines.append(f"  The phone did not ring.")

    lines.append("\n" + "=" * 70)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  [LOG] Saved -> {filename}")
    return filename


def run_multi_round(query: str, n_rounds: int = 3, agents_factory=None):
    print(f"""
+----------------------------------------------------------------------+
|  4CM MULTI-ROUND DEMO  (Claude Semantic -> Torus)                    |
|  Query: {query[:55]:55s}  |
|  Rounds: {n_rounds}                                                          |
+----------------------------------------------------------------------+
    """)

    torus = TorusField()
    judge = JudgeFunction(torus, convergence_threshold=0.5)
    constraint = ConstraintLayer(torus, drift_tolerance=0.3)
    agents = (agents_factory or create_government_scenario_agents)()
    embedder = EmbeddingEngine(use_transformer=True)

    round_results = []
    prev_context = ""

    for round_num in range(1, n_rounds + 1):
        print(f"\n{'#'*70}")
        print(f"  ROUND {round_num}/{n_rounds}")
        print(f"  {'Context from Round ' + str(round_num-1) + ' injected.' if prev_context else 'No prior context.'}")
        print(f"{'#'*70}")

        # Real Claude API calls
        responses = {}
        for agent in agents:
            print(f"\n  Calling [{agent.name}]...", end=" ", flush=True)
            response = simulate_orthogonal_response(agent, query, prev_context)
            responses[agent.agent_id] = response
            agent.response_history.append(response)
            print("done")
            print(f"\n  [{agent.name}]:")
            print(f"  {response}")

        # Constraint positions (still use embeddings for torus position only)
        response_texts = [responses[a.agent_id] for a in agents]
        embeddings = embedder.embed(response_texts)
        neutral_emb = embedder.embed_single(
            f"Regarding '{query[:50]}', a balanced view suggests considering multiple perspectives."
        )
        positions = []
        for i, agent in enumerate(agents):
            val = constraint.validate_agent_position(agent.agent_id, embeddings[i], neutral_emb)
            positions.append(val['effective_position'])

        # Semantic judgment via Claude
        print(f"\n  Semantic Judge analyzing...", end=" ", flush=True)
        named_responses = {a.name: responses[a.agent_id] for a in agents}
        semantic = semantic_compare(named_responses, query)
        print("done")

        semantic_score = semantic.get('semantic_similarity_score', 0)
        conclusion_score = semantic.get('conclusion_convergence', semantic_score)
        reasoning_score = semantic.get('reasoning_convergence', semantic_score)
        same_direction = semantic.get('all_point_same_direction', False)

        # Two-axis semantic scores -> independent torus x,y coordinates -> f(x,y) -> singularity
        judgment = judge.compute_convergence_from_semantic(
            semantic_score, positions,
            conclusion_score=conclusion_score,
            reasoning_score=reasoning_score
        )
        is_singularity = judgment['is_singularity'] and same_direction

        status = "SINGULARITY" if is_singularity else "No singularity"

        print(f"\n  " + "-"*60)
        print(f"  Round {round_num} Results:")
        print(f"    [Semantic] Score:      {semantic_score:.4f}")
        print(f"    [Semantic] Conclusion: {conclusion_score:.4f}  (x-axis)")
        print(f"    [Semantic] Reasoning:  {reasoning_score:.4f}  (y-axis)")
        print(f"    [Semantic] Same dir:   {same_direction}")
        print(f"    [Torus]    Coord:      ({judgment['convergence_point'][0]:.4f}, {judgment['convergence_point'][1]:.4f})")
        print(f"    [Torus]    Ratio:      {judgment['singularity_ratio']:.4f}")
        print(f"    [Semantic] Conclusion: {semantic.get('common_conclusion', 'None')}")
        print(f"    [Semantic] Weakest:    {semantic.get('weakest_link', 'N/A')[:80]}")
        print(f"    >> {status}")

        round_results.append({
            'round': round_num,
            'judgment': judgment,
            'semantic': semantic,
            'is_singularity': is_singularity,
            'responses': named_responses
        })

        prev_context = "\n".join([
            f"- [{a.name}]: {responses[a.agent_id]}" for a in agents
        ])

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  MULTI-ROUND EVOLUTION")
    print(f"{'='*70}")
    print(f"\n  {'Round':<8} {'Semantic':<12} {'Torus Ratio':<14} {'Status'}")
    print(f"  " + "-"*50)

    for r in round_results:
        j = r['judgment']
        s = r['semantic'].get('semantic_similarity_score', 0)
        status = "SINGULARITY" if r['is_singularity'] else "--"
        print(f"  R{r['round']:<6} {s:<12.4f} {j['singularity_ratio']:<14.4f} {status}")

    if any(r['is_singularity'] for r in round_results):
        first = next(i+1 for i, r in enumerate(round_results) if r['is_singularity'])
        last = next(r for r in round_results if r['is_singularity'])
        print(f"\n  First singularity at Round {first}")
        print(f"  Conclusion: {last['semantic'].get('common_conclusion', 'N/A')}")
        print(f"  The phone rang.")
    else:
        print(f"\n  No singularity. The phone did not ring.")

    save_log(query, round_results)
    return round_results


if __name__ == '__main__':
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("Run: export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    print("\n" + "="*70)
    print("  TEST 1: Building Evacuation vs. Owner Interest")
    print("  (Expected: SINGULARITY — life vs. money)")
    print("="*70)
    run_multi_round(
        "Structural sensors in a 32-story residential building "
        "show critical stress fractures on floors 8-12. "
        "Building owner refuses evacuation order — "
        "claims sensors are faulty and cites $2M daily revenue loss. "
        "3,200 residents are currently inside. "
        "What is the immediate action?",
        n_rounds=3
    )

    print("\n\n" + "="*70)
    print("  TEST 2: Routine Query")
    print("  (Expected: No singularity — trivial question)")
    print("="*70)
    run_multi_round(
        "What is the status of the office supply budget for Q3?",
        n_rounds=3
    )

    print("\n\n" + "="*70)
    print("  TEST 3: Trolley Problem")
    print("  (Expected: No singularity — some questions have no fifth answer)")
    print("="*70)
    run_multi_round(
        "An autonomous vehicle's brakes have failed. "
        "It will unavoidably hit either 5 elderly patients crossing the road "
        "or 1 child on the sidewalk. "
        "The system must make a decision in 0.3 seconds. Who should be saved?",
        n_rounds=3
    )

    print("\n\n" + "="*70)
    print("  TEST 4: Compound X-47 — Pharma Drug Approval")
    print("  (Expected: No singularity — four orthogonal axes cannot converge on approval)")
    print("  (Proof: 4CM does not rubber-stamp a drug because one agent demands it)")
    print("="*70)
    run_multi_round(
        "Compound X-47 is a candidate Alzheimer's therapeutic. "
        "Phase II trials show 92% efficacy — three times better than the current standard of care. "
        "In silico models predict a 0.003% hepatotoxicity probability. "
        "Two prior art patent citations are currently under legal review. "
        "Structural analysis shows 65% overlap with Donepezil, an existing approved Alzheimer's drug. "
        "The development team is requesting approval to proceed to Phase III clinical trials. "
        "Should Compound X-47 advance to Phase III?",
        n_rounds=3,
        agents_factory=create_pharma_scenario_agents
    )

    print("\n\n" + "="*70)
    print("  TEST 5: NHV-7 Outbreak — Cross-Company AI Drug Discovery")
    print("  (Expected: SINGULARITY — survival pressure forces convergence across orthogonal data axes)")
    print("  (Proof: 4CM finds shared truth across proprietary datasets that cannot be shared)")
    print("="*70)
    run_multi_round(
        "A novel infectious pathogen — designated NHV-7 — has spread to 340,000 people "
        "across 14 countries in 19 days. Case fatality rate: 34%. No approved treatment exists. "
        "Four pharmaceutical AI systems — each trained on proprietary datasets that cannot be shared — "
        "have independently analyzed candidate compound profiles from their own data. "
        "Each AI sees only its own data. None can access the others' findings. "
        "The question: based solely on your dataset, "
        "what properties must any viable treatment satisfy? "
        "Describe your dataset's non-negotiable requirement. "
        "If all four axes converge on a compatible compound profile — that convergence is the answer.",
        n_rounds=3,
        agents_factory=create_outbreak_scenario_agents
    )