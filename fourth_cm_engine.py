"""
4CM Engine - The Complete System
=================================
REAL LLM VERSION

Key change from prototype:
- Agent responses: Real Claude API calls (not hardcoded templates)
- Semantic comparison: Claude API judges meaning, not word overlap
- Embedding: sentence-transformers still used for torus math
  but semantic verdict comes from Claude itself

PhD Dissertation, 2011.
"""

import os
import json
import requests
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from torus_math import TorusField, JudgeFunction, ConstraintLayer
from orthogonal_agents import (
    OrthogonalAgent,
    create_government_scenario_agents,
    simulate_orthogonal_response,
    call_claude
)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"


# ============================================================
# Semantic Comparison Engine (Claude-powered)
# ============================================================

SEMANTIC_JUDGE_SYSTEM = """You are a rigorous semantic convergence analyst.

Your job: given responses from 4 agents with completely different perspectives,
determine whether they are converging toward the SAME CONCLUSION — 
not because they use the same words, but because they MEAN the same thing.

You must be brutally honest. Do not be fooled by:
- Same words but different meanings
- Superficial agreement masking deep disagreement
- One agent capitulating to others vs genuine independent convergence

You ARE looking for:
- Same recommended ACTION despite different reasoning
- Same target/subject despite different framing
- Same urgency level despite different justifications
- Genuine semantic alignment across all 4 agents

Output ONLY a JSON object with exactly these fields:
{
  "semantic_similarity_score": <float 0.0-1.0>,
  "all_point_same_direction": <true/false>,
  "common_conclusion": "<what all 4 are saying, in one sentence, or null>",
  "weakest_link": "<which agent diverges most and why>",
  "convergence_analysis": "<2-3 sentences explaining your judgment>"
}

No other text. No markdown. Just the JSON."""


def semantic_compare(responses: Dict[str, str], query: str) -> Dict:
    """
    Use Claude to semantically judge whether 4 agent responses
    are converging on the same conclusion.

    This is the core improvement over TF-IDF/word-overlap:
    Claude understands MEANING, not just vocabulary.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    formatted = "\n\n".join([
        f"[{name}]:\n{resp}"
        for name, resp in responses.items()
    ])

    user_message = (
        f"Original query: {query}\n\n"
        f"Four orthogonal agents responded:\n\n"
        f"{formatted}\n\n"
        f"Analyze whether these 4 responses are semantically converging "
        f"toward the same conclusion. Be harsh and precise."
    )

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    payload = {
        "model": MODEL,
        "max_tokens": 500,
        "system": SEMANTIC_JUDGE_SYSTEM,
        "messages": [{"role": "user", "content": user_message}]
    }

    response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Claude API error {response.status_code}: {response.text}")

    raw = response.json()["content"][0]["text"].strip()

    # Parse JSON
    # Strip markdown fences if present
    clean = raw
    if "```" in clean:
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()

    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        # Fallback if parsing fails
        result = {
            "semantic_similarity_score": 0.0,
            "all_point_same_direction": False,
            "common_conclusion": None,
            "weakest_link": "parse error",
            "convergence_analysis": f"Raw response: {raw[:200]}"
        }

    return result


# ============================================================
# Embedding Engine (for torus math)
# ============================================================

class EmbeddingEngine:
    """
    Converts text responses to vectors for torus geometry.
    Uses sentence-transformers for vector math.
    Semantic VERDICT comes from Claude (above), not from here.
    """

    def __init__(self, use_transformer=True):
        self.model = None
        self.use_transformer = use_transformer
        self._fitted = False
        self._init_model()

    def _init_model(self):
        if self.use_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embed_dim = 384
                print("    [Embedding] Sentence-Transformer loaded (all-MiniLM-L6-v2)")
            except Exception as e:
                print(f"    [Embedding] Transformer failed: {e}")
                self.use_transformer = False

        if not self.use_transformer:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=512)
            self.embed_dim = 512
            print("    [Embedding] TF-IDF initialized (fallback)")

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        if self.use_transformer and self.model:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [emb for emb in embeddings]
        else:
            if not self._fitted:
                tfidf = self.vectorizer.fit_transform(texts)
                self._fitted = True
            else:
                tfidf = self.vectorizer.transform(texts)
            return [tfidf[i].toarray().flatten() for i in range(len(texts))]

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ============================================================
# The 5th Response
# ============================================================

@dataclass
class FifthResponse:
    """
    The 5th Response.
    Born from semantic convergence of 4 orthogonals.
    No model. No weights. No memory.
    """
    exists: bool
    content: Optional[str]
    consensus_value: float
    singularity_ratio: float
    semantic_score: float
    common_conclusion: Optional[str]
    agent_responses: Dict[str, str]
    judgment: Dict
    semantic_judgment: Dict
    timestamp: str

    def __str__(self):
        if not self.exists:
            return (
                f"\n  [THE 5TH RESPONSE DOES NOT EXIST]\n"
                f"  Semantic score: {self.semantic_score:.4f}\n"
                f"  Weakest link: {self.semantic_judgment.get('weakest_link', 'N/A')}\n"
                f"  Analysis: {self.semantic_judgment.get('convergence_analysis', 'N/A')}\n"
            )
        return (
            f"\n╔══════════════════════════════════════════════════════════════╗\n"
            f"║  ★ THE 5TH RESPONSE                                         ║\n"
            f"║  Singularity Ratio:  {self.singularity_ratio:.4f}                           ║\n"
            f"║  Semantic Score:     {self.semantic_score:.4f}                           ║\n"
            f"╠══════════════════════════════════════════════════════════════╣\n"
            f"║  Common Conclusion:                                          ║\n"
            f"║  {(self.common_conclusion or 'N/A')[:60]:60s}║\n"
            f"╠══════════════════════════════════════════════════════════════╣\n"
            f"║  Analysis: {(self.semantic_judgment.get('convergence_analysis',''))[:58]:58s}║\n"
            f"╠══════════════════════════════════════════════════════════════╣\n"
            f"║  No model. No weights. No storage.                          ║\n"
            f"║  Born at: {self.timestamp:20s}                        ║\n"
            f"║  Already gone.                                               ║\n"
            f"╚══════════════════════════════════════════════════════════════╝"
        )


# ============================================================
# The 4CM Engine
# ============================================================

class FourCMEngine:
    """
    The complete 4 Councilmen Theory engine.

    Two-layer judgment:
    1. Torus math (geometry) — where are the agents on the field?
    2. Claude semantic judge — do they MEAN the same thing?

    Both must confirm for singularity.
    """

    def __init__(self, scenario: str = "government"):
        print("\n" + "="*70)
        print("  4CM ENGINE - INITIALIZATION (Claude API version)")
        print("="*70)

        print("\n  [1/5] Initializing Torus Field...")
        self.torus = TorusField()
        if self.torus.singularity:
            s = self.torus.singularity
            print(f"         Singularity threshold: {s['threshold']:.4f}")
            print(f"         Ring radius: {s['ring_radius']:.4f}")

        print(f"\n  [2/5] Loading Orthogonal Agents ({scenario} scenario)...")
        self.agents = create_government_scenario_agents()
        for agent in self.agents:
            print(f"         {agent.name} ({agent.role})")

        print(f"\n  [3/5] Loading Embedding Engine (for torus geometry)...")
        self.embedder = EmbeddingEngine(use_transformer=True)

        print(f"\n  [4/5] Initializing Constraint Layer...")
        self.constraint = ConstraintLayer(self.torus, drift_tolerance=0.3)

        print(f"\n  [5/5] Initializing Judge Function...")
        self.judge = JudgeFunction(self.torus)
        print(f"         Torus judge: pure math.")
        print(f"         Semantic judge: Claude {MODEL}.")

        print("\n" + "="*70)
        print("  4CM ENGINE READY")
        print("  The phone booth is waiting.")
        print("="*70 + "\n")

    def query(self, question: str, context: str = "") -> FifthResponse:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        print(f"\n{'─'*70}")
        print(f"  QUERY: {question[:65]}")
        print(f"  TIME:  {timestamp}")
        print(f"{'─'*70}")

        # Step 1: Real Claude API responses from each agent
        print(f"\n  [Step 1] Calling Claude API for each orthogonal agent...")
        responses = {}
        for agent in self.agents:
            print(f"    Calling {agent.name}...", end=" ", flush=True)
            response = simulate_orthogonal_response(agent, question, context)
            responses[agent.agent_id] = response
            agent.response_history.append(response)
            print("done")
            print(f"      → {response[:100]}...")

        # Step 2: Torus geometry (sentence-transformer embeddings)
        print(f"\n  [Step 2] Computing torus geometry embeddings...")
        response_texts = [responses[agent.agent_id] for agent in self.agents]
        embeddings = self.embedder.embed(response_texts)

        neutral_response = (
            f"Regarding '{question[:50]}', a balanced analysis suggests "
            f"multiple perspectives should be considered."
        )
        neutral_embedding = self.embedder.embed_single(neutral_response)

        # Step 3: Constraint validation
        print(f"\n  [Step 3] Validating agent extremity on torus...")
        positions = []
        for i, agent in enumerate(self.agents):
            validation = self.constraint.validate_agent_position(
                agent.agent_id, embeddings[i], neutral_embedding
            )
            positions.append(validation['effective_position'])
            status = "ORTHOGONAL" if validation['is_orthogonal_enough'] else "DRIFTING"
            icon = "■" if validation['is_orthogonal_enough'] else "□"
            print(f"    {icon} {agent.name}: neutrality={validation['neutrality']:.3f} → {status}")

        # Step 4: Semantic judgment via Claude
        # Claude semantic score is the SOLE input to the torus.
        # sentence-transformers vectors are no longer used for singularity.
        # Claude understands MEANING. The torus maps meaning to geometry.
        print(f"\n  [Step 4] Semantic Judge (Claude {MODEL})...")
        named_responses = {agent.name: responses[agent.agent_id] for agent in self.agents}
        semantic = semantic_compare(named_responses, question)

        semantic_score = semantic.get('semantic_similarity_score', 0)
        same_direction = semantic.get('all_point_same_direction', False)
        common_conclusion = semantic.get('common_conclusion')

        print(f"    Semantic score:    {semantic_score:.4f}")
        print(f"    Same direction:    {same_direction}")
        print(f"    Conclusion:        {common_conclusion}")
        print(f"    Weakest link:      {semantic.get('weakest_link', 'N/A')}")

        # Step 5: Feed Claude semantic score directly into torus f(x,y)
        # semantic_score 0.0~1.0 -> torus coordinate -> f(x,y) -> singularity
        print(f"\n  [Step 5] Torus Judge — semantic score -> f(x,y)...")
        judgment = self.judge.compute_convergence_from_semantic(semantic_score, positions)

        print(f"    Torus coordinate:  ({judgment['convergence_point'][0]:.4f}, {judgment['convergence_point'][1]:.4f})")
        print(f"    Consensus value:   {judgment['consensus_value']:.6f}")
        print(f"    Threshold:         {judgment['threshold']:.6f}")
        print(f"    Singularity ratio: {judgment['singularity_ratio']:.4f}")

        # Step 6: Final verdict — semantic drives torus, torus gives binary
        is_singularity = judgment['is_singularity'] and same_direction

        print(f"\n  [Verdict]")
        print(f"    Semantic score:   {semantic_score:.4f}  ({'converging' if same_direction else 'diverging'})")
        print(f"    Torus ratio:      {judgment['singularity_ratio']:.4f}")
        print(f"    FINAL:            {'★★ SINGULARITY' if is_singularity else 'No singularity'}")

        return FifthResponse(
            exists=is_singularity,
            content=common_conclusion,
            consensus_value=judgment['consensus_value'],
            singularity_ratio=judgment['singularity_ratio'],
            semantic_score=semantic_score,
            common_conclusion=common_conclusion,
            agent_responses=named_responses,
            judgment=judgment,
            semantic_judgment=semantic,
            timestamp=timestamp
        )
