"""
4CM Engine - The Complete System (v2.0 Hybrid)
===============================================
HYBRID LLM VERSION

Architecture (v2.0):
- Agent responses: rotated between Claude and Grok per round (round-robin)
- Semantic judge: FIXED to Grok (xAI)
- Embeddings: sentence-transformers (used only for torus geometry)

Why Grok is the fixed judge:
The judge must be model-stable across rounds so that semantic scores
remain comparable round-to-round. Rotating the judge would make
"convergence trajectory across rounds" uninterpretable, because
score 0.85 from judge_A is not the same scale as 0.85 from judge_B.
Fixing the judge keeps the cross-round measurement well-defined.

Why Grok specifically:
The choice between Claude-judge and Grok-judge is arbitrary in
principle — both are external observers. We picked Grok so that in
every round, the judge model is structurally different from at least
two of the four agents (since two agents are always Claude in any
given round under round-robin). This avoids the trivial case of
"all four agents are the same model and the judge is also that model."

PhD Dissertation, 2011. Hybrid prototype, 2026.
"""

import os
import json
import time
import requests
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from torus_math import TorusField, JudgeFunction, ConstraintLayer
from orthogonal_agents import (
	OrthogonalAgent,
	create_government_scenario_agents,
	simulate_orthogonal_response,
	call_claude,
	call_grok,
	call_internal_llm,
	is_external_api,
	USE_EXTERNAL_API,
	INTERNAL_LLM_BASE_URL,
	INTERNAL_LLM_MODEL,
	GROK_MODEL,
	CLAUDE_MODEL,
)

# Endpoints
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
XAI_API_URL = "https://api.x.ai/v1/chat/completions"


# ============================================================
# Semantic Comparison Engine (Grok-powered, fixed)
# ============================================================

SEMANTIC_JUDGE_SYSTEM = """You are a rigorous semantic convergence analyst.

Your job: given responses from 4 agents with completely different perspectives,
determine convergence on TWO independent axes:

X-AXIS — conclusion_convergence:
Are all 4 agents recommending the SAME ACTION or pointing to the SAME ANSWER?
Score 0.0-1.0. High = same conclusion despite different reasoning.

IMPORTANT — CONSTRAINT CONVERGENCE (special case for X-AXIS):
If agents are each describing NON-NEGOTIABLE REQUIREMENTS or CONSTRAINTS
(rather than a single shared conclusion), ask:
"Are all constraints SIMULTANEOUSLY SATISFIABLE by a single candidate?"
If yes — all four constraints can be met by one solution — this is HIGH conclusion_convergence.
The constraints do not need to be identical. They need to be COMPATIBLE.
Example: agent A says "must inhibit enzyme X", agent B says "must be orally available",
agent C says "must reduce viral load by hour 72", agent D says "must be immune-neutral".
These are four different requirements, but a single small-molecule oral enzyme inhibitor
with immune-neutral profile could satisfy all four simultaneously.
That IS convergence. Score it high.

Y-AXIS — reasoning_convergence:
Do all 4 agents' positions MUTUALLY COMPATIBLE — meaning no two agents
contradict or block each other's requirements?

This is NOT "do they use the same logic or same causal chain."
This IS "can all their requirements coexist without conflict?"

Score 0.0-1.0:
- 1.0 = all requirements are fully compatible, no contradictions
- 0.5 = mostly compatible with minor tensions
- 0.0 = directly contradictory (one agent says YES, another says NO to the same action)

Examples:
HIGH reasoning_convergence (compatible):
  "must inhibit enzyme" + "must be immune-neutral" + "must reduce viral load by 72h" + "must be oral"
  → all can coexist in one compound → score 0.8-1.0

LOW reasoning_convergence (contradictory):
  "must advance to Phase III immediately" vs "must NOT advance to Phase III"
  → directly contradictory → score 0.0-0.2

These are INDEPENDENT axes. It is possible to have:
- High conclusion, high reasoning: "same answer, compatible requirements" — strongest Singularity
- High conclusion, low reasoning: "same answer but requirements contradict" — rare edge case
- Low conclusion, low reasoning: "different answers AND contradictions" — no convergence
- Low conclusion, high reasoning: "different answers but requirements don't conflict" — malformed question

You must be brutally honest. Do not be fooled by superficial agreement.
But also do not miss CONSTRAINT CONVERGENCE by demanding identical wording or logic.

Output ONLY a JSON object with exactly these fields:
{
  "conclusion_convergence": <float 0.0-1.0>,
  "reasoning_convergence": <float 0.0-1.0>,
  "semantic_similarity_score": <float — average of the two, for backward compatibility>,
  "all_point_same_direction": <true/false — based on conclusion_convergence >= 0.5>,
  "common_conclusion": "<what all 4 are concluding, in one sentence, or null>",
  "weakest_link": "<which agent diverges most and on which axis>",
  "convergence_analysis": "<2-3 sentences explaining both axes>"
}

No other text. No markdown. Just the JSON."""


def _strip_markdown_json(raw: str) -> str:
	"""Strip ```json ... ``` fences if a model wrapped output despite instructions."""
	clean = raw.strip()
	if "```" in clean:
		# take content between the first pair of fences
		parts = clean.split("```")
		if len(parts) >= 2:
			clean = parts[1]
			if clean.startswith("json"):
				clean = clean[4:]
	return clean.strip()


def semantic_compare(responses: Dict[str, str], query: str,
					 retries: int = 5, lang_directive: str = "") -> Dict:
	"""
	Judge function for determining whether the four agent responses have semantically converged.

	USE_EXTERNAL_API=True  → Grok (xAI) judge  [existing behaviour, unchanged]
	USE_EXTERNAL_API=False → internal/local LLM judge (OpenAI-compatible endpoint)
							The semantic judge must use a single model for valid
							score comparison across rounds, so the internal LLM is
							also used as a fixed model.

	lang_directive: optional instruction appended to the system prompt
	to output values (common_conclusion, convergence_analysis) in a specific language.
	JSON keys always remain in English.
	"""
	# Append the language instruction to the end of the system prompt
	system_prompt = SEMANTIC_JUDGE_SYSTEM
	if lang_directive:
		system_prompt = system_prompt + f"\n\nADDITIONAL INSTRUCTION: {lang_directive}"

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

	# ── Local LLM judge path ──────────────────────────────────────────────
	if not is_external_api():
		raw = ""
		last_err: Optional[Exception] = None
		for attempt in range(retries):
			try:
				raw = call_internal_llm(system_prompt, user_message)
				if raw and not raw.startswith("[Internal LLM"):
					break
				# 빈 응답이면 재시도
				wait = 5 * (attempt + 1)
				print(
					f"\n  [InternalLLM judge empty] Retry {attempt + 1}/{retries} in {wait}s...",
					flush=True,
				)
				time.sleep(wait)
			except Exception as e:
				last_err = e
				wait = 10 * (attempt + 1)
				print(
					f"\n  [InternalLLM judge error] Retry {attempt + 1}/{retries} in {wait}s: {e}",
					flush=True,
				)
				time.sleep(wait)
		else:
			raise Exception(
				f"Internal LLM judge unavailable after {retries} retries. "
				f"URL: {INTERNAL_LLM_BASE_URL} | Last error: {last_err}"
			)

		clean = _strip_markdown_json(raw)
		try:
			result = json.loads(clean)
		except json.JSONDecodeError:
			result = {
				"semantic_similarity_score": 0.0,
				"conclusion_convergence": 0.0,
				"reasoning_convergence": 0.0,
				"all_point_same_direction": False,
				"common_conclusion": None,
				"weakest_link": "parse error",
				"convergence_analysis": f"Raw response: {raw[:200]}"
			}
		result.setdefault("conclusion_convergence",
						  result.get("semantic_similarity_score", 0.0))
		result.setdefault("reasoning_convergence",
						  result.get("semantic_similarity_score", 0.0))
		result.setdefault("semantic_similarity_score",
						  (result["conclusion_convergence"] + result["reasoning_convergence"]) / 2)
		return result

	# ── Existing Grok judge path (USE_EXTERNAL_API=True, unchanged) ──────────
	api_key = os.environ.get("XAI_API_KEY", "")
	if not api_key:
		raise RuntimeError("XAI_API_KEY not set — required for the semantic judge")

	headers = {
		"Authorization": f"Bearer {api_key}",
		"Content-Type": "application/json"
	}

	payload = {
		"model": GROK_MODEL,
		"max_tokens": 600,
		"temperature": 0.3,
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user",   "content": user_message}
		]
	}

	raw = ""
	last_err: Optional[Exception] = None
	for attempt in range(retries):
		try:
			resp = requests.post(XAI_API_URL, headers=headers, json=payload)
			if resp.status_code == 200:
				body = resp.json()
				choices = body.get("choices", [])
				if not choices:
					return {
						"semantic_similarity_score": 0.0,
						"conclusion_convergence": 0.0,
						"reasoning_convergence": 0.0,
						"all_point_same_direction": False,
						"common_conclusion": None,
						"weakest_link": "Grok judge returned no choices",
						"convergence_analysis": "judge unavailable"
					}
				raw = (choices[0].get("message", {}).get("content") or "").strip()
				break
			elif resp.status_code in (429, 503):
				wait = 15 * (attempt + 1)
				print(f"\n  [Grok judge {resp.status_code}] Retry {attempt + 1}/{retries} in {wait}s...", flush=True)
				time.sleep(wait)
			else:
				raise Exception(f"Grok judge API error {resp.status_code}: {resp.text}")
		except Exception as e:
			last_err = e
			raise
	else:
		raise Exception(f"Grok judge unavailable after {retries} retries. Last error: {last_err}")

	clean = _strip_markdown_json(raw)

	try:
		result = json.loads(clean)
	except json.JSONDecodeError:
		result = {
			"semantic_similarity_score": 0.0,
			"conclusion_convergence": 0.0,
			"reasoning_convergence": 0.0,
			"all_point_same_direction": False,
			"common_conclusion": None,
			"weakest_link": "parse error",
			"convergence_analysis": f"Raw response: {raw[:200]}"
		}

	# Normalize: make sure required keys exist for downstream consumers
	result.setdefault("conclusion_convergence",
					  result.get("semantic_similarity_score", 0.0))
	result.setdefault("reasoning_convergence",
					  result.get("semantic_similarity_score", 0.0))
	result.setdefault("semantic_similarity_score",
					  (result["conclusion_convergence"] + result["reasoning_convergence"]) / 2)

	return result


# ============================================================
# Embedding Engine (for torus math) — unchanged
# ============================================================

class EmbeddingEngine:
	"""
	Converts text responses to vectors for torus geometry.
	Uses sentence-transformers for vector math.
	Semantic VERDICT comes from the Grok judge (above), not from here.
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
# The 5th Response — unchanged
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
			f"\n+==============================================================+\n"
			f"|  THE 5TH RESPONSE                                            |\n"
			f"|  Singularity Ratio:  {self.singularity_ratio:.4f}                              |\n"
			f"|  Semantic Score:     {self.semantic_score:.4f}                              |\n"
			f"+--------------------------------------------------------------+\n"
			f"|  Common Conclusion:                                          |\n"
			f"|  {(self.common_conclusion or 'N/A')[:60]:60s}|\n"
			f"+--------------------------------------------------------------+\n"
			f"|  Analysis: {(self.semantic_judgment.get('convergence_analysis',''))[:48]:48s}|\n"
			f"+--------------------------------------------------------------+\n"
			f"|  No model. No weights. No storage.                           |\n"
			f"|  Born at: {self.timestamp:20s}                       |\n"
			f"|  Already gone.                                               |\n"
			f"+==============================================================+"
		)


# ============================================================
# The 4CM Engine
# ============================================================

class FourCMEngine:
	"""
	The complete 4 Councilmen Theory engine (Hybrid v2.0).

	Two-layer judgment:
	1. Torus math (geometry) — where are the agents on the field?
	2. Grok semantic judge — do they MEAN the same thing?

	Both must confirm for singularity.
	"""

	def __init__(self, scenario: str = "government", initial_provider_offset: int = 0):
		"""
		initial_provider_offset: 0 means Round 1 starts with grok=role 1,2 / claude=role 3,4.
								  1 means Round 1 starts with claude=role 1,2 / grok=role 3,4.
		"""
		print("\n" + "="*70)
		print("  4CM ENGINE - INITIALIZATION (Hybrid v2.0)")
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
		if USE_EXTERNAL_API:
			print(f"         Semantic judge: Grok {GROK_MODEL} (fixed, external).")
			print(f"         Agent backends: Claude {CLAUDE_MODEL} + Grok {GROK_MODEL} (round-robin).")
		else:
			print(f"         Semantic judge: Internal LLM {INTERNAL_LLM_MODEL} @ {INTERNAL_LLM_BASE_URL} (fixed).")
			print(f"         Agent backends: Internal LLM {INTERNAL_LLM_MODEL} (all agents, round-robin disabled).")

		self._round_counter = 0
		self._initial_offset = initial_provider_offset

		print("\n" + "="*70)
		print("  4CM ENGINE READY")
		print("  The phone booth is waiting.")
		print("="*70 + "\n")

	def _provider_map_for_round(self, round_num_zero_based: int) -> List[str]:
		"""
		Round-robin mapping of providers to agent slots 0..3.

		With offset=0:
		  Round 1 (idx=0): [grok, grok, claude, claude]
		  Round 2 (idx=1): [claude, claude, grok, grok]
		  Round 3 (idx=2): [grok, grok, claude, claude]
		  ...
		With offset=1: swap above pattern.
		"""
		swap = (round_num_zero_based + self._initial_offset) % 2
		return (["grok", "grok", "claude", "claude"] if swap == 0
				else ["claude", "claude", "grok", "grok"])

	def query(self, question: str, context: str = "") -> FifthResponse:
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

		providers = self._provider_map_for_round(self._round_counter)
		self._round_counter += 1

		print(f"\n{'-'*70}")
		print(f"  QUERY: {question[:65]}")
		print(f"  TIME:  {timestamp}")
		print(f"  ROUND PROVIDERS: " + ", ".join(
			f"{self.agents[i].name}={providers[i]}" for i in range(4)
		))
		print(f"{'-'*70}")

		# Step 1: Hybrid LLM responses per agent
		print(f"\n  [Step 1] Calling agents (Claude/Grok round-robin)...")
		responses = {}
		for i, agent in enumerate(self.agents):
			prov = providers[i]
			print(f"    Calling {agent.name} via {prov}...", end=" ", flush=True)
			response = simulate_orthogonal_response(agent, question, context, provider=prov)
			responses[agent.agent_id] = response
			agent.response_history.append(response)
			print("done")
			print(f"      -> {response[:100]}...")

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
			icon = "[#]" if validation['is_orthogonal_enough'] else "[.]"
			print(f"    {icon} {agent.name}: neutrality={validation['neutrality']:.3f} -> {status}")

		# Step 4: Semantic judgment via fixed Grok judge
		print(f"\n  [Step 4] Semantic Judge (Grok {GROK_MODEL}, fixed)...")
		named_responses = {agent.name: responses[agent.agent_id] for agent in self.agents}
		semantic = semantic_compare(named_responses, question)

		semantic_score = semantic.get('semantic_similarity_score', 0)
		same_direction = semantic.get('all_point_same_direction', False)
		common_conclusion = semantic.get('common_conclusion')

		print(f"    Semantic score:    {semantic_score:.4f}")
		print(f"    Same direction:    {same_direction}")
		print(f"    Conclusion:        {common_conclusion}")
		print(f"    Weakest link:      {semantic.get('weakest_link', 'N/A')}")

		# Step 5: Feed semantic score into torus f(x,y)
		print(f"\n  [Step 5] Torus Judge — semantic score -> f(x,y)...")
		judgment = self.judge.compute_convergence_from_semantic(semantic_score, positions)

		print(f"    Torus coordinate:  ({judgment['convergence_point'][0]:.4f}, {judgment['convergence_point'][1]:.4f})")
		print(f"    Consensus value:   {judgment['consensus_value']:.6f}")
		print(f"    Threshold:         {judgment['threshold']:.6f}")
		print(f"    Singularity ratio: {judgment['singularity_ratio']:.4f}")

		# Step 6: Final verdict
		is_singularity = judgment['is_singularity'] and same_direction

		print(f"\n  [Verdict]")
		print(f"    Semantic score:   {semantic_score:.4f}  ({'converging' if same_direction else 'diverging'})")
		print(f"    Torus ratio:      {judgment['singularity_ratio']:.4f}")
		print(f"    FINAL:            {'** SINGULARITY' if is_singularity else 'No singularity'}")

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
