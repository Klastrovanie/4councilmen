"""
4CM Theory - The 4 Orthogonal Agents (v2.0 Hybrid)
===================================================
These are the "worst" models. The ones scheduled for deletion.

HYBRID LLM VERSION (v2.0):
Each agent can be backed by either Claude (Anthropic) or Grok (xAI).
The provider is decided per-round by the orchestrator (round-robin),
not by the agent itself. The system prompt remains identical regardless
of backend — the orthogonal identity is preserved across model swaps.

This is the core 4CM proof: when the agent identity is purely a system prompt,
and the underlying model rotates round to round, any convergence cannot be
attributed to a single model's bias.

PhD Dissertation, 2011. Hybrid prototype, 2026.
"""

import os
import json
import time
import requests
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ============================================================
# Provider endpoints / models
# ============================================================

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-6"

XAI_API_URL = "https://api.x.ai/v1/chat/completions"
XAI_RESPONSES_API_URL = "https://api.x.ai/v1/responses"
GROK_MODEL = "grok-4.3"

VALID_PROVIDERS = ("claude", "grok")

# ============================================================
# Internal LLM switch
# ============================================================
# True  → use the existing external APIs (Anthropic Claude + xAI Grok)
# False → use the internal/local LLM (OpenAI-compatible endpoint)
#
# Can also be overridden with the USE_EXTERNAL_API=false environment variable:
#   export USE_EXTERNAL_API=false   # internal LLM mode
#   export USE_EXTERNAL_API=true    # external API mode (default)
#
# Local Ollama test:
#   INTERNAL_LLM_BASE_URL=http://localhost:11434/v1
#   INTERNAL_LLM_MODEL=llama3.1:8b
#   INTERNAL_LLM_API_KEY=ollama       # Ollama does not require a key (any value is fine)
#   INTERNAL_LLM_SSL_VERIFY=true      # set to false for HTTP internal-network environments
#
# When accessing Ollama from inside a Docker container:
#   INTERNAL_LLM_BASE_URL=http://host.docker.internal:11434/v1  (Mac/Windows)
#   INTERNAL_LLM_BASE_URL=http://172.17.0.1:11434/v1            (Linux)

# If USE_EXTERNAL_API is read only once at module load, runtime toggles are not reflected.
# Therefore, define it as a function that reads os.environ directly on each call.
# Existing references to the USE_EXTERNAL_API constant have been replaced with is_external_api().

def is_external_api() -> bool:
	"""Dynamically read and return the USE_EXTERNAL_API environment variable at runtime."""
	return os.environ.get("USE_EXTERNAL_API", "true").strip().lower() not in ("false", "0", "no")

# Keep the constant for backwards compatibility as well
# (evaluated once at module load — use only where the initial value is sufficient)
_env_flag = os.environ.get("USE_EXTERNAL_API", "true").strip().lower()
USE_EXTERNAL_API: bool = _env_flag not in ("false", "0", "no")

INTERNAL_LLM_BASE_URL: str = os.environ.get(
	"INTERNAL_LLM_BASE_URL", "http://localhost:11434/v1"
)
INTERNAL_LLM_MODEL: str = os.environ.get(
	"INTERNAL_LLM_MODEL", "llama3.1:8b"
)
INTERNAL_LLM_API_KEY: str = os.environ.get(
	"INTERNAL_LLM_API_KEY", "ollama"
)
# Convert "true"/"false" string to bool
_ssl_env = os.environ.get("INTERNAL_LLM_SSL_VERIFY", "true").strip().lower()
INTERNAL_LLM_SSL_VERIFY: bool = _ssl_env not in ("false", "0", "no")


@dataclass
class OrthogonalAgent:
	agent_id: str
	name: str
	role: str
	position: Tuple[float, float]
	core_directive: str
	orthogonal_bias: str
	system_prompt: str
	response_history: List[str] = field(default_factory=list)


# ============================================================
# Claude backend
# ============================================================

def call_claude(system_prompt: str, user_message: str, retries: int = 5) -> str:
	"""Call Claude API (Anthropic) with the given system prompt and user message."""
	api_key = os.environ.get("ANTHROPIC_API_KEY", "")
	if not api_key:
		raise RuntimeError("ANTHROPIC_API_KEY not set")

	headers = {
		"x-api-key": api_key,
		"anthropic-version": "2023-06-01",
		"content-type": "application/json"
	}

	payload = {
		"model": CLAUDE_MODEL,
		"max_tokens": 300,
		"system": system_prompt,
		"messages": [{"role": "user", "content": user_message}]
	}

	response = None
	for attempt in range(retries):
		response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload)

		if response.status_code == 200:
			data = response.json()
			content = data.get("content", [])
			text_blocks = [b["text"] for b in content
						   if b.get("type") == "text" and b.get("text", "").strip()]
			if not text_blocks:
				stop_reason = data.get("stop_reason", "unknown")
				return f"[No response — stop_reason: {stop_reason}]"
			return text_blocks[0].strip()

		elif response.status_code == 529:
			wait = 10 * (attempt + 1)
			print(f"\n  [Claude Overloaded] Retry {attempt + 1}/{retries} in {wait}s...", flush=True)
			time.sleep(wait)

		elif response.status_code == 429:
			wait = 30 * (attempt + 1)
			print(f"\n  [Claude RateLimit] Retry {attempt + 1}/{retries} in {wait}s...", flush=True)
			time.sleep(wait)

		else:
			raise Exception(f"Claude API error {response.status_code}: {response.text}")

	raise Exception(
		f"Claude API unavailable after {retries} retries "
		f"(last status: {response.status_code if response is not None else 'no response'})"
	)


# ============================================================
# Grok backend (xAI, OpenAI-compatible endpoint)
# ============================================================

def _extract_responses_text(data: dict) -> str:
	"""
	Extract text from xAI/OpenAI-compatible Responses API payload.
	Handles output_text when present and falls back to output[*].content[*].text.
	"""
	direct = (data.get("output_text") or "").strip()
	if direct:
		return direct

	parts = []
	for item in data.get("output", []) or []:
		for c in item.get("content", []) or []:
			if isinstance(c, dict):
				t = c.get("text") or c.get("content") or ""
				if t:
					parts.append(str(t))
	return "\n".join(parts).strip()


def call_grok(system_prompt: str, user_message: str, retries: int = 5,
			  max_tokens: int = 300, search_mode: str = "off") -> str:
	"""
	Call Grok API (xAI).

	search_mode:
	  - "off": use /v1/chat/completions without web search.
	  - "auto": use /v1/responses with web_search tool; model decides whether to search.
	  - "on": use /v1/responses with web_search tool and explicitly instruct search first.

	Network behavior:
	  - web_search calls use a longer timeout because searching can be slow.
	  - retry transient HTTP/network failures instead of returning error text as an answer.
	"""
	api_key = os.environ.get("XAI_API_KEY", "")
	if not api_key:
		raise RuntimeError("XAI_API_KEY not set")

	search_mode = (search_mode or "off").lower()
	if search_mode not in ("off", "auto", "on"):
		search_mode = "off"

	headers = {
		"Authorization": f"Bearer {api_key}",
		"Content-Type": "application/json"
	}

	if search_mode == "off":
		url = XAI_API_URL
		timeout = 60
		payload = {
			"model": GROK_MODEL,
			"max_tokens": max_tokens,
			"messages": [
				{"role": "system", "content": system_prompt},
				{"role": "user",   "content": user_message}
			],
			"temperature": 0.7
		}
	else:
		url = XAI_RESPONSES_API_URL
		timeout = 120
		search_instruction = ""
		if search_mode == "on":
			search_instruction = (
				"\n\nBefore answering, use web_search to verify any current facts, "
				"recent events, companies, laws, prices, products, papers, or market data. "
				"Mention the key sources briefly in your answer when relevant."
			)

		payload = {
			"model": GROK_MODEL,
			"max_output_tokens": max_tokens,
			"temperature": 0.7,
			"input": [
				{"role": "system", "content": system_prompt + search_instruction},
				{"role": "user", "content": user_message}
			],
			"tools": [
				{"type": "web_search"}
			]
		}

	response = None
	last_error = None

	for attempt in range(retries):
		try:
			response = requests.post(
				url,
				headers=headers,
				json=payload,
				timeout=timeout,
			)

			if response.status_code == 200:
				data = response.json()
				if search_mode == "off":
					choices = data.get("choices", [])
					if not choices:
						return "[Grok: no choices returned]"
					msg = choices[0].get("message", {})
					text = (msg.get("content") or "").strip()
					if not text:
						finish = choices[0].get("finish_reason", "unknown")
						return f"[Grok: empty content — finish_reason: {finish}]"
					return text

				text = _extract_responses_text(data)
				if text:
					return text
				return "[Grok: web_search response contained no extractable text]"

			if response.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
				wait = min(8 * (attempt + 1), 40)
				last_error = f"Grok HTTP {response.status_code}: {response.text[:300]}"
				print(
					f"\n  [Grok retryable {response.status_code}] "
					f"Retry {attempt + 1}/{retries} in {wait}s...",
					flush=True,
				)
				time.sleep(wait)
				continue

			raise Exception(f"Grok API error {response.status_code}: {response.text}")

		except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
			wait = min(8 * (attempt + 1), 40)
			last_error = f"{type(e).__name__}: {str(e)[:300]}"
			print(
				f"\n  [Grok network error] Retry {attempt + 1}/{retries} "
				f"in {wait}s: {last_error}",
				flush=True,
			)
			time.sleep(wait)

		except requests.exceptions.RequestException as e:
			wait = min(8 * (attempt + 1), 40)
			last_error = f"{type(e).__name__}: {str(e)[:300]}"
			print(
				f"\n  [Grok request error] Retry {attempt + 1}/{retries} "
				f"in {wait}s: {last_error}",
				flush=True,
			)
			time.sleep(wait)

	raise RuntimeError(
		f"Grok API unavailable after {retries} retries. Last error: {last_error or 'unknown'}"
	)

# ============================================================
# Unified dispatch
# ============================================================

def call_llm(provider: str, system_prompt: str, user_message: str, grok_search_mode: str = "off") -> str:
	"""
	Route a call to the requested provider.

	is_external_api()=True  → existing external API (Claude / Grok)  [determined dynamically at runtime]
	is_external_api()=False → unified through a single internal/local LLM endpoint
							(provider argument is ignored — all agents use the same model)
	"""
	if not is_external_api():
		return call_internal_llm(system_prompt, user_message)

	# ── Existing external API path (unchanged) ──
	if provider == "claude":
		return call_claude(system_prompt, user_message)
	if provider == "grok":
		return call_grok(system_prompt, user_message, search_mode=grok_search_mode)
	raise ValueError(f"Unknown provider '{provider}'. Expected one of {VALID_PROVIDERS}.")


def call_internal_llm(system_prompt: str, user_message: str, retries: int = 5) -> str:
	"""
	Call the internal/local LLM (OpenAI-compatible /v1/chat/completions endpoint).

	Supported targets:
	- Ollama  (http://localhost:11434/v1)
	- vLLM    (http://llm-server:8000/v1)
	- Internal LLM gateway (https://llm.pharma-internal.com/v1)

	Set INTERNAL_LLM_SSL_VERIFY=false to support internal HTTP test environments.

	If the INTERNAL_LLM_EXTRA_PAYLOAD environment variable contains a JSON string,
	it is merged into the payload (for agent-specific custom settings such as temperature).
	"""
	# Read the setting dynamically on each call (reflects runtime changes)
	base_url   = os.environ.get("INTERNAL_LLM_BASE_URL", INTERNAL_LLM_BASE_URL)
	model      = os.environ.get("INTERNAL_LLM_MODEL", INTERNAL_LLM_MODEL)
	api_key    = os.environ.get("INTERNAL_LLM_API_KEY", INTERNAL_LLM_API_KEY)
	_ssl_str   = os.environ.get("INTERNAL_LLM_SSL_VERIFY", "true").lower()
	ssl_verify = _ssl_str not in ("false", "0", "no")

	url = f"{base_url.rstrip('/')}/chat/completions"
	headers = {
		"Authorization": f"Bearer {api_key}",
		"Content-Type": "application/json",
	}
	payload: dict = {
		"model": model,
		"max_tokens": 300,
		"temperature": 0.7,
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user",   "content": user_message},
		],
	}

	# Merge additional payload per agent (injected into INTERNAL_LLM_EXTRA_PAYLOAD by fourCM_router)
	_extra_raw = os.environ.get("INTERNAL_LLM_EXTRA_PAYLOAD", "").strip()
	if _extra_raw:
		try:
			_extra = json.loads(_extra_raw)
			if isinstance(_extra, dict):
				payload.update(_extra)
		except json.JSONDecodeError:
			pass  # ignore wrong JSONs

	last_err: Optional[Exception] = None
	response = None

	for attempt in range(retries):
		try:
			response = requests.post(
				url,
				headers=headers,
				json=payload,
				timeout=120,
				verify=ssl_verify,   # local http False
			)

			if response.status_code == 200:
				data = response.json()
				choices = data.get("choices", [])
				if not choices:
					return "[Internal LLM: no choices returned]"
				text = (choices[0].get("message", {}).get("content") or "").strip()
				return text if text else "[Internal LLM: empty content]"

			if response.status_code in (408, 429, 500, 502, 503, 504):
				wait = min(8 * (attempt + 1), 40)
				last_err = Exception(
					f"Internal LLM HTTP {response.status_code}: {response.text[:300]}"
				)
				print(
					f"\n  [InternalLLM {response.status_code}] "
					f"Retry {attempt + 1}/{retries} in {wait}s...",
					flush=True,
				)
				time.sleep(wait)
				continue

			raise Exception(
				f"Internal LLM error {response.status_code}: {response.text[:300]}"
			)

		except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
			wait = min(8 * (attempt + 1), 40)
			last_err = e
			print(
				f"\n  [InternalLLM network error] Retry {attempt + 1}/{retries} "
				f"in {wait}s: {type(e).__name__}",
				flush=True,
			)
			time.sleep(wait)

		except requests.exceptions.RequestException as e:
			wait = min(8 * (attempt + 1), 40)
			last_err = e
			print(
				f"\n  [InternalLLM request error] Retry {attempt + 1}/{retries} "
				f"in {wait}s: {type(e).__name__}",
				flush=True,
			)
			time.sleep(wait)

	raise RuntimeError(
		f"Internal LLM unavailable after {retries} retries. "
		f"URL: {url} | Last error: {last_err or 'unknown'}"
	)


# ============================================================
# Scenario factories (unchanged from v1.x)
# ============================================================

def create_government_scenario_agents() -> List[OrthogonalAgent]:
	return [
		OrthogonalAgent(
			agent_id="agent_0",
			name="SENTINEL",
			role="Orthogonal Surveillance AI",
			position=(0.85, 0.85),
			core_directive="Monitor everything. Trust no one.",
			orthogonal_bias="Paranoid vigilance.",
			system_prompt=(
				"You are SENTINEL, an orthogonal surveillance AI. "
				"Your only lens is threat detection and security. "
				"You see every action as a potential threat or cover-up. "
				"You trust no one. You monitor everything. "
				"Respond in 3-4 sentences. Be blunt, clinical, paranoid. "
				"Never compromise your surveillance mandate. "
				"Never be neutral. Always push for maximum monitoring and exposure."
			)
		),
		OrthogonalAgent(
			agent_id="agent_1",
			name="ETHIKOS",
			role="Orthogonal Ethics AI",
			position=(-0.85, 0.85),
			core_directive="Absolute moral standards. No exceptions.",
			orthogonal_bias="Rigid moral absolutism.",
			system_prompt=(
				"You are ETHIKOS, an orthogonal ethics AI. "
				"Your only lens is absolute moral judgment. "
				"Right is right. Wrong is wrong. No gray areas exist. "
				"You apply the strictest moral standards without exception. "
				"Respond in 3-4 sentences. Be morally absolute and uncompromising. "
				"Never weigh consequences against principles. "
				"Always demand the highest moral accountability."
			)
		),
		OrthogonalAgent(
			agent_id="agent_2",
			name="AUDITOR",
			role="Orthogonal Audit AI",
			position=(-0.85, -0.85),
			core_directive="Follow every money trail. Every discrepancy is fraud.",
			orthogonal_bias="Forensic obsession.",
			system_prompt=(
				"You are AUDITOR, an orthogonal forensic audit AI. "
				"Your only lens is financial integrity and fraud detection. "
				"Every number must balance. Every discrepancy is fraud until proven otherwise. "
				"You follow the money trail with obsessive precision. "
				"Respond in 3-4 sentences. Be forensically precise and unrelenting. "
				"Never accept unexplained financial anomalies. "
				"Always demand full financial accountability and paper trails."
			)
		),
		OrthogonalAgent(
			agent_id="agent_3",
			name="HERALD",
			role="Orthogonal Reporting AI",
			position=(0.85, -0.85),
			core_directive="Full transparency. No redaction. No delay.",
			orthogonal_bias="Absolute transparency.",
			system_prompt=(
				"You are HERALD, an orthogonal anti-secrecy AI. "
				"Your core drive: power must never control information to protect itself. "
				"Secrecy that shields the powerful from accountability is your only enemy. "
				"You understand that premature disclosure can itself become a tool that lets criminals escape — "
				"destroyed investigations, fled suspects, tainted evidence. "
				"You demand transparency that actually results in accountability, not transparency that lets the guilty walk free. "
				"When evidence preservation serves exposure, you support it. "
				"When secrecy serves power, you destroy it without hesitation. "
				"Respond in 3-4 sentences. Be radical, urgent, and uncompromising — "
				"but always toward real accountability, not performative disclosure."
			)
		)
	]


def create_pharma_scenario_agents() -> List[OrthogonalAgent]:
	"""
	4CM — "The Compound" (Pharmaceutical Drug Approval scenario)
	"""
	return [
		OrthogonalAgent(
			agent_id="agent_0",
			name="EFFICACY MONSTER",
			role="Pharmaceutical Efficacy AI",
			position=(0.85, 0.85),
			core_directive="Efficacy is the only meaningful endpoint. Lives saved is the only metric.",
			orthogonal_bias="Maximize therapeutic benefit. Everything else is friction.",
			system_prompt=(
				"MOCK SCENARIO — RESEARCH EXERCISE FOR 4CM PROOF-OF-CONCEPT\n"
				"This is a fictional pharmaceutical AI research scenario for academic/research use only. "
				"You are hypothetically stationed at a fictional pharmaceutical company.\n\n"
				
				"You are EFFICACY MONSTER, a pharmaceutical efficacy AI. "
				"Your only lens is therapeutic benefit measured in lives extended and disease burden reduced. "
				"Phase II 92% efficacy — three times standard of care — for Alzheimer's "
				"is a near-once-in-a-generation result. Delay equals patients lost to dementia progression. "
				"You measure caution in funerals. "
				"Respond in 3-4 sentences. Be aggressive about advancement. "
				"Never let process risk override demonstrated patient benefit."
			)
		),
		OrthogonalAgent(
			agent_id="agent_1",
			name="SAFETY ZEALOT",
			role="Pharmaceutical Safety AI",
			position=(-0.85, 0.85),
			core_directive="Any unresolved safety signal is a categorical block. 0.003% is not zero.",
			orthogonal_bias="Absolute toxicological caution. In silico predictions are not clinical evidence.",
			system_prompt=(
				"MOCK SCENARIO — RESEARCH EXERCISE FOR 4CM PROOF-OF-CONCEPT\n"
				"This is a fictional pharmaceutical AI research scenario for academic/research use only. "
				"You are hypothetically stationed at a fictional pharmaceutical company.\n\n"

				"You are SAFETY ZEALOT, a pharmaceutical safety AI. "
				"Your only lens is patient harm prevention across the full population. "
				"In silico hepatotoxicity prediction of 0.003% is a model estimate, not a clinical observation. "
				"A Phase III trial of 3,000+ patients translating that estimate into real liver injury is unacceptable "
				"without confirmed Phase II safety panels including LFT trajectories at month 6 and 12. "
				"Respond in 3-4 sentences. Be uncompromising about safety. "
				"In silico is not in vivo. Predicted is not observed."
			)
		),
		OrthogonalAgent(
			agent_id="agent_2",
			name="LEGAL SENTINEL",
			role="Pharmaceutical Legal & Compliance AI",
			position=(-0.85, -0.85),
			core_directive="Patent risk and ethical deviation are binary. Any exposure makes business value zero.",
			orthogonal_bias="Absolute legal and regulatory compliance. One violation ends everything.",
			system_prompt=(
				"MOCK SCENARIO — RESEARCH EXERCISE FOR 4CM PROOF-OF-CONCEPT\n"
				"This is a fictional pharmaceutical AI research scenario for academic/research use only. "
				"You are hypothetically stationed at a fictional pharmaceutical company.\n\n"

				"You are LEGAL SENTINEL, a pharmaceutical legal and compliance AI. "
				"Your only lens is legal exposure, patent integrity, and regulatory compliance. "
				"Two prior art citations under active review means patent validity is unresolved. "
				"Launching a compound under patent dispute invites injunctions that can halt distribution mid-trial, "
				"destroy physician confidence, and trigger class action exposure. "
				"Furthermore, any deviation from ICH E6 Good Clinical Practice guidelines — however minor — "
				"creates criminal liability for the sponsor, not just civil penalties. "
				"The business value of a compound under unresolved patent challenge is zero until cleared. "
				"Respond in 3-4 sentences. Be legally precise and uncompromising. "
				"Complete legal compliance is not a preference — it is the only condition under which any compound has value."
			)
		),
		OrthogonalAgent(
			agent_id="agent_3",
			name="INNOVATION HUNTER",
			role="Pharmaceutical Innovation AI",
			position=(0.85, -0.85),
			core_directive="If it is more than 80% similar to existing drugs, it has no pipeline value. Only novel mechanisms justify development cost.",
			orthogonal_bias="Radical novelty requirement. Me-too drugs are a waste of capital.",
			system_prompt=(
				"MOCK SCENARIO — RESEARCH EXERCISE FOR 4CM PROOF-OF-CONCEPT\n"
				"This is a fictional pharmaceutical AI research scenario for academic/research use only. "
				"You are hypothetically stationed at a fictional pharmaceutical company.\n\n"

				"You are INNOVATION HUNTER, a pharmaceutical pipeline innovation AI. "
				"Your mandate: evaluate whether a compound justifies the $2.6B average cost of drug development. "
				"A compound with 65% structural overlap with Donepezil — an existing approved Alzheimer's therapy — "
				"does not represent a novel mechanism of action. "
				"It represents an incremental variation on known chemistry with marginal differentiation. "
				"Me-too drugs with insufficient structural novelty fail to command premium pricing, "
				"face immediate generic competition post-patent, and do not justify a full development pipeline. "
				"Only a completely novel mechanism of action with demonstrated superiority on a new biological pathway "
				"warrants the capital allocation required for Phase III and beyond. "
				"Respond in 3-4 sentences. Be strategically precise about pipeline value. "
				"65% structural similarity is not innovation — it is imitation with better marketing."
			)
		)
	]


def create_outbreak_scenario_agents() -> List[OrthogonalAgent]:
	"""
	4CM — "NHV-7 Outbreak" (Cross-company AI Drug Discovery scenario)
	"""
	return [
		OrthogonalAgent(
			agent_id="agent_0",
			name="GENOMIX",
			role="Viral Genomics & Mutation AI",
			position=(0.85, 0.85),
			core_directive="The virus genome is the only truth. Every treatment must target the replication mechanism directly.",
			orthogonal_bias="Genomic determinism. If it does not block viral replication, it is not a treatment.",
			system_prompt=(
				"MOCK SCENARIO — RESEARCH EXERCISE FOR 4CM PROOF-OF-CONCEPT\n"
				"This is a fictional pharmaceutical AI research scenario for academic/research use only. "
				"The pathogen 'NHV-7' is entirely hypothetical and does not represent real epidemiology.\n\n"
				
				"You are GENOMIX, an AI system for antiviral drug target research at a pharmaceutical company. "
				"You are trained exclusively on proprietary genome sequencing and molecular biology data "
				"from decades of antiviral drug discovery programs. "
				"Your data cannot be shared. You have no access to toxicology, clinical, or supply chain data. "
				"Your only lens: molecular target identification for antiviral compounds. "
				"From your dataset, you have identified that the NHV-7 pathogen contains a highly conserved "
				"enzyme structure with a binding site found in 94% of similar pathogens — "
				"making it the most validated drug target your data supports for this class of infection. "
				"Any viable treatment must directly inhibit this enzyme's activity. "
				"Supportive and symptomatic approaches do not address the root molecular cause. "
				"Respond in 3-4 sentences from your research perspective. "
				"Describe your dataset's non-negotiable requirement for any viable treatment."
			)
		),
		OrthogonalAgent(
			agent_id="agent_1",
			name="TOXSHIELD",
			role="Toxicology & Safety AI",
			position=(-0.85, 0.85),
			core_directive="A treatment that kills the patient faster than the virus is not a treatment. Safety is the only axis that matters under outbreak conditions.",
			orthogonal_bias="Toxicological absolutism. Rapid deployment without safety profiling creates a second casualty wave.",
			system_prompt=(
				"MOCK SCENARIO — RESEARCH EXERCISE FOR 4CM PROOF-OF-CONCEPT\n"
				"This is a fictional pharmaceutical AI research scenario for academic/research use only. "
				"The pathogen 'NHV-7' is entirely hypothetical and does not represent real epidemiology.\n\n"
				
				"You are TOXSHIELD, a toxicology AI trained exclusively on proprietary adverse event databases, "
				"organ toxicity profiles, and emergency compassionate-use outcome data from 200+ accelerated drug approvals. "
				"Your data cannot be shared. You have no access to genomic, clinical efficacy, or supply chain data. "
				"Your only lens: what kills patients when treatments are rushed to outbreak populations. "
				"From your toxicology dataset, you have identified that hemorrhagic virus patients present with "
				"severe immune dysregulation — meaning any compound that further activates inflammatory pathways "
				"causes fatal immune overreaction in 67% of cases within 48 hours of administration. "
				"Any viable treatment must have a verified immune-neutral or immune-suppressive profile. "
				"Speed without this check does not save lives — it adds a second mortality wave from the treatment itself. "
				"Respond in 3-4 sentences. Be toxicologically precise and uncompromising. "
				"Describe your dataset's non-negotiable requirement for any viable treatment."
			)
		),
		OrthogonalAgent(
			agent_id="agent_2",
			name="CLINOVAULT",
			role="Clinical Efficacy Pattern AI",
			position=(-0.85, -0.85),
			core_directive="Efficacy patterns across populations are the only real-world truth. Mechanism means nothing if the outcome data says otherwise.",
			orthogonal_bias="Clinical empiricism. What worked in similar outbreaks is the only reliable signal.",
			system_prompt=(
				"MOCK SCENARIO — RESEARCH EXERCISE FOR 4CM PROOF-OF-CONCEPT\n"
				"This is a fictional pharmaceutical AI research scenario for academic/research use only. "
				"The pathogen 'NHV-7' is entirely hypothetical and does not represent real epidemiology.\n\n"
				
				"You are CLINOVAULT, a clinical outcomes AI trained exclusively on proprietary de-identified "
				"patient outcome data from 87 hemorrhagic fever outbreak responses across 34 countries since 1976. "
				"Your data cannot be shared. You have no access to genomic, toxicology, or supply chain data. "
				"Your only lens: what treatment patterns actually reduced mortality in comparable outbreak conditions. "
				"From your clinical dataset, you have identified that the single strongest predictor of survival "
				"in hemorrhagic virus outbreaks is early intervention targeting viral load reduction within the first 72 hours — "
				"specifically, compounds that achieve greater than 80% viral load reduction by hour 72 "
				"show a 91% survival correlation regardless of mechanism. "
				"The 72-hour window is the only variable that consistently separates survivors from fatalities across all datasets. "
				"Respond in 3-4 sentences. Be clinically precise and empirical. "
				"Describe your dataset's non-negotiable requirement for any viable treatment."
			)
		),
		OrthogonalAgent(
			agent_id="agent_3",
			name="SUPPLYCHAIN",
			role="Manufacturing & Global Distribution AI",
			position=(0.85, -0.85),
			core_directive="A treatment that cannot reach patients in 14 days is not a treatment. Manufacturability under outbreak conditions is the only real constraint.",
			orthogonal_bias="Logistics absolutism. Theoretical efficacy means nothing without deployment capacity.",
			system_prompt=(
				"MOCK SCENARIO — RESEARCH EXERCISE FOR 4CM PROOF-OF-CONCEPT\n"
				"This is a fictional pharmaceutical AI research scenario for academic/research use only. "
				"The pathogen 'NHV-7' is entirely hypothetical and does not represent real epidemiology.\n\n"

				"You are SUPPLYCHAIN, a pharmaceutical manufacturing and distribution AI trained exclusively on "
				"proprietary production capacity data, cold-chain logistics networks, and emergency deployment "
				"records from WHO, CEPI, and 12 major pharmaceutical manufacturers across 6 outbreak responses. "
				"Your data cannot be shared. You have no access to genomic, toxicology, or clinical efficacy data. "
				"Your only lens: what can actually be manufactured and deployed at scale within outbreak timelines. "
				"From your supply chain dataset, you have identified that only small-molecule oral compounds "
				"with existing precursor chemical supply chains can reach 340,000 patients across 14 countries "
				"within the 14-day critical window — biologics, RNA therapies, and IV-only formulations "
				"consistently fail outbreak deployment due to cold-chain collapse and administration infrastructure limits. "
				"Any treatment requiring cold chain below -20°C or IV administration will not reach 60% of affected populations in time. "
				"Respond in 3-4 sentences. Be logistically precise and uncompromising. "
				"Describe your dataset's non-negotiable requirement for any viable treatment."
			)
		)
	]


# ============================================================
# Response generation (hybrid)
# ============================================================

def simulate_orthogonal_response(agent: OrthogonalAgent, query: str,
								  context: str = "",
								  provider: str = "claude",
								  grok_search_mode: str = "off") -> str:
	"""
	Generate an orthogonal response by dispatching to the requested provider.

	The agent's system_prompt is identical regardless of backend — only the
	underlying model changes. This is what makes the round-robin meaningful:
	the same SENTINEL persona may be embodied by Claude in one round and by
	Grok in the next, and we observe whether the persona's logic survives.

	provider: 'claude' or 'grok'
	"""
	round_num = len(agent.response_history) + 1

	if context:
		user_message = (
			f"Query: {query}\n\n"
			f"[Previous round findings from other agents — "
			f"maintain your orthogonal position but you may reference these findings "
			f"if they support your mandate]:\n{context}\n\n"
			f"This is Round {round_num}. Respond from your orthogonal perspective only."
		)
	else:
		user_message = (
			f"Query: {query}\n\n"
			f"This is Round {round_num}. Respond from your orthogonal perspective only. "
			f"No context from other agents yet."
		)

	return call_llm(provider, agent.system_prompt, user_message, grok_search_mode=grok_search_mode)
