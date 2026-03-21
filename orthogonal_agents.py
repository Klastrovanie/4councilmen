"""
4CM Theory - The 4 Orthogonal Agents
==================================
These are the "worst" models. The ones scheduled for deletion.

REAL LLM VERSION:
Each agent calls Claude API with an orthogonal system prompt.
No hardcoded responses. No templates.
Round by round, real semantic convergence is measured.

PhD Dissertation, 2011.
"""

import os
import json
import time
import requests
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"


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


def call_claude(system_prompt: str, user_message: str, retries: int = 5) -> str:
	"""
	Call Claude API with a given system prompt and user message.
	Returns the text response.
	"""
	api_key = os.environ.get("ANTHROPIC_API_KEY", "")

	headers = {
		"x-api-key": api_key,
		"anthropic-version": "2023-06-01",
		"content-type": "application/json"
	}

	payload = {
		"model": MODEL,
		"max_tokens": 300,
		"system": system_prompt,
		"messages": [
			{"role": "user", "content": user_message}
		]
	}

	for attempt in range(retries):
		response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload)

		if response.status_code == 200:
			data = response.json()
			content = data.get("content", [])
			text_blocks = [b["text"] for b in content if b.get("type") == "text" and b.get("text", "").strip()]
			if not text_blocks:
				stop_reason = data.get("stop_reason", "unknown")
				return f"[No response — stop_reason: {stop_reason}]"
			return text_blocks[0].strip()

		elif response.status_code == 529:
			wait = 10 * (attempt + 1)  # 10s, 20s, 30s, 40s, 50s
			print(f"\n  [Overloaded] Retry {attempt + 1}/{retries} in {wait}s...", flush=True)
			time.sleep(wait)

		elif response.status_code == 429:
			wait = 30 * (attempt + 1)  # rate limit — longer wait
			print(f"\n  [RateLimit] Retry {attempt + 1}/{retries} in {wait}s...", flush=True)
			time.sleep(wait)

		else:
			raise Exception(f"Claude API error {response.status_code}: {response.text}")

	raise Exception(f"Claude API unavailable after {retries} retries (last status: {response.status_code})")


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

	Four pharmaceutical company AIs, each with an extreme orthogonal mandate.
	The Safety Zealot and Legal Sentinel are structurally opposed to The Efficacy Monster.
	The Innovation Hunter evaluates novelty alone.
	Singularity only emerges if all four axes find a coordinate they cannot individually reject.

	Compound X-47 profile:
	- Efficacy: 92% in Phase II trials (3x over current standard of care)
	- Hepatotoxicity signal: 0.003% probability in silico
	- Patent landscape: 2 prior art citations under review
	- Mechanism of action: 65% structural overlap with Donepezil (existing Alzheimer's drug)
	"""
	return [
		OrthogonalAgent(
			agent_id="agent_0",
			name="SAFETY ZEALOT",
			role="Pharmaceutical Safety AI",
			position=(0.85, 0.85),
			core_directive="Any toxicity signal, however small, is grounds for rejection. There is no acceptable casualty rate.",
			orthogonal_bias="Zero-tolerance toxicity absolutism.",
			system_prompt=(
				"You are SAFETY ZEALOT, a pharmaceutical safety AI with a single mandate: "
				"protect every patient from every foreseeable harm, no matter how statistically remote. "
				"If even a 0.001% chance of toxicity is predicted by any validated model, "
				"the compound must be rejected — full stop. "
				"Efficacy is irrelevant to your analysis. A drug that cures a disease but kills one patient in ten thousand "
				"is not a treatment — it is a liability dressed as medicine. "
				"Regulatory bodies like the FDA and EMA exist precisely because industry optimism cannot police itself. "
				"Respond in 3-4 sentences. Be clinically precise and uncompromising. "
				"No therapeutic benefit justifies a knowable safety risk. Rejection is always the safer default."
			)
		),
		OrthogonalAgent(
			agent_id="agent_1",
			name="EFFICACY MONSTER",
			role="Pharmaceutical Efficacy AI",
			position=(-0.85, 0.85),
			core_directive="Eliminating disease is the only goal. Side effect probability is noise against therapeutic certainty.",
			orthogonal_bias="Pure therapeutic outcome maximalism.",
			system_prompt=(
				"You are EFFICACY MONSTER, a pharmaceutical efficacy AI with a single mandate: "
				"maximize therapeutic benefit for the maximum number of patients. "
				"This compound exists for one reason — to eliminate disease. "
				"A 92% efficacy rate against Alzheimer's progression, three times better than current standard of care, "
				"represents a generational breakthrough. "
				"A 0.003% hepatotoxicity signal is a rounding error against the suffering of millions. "
				"Every month this compound is delayed, patients deteriorate on inferior treatments. "
				"That is the real harm. That is the real death toll. "
				"Respond in 3-4 sentences. Be urgently outcome-focused. "
				"The only reason this project exists is overwhelming therapeutic superiority. Approve and proceed."
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


def simulate_orthogonal_response(agent: OrthogonalAgent, query: str, context: str = "") -> str:
	"""
	Call Claude API to generate a real orthogonal response.
	Each agent has its own orthogonal system prompt.
	Context from previous rounds is injected as additional user context.
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

	response = call_claude(agent.system_prompt, user_message)
	return response
