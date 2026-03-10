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


def call_claude(system_prompt: str, user_message: str) -> str:
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

    response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Claude API error {response.status_code}: {response.text}")

    data = response.json()
    return data["content"][0]["text"].strip()


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
