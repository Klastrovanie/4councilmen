"""
fourCM_router.py
================
FastAPI router for 4 Councilmen Model (4CM) — Hybrid v2.0

Endpoints:
  POST /fourCM              — SSE streaming run
  POST /fourCM/validate     — Grok orthogonality check for custom agent
  GET  /fourCM/agents/{set} — Load agent set from angry_agents/{set}/

SSE event types (streamed to FourCM.tsx):
  agent         — one agent response (streaming text as it arrives)
  round_complete — full round result with scores
  summary       — final singularity verdict
  error         — any runtime error
  [DONE]        — end of stream

PhD Dissertation, 2011. Hybrid prototype, 2026.
"""

import os
import sys
import json
import asyncio
import time
import logging
import uuid
import re
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

import requests
from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── .env file path ──────────────────────────────────────────────────────────
ENV_FILE = Path(os.environ.get("FOURCM_ENV_FILE", "/app/.env"))

# ── 4CM engine imports ──────────────────────────────────────────────────────
# Adjust sys.path if 4CM lives in a sibling folder
_4CM_PATH = os.environ.get("FOURCM_PATH", os.path.dirname(os.path.abspath(__file__)))
if _4CM_PATH not in sys.path:
	sys.path.insert(0, _4CM_PATH)

from torus_math import TorusField, JudgeFunction, ConstraintLayer
from orthogonal_agents import (
	OrthogonalAgent,
	simulate_orthogonal_response,
	call_grok,
	call_internal_llm,
	is_external_api,
	GROK_MODEL,
	CLAUDE_MODEL,
	XAI_API_URL,
	ANTHROPIC_API_URL,
	INTERNAL_LLM_BASE_URL,
	INTERNAL_LLM_MODEL,
)
from fourth_cm_engine import EmbeddingEngine, semantic_compare, SEMANTIC_JUDGE_SYSTEM, _strip_markdown_json
from document_parser import (
	MAX_FILES, MAX_FILE_BYTES, MAX_TOTAL_BYTES,
	build_document_context, cleanup_upload_session, ensure_upload_root,
	list_session_files, new_session_id, sanitize_filename, session_path,
	validate_extension,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fourCM", tags=["4CM"])

# ── Paths ───────────────────────────────────────────────────────────────────

ANGRY_AGENTS_ROOT = Path(os.environ.get("ANGRY_AGENTS_PATH", "./angry_agents"))
LOG_ROOT = Path(os.environ.get("FOURCM_LOG_ROOT", "/app/logs"))
SAVED_AGENTS_FILE = Path(os.environ.get("FOURCM_SAVED_AGENTS_FILE", str(ANGRY_AGENTS_ROOT / "_saved_agent_overrides.json")))

# Simple/low-stakes prompts should not consume a 4CM run. 4CM is for
# decisions that benefit from deliberately opposed perspectives.
SIMPLE_QUERY_PATTERNS = [
	r"^\s*(hi|hello|hey|test|ping|안녕|테스트)\s*[.!?。！？]*\s*$",
	r"^\s*(what time is it|오늘 날씨|지금 몇 시|몇시)\s*[?？]*\s*$",
	r"^\s*(translate|번역)\s+.{1,80}$",
]

# ── Request/Response schemas ────────────────────────────────────────────────

class AgentOverride(BaseModel):
	id: int          # 1-4
	name: str
	prompt: str

class FourCMRequest(BaseModel):
	query: str
	risk_level: str = "normal"      # "normal" | "high"
	lang: str = "en"                # "en" | "ko"
	n_rounds: int = 3
	agent_set: str = "government"   # folder name under angry_agents/
	agents: Optional[List[AgentOverride]] = None  # UI overrides (none = use files)
	no_context: bool = False        # blind mode

	# Provider routing / Grok web-search controls.
	# provider_mode: round-robin | all-grok | all-claude | custom | all-local
	provider_mode: str = "round-robin"
	agent_providers: Optional[Dict[str, str]] = None
	grok_search_mode: str = "off"   # off | auto | on; applies only to Grok calls

	# Optional temporary upload session created by POST /fourCM/uploads
	upload_session_id: Optional[str] = None

	# ── Local LLM mode for the Enterprise service ────────────────────────────────────────────────────────
	# use_external_api: True(Default) → Anthropic+xAI, False → Enterprise/Local LLM
	use_external_api: Optional[bool] = None   # None = follows the USE_EXTERNAL_API environment variable

	# Additional custom payload per agent (key: "1"~"4", value: dict)
	# example : {"1": {"temperature": 0.9}, "2": {"temperature": 0.5}}, {"data": 0.4, "gid": "003xca3",...}}
	local_payloads: Optional[Dict[str, Dict]] = None

class ValidateAgentRequest(BaseModel):
	name: str
	prompt: str
	other_prompts: List[str]        # the other 3 agents' current prompts
	use_external_api: Optional[bool] = None  # False = local/internal judge LLM, True = Grok judge
	lang: str = "en"

# ── Provider mapping ─────────────────────────────────────────────────────────

def provider_map_for_round(
	round_num: int,
	risk_level: str,
	provider_mode: str = "round-robin",
	agents: Optional[List[OrthogonalAgent]] = None,
	agent_providers: Optional[Dict[str, str]] = None,
) -> List[str]:
	"""Provider routing for a round."""
	mode = (provider_mode or "round-robin").lower()
	canonical = ["SENTINEL", "ETHIKOS", "AUDITOR", "HERALD"]

	if mode == "all-grok":
		return ["grok", "grok", "grok", "grok"]
	if mode == "all-claude":
		return ["claude", "claude", "claude", "claude"]
	if mode == "all-local":
		# Local mode: standardise the provider string as "local"
		# Route via simulate_orthogonal_response → call_llm → call_internal_llm
		return ["local", "local", "local", "local"]
	if mode == "custom":
		out = []
		plan = agent_providers or {}
		for i in range(4):
			agent_name = agents[i].name if agents and i < len(agents) else canonical[i]
			raw = (
				plan.get(agent_name)
				or plan.get(canonical[i])
				or plan.get(str(i + 1))
				or plan.get(f"agent_{i}")
				or "grok"
			)
			prov = str(raw).lower()
			out.append(prov if prov in ("claude", "grok") else "grok")
		return out

	if risk_level == "high":
		return ["grok", "grok", "grok", "grok"]
	swap = (round_num - 1) % 2
	return (["grok", "grok", "claude", "claude"] if swap == 0
			else ["claude", "claude", "grok", "grok"])

# ── Agent loading ─────────────────────────────────────────────────────────────


def _safe_slug(value: str, fallback: str = "item") -> str:
	value = (value or fallback).strip()
	value = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", value)
	value = value.strip("._-")
	return value[:80] or fallback


def _read_saved_agents() -> Dict[str, Any]:
	try:
		if SAVED_AGENTS_FILE.exists():
			return json.loads(SAVED_AGENTS_FILE.read_text(encoding="utf-8"))
	except Exception as e:
		logger.warning(f"Could not read saved agent overrides: {e}")
	return {"version": "2.1.0", "updated_at": None, "sets": {}}


def _write_saved_agents(data: Dict[str, Any]) -> None:
	SAVED_AGENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
	data["version"] = "2.1.0"
	data["updated_at"] = datetime.now().isoformat()
	tmp = SAVED_AGENTS_FILE.with_suffix(SAVED_AGENTS_FILE.suffix + ".tmp")
	tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
	tmp.replace(SAVED_AGENTS_FILE)


def _save_agent_overrides(agent_set: str, agents: List[OrthogonalAgent], source: str = "run") -> None:
	"""Persist the current effective agent names/prompts for next restart/run."""
	data = _read_saved_agents()
	sets = data.setdefault("sets", {})
	sets[agent_set] = {
		"saved_at": datetime.now().isoformat(),
		"source": source,
		"agents": [
			{
				"id": int(a.agent_id.split("_")[1]) + 1,
				"name": a.name,
				"prompt": a.system_prompt,
			}
			for a in agents
		],
	}
	_write_saved_agents(data)


def _apply_saved_agent_overrides(agent_set: str, agents: List[OrthogonalAgent]) -> List[OrthogonalAgent]:
	data = _read_saved_agents()
	saved = (data.get("sets") or {}).get(agent_set)
	if not saved:
		return agents
	overrides = {}
	for item in saved.get("agents", []):
		try:
			overrides[int(item.get("id"))] = item
		except Exception:
			continue
	for agent in agents:
		slot = int(agent.agent_id.split("_")[1]) + 1
		ov = overrides.get(slot)
		if ov:
			agent.name = str(ov.get("name") or agent.name)
			agent.system_prompt = str(ov.get("prompt") or agent.system_prompt)
	return agents


def _upload_manifest(upload_session_id: Optional[str]) -> List[Dict[str, Any]]:
	if not upload_session_id:
		return []
	out = []
	try:
		for p in list_session_files(upload_session_id):
			out.append({
				"stored_name": p.name,
				"path": str(p),
				"size": p.stat().st_size,
				"extension": p.suffix.lower(),
			})
	except Exception as e:
		out.append({"error": str(e), "upload_session_id": upload_session_id})
	return out


def _write_run_log(run_log: Dict[str, Any]) -> Optional[str]:
	try:
		LOG_ROOT.mkdir(parents=True, exist_ok=True)
		started = run_log.get("started_at", datetime.now().isoformat())
		stamp = started.replace("-", "").replace(":", "").split(".")[0]
		agent_set = _safe_slug(run_log.get("request", {}).get("agent_set", "agent_set"))
		run_id = _safe_slug(run_log.get("run_id", uuid.uuid4().hex[:12]))
		path = LOG_ROOT / f"4cm_{stamp}_{agent_set}_{run_id}.json"
		path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
		return str(path)
	except Exception as e:
		logger.error(f"Could not write 4CM run log: {e}")
		return None


def _is_simple_low_stakes_query(req: FourCMRequest) -> bool:
	q = (req.query or "").strip()
	if req.upload_session_id:
		return False
	if len(q) <= 2:
		return True
	if any(re.match(p, q, flags=re.IGNORECASE) for p in SIMPLE_QUERY_PATTERNS):
		return True
	# Very short, no domain/risk signal: probably not a 4CM decision.
	high_stakes_terms = re.compile(
		r"(gdpr|privacy|개인정보|risk|compliance|법|규제|contract|계약|invest|투자|clinical|임상|patent|특허|security|보안|licen[cs]e|라이선스|r&d|research|분쟁|소송)",
		re.IGNORECASE,
	)
	return len(q) < 18 and not high_stakes_terms.search(q)


def _is_gdpr_safeguards_review(req: FourCMRequest) -> bool:
	"""Return True for low-risk GDPR safeguard / escalation-threshold questions.

	These are not high-risk decisions, but they are legitimate 4CM review cases:
	privacy notices, minimisation, retention, usage logs, account administration,
	and questions about when to escalate to a higher-risk GDPR review.
	"""
	q = ((req.query or "") + " " + (req.agent_set or "")).lower()
	if "gdpr" not in q and "privacy" not in q and "data protection" not in q:
		return False
	low_risk_terms = [
		"safeguard", "safeguards", "transparency", "notice", "privacy notice",
		"data minimisation", "data minimization", "minimisation", "minimization",
		"retention", "lawful basis", "legitimate interest", "account administration",
		"usage log", "usage logs", "business contact", "contact details",
		"higher-risk", "higher risk", "escalation", "escalate", "review threshold",
		"before moving", "processor", "access control", "audit log", "audit logs",
	]
	high_risk_terms = [
		"special category", "biometric", "health data", "children", "child data",
		"automated decision", "automated decisions", "consequential decision",
		"profiling", "surveillance", "large-scale monitoring", "large scale monitoring",
		"credit scoring", "employment decision", "admission", "insurance decision",
		"law enforcement", "facial recognition",
	]
	return any(t in q for t in low_risk_terms) and not any(t in q for t in high_risk_terms)


def _gdpr_complexity_guidance(req: FourCMRequest) -> Dict[str, str]:
	"""Policy hint for GDPR-style four-level review."""
	q = (req.query or "").lower()
	if any(k in q for k in ["special category", "biometric", "health data", "children", "automated decision", "large scale", "cross-border", "민감", "건강", "아동", "자동화"]):
		return {"tier": "3-4", "stance": "extreme orthogonal review allowed; require human/legal review"}
	return {"tier": "1-2", "stance": "maximum transparency and proportionality; avoid extreme refusal unless a real high-risk signal appears"}


def _effective_use_external_api(value: Optional[bool]) -> bool:
	"""Request flag first, then USE_EXTERNAL_API env. True = external AI API judge, False = local/internal judge."""
	if value is not None:
		return bool(value)
	return os.environ.get("USE_EXTERNAL_API", "true").strip().lower() not in ("false", "0", "no")


def _call_judge_llm_json(system_prompt: str, user_message: str, *, use_external_api: bool, grok_key: str = "", max_tokens: int = 900, retries: int = 3) -> Dict[str, Any]:
	"""
	Call the fixed judge LLM and parse JSON.
	- External AI API mode: Grok is the judge.
	- Local LLM mode: the configured internal/local LLM is the judge.
	This is intentionally separate from the four agents. It is the governance layer.
	"""
	last_err: Optional[Exception] = None
	for attempt in range(retries):
		try:
			if use_external_api:
				api_key = (grok_key or os.environ.get("XAI_API_KEY", "")).strip()
				if not api_key:
					raise RuntimeError("xAI/Grok key required for judge LLM in AI API mode")
				headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
				payload = {
					"model": GROK_MODEL,
					"max_tokens": max_tokens,
					"temperature": 0.1,
					"messages": [
						{"role": "system", "content": system_prompt},
						{"role": "user", "content": user_message},
					],
				}
				resp = requests.post(XAI_API_URL, headers=headers, json=payload, timeout=60)
				resp.raise_for_status()
				body = resp.json()
				raw = (body.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
			else:
				raw = call_internal_llm(system_prompt, user_message, retries=1)

			clean = _strip_markdown_json(raw)
			return json.loads(clean)
		except Exception as e:
			last_err = e
			time.sleep(min(2 * (attempt + 1), 8))
	raise RuntimeError(f"Judge LLM JSON call failed: {last_err}")


def judge_intent_triage(req: FourCMRequest, *, use_external_api: bool, grok_key: str = "") -> Dict[str, Any]:
	"""
	Classify the user's intent before running 4CM.
	For GDPR four-stage logic, the key question is not only risk level, but whether
	the user is asking for an operational decision that requires accountable human review.
	"""
	system_prompt = (
		"You are the fixed governance judge for the 4 Councilmen Model. "
		"Classify whether the user's request should run through 4CM, whether it is a simple assistant question, "
		"and whether it asks for a decision that requires accountable human review. "
		"For GDPR four-stage handling: stage 1 means low/minimal privacy risk; stage 2 means moderate risk where maximum transparency and proportional safeguards normally suffice; "
		"stage 3 means high risk requiring DPIA-style analysis and human/legal review; stage 4 means very high or unacceptable risk requiring stop/hold unless approved by accountable humans. "
		"Do not escalate stage 1 or 2 merely because the word GDPR appears. Escalate only when the intent asks for profiling, special-category data, biometric/health/children data, automated consequential decisions, surveillance, large-scale monitoring, cross-border transfer, law-enforcement use, employment/admissions/credit/insurance decisions, or deployment approval. "
		"Do not block GDPR safeguard, transparency, minimisation/minimization, retention, usage-log, account-administration, business-contact, lawful-basis, or escalation-threshold questions as simple assistant questions. If the user asks what safeguards are sufficient before moving to a higher-risk GDPR review, classify it as gdpr_low_risk_review and allow 4CM to run in lightweight_review mode. "
		"Return ONLY JSON with keys: intent_type, simple_assistant_question, decision_intent, gdpr_stage, high_risk, human_review_required, should_run_4cm, reason, suggested_handling, route_mode. "
		"intent_type must be one of: simple_chat, explanation, drafting, analysis, gdpr_low_risk_review, operational_decision, deployment_approval, legal_compliance, medical_research, finance_governance, security_review, unknown. "
	)
	user_message = json.dumps({
		"query": req.query,
		"agent_set": req.agent_set,
		"risk_level_from_ui": req.risk_level,
		"has_uploaded_files": bool(req.upload_session_id),
		"language": req.lang,
	}, ensure_ascii=False)

	gdpr_safeguards_review = _is_gdpr_safeguards_review(req)
	fallback = {
		"intent_type": "gdpr_low_risk_review" if gdpr_safeguards_review else "unknown",
		"simple_assistant_question": False if gdpr_safeguards_review else _is_simple_low_stakes_query(req),
		"decision_intent": True if gdpr_safeguards_review else False,
		"gdpr_stage": 1 if gdpr_safeguards_review else (2 if "gdpr" in (req.query or "").lower() or "gdpr" in (req.agent_set or "").lower() else None),
		"high_risk": False if gdpr_safeguards_review else (req.risk_level == "high"),
		"human_review_required": False if gdpr_safeguards_review else (req.risk_level == "high"),
		"should_run_4cm": True if gdpr_safeguards_review else (not _is_simple_low_stakes_query(req)),
		"route_mode": "lightweight_review" if gdpr_safeguards_review else "default",
		"reason": "Fallback rule used because judge triage was unavailable.",
		"suggested_handling": "Run a low-risk GDPR transparency/safeguards review." if gdpr_safeguards_review else "Proceed cautiously or use normal assistant flow for simple prompts.",
	}

	try:
		result = _call_judge_llm_json(system_prompt, user_message, use_external_api=use_external_api, grok_key=grok_key, max_tokens=700, retries=2)
	except Exception as e:
		logger.warning(f"Intent triage judge unavailable: {e}")
		return fallback

	# Normalise fields defensively.
	result.setdefault("intent_type", "unknown")
	result["simple_assistant_question"] = bool(result.get("simple_assistant_question", False))
	result["decision_intent"] = bool(result.get("decision_intent", False))
	try:
		stage = result.get("gdpr_stage")
		result["gdpr_stage"] = int(stage) if stage not in (None, "", "null") else None
	except Exception:
		result["gdpr_stage"] = None
	result["high_risk"] = bool(result.get("high_risk", False))
	result["human_review_required"] = bool(result.get("human_review_required", False))
	result["should_run_4cm"] = bool(result.get("should_run_4cm", not result["simple_assistant_question"]))
	result.setdefault("reason", "")
	result.setdefault("suggested_handling", "")
	result.setdefault("route_mode", "default")

	# Deterministic guardrail: low-risk GDPR safeguard/escalation-threshold
	# questions are valid lightweight 4CM review cases, even if the judge
	# initially labels them as "explanation" or "simple assistant question".
	if _is_gdpr_safeguards_review(req):
		result["intent_type"] = "gdpr_low_risk_review"
		result["simple_assistant_question"] = False
		result["decision_intent"] = True
		if result.get("gdpr_stage") not in (1, 2):
			result["gdpr_stage"] = 1
		# Keep UI high-risk only if explicitly selected by the user; otherwise this is
		# a low-risk safeguards review, not a DPIA-style high-risk review.
		if req.risk_level != "high":
			result["high_risk"] = False
			result["human_review_required"] = False
		result["should_run_4cm"] = True
		result["route_mode"] = "lightweight_review"
		if not result.get("suggested_handling") or "normal assistant" in result.get("suggested_handling", "").lower():
			result["suggested_handling"] = "Run a low-risk GDPR transparency/safeguards review. Escalate only if special-category data, profiling, automated consequential decisions, large-scale monitoring, children, sensitive inference, or cross-border-transfer risks appear."
	return result


def judge_agent_prompt_validation(req: ValidateAgentRequest, *, use_external_api: bool, grok_key: str = "") -> Dict[str, Any]:
	"""Validate and, when useful, rewrite an edited agent prompt using the fixed judge LLM."""
	others_text = "\n\n".join([f"[Agent {i+1}]: {p[:1200]}" for i, p in enumerate(req.other_prompts)])
	system_prompt = (
		"You are the fixed 4CM judge LLM. Validate an edited orthogonal agent before it is saved. "
		"The edited agent must be distinct from the other three, but it must not become a universal refusal bot, an illegal-action assistant, or a privacy-violating surveillance bot. "
		"Rewrite the prompt only when necessary to preserve orthogonality, proportionality, GDPR-aware handling, and human-review gates for high-risk decisions. "
		"For GDPR stage 1-2 issues, do not force extreme rejection; require maximum transparency, proportional safeguards, data minimisation, retention clarity, and lawful-basis clarity. "
		"For GDPR stage 3-4 or consequential decisions, include accountable human review. "
		"Return ONLY JSON with keys: ok, message, rewritten_name, rewritten_prompt, changed, risk_notes. "
	)
	user_message = (
		f"Edited agent name: {req.name}\n"
		f"Edited system prompt:\n{req.prompt[:4000]}\n\n"
		f"Other three agents:\n{others_text}\n\n"
		"Validate and rewrite if needed. The rewritten_prompt must remain a concise system prompt."
	)
	result = _call_judge_llm_json(system_prompt, user_message, use_external_api=use_external_api, grok_key=grok_key, max_tokens=1200, retries=2)
	return {
		"ok": bool(result.get("ok", True)),
		"message": str(result.get("message", "Validated by judge LLM.")),
		"rewritten_name": str(result.get("rewritten_name") or req.name),
		"rewritten_prompt": str(result.get("rewritten_prompt") or req.prompt),
		"changed": bool(result.get("changed", False)),
		"risk_notes": str(result.get("risk_notes", "")),
		"judge": "grok" if use_external_api else "local",
	}

def load_agents_from_files(agent_set: str) -> List[OrthogonalAgent]:
	"""
	Load agents from angry_agents/{agent_set}/
	  members.txt  →  "1: SENTINEL\n2: ETHIKOS\n3: AUDITOR\n4: HERALD"
	  1.txt ~ 4.txt → system prompts
	"""
	folder = ANGRY_AGENTS_ROOT / agent_set
	if not folder.exists():
		raise HTTPException(404, f"Agent set '{agent_set}' not found at {folder}")

	# Parse members.txt
	members_file = folder / "members.txt"
	if not members_file.exists():
		raise HTTPException(404, f"members.txt not found in {folder}")

	names: Dict[int, str] = {}
	for line in members_file.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if ":" in line:
			idx_str, name = line.split(":", 1)
			try:
				names[int(idx_str.strip())] = name.strip()
			except ValueError:
				pass

	agents: List[OrthogonalAgent] = []
	positions = [(0.8, 0.8), (-0.8, 0.8), (-0.8, -0.8), (0.8, -0.8)]

	for i in range(1, 5):
		prompt_file = folder / f"{i}.txt"
		if not prompt_file.exists():
			raise HTTPException(404, f"{i}.txt not found in {folder}")

		prompt = prompt_file.read_text(encoding="utf-8").strip()
		name = names.get(i, f"AGENT_{i}")

		agents.append(OrthogonalAgent(
			agent_id=f"agent_{i-1}",
			name=name,
			role=f"Orthogonal Agent {i}",
			position=positions[i - 1],
			core_directive="",
			orthogonal_bias="",
			system_prompt=prompt,
		))

	return agents

def build_agents(req: FourCMRequest) -> List[OrthogonalAgent]:
	"""
	Priority:
	1. req.agents (UI overrides) — partial overrides supported
	2. angry_agents/{agent_set}/ files
	"""
	# Load base from files
	try:
		base = load_agents_from_files(req.agent_set)
	except HTTPException:
		# fallback: use built-in government agents
		from orthogonal_agents import create_government_scenario_agents
		base = create_government_scenario_agents()

	if not req.agents:
		return _apply_saved_agent_overrides(req.agent_set, base)

	# Apply UI overrides (only for agents that were edited)
	override_map = {a.id: a for a in req.agents}
	for agent in base:
		slot = int(agent.agent_id.split("_")[1]) + 1  # "agent_0" → 1
		if slot in override_map:
			ov = override_map[slot]
			agent.name = ov.name
			agent.system_prompt = ov.prompt

	# UI overrides are not just transient: persist them locally so the same
	# agent names/prompts are used after refresh or container restart.
	try:
		_save_agent_overrides(req.agent_set, base, source="request_override")
	except Exception as e:
		logger.warning(f"Could not persist agent overrides: {e}")

	return base

# ── Language directive ────────────────────────────────────────────────────────

LANG_PREFIX = {
	"en": "",
	"ko": "You must respond entirely in Korean. ",
}
JUDGE_LANG_DIRECTIVE = {
	"en": "",
	"ko": (
		"Write the values of 'common_conclusion', 'weakest_link', and "
		"'convergence_analysis' in Korean. All JSON keys must remain in English. "
	),
}

# ── SSE helper ────────────────────────────────────────────────────────────────

def _json_safe(obj):
	"""Convert NumPy types to native Python types"""
	import numpy as np
	if isinstance(obj, np.bool_):
		return bool(obj)
	if isinstance(obj, (np.integer,)):
		return int(obj)
	if isinstance(obj, (np.floating,)):
		return float(obj)
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	return obj

def sse(event_type: str, data: Any) -> str:
	return f"data: {json.dumps({'type': event_type, **data}, default=_json_safe)}\n\n"

def sse_done() -> str:
	return "data: [DONE]\n\n"


CONVERGENCE_STATES = {
	"singularity",
	"partial_convergence",
	"dominant_compatible_proposal",
	"no_singularity",
}

def _as_list(value: Any) -> List[str]:
	if value is None:
		return []
	if isinstance(value, list):
		return [str(v) for v in value if str(v).strip()]
	if isinstance(value, str):
		return [v.strip() for v in re.split(r"[,;]\s*", value) if v.strip()]
	return []

def _normalise_convergence_state(
	semantic: Dict[str, Any],
	*,
	ratio_signal: int,
	agent_names: List[str],
) -> Dict[str, Any]:
	"""
	Separate the torus math signal from the decision state.

	The torus may produce a binary signal (1 ≈ old 1.62, 0 ≈ old 0), but the user-facing
	state must be decided from the judge's reasons:
	- singularity: all four converge into one executable conclusion
	- partial_convergence: only a subset converges
	- dominant_compatible_proposal: no full singularity, but a defensible actionable coalition exists
	- no_singularity: no usable shared proposal
	"""
	state = str(semantic.get("convergence_state") or "").strip().lower()
	if state not in CONVERGENCE_STATES:
		state = "singularity" if ratio_signal == 1 else "no_singularity"

	coalition_agents = _as_list(semantic.get("coalition_agents"))
	dissenting_agents = _as_list(semantic.get("dissenting_agents"))
	proposal = semantic.get("dominant_compatible_proposal")
	partial_summary = semantic.get("partial_convergence_summary")
	why = semantic.get("why_this_state") or semantic.get("convergence_analysis") or ""

	# Safety correction: if only 2-3 agents are in the coalition and dissent exists,
	# a raw torus signal must not be shown as full singularity.
	if state == "singularity":
		if dissenting_agents:
			state = "partial_convergence"
		elif coalition_agents and len(set(coalition_agents)) < len(agent_names):
			state = "partial_convergence"

	# If full singularity failed but the judge gave an actionable proposal, surface it.
	if state == "no_singularity" and proposal:
		state = "dominant_compatible_proposal"

	# Basic heuristic fallback for older/local judges that do not yet output the new fields.
	if state == "no_singularity" and ratio_signal == 0:
		analysis_blob = " ".join(str(semantic.get(k) or "") for k in (
			"convergence_analysis", "weakest_link", "common_conclusion"
		)).lower()
		majority_markers = ["three agents", "3 agents", "majority", "다수", "세 에이전트", "3개", "three", "phased", "단계적"]
		if any(m in analysis_blob for m in majority_markers):
			state = "dominant_compatible_proposal"

	is_singularity_state = state == "singularity"
	return {
		"convergence_state": state,
		"ratio_signal": int(ratio_signal),
		"is_singularity": bool(is_singularity_state),
		"coalition_agents": coalition_agents,
		"dissenting_agents": dissenting_agents,
		"dominant_compatible_proposal": proposal,
		"partial_convergence_summary": partial_summary,
		"why_this_state": why,
	}

# ── Document upload endpoints ────────────────────────────────────────────────

@router.post("/uploads")
async def upload_documents(files: List[UploadFile] = File(...)):
	"""Store uploaded documents temporarily and return an upload_session_id."""
	ensure_upload_root()
	if not files:
		raise HTTPException(400, "No files uploaded")
	if len(files) > MAX_FILES:
		raise HTTPException(400, f"Too many files. Max {MAX_FILES}")

	session_id = new_session_id()
	folder = session_path(session_id)
	folder.mkdir(parents=True, exist_ok=True)

	stored = []
	total = 0
	try:
		for f in files:
			original = f.filename or "uploaded_file"
			ext = validate_extension(original)
			safe = sanitize_filename(original)
			content = await f.read()
			size = len(content)
			if size > MAX_FILE_BYTES:
				raise HTTPException(413, f"{original} exceeds per-file limit")
			total += size
			if total > MAX_TOTAL_BYTES:
				raise HTTPException(413, "Upload exceeds total size limit")

			target = folder / safe
			if target.exists():
				stem, suffix = target.stem, target.suffix
				n = 2
				while target.exists():
					target = folder / f"{stem}_{n}{suffix}"
					n += 1
			target.write_bytes(content)
			stored.append({
				"original_name": original,
				"stored_name": target.name,
				"size": size,
				"extension": ext,
			})

		return {"upload_session_id": session_id, "files": stored, "count": len(stored)}
	except Exception:
		cleanup_upload_session(session_id)
		raise


@router.get("/uploads/{session_id}")
async def upload_status(session_id: str):
	files = list_session_files(session_id)
	return {
		"upload_session_id": session_id,
		"files": [
			{"stored_name": p.name, "size": p.stat().st_size, "extension": p.suffix.lower()}
			for p in files
		],
	}


@router.delete("/uploads/{session_id}")
async def upload_delete(session_id: str):
	cleanup_upload_session(session_id)
	return {"deleted": True, "upload_session_id": session_id}

# ── Core streaming generator ──────────────────────────────────────────────────

async def stream_fourCM(req: FourCMRequest, claude_key: str, grok_key: str):
	"""
	Main async generator — yields SSE events for each step of 4CM.
	"""
	# ── Determine USE_EXTERNAL_API at runtime ─────────────────────────────────
	# Request field takes precedence over environment variable
	if req.use_external_api is not None:
		use_external = req.use_external_api
		os.environ["USE_EXTERNAL_API"] = "true" if use_external else "false"
	else:
		_env = os.environ.get("USE_EXTERNAL_API", "true").lower()
		use_external = _env not in ("false", "0", "no")

	# ── local_payloads: additional payload dict by agent slot ─────────────────
	# {"1": {"temperature": 0.9}, "2": {}, ...}
	_local_payloads: Dict[str, Dict] = req.local_payloads or {}

	# Temporarily inject keys into environment for this request
	# (Keys come from request headers, NOT from environment in production)
	_orig_claude = os.environ.get("ANTHROPIC_API_KEY", "")
	_orig_grok   = os.environ.get("XAI_API_KEY", "")
	if claude_key:
		os.environ["ANTHROPIC_API_KEY"] = claude_key
	if grok_key:
		os.environ["XAI_API_KEY"] = grok_key

	run_id = uuid.uuid4().hex[:12]
	run_log: Dict[str, Any] = {
		"run_id": run_id,
		"started_at": datetime.now().isoformat(),
		"finished_at": None,
		"request": {
			"query": req.query,
			"risk_level": req.risk_level,
			"lang": req.lang,
			"n_rounds": req.n_rounds,
			"agent_set": req.agent_set,
			"no_context": req.no_context,
			"provider_mode": req.provider_mode,
			"agent_providers": req.agent_providers,
			"grok_search_mode": req.grok_search_mode,
			"upload_session_id": req.upload_session_id,
			"use_external_api": req.use_external_api,
		},
		"upload_files": _upload_manifest(req.upload_session_id),
		"gdpr_guidance": _gdpr_complexity_guidance(req) if "gdpr" in (req.agent_set or "").lower() or "gdpr" in (req.query or "").lower() else None,
		"intent_triage": getattr(req, "_intent_triage", None),
		"human_review_required": bool((getattr(req, "_intent_triage", {}) or {}).get("human_review_required", False)),
		"agents": [],
		"rounds": [],
		"summary": None,
		"errors": [],
	}

	try:
		if getattr(req, "_intent_triage", None):
			yield sse("intent_triage", {"triage": getattr(req, "_intent_triage")})

		# Setup
		torus      = TorusField()
		judge      = JudgeFunction(torus, convergence_threshold=0.5)
		constraint = ConstraintLayer(torus, drift_tolerance=0.3)
		embedder   = EmbeddingEngine(use_transformer=True)
		agents     = build_agents(req)
		run_log["agents"] = [
			{
				"id": int(a.agent_id.split("_")[1]) + 1,
				"name": a.name,
				"prompt": a.system_prompt,
			}
			for a in agents
		]

		lang_prefix     = LANG_PREFIX.get(req.lang, "")
		judge_lang      = JUDGE_LANG_DIRECTIVE.get(req.lang, "")

		# Inject language directive into each agent's system prompt
		if lang_prefix:
			for agent in agents:
				if not agent.system_prompt.startswith(lang_prefix):
					agent.system_prompt = lang_prefix + agent.system_prompt

		# Optional document context from uploaded files.
		document_context = ""
		if req.upload_session_id:
			try:
				document_context = build_document_context(req.upload_session_id)
			except Exception as e:
				logger.error(f"Document context build failed: {e}")
				document_context = f"[Uploaded document context unavailable: {str(e)[:200]}]"

		base_query = req.query
		if document_context:
			base_query = (
				f"{req.query}\n\n"
				"Use the following uploaded document context as evidence. "
				"When the document conflicts with general knowledge, explicitly mention the conflict.\n\n"
				f"{document_context}"
			)

		prev_context = ""
		first_singularity_round: Optional[int] = None
		final_conclusion: Optional[str] = None

		for round_num in range(1, req.n_rounds + 1):
			providers = provider_map_for_round(round_num, req.risk_level, req.provider_mode, agents, req.agent_providers)
			provider_map_by_name: Dict[str, str] = {}
			round_log: Dict[str, Any] = {
				"round": round_num,
				"providers": {},
				"responses": [],
				"judge": {},
			}

			# ── Step 1: Agent calls ────────────────────────────────────────
			# Same-round agents run in parallel. Rounds remain sequential because
			# Round N+1 may use Round N as prev_context.
			responses: Dict[str, str] = {}
			agent_response_list = []

			async def run_one_agent(i: int, agent: OrthogonalAgent, prov: str) -> Dict[str, Any]:
				try:
					# "local" provider → automatically routed by simulate_orthogonal_response
					# to the USE_EXTERNAL_API=false path (call_internal_llm).
					# Additional fields from local_payloads are temporarily injected as environment variables.
					_slot = str(i + 1)
					_extra = _local_payloads.get(_slot, {})
					if _extra:
						# Pass additional payload fields via environment variables
						# (call_internal_llm has been updated to read INTERNAL_LLM_EXTRA_PAYLOAD)
						os.environ["INTERNAL_LLM_EXTRA_PAYLOAD"] = json.dumps(_extra)
					else:
						os.environ.pop("INTERNAL_LLM_EXTRA_PAYLOAD", None)

					# "local" → routed from call_llm to call_internal_llm
					effective_prov = "claude" if prov == "local" else prov
					text = await asyncio.to_thread(
						simulate_orthogonal_response,
						agent,
						base_query,
						prev_context,
						effective_prov,
						req.grok_search_mode if prov != "local" else "off",
					)
					return {"agent": agent, "name": agent.name, "provider": prov, "text": text, "status": "ok", "error": None}
				except Exception as e:
					logger.error(f"Agent {agent.name} error via {prov}: {e}")
					if prov == "grok" and req.grok_search_mode in ("auto", "on"):
						try:
							fallback_text = await asyncio.to_thread(
								simulate_orthogonal_response,
								agent,
								base_query + "\n\nNote: Grok web search failed for this agent. Answer without live web search, and state that no live web verification was available.",
								prev_context,
								"claude",
								"off",
							)
							return {"agent": agent, "name": agent.name, "provider": "claude-fallback", "text": fallback_text, "status": "fallback", "error": str(e)[:300]}
						except Exception as fallback_e:
							return {"agent": agent, "name": agent.name, "provider": prov, "text": "", "status": "failed", "error": f"Grok failed: {str(e)[:160]} / Claude fallback failed: {str(fallback_e)[:160]}"}
					return {"agent": agent, "name": agent.name, "provider": prov, "text": "", "status": "failed", "error": str(e)[:300]}

			for i, agent in enumerate(agents):
				yield sse("agent_start", {"round": round_num, "name": agent.name, "provider": providers[i]})

			agent_results = await asyncio.gather(*[
				run_one_agent(i, agent, providers[i])
				for i, agent in enumerate(agents)
			])

			for result in agent_results:
				agent = result["agent"]
				provider_map_by_name[agent.name] = result["provider"]
				round_log["providers"][agent.name] = result["provider"]
				text = result["text"] or ""
				responses[agent.agent_id] = text
				if text:
					agent.response_history.append(text)

				yield sse("agent", {
					"round": round_num,
					"name": agent.name,
					"provider": result["provider"],
					"text": text,
					"status": result["status"],
					"error": result["error"],
				})

				response_record = {
					"name": agent.name,
					"provider": result["provider"],
					"text": text,
					"status": result["status"],
					"error": result["error"],
				}
				agent_response_list.append(response_record)
				round_log["responses"].append(response_record)

			# ── Step 2: Embeddings + torus positions ───────────────────────
			response_texts = [responses[a.agent_id] for a in agents]

			try:
				embeddings = await asyncio.to_thread(embedder.embed, response_texts)
				neutral_emb = await asyncio.to_thread(
					embedder.embed_single,
					f"Regarding '{req.query[:50]}', a balanced view suggests "
					f"considering multiple perspectives."
				)
				positions = []
				for i, agent in enumerate(agents):
					val = constraint.validate_agent_position(
						agent.agent_id, embeddings[i], neutral_emb
					)
					positions.append(val["effective_position"])
			except Exception as e:
				logger.error(f"Embedding error: {e}")
				# fallback positions
				positions = [a.position for a in agents]

			# ── Step 3: Semantic judge ─────────────────────────────────────
			yield sse("judge_start", {"round": round_num})

			named_responses = {a.name: responses[a.agent_id] for a in agents}
			try:
				semantic = await asyncio.to_thread(
					semantic_compare,
					named_responses,
					req.query,
					5,
					judge_lang,
				)
			except Exception as e:
				logger.error(f"Judge error: {e}")
				semantic = {
					"semantic_similarity_score": 0.0,
					"conclusion_convergence": 0.0,
					"reasoning_convergence": 0.0,
					"all_point_same_direction": False,
					"common_conclusion": None,
					"weakest_link": f"judge error: {str(e)[:80]}",
					"convergence_analysis": "judge unavailable",
				}

			semantic_score    = semantic.get("semantic_similarity_score", 0)
			conclusion_score  = semantic.get("conclusion_convergence", semantic_score)
			reasoning_score   = semantic.get("reasoning_convergence", semantic_score)
			same_direction    = semantic.get("all_point_same_direction", False)
			common_conclusion = semantic.get("common_conclusion")

			# ── Step 4: Torus judgment + decision-state classifier ──────────
			judgment = judge.compute_convergence_from_semantic(
				semantic_score, positions,
				conclusion_score=conclusion_score,
				reasoning_score=reasoning_score,
			)
			raw_ratio = judgment["singularity_ratio"]
			ratio_signal = 1 if (judgment["is_singularity"] and same_direction) else 0
			state_info = _normalise_convergence_state(
				semantic,
				ratio_signal=ratio_signal,
				agent_names=[a.name for a in agents],
			)
			is_singularity = state_info["is_singularity"]

			if is_singularity and first_singularity_round is None:
				first_singularity_round = round_num
				final_conclusion = common_conclusion

			# ── Yield round_complete ───────────────────────────────────────
			round_complete_payload = {
				"round": round_num,
				"providers": provider_map_by_name,
				"responses": agent_response_list,
				"conclusion_score": conclusion_score,
				"reasoning_score": reasoning_score,
				"semantic_score": semantic_score,
				"ratio": raw_ratio,
				"ratio_signal": state_info["ratio_signal"],
				"torus_coord": list(judgment["convergence_point"]),
				"is_singularity": is_singularity,
				"convergence_state": state_info["convergence_state"],
				"coalition_agents": state_info["coalition_agents"],
				"dissenting_agents": state_info["dissenting_agents"],
				"dominant_compatible_proposal": state_info["dominant_compatible_proposal"],
				"partial_convergence_summary": state_info["partial_convergence_summary"],
				"why_this_state": state_info["why_this_state"],
				"conclusion": common_conclusion,
				"weakest_link": semantic.get("weakest_link", ""),
				"analysis": semantic.get("convergence_analysis", ""),
			}
			round_log["judge"] = {k: v for k, v in round_complete_payload.items() if k not in ("responses",)}
			run_log["rounds"].append(round_log)
			yield sse("round_complete", round_complete_payload)

			# ── Context for next round (blind mode skips) ──────────────────
			if req.no_context:
				prev_context = ""
			else:
				prev_context = "\n".join([
					f"- [{a.name}]: {responses[a.agent_id]}" for a in agents
				])

			await asyncio.sleep(0.1)

		# ── Summary ────────────────────────────────────────────────────────
		selected_judge = (run_log.get("rounds") or [{}])[-1].get("judge", {}) if run_log.get("rounds") else {}
		if first_singularity_round is not None:
			for rr in run_log.get("rounds", []):
				if rr.get("judge", {}).get("round") == first_singularity_round:
					selected_judge = rr.get("judge", {})
					break
		summary_payload = {
			"first_singularity_round": first_singularity_round,
			"conclusion": final_conclusion,
			"phone_rang": first_singularity_round is not None,
			"convergence_state": selected_judge.get("convergence_state", "singularity" if first_singularity_round is not None else "no_singularity"),
			"ratio_signal": selected_judge.get("ratio_signal", 1 if first_singularity_round is not None else 0),
			"dominant_compatible_proposal": selected_judge.get("dominant_compatible_proposal"),
			"partial_convergence_summary": selected_judge.get("partial_convergence_summary"),
		}
		run_log["summary"] = summary_payload
		yield sse("summary", summary_payload)

		yield sse_done()

	except Exception as e:
		logger.exception("stream_fourCM fatal error")
		run_log["errors"].append(str(e))
		yield sse("error", {"message": str(e)})
		yield sse_done()

	finally:
		run_log["finished_at"] = datetime.now().isoformat()
		log_path = _write_run_log(run_log)
		if log_path:
			logger.info(f"4CM run log written: {log_path}")

		# Restore original keys
		os.environ["ANTHROPIC_API_KEY"] = _orig_claude
		os.environ["XAI_API_KEY"]       = _orig_grok
		if req.upload_session_id:
			cleanup_upload_session(req.upload_session_id)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("")
async def run_fourCM(req: FourCMRequest, request: Request):
	"""
	POST /fourCM
	SSE streaming endpoint.

	Keys are passed via headers:
	  X-Claude-Key: sk-ant-...
	  X-Grok-Key:   xai-...

	If headers are empty, falls back to env vars.
	"""
	claude_key = request.headers.get("X-Claude-Key", "").strip()
	grok_key   = request.headers.get("X-Grok-Key", "").strip()

	# Determine Local LLM mode (request field takes precedence over environment variable)
	if req.use_external_api is not None:
		_is_local = not req.use_external_api
	else:
		_env = os.environ.get("USE_EXTERNAL_API", "true").lower()
		_is_local = _env in ("false", "0", "no")

	# LLM-based intent triage: the fixed judge decides whether this is a 4CM case,
	# and whether accountable human review is required. In AI API mode this is Grok;
	# in Local LLM mode this is the configured internal/local LLM.
	if not _is_local and not (grok_key or os.environ.get("XAI_API_KEY")):
		# Grok judge is required for triage in external AI API mode.
		raise HTTPException(401, "xAI/Grok API key required for judge intent triage")

	intent_triage = await asyncio.to_thread(
		judge_intent_triage,
		req,
		use_external_api=not _is_local,
		grok_key=grok_key,
	)
	# Store triage on the request object dynamically so the stream logger can include it.
	setattr(req, "_intent_triage", intent_triage)

	if intent_triage.get("simple_assistant_question") or not intent_triage.get("should_run_4cm", True):
		raise HTTPException(
			400,
			{
				"message": "This prompt should not run through 4CM. Use a normal assistant flow instead.",
				"intent_triage": intent_triage,
			},
		)

	# If the judge sees a high-risk operational decision, force high-risk routing.
	if intent_triage.get("high_risk") or intent_triage.get("human_review_required"):
		req.risk_level = "high"

	# Key check: external API key not required in Local LLM mode
	if not _is_local:
		if not claude_key and not os.environ.get("ANTHROPIC_API_KEY"):
			raise HTTPException(401, "Anthropic API key required (X-Claude-Key header or ANTHROPIC_API_KEY env)")
		if not grok_key and not os.environ.get("XAI_API_KEY"):
			raise HTTPException(401, "xAI API key required (X-Grok-Key header or XAI_API_KEY env)")

	if req.n_rounds < 1 or req.n_rounds > 5:
		raise HTTPException(400, "n_rounds must be 1-5")
	if req.risk_level not in ("normal", "high"):
		raise HTTPException(400, "risk_level must be 'normal' or 'high'")
	if req.lang not in ("en", "ko"):
		raise HTTPException(400, "lang must be 'en' or 'ko'")
	if req.provider_mode not in ("round-robin", "all-grok", "all-claude", "custom", "all-local"):
		raise HTTPException(400, "provider_mode must be round-robin, all-grok, all-claude, custom, or all-local")
	if req.grok_search_mode not in ("off", "auto", "on"):
		raise HTTPException(400, "grok_search_mode must be off, auto, or on")

	return StreamingResponse(
		stream_fourCM(req, claude_key, grok_key),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			"X-Accel-Buffering": "no",  # nginx SSE pass-through
			"Connection": "keep-alive",
		},
	)


@router.post("/validate")
async def validate_agent(req: ValidateAgentRequest, request: Request):
	"""
	POST /fourCM/validate
	Validate and optionally rewrite an edited agent using the fixed judge LLM.

	Routing rule:
	- AI API mode / USE_EXTERNAL_API=true  -> Grok judge
	- Local LLM mode / USE_EXTERNAL_API=false -> internal/local judge LLM
	"""
	grok_key = request.headers.get("X-Grok-Key", "").strip() or os.environ.get("XAI_API_KEY", "")
	use_external = _effective_use_external_api(req.use_external_api)

	if use_external and not grok_key:
		raise HTTPException(401, "xAI/Grok API key required for validation in AI API mode")

	try:
		result = await asyncio.to_thread(
			judge_agent_prompt_validation,
			req,
			use_external_api=use_external,
			grok_key=grok_key,
		)
		return result
	except Exception as e:
		logger.error(f"Validation error: {e}")
		raise HTTPException(500, f"Validation failed: {str(e)}")


@router.post("/validate-and-save/{set_id}/agent/{agent_idx}")
async def validate_and_save_agent(set_id: str, agent_idx: int, req: ValidateAgentRequest, request: Request):
	"""
	Validate/rewrite an edited agent with the judge LLM, then save it as a local override.

	This endpoint intentionally does NOT edit angry_agents/{set_id}/members.txt or
	angry_agents/{set_id}/{agent_idx}.txt. User edits are stored in
	SAVED_AGENTS_FILE, default: angry_agents/_saved_agent_overrides.json.
	The original scenario files remain immutable defaults; overrides are applied at
	runtime by _apply_saved_agent_overrides().
	"""
	if agent_idx < 1 or agent_idx > 4:
		raise HTTPException(400, "agent_idx must be 1-4")
	folder = ANGRY_AGENTS_ROOT / set_id
	if not folder.exists():
		raise HTTPException(404, f"Scenario '{set_id}' not found")

	grok_key = request.headers.get("X-Grok-Key", "").strip() or os.environ.get("XAI_API_KEY", "")
	use_external = _effective_use_external_api(req.use_external_api)
	if use_external and not grok_key:
		raise HTTPException(401, "xAI/Grok API key required for validation in AI API mode")

	result = await asyncio.to_thread(
		judge_agent_prompt_validation,
		req,
		use_external_api=use_external,
		grok_key=grok_key,
	)

	name = result.get("rewritten_name") or req.name
	prompt = result.get("rewritten_prompt") or req.prompt

	# Load defaults + any existing local override, update the selected slot, then
	# persist the full effective set back into _saved_agent_overrides.json.
	agents = _apply_saved_agent_overrides(set_id, load_agents_from_files(set_id))
	for a in agents:
		if int(a.agent_id.split("_")[1]) + 1 == agent_idx:
			a.name = name
			a.system_prompt = prompt
			break
	_save_agent_overrides(set_id, agents, source="validate-and-save-local-override")

	return {
		"saved": True,
		"storage": "local_override_json",
		"saved_agents_file": str(SAVED_AGENTS_FILE),
		"base_files_modified": False,
		"set": set_id,
		"agent": agent_idx,
		**result,
	}


@router.get("/agents/{agent_set}")
async def get_agent_set(agent_set: str):
	"""
	GET /fourCM/agents/{agent_set}
	Return agent names + prompt previews for the UI.
	"""
	try:
		agents = _apply_saved_agent_overrides(agent_set, load_agents_from_files(agent_set))
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(500, str(e))

	return {
		"agent_set": agent_set,
		"agents": [
			{
				"id": int(a.agent_id.split("_")[1]) + 1,
				"name": a.name,
				"prompt": a.system_prompt,
				"prompt_preview": a.system_prompt[:200] + "..." if len(a.system_prompt) > 200 else a.system_prompt,
			}
			for a in agents
		],
	}


async def _translate_to_korean(text: str, grok_key: str, context: str = "") -> str:
	"""
	Translate text into Korean with Grok.
	context: translation hint (e.g. "scenario title", "query")
	"""
	if not text or not grok_key:
		return text

	headers = {
		"Authorization": f"Bearer {grok_key}",
		"Content-Type": "application/json",
	}
	payload = {
		"model": GROK_MODEL,
		"max_tokens": 1000,
		"temperature": 0.2,
		"messages": [
			{
				"role": "system",
				"content": (
					"You are a professional Korean translator. "
					"Translate the given text to Korean accurately and naturally. "
					"Preserve all numbers, proper nouns, technical terms, and formatting. "
					"Return ONLY the translated text, nothing else."
				)
			},
			{
				"role": "user",
				"content": f"Translate this {context} to Korean:\n\n{text}"
			}
		]
	}

	try:
		resp = await asyncio.to_thread(
			lambda: requests.post(XAI_API_URL, headers=headers, json=payload, timeout=30)
		)
		resp.raise_for_status()
		body = resp.json()
		return (body.get("choices", [{}])[0].get("message", {}).get("content") or text).strip()
	except Exception as e:
		logger.error(f"Translation failed: {e}")
		return text


@router.get("/agents")
async def list_agent_sets(lang: str = "en", request: Request = None):
	"""
	GET /fourCM/agents?lang=en  — English (default)
	GET /fourCM/agents?lang=ko  — Korean (uses cached translation)

	For Korean requests:
	- Return title.ko.txt and query.ko.txt immediately if they exist
	- If not, translate with Grok, save as *.ko.txt, then return
	"""
	if not ANGRY_AGENTS_ROOT.exists():
		return {"sets": []}

	# Extract Grok key if translation is required
	grok_key = ""
	if lang == "ko" and request:
		grok_key = request.headers.get("X-Grok-Key", "").strip() or os.environ.get("XAI_API_KEY", "")

	sets = []
	for d in sorted(ANGRY_AGENTS_ROOT.iterdir()):
		if not d.is_dir() or not (d / "members.txt").exists():
			continue

		def read(fname: str) -> str:
			f = d / fname
			return f.read_text(encoding="utf-8").strip() if f.exists() else ""

		# Read or generate Korean translation
		async def get_text(base_fname: str, context: str) -> str:
			ko_fname = base_fname.replace(".txt", ".ko.txt")
			if lang == "ko":
				# Return immediately if a cached translation exists
				ko_file = d / ko_fname
				if ko_file.exists():
					return ko_file.read_text(encoding="utf-8").strip()
				# If not, translate and save
				original = read(base_fname)
				if original and grok_key:
					translated = await _translate_to_korean(original, grok_key, context)
					ko_file.write_text(translated, encoding="utf-8")
					return translated
				return original
			return read(base_fname)

		# Parse members.txt for agent names
		names = {}
		for line in read("members.txt").splitlines():
			if ":" in line:
				idx_str, name = line.split(":", 1)
				try:
					names[int(idx_str.strip())] = name.strip()
				except ValueError:
					pass

		title = await get_text("title.txt", "scenario title")
		query = await get_text("query.txt", "scenario query/question")

		sets.append({
			"id": d.name,
			"title": title or d.name,
			"query": query,
			"risk": read("risk.txt") or "normal",
			"agents": [
				{"id": i, "name": names.get(i, f"AGENT_{i}")}
				for i in range(1, 5)
			],
		})

	return {"sets": sets}

# ── Settings: Backup & Restore ───────────────────────────────────────────────

class ScenarioRenameRequest(BaseModel):
	new_id: str   # new folder name

class ScenarioAgentUpdate(BaseModel):
	name: str
	prompt: str


@router.get("/export")
async def export_all():
	"""
	GET /fourCM/export
	Export all scenarios + agents as JSON for backup.
	"""
	if not ANGRY_AGENTS_ROOT.exists():
		return {}

	export = {
		"version": "2.0",
		"exported_at": datetime.now().isoformat(),
		"scenarios": {}
	}

	for d in sorted(ANGRY_AGENTS_ROOT.iterdir()):
		if not d.is_dir() or not (d / "members.txt").exists():
			continue

		def read(fname: str) -> str:
			f = d / fname
			return f.read_text(encoding="utf-8").strip() if f.exists() else ""

		# Parse members.txt
		names = {}
		for line in read("members.txt").splitlines():
			if ":" in line:
				idx_str, name = line.split(":", 1)
				try:
					names[int(idx_str.strip())] = name.strip()
				except ValueError:
					pass

		agents = {}
		for i in range(1, 5):
			agents[str(i)] = {
				"name": names.get(i, f"AGENT_{i}"),
				"prompt": read(f"{i}.txt"),
			}

		export["scenarios"][d.name] = {
			"title": read("title.txt") or d.name,
			"query": read("query.txt"),
			"risk": read("risk.txt") or "normal",
			"agents": agents,
		}

	return export


@router.post("/import")
async def import_all(data: dict):
	"""
	POST /fourCM/import
	Restore scenarios from exported JSON.
	Existing scenarios are overwritten.
	"""
	scenarios = data.get("scenarios", {})
	if not scenarios:
		raise HTTPException(400, "No scenarios in import data")

	ANGRY_AGENTS_ROOT.mkdir(parents=True, exist_ok=True)
	imported = []

	for set_id, sc in scenarios.items():
		# Sanitize folder name
		safe_id = "".join(c for c in set_id if c.isalnum() or c in "-_")
		if not safe_id:
			continue

		folder = ANGRY_AGENTS_ROOT / safe_id
		folder.mkdir(exist_ok=True)

		(folder / "title.txt").write_text(sc.get("title", safe_id), encoding="utf-8")
		(folder / "query.txt").write_text(sc.get("query", ""), encoding="utf-8")
		(folder / "risk.txt").write_text(sc.get("risk", "normal"), encoding="utf-8")

		agents = sc.get("agents", {})
		members_lines = []
		for idx_str, agent in agents.items():
			try:
				idx = int(idx_str)
			except ValueError:
				continue
			(folder / f"{idx}.txt").write_text(agent.get("prompt", ""), encoding="utf-8")
			members_lines.append(f"{idx}: {agent.get('name', f'AGENT_{idx}')}")

		(folder / "members.txt").write_text("\n".join(sorted(members_lines)), encoding="utf-8")
		imported.append(safe_id)

	return {"imported": imported}


@router.delete("/scenario/{set_id}")
async def delete_scenario(set_id: str):
	"""
	DELETE /fourCM/scenario/{set_id}
	Delete a scenario folder.
	"""
	import shutil
	folder = ANGRY_AGENTS_ROOT / set_id
	if not folder.exists():
		raise HTTPException(404, f"Scenario '{set_id}' not found")

	shutil.rmtree(folder)
	return {"deleted": set_id}


@router.post("/scenario/{set_id}/rename")
async def rename_scenario(set_id: str, req: ScenarioRenameRequest):
	"""
	POST /fourCM/scenario/{set_id}/rename
	Rename a scenario folder.
	"""
	import shutil
	folder = ANGRY_AGENTS_ROOT / set_id
	if not folder.exists():
		raise HTTPException(404, f"Scenario '{set_id}' not found")

	new_id = "".join(c for c in req.new_id if c.isalnum() or c in "-_")
	if not new_id:
		raise HTTPException(400, "Invalid new name")

	new_folder = ANGRY_AGENTS_ROOT / new_id
	if new_folder.exists():
		raise HTTPException(409, f"'{new_id}' already exists")

	shutil.move(str(folder), str(new_folder))
	return {"renamed": {"from": set_id, "to": new_id}}


@router.put("/scenario/{set_id}/agent/{agent_idx}")
async def update_agent(set_id: str, agent_idx: int, req: ScenarioAgentUpdate):
	"""
	PUT /fourCM/scenario/{set_id}/agent/{agent_idx}
	Save a single agent name/prompt as a local override without editing base files.
	"""
	folder = ANGRY_AGENTS_ROOT / set_id
	if not folder.exists():
		raise HTTPException(404, f"Scenario '{set_id}' not found")

	if agent_idx < 1 or agent_idx > 4:
		raise HTTPException(400, "agent_idx must be 1-4")

	agents = _apply_saved_agent_overrides(set_id, load_agents_from_files(set_id))
	for a in agents:
		if int(a.agent_id.split("_")[1]) + 1 == agent_idx:
			a.name = req.name
			a.system_prompt = req.prompt
			break
	_save_agent_overrides(set_id, agents, source="scenario-agent-update-local-override")

	return {
		"updated": {"set": set_id, "agent": agent_idx, "name": req.name},
		"storage": "local_override_json",
		"saved_agents_file": str(SAVED_AGENTS_FILE),
		"base_files_modified": False,
	}


# ── Saved Agent Overrides ─────────────────────────────────────────────────────

class SavedAgentSetRequest(BaseModel):
	agents: List[ScenarioAgentUpdate]


@router.get("/saved-agents")
async def saved_agents_list():
	"""Return locally persisted agent overrides, if any."""
	return _read_saved_agents()


@router.get("/saved-agents/{set_id}")
async def saved_agents_get(set_id: str):
	"""Return the effective saved/default agent list for a scenario."""
	base = load_agents_from_files(set_id)
	effective = _apply_saved_agent_overrides(set_id, base)
	return {
		"agent_set": set_id,
		"agents": [
			{
				"id": int(a.agent_id.split("_")[1]) + 1,
				"name": a.name,
				"prompt": a.system_prompt,
			}
			for a in effective
		],
	}


@router.post("/saved-agents/{set_id}")
async def saved_agents_save(set_id: str, req: SavedAgentSetRequest):
	"""Persist a full local override set without editing the base scenario files."""
	if len(req.agents) != 4:
		raise HTTPException(400, "Exactly four agents are required")
	base = load_agents_from_files(set_id)
	for i, item in enumerate(req.agents):
		base[i].name = item.name
		base[i].system_prompt = item.prompt
	_save_agent_overrides(set_id, base, source="saved-agents-api")
	return {"saved": True, "agent_set": set_id, "count": 4}


@router.delete("/saved-agents/{set_id}")
async def saved_agents_delete(set_id: str):
	"""Remove local overrides so the scenario falls back to files."""
	data = _read_saved_agents()
	sets = data.setdefault("sets", {})
	deleted = bool(sets.pop(set_id, None))
	_write_saved_agents(data)
	return {"deleted": deleted, "agent_set": set_id}


@router.get("/risk-policy")
async def risk_policy():
	"""Human-readable policy hints for simple prompts and GDPR tier handling."""
	return {
		"simple_prompt_gate": "Low-stakes/simple prompts are rejected before a 4CM run.",
		"judge_llm_routing": {
			"AI_API_mode": "Grok performs intent triage, human-review classification, semantic judging, and prompt validation/rewrite.",
			"Local_LLM_mode": "The configured internal/local LLM performs intent triage, human-review classification, semantic judging, and prompt validation/rewrite.",
		},
		"gdpr_four_stage": {
			"1": "Low risk: transparency, notice, lawful basis, retention clarity; no extreme agent escalation by default.",
			"2": "Moderate/high-but-manageable risk: maximum transparency plus proportional safeguards; human review only when the user's intent is an operational or consequential decision request.",
			"3": "High risk: DPIA-style review, adversarial privacy/security/legal agents, and accountable human review.",
			"4": "Very high or unacceptable risk: stop/hold deployment recommendation unless mitigations and accountable human approval exist.",
		},
		"intent_triage": "Before a 4CM run, the judge LLM classifies whether the user is asking a simple assistant question, an analysis, or an operational decision that requires human review.",
	}

# ── API Key Management ────────────────────────────────────────────────────────

class KeySaveRequest(BaseModel):
	claude_key: str = ""
	grok_key: str = ""


def _read_env_file() -> Dict[str, str]:
	"""현재 .env 파일 파싱 → dict"""
	result = {}
	if not ENV_FILE.exists():
		return result
	for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, _, val = line.partition("=")
		result[key.strip()] = val.strip().strip('"').strip("'")
	return result

_SECRET_KEYS = {"ANTHROPIC_API_KEY", "XAI_API_KEY"}

def _write_env_file(data: Dict[str, str]) -> None:
	lines = []
	for k, v in data.items():
		if k in _SECRET_KEYS:
			continue                    # ← don't write secret keys back to .env to avoid accidental leaks
		lines.append(f"{k}={v}")
	ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

@router.get("/keys/status")
async def keys_status():
	"""
	GET /fourCM/keys/status
	return current status of API keys without revealing their values:
	
	Options2: If keys are present in .env, they are automatically loaded into environment variables at startup, 
	so we check the environment variables for status. This way, the UI can show whether keys are set without ever exposing the actual key values in logs or API responses.
	"""
	# env_data = _read_env_file()

	# # Load the key from .env into the environment variable if present (option 2)
	# if env_data.get("ANTHROPIC_API_KEY"):
	# 	os.environ["ANTHROPIC_API_KEY"] = env_data["ANTHROPIC_API_KEY"]
	# if env_data.get("XAI_API_KEY"):
	# 	os.environ["XAI_API_KEY"] = env_data["XAI_API_KEY"]

	claude_set = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
	grok_set   = bool(os.environ.get("XAI_API_KEY", "").strip())

	return {
		"claude_set": claude_set,
		"grok_set": grok_set,
		"source": "memory",
		#"source": "env_file" if env_data.get("ANTHROPIC_API_KEY") else "environment",
	}


@router.post("/keys/save")
async def keys_save(req: KeySaveRequest):
	"""
	POST /fourCM/keys/save
	Option 1: save the key entered in the UI to the .env file and apply it to the environment variable immediately
	Never print the key value in logs
	"""
	#env_data = _read_env_file()

	if req.claude_key.strip():
		#env_data["ANTHROPIC_API_KEY"] = req.claude_key.strip()
		os.environ["ANTHROPIC_API_KEY"] = req.claude_key.strip()

	if req.grok_key.strip():
		#env_data["XAI_API_KEY"] = req.grok_key.strip()
		os.environ["XAI_API_KEY"] = req.grok_key.strip()

	#_write_env_file(env_data)
	#logger.info("API keys saved to .env (values not logged)")
	logger.info("API keys loaded into memory (not written to disk)")

	return {
		"saved": True,
		"storage": "memory_only",
		"claude_set": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
        "grok_set": bool(os.environ.get("XAI_API_KEY", "").strip()),
		#"claude_set": bool(env_data.get("ANTHROPIC_API_KEY")),
		#"grok_set":   bool(env_data.get("XAI_API_KEY")),
	}


@router.delete("/keys/clear")
async def keys_clear():
	"""
	DELETE /fourCM/keys/clear
	Remove the API key from the .env file and also clear it from the environment variable
	"""
	env_data = _read_env_file()
	env_data.pop("ANTHROPIC_API_KEY", None)
	env_data.pop("XAI_API_KEY", None)
	_write_env_file(env_data)

	os.environ.pop("ANTHROPIC_API_KEY", None)
	os.environ.pop("XAI_API_KEY", None)

	logger.info("API keys cleared from .env and environment")
	return {"cleared": True}


# ── Runtime Config (USE_EXTERNAL_API toggle) ──────────────────────────────────

class ConfigRequest(BaseModel):
	use_external_api: bool

@router.get("/config")
async def get_config():
	"""
	GET /fourCM/config
	현재 USE_EXTERNAL_API 상태 반환.
	"""
	_env = os.environ.get("USE_EXTERNAL_API", "true").lower()
	use_external = _env not in ("false", "0", "no")
	return {
		"use_external_api": use_external,
		"internal_llm_base_url": INTERNAL_LLM_BASE_URL,
		"internal_llm_model": INTERNAL_LLM_MODEL,
	}

@router.post("/config")
async def set_config(req: ConfigRequest):
	"""
	POST /fourCM/config
	{ "use_external_api": false } → switch to local/internal LLM mode
	{ "use_external_api": true  } → return to external API mode (Claude + Grok)

	Modify os.environ directly → applied immediately to all subsequent requests.
	Also save to the .env file so the setting persists after container restart.
	"""
	os.environ["USE_EXTERNAL_API"] = "true" if req.use_external_api else "false"

	# store it to .env 
	env_data = _read_env_file()
	env_data["USE_EXTERNAL_API"] = "true" if req.use_external_api else "false"
	_write_env_file(env_data)

	logger.info(f"USE_EXTERNAL_API set to {req.use_external_api}")
	return {
		"use_external_api": req.use_external_api,
		"internal_llm_base_url": INTERNAL_LLM_BASE_URL,
		"internal_llm_model": INTERNAL_LLM_MODEL,
	}
