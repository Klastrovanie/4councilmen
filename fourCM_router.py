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
	GROK_MODEL,
	CLAUDE_MODEL,
	XAI_API_URL,
	ANTHROPIC_API_URL,
	INTERNAL_LLM_BASE_URL,
	INTERNAL_LLM_MODEL,
)
from fourth_cm_engine import EmbeddingEngine, semantic_compare, SEMANTIC_JUDGE_SYSTEM
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
	positions = [(0.85, 0.85), (-0.85, 0.85), (-0.85, -0.85), (0.85, -0.85)]

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
		return base

	# Apply UI overrides (only for agents that were edited)
	override_map = {a.id: a for a in req.agents}
	for agent in base:
		slot = int(agent.agent_id.split("_")[1]) + 1  # "agent_0" → 1
		if slot in override_map:
			ov = override_map[slot]
			agent.name = ov.name
			agent.system_prompt = ov.prompt

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

	try:
		# Setup
		torus      = TorusField()
		judge      = JudgeFunction(torus, convergence_threshold=0.5)
		constraint = ConstraintLayer(torus, drift_tolerance=0.3)
		embedder   = EmbeddingEngine(use_transformer=True)
		agents     = build_agents(req)

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

				agent_response_list.append({
					"name": agent.name,
					"provider": result["provider"],
					"text": text,
					"status": result["status"],
					"error": result["error"],
				})

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

			# ── Step 4: Torus judgment ─────────────────────────────────────
			judgment = judge.compute_convergence_from_semantic(
				semantic_score, positions,
				conclusion_score=conclusion_score,
				reasoning_score=reasoning_score,
			)
			is_singularity = judgment["is_singularity"] and same_direction

			if is_singularity and first_singularity_round is None:
				first_singularity_round = round_num
				final_conclusion = common_conclusion

			# ── Yield round_complete ───────────────────────────────────────
			yield sse("round_complete", {
				"round": round_num,
				"providers": provider_map_by_name,
				"responses": agent_response_list,
				"conclusion_score": conclusion_score,
				"reasoning_score": reasoning_score,
				"semantic_score": semantic_score,
				"ratio": judgment["singularity_ratio"],
				"torus_coord": list(judgment["convergence_point"]),
				"is_singularity": is_singularity,
				"conclusion": common_conclusion,
				"weakest_link": semantic.get("weakest_link", ""),
				"analysis": semantic.get("convergence_analysis", ""),
			})

			# ── Context for next round (blind mode skips) ──────────────────
			if req.no_context:
				prev_context = ""
			else:
				prev_context = "\n".join([
					f"- [{a.name}]: {responses[a.agent_id]}" for a in agents
				])

			await asyncio.sleep(0.1)

		# ── Summary ────────────────────────────────────────────────────────
		yield sse("summary", {
			"first_singularity_round": first_singularity_round,
			"conclusion": final_conclusion,
			"phone_rang": first_singularity_round is not None,
		})

		yield sse_done()

	except Exception as e:
		logger.exception("stream_fourCM fatal error")
		yield sse("error", {"message": str(e)})
		yield sse_done()

	finally:
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
	Ask Grok if the proposed agent prompt is sufficiently orthogonal
	to the other three. Returns {ok: bool, message: str}.
	"""
	grok_key = request.headers.get("X-Grok-Key", "").strip() or os.environ.get("XAI_API_KEY", "")
	if not grok_key:
		raise HTTPException(401, "xAI API key required for validation")

	others_text = "\n\n".join([
		f"[Agent {i+1}]: {p[:400]}"
		for i, p in enumerate(req.other_prompts)
	])

	system_prompt = (
		"You are an orthogonality validator for the 4 Councilmen Model (4CM). "
		"Your job: check whether a proposed agent is genuinely orthogonal to "
		"the other three agents. "
		"Orthogonal means: the proposed agent approaches problems from a FUNDAMENTALLY "
		"DIFFERENT perspective that cannot be subsumed by any of the others. "
		"Same topic, different angle is fine. Same angle is NOT orthogonal. "
		"Be strict. Return ONLY a JSON object: "
		'{"ok": true/false, "message": "one sentence explanation"}'
	)

	user_message = (
		f"Proposed new agent:\n"
		f"Name: {req.name}\n"
		f"Prompt: {req.prompt[:600]}\n\n"
		f"Existing three agents:\n{others_text}\n\n"
		f"Is the proposed agent sufficiently orthogonal to the other three?"
	)

	headers = {
		"Authorization": f"Bearer {grok_key}",
		"Content-Type": "application/json",
	}
	payload = {
		"model": GROK_MODEL,
		"max_tokens": 200,
		"temperature": 0.2,
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user",   "content": user_message},
		],
	}

	try:
		resp = await asyncio.to_thread(
			lambda: requests.post(XAI_API_URL, headers=headers, json=payload, timeout=30)
		)
		resp.raise_for_status()
		body = resp.json()
		raw = (body.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

		# strip markdown fences if any
		if "```" in raw:
			parts = raw.split("```")
			raw = parts[1] if len(parts) >= 2 else raw
			if raw.startswith("json"):
				raw = raw[4:]
		raw = raw.strip()

		result = json.loads(raw)
		return {"ok": result.get("ok", False), "message": result.get("message", "")}

	except json.JSONDecodeError:
		return {"ok": True, "message": "Validation parsing issue — proceeding with caution."}
	except Exception as e:
		logger.error(f"Validation error: {e}")
		raise HTTPException(500, f"Validation failed: {str(e)}")


@router.get("/agents/{agent_set}")
async def get_agent_set(agent_set: str):
	"""
	GET /fourCM/agents/{agent_set}
	Return agent names + prompt previews for the UI.
	"""
	try:
		agents = load_agents_from_files(agent_set)
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
	Update a single agent's name and prompt.
	"""
	folder = ANGRY_AGENTS_ROOT / set_id
	if not folder.exists():
		raise HTTPException(404, f"Scenario '{set_id}' not found")

	if agent_idx < 1 or agent_idx > 4:
		raise HTTPException(400, "agent_idx must be 1-4")

	# Update prompt file
	(folder / f"{agent_idx}.txt").write_text(req.prompt, encoding="utf-8")

	# Update members.txt
	members_file = folder / "members.txt"
	lines = []
	if members_file.exists():
		for line in members_file.read_text(encoding="utf-8").splitlines():
			if ":" in line:
				idx_str, _ = line.split(":", 1)
				if idx_str.strip() == str(agent_idx):
					lines.append(f"{agent_idx}: {req.name}")
				else:
					lines.append(line)
			else:
				lines.append(line)
	else:
		lines = [f"{agent_idx}: {req.name}"]

	members_file.write_text("\n".join(lines), encoding="utf-8")

	# Clear Korean translation cache for this scenario
	for ko_file in folder.glob("*.ko.txt"):
		ko_file.unlink()

	return {"updated": {"set": set_id, "agent": agent_idx, "name": req.name}}

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
