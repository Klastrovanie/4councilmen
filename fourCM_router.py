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
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── .env 파일 경로 ──────────────────────────────────────────────────────────
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
)
from fourth_cm_engine import EmbeddingEngine, semantic_compare, SEMANTIC_JUDGE_SYSTEM

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

class ValidateAgentRequest(BaseModel):
    name: str
    prompt: str
    other_prompts: List[str]        # the other 3 agents' current prompts

# ── Provider mapping ─────────────────────────────────────────────────────────

def provider_map_for_round(round_num: int, risk_level: str) -> List[str]:
    """
    risk=high  → all Grok (avoids Claude safety refusals on medical/pharma scenarios)
    risk=normal → round-robin Claude+Grok
    """
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
    """numpy 타입을 Python 기본 타입으로 변환"""
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

# ── Core streaming generator ──────────────────────────────────────────────────

async def stream_fourCM(req: FourCMRequest, claude_key: str, grok_key: str):
    """
    Main async generator — yields SSE events for each step of 4CM.
    """
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

        prev_context = ""
        first_singularity_round: Optional[int] = None
        final_conclusion: Optional[str] = None

        for round_num in range(1, req.n_rounds + 1):
            providers = provider_map_for_round(round_num, req.risk_level)
            provider_map_by_name: Dict[str, str] = {}

            # ── Step 1: Agent calls ────────────────────────────────────────
            responses: Dict[str, str] = {}
            agent_response_list = []

            for i, agent in enumerate(agents):
                prov = providers[i]
                provider_map_by_name[agent.name] = prov

                # yield "calling" signal
                yield sse("agent_start", {
                    "round": round_num,
                    "name": agent.name,
                    "provider": prov,
                })

                # Run in thread pool (blocking HTTP call)
                try:
                    text = await asyncio.to_thread(
                        simulate_orthogonal_response,
                        agent,
                        req.query,
                        prev_context,
                        prov,
                    )
                except Exception as e:
                    logger.error(f"Agent {agent.name} error: {e}")
                    text = f"[{agent.name} error: {str(e)[:100]}]"

                responses[agent.agent_id] = text
                agent.response_history.append(text)

                # yield completed response
                yield sse("agent", {
                    "round": round_num,
                    "name": agent.name,
                    "provider": prov,
                    "text": text,
                })

                agent_response_list.append({
                    "name": agent.name,
                    "provider": prov,
                    "text": text,
                })

                # tiny pause to avoid thundering herd
                await asyncio.sleep(0.05)

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
    Grok으로 텍스트를 한국어로 번역.
    context: 번역 힌트 (e.g. "scenario title", "query")
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
    GET /fourCM/agents?lang=en  — 영어 (기본)
    GET /fourCM/agents?lang=ko  — 한국어 (번역본 캐시 사용)

    한국어 요청 시:
    - title.ko.txt, query.ko.txt 있으면 바로 반환
    - 없으면 Grok으로 번역 후 *.ko.txt 저장 → 반환
    """
    if not ANGRY_AGENTS_ROOT.exists():
        return {"sets": []}

    # 번역 필요 시 Grok 키 추출
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

        # 한국어 번역본 읽기 또는 생성
        async def get_text(base_fname: str, context: str) -> str:
            ko_fname = base_fname.replace(".txt", ".ko.txt")
            if lang == "ko":
                # 캐시된 번역본 있으면 바로 반환
                ko_file = d / ko_fname
                if ko_file.exists():
                    return ko_file.read_text(encoding="utf-8").strip()
                # 없으면 번역 후 저장
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


def _write_env_file(data: Dict[str, str]) -> None:
    """dict → .env 파일 저장"""
    lines = []
    for k, v in data.items():
        lines.append(f"{k}={v}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


@router.get("/keys/status")
async def keys_status():
    """
    GET /fourCM/keys/status
    현재 API 키 설정 여부 반환 (값은 절대 반환하지 않음)
    
    옵션 2: .env에 키가 있으면 자동으로 환경변수에 로드
    """
    env_data = _read_env_file()

    # .env에 키가 있으면 환경변수에 로드 (옵션 2)
    if env_data.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = env_data["ANTHROPIC_API_KEY"]
    if env_data.get("XAI_API_KEY"):
        os.environ["XAI_API_KEY"] = env_data["XAI_API_KEY"]

    claude_set = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    grok_set   = bool(os.environ.get("XAI_API_KEY", "").strip())

    return {
        "claude_set": claude_set,
        "grok_set": grok_set,
        "source": "env_file" if env_data.get("ANTHROPIC_API_KEY") else "environment",
    }


@router.post("/keys/save")
async def keys_save(req: KeySaveRequest):
    """
    POST /fourCM/keys/save
    옵션 1: UI에서 입력한 키를 .env 파일에 저장 + 환경변수에 즉시 반영
    키값은 로그에 절대 출력하지 않음
    """
    env_data = _read_env_file()

    if req.claude_key.strip():
        env_data["ANTHROPIC_API_KEY"] = req.claude_key.strip()
        os.environ["ANTHROPIC_API_KEY"] = req.claude_key.strip()

    if req.grok_key.strip():
        env_data["XAI_API_KEY"] = req.grok_key.strip()
        os.environ["XAI_API_KEY"] = req.grok_key.strip()

    _write_env_file(env_data)
    logger.info("API keys saved to .env (values not logged)")

    return {
        "saved": True,
        "claude_set": bool(env_data.get("ANTHROPIC_API_KEY")),
        "grok_set":   bool(env_data.get("XAI_API_KEY")),
    }


@router.delete("/keys/clear")
async def keys_clear():
    """
    DELETE /fourCM/keys/clear
    .env 파일에서 API 키 삭제 + 환경변수에서도 제거
    """
    env_data = _read_env_file()
    env_data.pop("ANTHROPIC_API_KEY", None)
    env_data.pop("XAI_API_KEY", None)
    _write_env_file(env_data)

    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("XAI_API_KEY", None)

    logger.info("API keys cleared from .env and environment")
    return {"cleared": True}
