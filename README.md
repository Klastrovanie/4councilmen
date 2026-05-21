# 4 Councilmen Model (4CM) — Prudentia

> *"Four models that never agree — until they do."*

**Prudentia** is the first commercial release of the 4 Councilmen Model (4CM) — a document & web grounded advisory system powered by four orthogonal AI agents.

PhD Dissertation, 2011 · Hybrid Prototype, 2026

**First Public Release of 4CM:** 2026-03-09  
**First Public Release of Prudentia Branch:** 2026-05-19  
**Last Updated:** 2026-05-21

---

## What is Prudentia?

**Prudentia** (Latin: *wisdom, prudence*) is a **Document & Web Grounded Advisory** system:

- **Document-grounded** — upload PDFs, spreadsheets, and documents as evidence
- **Web-grounded** — real-time web search via Grok
- **Evidence-based** — grounded in real data, not hallucinated
- **Advisory** — four orthogonal AI agents debate, a judge decides

Four AI agents — each locked to a single extreme perspective — independently analyze your question. A fixed Grok judge measures convergence. When all four axes point the same direction despite their orthogonality: **the phone rings.**

---

## How It Works

```
Your Question + Documents + Web Search
        ↓
┌─────────────────────────────────────┐
│  SENTINEL   │  ETHIKOS              │
│  (security) │  (ethics)             │
├─────────────┼───────────────────────┤
│  AUDITOR    │  HERALD               │
│  (finance)  │  (anti-secrecy)       │
└─────────────────────────────────────┘
        ↓
   Grok Judge (fixed, semantic)
        ↓
   Singularity? → The phone rang.
   No convergence? → The phone did not ring.
```

**Singularity** = all four orthogonal axes converge on a compatible conclusion despite different reasoning paths.

---

## Features

- **4 orthogonal AI agents** — Claude + Grok hybrid, round-robin per round
- **Web search** — Grok real-time search per agent
- **Document upload** — PDF, DOCX, XLSX, images as grounding evidence
- **Multi-round** — 1–5 rounds, blind or context-aware
- **PDF export** — structured report, not a screenshot
- **Korean/English** — UI and output language toggle
- **4 themes** — Light, Light Warm, Dark Classic, Monokai
- **Custom agents** — edit via `.txt` files, no code required
- **Docker-based** — one command deployment

---

## Quick Start

### Requirements
- Docker + Docker Compose
- Anthropic API key (Claude)
- xAI API key (Grok), for API-based web search, since Claude API does not support API-based web search.

### Run

```bash
git clone https://github.com/Klastrovanie/4councilmen.git
cd 4councilmen
git checkout Prudentia

# Create empty .env (keys entered via UI, however, it is already provided)
touch .env

# Start
docker compose up --build -d

# Shutdown
docker compose down
```

Open: **http://localhost/4councilmen/**

API keys are entered via the UI — never stored to disk.

---

## Custom Agents

No code required. Edit `.txt` files directly:

```
angry_agents/
└── your_scenario/
    ├── members.txt     # "1: AGENT_NAME" per line
    ├── title.txt       # Scenario title
    ├── query.txt       # Default question
    ├── risk.txt        # "normal" or "high"
    ├── 1.txt           # Agent 1 system prompt
    ├── 2.txt           # Agent 2 system prompt
    ├── 3.txt           # Agent 3 system prompt
    └── 4.txt           # Agent 4 system prompt
```

Changes apply immediately — no restart required.

`risk: high` → all agents use Grok (bypasses Claude safety filters for medical/pharma scenarios).

---

## Default Scenarios

| Scenario | Risk | Expected |
|----------|------|----------|
| Building evacuation | normal | Singularity |
| Routine budget | normal | No singularity |
| Compound X-47 (pharma) | high | No singularity |
| NHV-7 outbreak | high | Singularity |
| M&A — Should we acquire? | normal | Depends |
| Oppenheimer — Drop the bomb? | normal | Depends |
| Whistleblower | normal | Singularity |
| Key talent — fight or let go? | normal | Depends |
| **Custom** | user-defined | File upload + web search enabled |

---

## Architecture

```
docker compose up
├── nginx :80
│   ├── /4councilmen/        → React UI (dist/)
│   └── /4councilmen/fourCM  → FastAPI SSE proxy
└── backend :8000 (internal)
    ├── main.py              FastAPI entry
    ├── fourCM_router.py     SSE streaming + agent API
    ├── fourth_cm_engine.py  Semantic judge (Grok fixed)
    ├── orthogonal_agents.py Claude + Grok agent calls
    ├── torus_math.py        Convergence math (f(x,y))
    └── document_parser.py   File upload + parsing
```

---

## Product Suite

Prudentia is the first release in the Klastrovanie advisory suite: (more will be coming soon)

| Product | Latin | Role |
|---------|-------|------|
| **Prudentia** | *wisdom* | Document & web grounded advisory (this) |  

---

## License

**AGPL v3**  
Free for research and non-commercial use.  
Commercial use requires a separate agreement.

This code is released to encourage collaboration across AI systems — not competition.  
The goal is shared solutions, not shared resources.

For commercial on-premise deployment without AGPL obligations, contact: [contact@klastrovanie.com](mailto:contact@klastrovanie.com)



---

## Theory

Original theory: 4 Councilmen Model (4CM)
Author's PhD Dissertation, 2011
ACM Digital Library: [https://dl.acm.org/doi/book/10.5555/2231522](https://dl.acm.org/doi/book/10.5555/2231522)

> *"No model. No weights. No storage. Born from non-convergence. Already gone."*

