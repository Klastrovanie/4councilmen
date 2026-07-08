# KlastroHeron Decision Intelligence — Prudentia (4CM)
### On-Premise Multi-Agent Decision Layer

**First Public Release of 4CM:** 2026-03-09  
**First Public Release of Prudentia Branch:** 2026-05-19  
**Last Updated:** 2026-07-02

> *"Four models that never agree — until they do."*

**Prudentia** (Latin: *wisdom*) is the first commercial release of the **4 Councilmen Model (4CM)**: a document & web grounded advisory system in which four structurally independent AI agents — each locked to a single orthogonal perspective (surveillance, ethics, audit, transparency) — analyze the same question without ever negotiating. A fixed semantic judge measures convergence. Agreement is never assumed; when it happens despite the agents' hostility to one another's logic, **the phone rings.**

Built for organizations that must demonstrate **independent review, human oversight support, and documented decision trails** for AI-assisted decisions — deployable **fully on-premises and air-gapped**, with no data ever leaving your infrastructure.

**Version 2.1.1** · PhD Dissertation, 2011 · Hybrid Prototype, 2026
📖 **[Full User Manual (PDF)](https://klastrovanie-shared-temp.s3.fr-par.scw.cloud/manuals/4CM_Prudentia_User_Manual.pdf)**

---

## Why single-model review fails

Every production AI system today is optimized for convergence: trained on human feedback, aligned to consensus, tuned for agreement. That makes a single model — or an ensemble of similar models — structurally blind to its own correlated failure modes. Regulators have noticed: high-risk AI frameworks increasingly require *effective oversight*, *risk identification from multiple angles*, and *traceable decision records*.

4CM addresses this at the architecture level:

- **Radical orthogonality.** Each agent operates from a fixed, mutually hostile perspective. No shared context, no negotiation channel.
- **Non-convergent by design.** Agents never move toward consensus. They remain fixed at their extreme positions on a torus field; the *convergence point* moves toward them only as their independent conclusions and reasoning align. A singularity is recognized only when that point reaches the peak region all four extremes share.
- **Independent judging.** Semantic scoring is performed by a single fixed judge model, separate from the agents, ensuring cross-round score comparability.
- **Evidence-grounded.** Conclusions are grounded in uploaded documents (PDF, DOCX, XLSX, CSV, PPTX, images) and web search — not free-floating generation.
- **Human-in-command.** The fifth response (common conclusion) is a recommendation, never a binding determination. The operator may reject, modify, or override any output at any time.

## How It Works

```
Your Question + Documents + Web Search
        ↓
┌─────────────────────────────────────┐
│  SENTINEL       │  ETHIKOS          │
│  (surveillance) │  (ethics)         │
├─────────────────┼───────────────────┤
│  AUDITOR        │  HERALD           │
│  (audit)        │  (transparency)   │
└─────────────────────────────────────┘
        ↓
   Fixed Semantic Judge
        ↓
   Singularity? → The phone rang.
   No convergence? → The phone did not ring.
```

Under the hood: agent responses are embedded (Sentence Transformer, local-only), scored by the judge on conclusion- and reasoning-convergence, and mapped onto the torus field **f(x,y) = (x⁴+y⁴)·e^(−(x⁸+y⁸))** from the 2011 dissertation. Four peaks, one deep central well; a closed consensus ring becomes topologically possible only when all four opposed perspectives are present. See the [User Manual](https://klastrovanie-shared-temp.s3.fr-par.scw.cloud/manuals/4CM_Prudentia_User_Manual.pdf), §2.3, for the full mathematics.

## Provider Modes & Data Flow

| Mode | Agents | Judge | Data flow |
|---|---|---|---|
| **All Local LLM** | Local model ×4 (Ollama / vLLM / llama.cpp) | Same local model (fixed) | **No data leaves your infrastructure** |
| Round-robin | Claude → Grok → Claude → Grok | Grok (fixed) | Anthropic + xAI |
| All Claude | Claude ×4 | Grok (fixed) | Anthropic + xAI |
| All Grok | Grok ×4 | Grok (fixed) | xAI |
| Per-agent | User-assigned per agent | Grok (fixed) | Selected external providers |

In every mode: embeddings, torus computation, and decision logs stay on the customer's servers. **Nothing is ever transmitted to Klastrovanie.**

## Security

- **Memory-only API keys (v2.0.4+).** Keys entered via the UI live in process memory only — never written to disk, never printed to logs, cleared on container restart.
- **Secret manager integration.** Docker Secrets, HashiCorp Vault, AWS/Azure/GCP secret managers, or host env vars for persistent enterprise key management.
- **Ephemeral uploads.** Documents are parsed server-side and auto-deleted after TTL (default 1 hour); no volume mount.
- **Internal-only networking.** Port 80 via nginx, no public exposure required. Administrator security checklist in Manual §5.4.

## Regulatory Posture

**As deployed, the Software itself** (per Manual §6 — not legal advice):

| Requirement | Status |
|---|---|
| EU AI Act — GPAI provider obligations (Art. 51–56) | Not applicable — no foundation model provided |
| EU AI Act — High-risk obligations (Art. 6–49) | Not applicable as deployed — limited-risk classification |
| EU AI Act — Transparency (Art. 50) | Satisfied — UI discloses AI-generated content |
| GDPR | Internal LLM Mode: all data stays in customer infrastructure; Klastrovanie is neither controller nor processor. External API Mode: customer establishes DPAs with providers |
| Cyber Resilience Act (2024/2847) | Committed: vulnerability disclosure channel, SBOM delivery, security patch notifications within support SLA |

⚠️ If deployed for Annex III high-risk purposes (employment decisions, credit scoring, etc.), the deploying organization assumes the corresponding provider/deployer obligations under Article 25.

**How Prudentia supports *your* governance obligations:**

| Framework | How Prudentia helps |
|---|---|
| EU AI Act Art. 9 (Risk management) | Four orthogonal reviewers stress-test the same decision; divergence surfaces unresolved risk before action |
| EU AI Act Art. 12 (Record-keeping) | Structured decision records: per-agent positions, judge scores, grounding evidence, exportable PDF reports |
| EU AI Act Art. 14 (Human oversight) | The singularity score is a calibrated trust signal; "the phone did not ring" explicitly tells the overseer *not* to rely on the AI conclusion |
| ISO/IEC 42001 · NIST AI RMF | Repeatable, logged, multi-perspective review process that slots into an AI management system as an operational control |

## Features

- 4 orthogonal AI agents — local LLM, Claude + Grok hybrid, or per-agent assignment
- Real-time web search (external API mode)
- Document upload as grounding evidence — PDF, DOCX, XLSX, CSV, PPTX, TXT, images (25 MB)
- Multi-round (1–5), blind or context-aware
- Structured PDF decision reports
- Korean/English UI and output · 4 themes
- Custom scenarios via plain `.txt` files — no code, no restart
- One-command Docker deployment · BYOK

## Quick Start

**Requirements:** Docker Engine 20+ with Compose V2 · Ollama (local mode) or API keys (Claude/Grok) · for local LLM: GPU with 8 GB+ VRAM recommended

```bash
git clone https://github.com/Klastrovanie/4councilmen.git
cd 4councilmen
git checkout Prudentia

cp the.env.example .env
# Edit .env — key variables:
#   INTERNAL_LLM_BASE_URL   (Ollama: http://host.docker.internal:11434/v1)
#   INTERNAL_LLM_MODEL      (e.g., llama3.1:8b — as shown in `ollama list`)

docker compose up -d --build
```

Access at **http://localhost**. API keys (if using external providers) are entered via the UI — memory-only, never written to disk. Full configuration reference: Manual §3–4.

## Custom Scenarios

Scenarios live in `angry_agents/` (Docker volume). Edit `.txt` files directly — changes apply immediately, no restart:

```
angry_agents/your_scenario/
├── members.txt   # "1: SENTINEL" per line
├── title.txt     # Display title
├── query.txt     # Default question
├── risk.txt      # normal / high
└── 1.txt … 4.txt # System prompt per agent
```

Pre-built scenarios (pharma, outbreak, government, M&A, Oppenheimer, whistleblower, key talent) are created on first run. Scenarios marked **[MOCK RESEARCH SCENARIO]** are for research and simulation only — not validated for clinical, financial, legal, or regulatory decision-making. Model selection for specialized domains (including judge/agent policy fit) is an operator responsibility; see Manual §8.

## API

FastAPI backend with SSE streaming. Key endpoints: `GET /health`, `GET /fourCM/agents`, `POST /fourCM` (SSE analysis stream), `POST /fourCM/validate` (orthogonality check), `POST /fourCM/upload`, key management under `/fourCM/keys/*` (values never returned). Full reference: Manual §7.

## Architecture

```
docker compose up
├── nginx :80
│   ├── React UI (Prudentia, FourCM.tsx)
│   └── FastAPI SSE proxy
└── backend :8000 (internal)
    ├── fourCM_router.py     SSE streaming + key management + scenarios
    ├── fourth_cm_engine.py  Semantic judge
    ├── orthogonal_agents.py Agent calls (local / Claude / Grok)
    ├── torus_math.py        Convergence f(x,y), singularity detection
    └── document_parser.py   Upload parsing + context injection
```

## Product Suite

Prudentia is the first release in the Klastrovanie advisory suite (more coming soon):

| Product | Latin | Role |
|---|---|---|
| **Prudentia** | *wisdom* | Document & web grounded advisory (this) |

---

## Licensing

Prudentia / 4CM is **dual-licensed**.

**Open source (AGPL-3.0).** You may use, modify, and deploy this software — including commercially — under the terms of the GNU Affero General Public License v3. Note that AGPL's copyleft applies to network use: if you offer this software, or a product that incorporates it, as a service over a network, you must make the complete corresponding source of that combined work available under AGPL-3.0.

**Commercial license.** For organizations that cannot meet AGPL obligations — e.g., embedding 4CM in proprietary products or SaaS platforms without source disclosure, or on-premise deployment free of copyleft obligations — Klastrovanie Co., Ltd. offers commercial and OEM licenses, including deployment support and integration assistance.

📧 Commercial licensing & OEM inquiries: **contact@klastrovanie.com**

This code is released to encourage collaboration across AI systems — not competition. The goal is shared solutions, not shared resources.

Copyright © 2026 Klastrovanie Co., Ltd. All rights reserved.

## Live Demo 
https://klastrovanie.github.io/4councilmen/   

## Theory & Research Background

Original theory: 4 Councilmen Model (4CM) — Author's PhD Dissertation, 2011
ACM Digital Library: https://dl.acm.org/doi/book/10.5555/2231522

The design premise: think of a politician, a religious leader, an environmental activist, and a labor organizer at the same table. They disagree by identity and conviction. Yet when all four, independently, point to the same conclusion — that conclusion is worth listening to.

Imagine a phone booth designed never to ring. No one has the number. In 4CM, that is the phone. When four orthogonal minds converge — it rings. When the question is trivial or not ready — it doesn't. *The phone did not ring* — that is also an answer.

> *"No model. No weights. No storage. Born from non-convergence. Already gone."*

---

*Prudentia is a multi-perspective decision-support and simulation tool. It does not automate any decision; all final decisions rest with the human operator. All agent outputs and the fifth response are AI-generated content. It is not a certification, legal advice, or a substitute for the human and organizational measures required by applicable AI regulation.*
