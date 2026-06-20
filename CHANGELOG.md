# Changelog

All notable changes to the 4 Councilmen Model (4CM) are documented here.

## [2.1.0] — 2026-06-20

A consolidated release covering theory alignment, local/internal LLM operation,
governance triage, audit logging, scenario provisioning, and a set of UI/PDF
report fixes. The folded-in work previously tracked as `v2.1.0-patched`,
`2.1.0-patched-2`, `2.1.0-patched-4`, and `Patch 5` is now part of this release.

### Theory alignment — constants restored to dissertation values

Implementation constants now match the original 2011 PhD dissertation.
**Behavior and outputs are unchanged** — the singularity ratio still resolves to
≈ 1.62 and all demo results reproduce identically. This is a provenance pass, not
a functional change.

- **Agent placement coordinates → (±0.8, ±0.8)** — restored the four orthogonal
  agent positions from `±0.85` to the canonical `±0.8` across all scenario
  definitions (`orthogonal_agents.py`, `torus_math.py`, `fourCM_router.py`).
  `±0.8` encodes how far each agent sits at its extreme; the four extremes are
  what force the consensus ring to exist. Max neutrality drift bound aligned to
  `0.8` for consistency.
- **Singularity peak constant → (1/4)^(1/8)** — the on-axis peak coordinate,
  previously hard-coded as an approximation (`0.83`, later `0.841`), is now the
  exact analytic value `(1/4)**(1/8) ≈ 0.8409`. Derivation: along the diagonal
  `f(x,x) = 2·x⁴·e^(−2x⁸)`; solving `df/dx = 0` gives `x⁸ = 1/4`. The constant is
  now derived rather than a magic number.
- Updated inline comments to state the analytic origin of the peak and removed an
  inaccurate ring-radius note.
- No change to gate logic, thresholds, convergence behavior, or API. Singularity
  ratio: `f(peak,peak) / threshold ≈ 0.6065 / 0.3743 ≈ 1.62` (unchanged).
  Existing reports and demo screenshots remain valid.

### Local / internal LLM persistence and agent overrides

- UI-edited agent names and prompts are persisted to
  `angry_agents/_saved_agent_overrides.json` and re-applied on later runs and
  after container restart. Default scenario files remain **immutable**; overrides
  are layered on top at load time.
- `POST /fourCM/validate-and-save/{set_id}/agent/{agent_idx}` no longer modifies
  `angry_agents/{set_id}/{idx}.txt` or `members.txt`.
- `PUT /fourCM/scenario/{set_id}/agent/{agent_idx}` stores name/prompt edits only
  in the override JSON.
- Responses now include `storage: local_override_json` and
  `base_files_modified: false` for clarity.
- Added `GET/POST/DELETE /fourCM/saved-agents` to list, save, and clear local
  agent overrides.

### Governance — judge-LLM intent triage

- Added a judge-LLM intent triage step before `/fourCM` execution.
  - External API mode uses Grok as the governance judge.
  - Local LLM mode uses the configured internal/local LLM as the governance judge.
- Classification fields: simple-prompt detection, operational-decision intent,
  GDPR stage, high-risk status, and human-review requirement.
- Forces `risk_level=high` when the judge classifies the request as high-risk or
  as requiring accountable human review.
- Added an SSE `intent_triage` event; the triage result is persisted into the
  JSON run log.
- Reworked `/fourCM/validate` so agent prompt validation and rewriting are done by
  the same judge LLM.
- Updated `/fourCM/risk-policy` to clarify GDPR stage 1–2 transparency handling
  versus stage 3–4 human-review gates.
- Added a lightweight prompt gate so trivial questions do not consume a 4CM
  high-stakes decision workflow.

### Convergence state classifier

- Separates the raw torus `ratio` from the user-facing convergence state.
- Adds `ratio_signal` (binary): 1 for a raw singularity signal, 0 otherwise.
- Adds judge-driven `convergence_state`: `singularity`, `partial_convergence`,
  `dominant_compatible_proposal`, or `no_singularity`.
- Adds coalition / dissent / proposal fields to SSE, JSON logs, the summary
  payload, and the report UI.
- Prevents 2-of-3 agreement from being shown as full Singularity even when the
  raw torus ratio reaches the old max value.

### Audit logging

- Added immutable per-run JSON audit logs under `/app/logs` (host `./logs`).
  Each run records: `run_id`, `started_at`/`finished_at`, full `request`
  (including the user's custom query verbatim), `upload_files` manifest,
  `intent_triage` decision (incl. `human_review_required`), per-round agent
  responses, judge outputs, `summary`, and `errors`.
- Designed for post-hoc audit and traceability; supports EU AI Act
  record-keeping (Art. 12) for high-risk-classified runs.
- **Privacy note:** custom queries are stored in plaintext on disk. Operators are
  responsible for log retention limits, access control, and advising users not to
  paste personal/sensitive data into custom queries.

### Scenarios and provisioning

- Added scenario provisioning scripts: `setup_new_scenarios.sh` (agent sets:
  `members.txt`, `1–4.txt` prompts) and `setup_scenario_files.sh`
  (`title.txt` / `query.txt` / `risk.txt`).
- `entrypoint.sh` seeds `angry_agents/` on first boot and idempotently backfills
  missing scenario sets on restart; default files remain immutable (UI edits
  persist separately via `_saved_agent_overrides.json`).
- Added a GDPR tier-guidance endpoint and a GDPR transparency scenario for lower
  tier 1–2 cases where maximum transparency/proportionality is preferred over
  extreme escalation.
- Scenario inventory: government, key talent, M&A, Oppenheimer, NHV-7 outbreak,
  pharma (Compound X-47, Pharma R&D evidence gate), whistleblower, plus
  business-domain sets: GDPR transparency, enterprise security, legal contract,
  finance governance.
- Mock/research scenarios are explicitly framed with a `[MOCK RESEARCH SCENARIO]`
  marker in both the scenario `query` and the agent prompts.

### UI and PDF report fixes

- **Fixed:** opening Settings → PDF crashed the app to a blank white page. The
  PDF tab referenced an out-of-scope `visibleResults`; corrected to the
  component's `results` prop.
- **Fixed:** in the PDF report, the Consensus Result box used colored
  (green/amber) backgrounds but kept dark-theme light text, rendering the text
  nearly invisible in Monokai / Dark Classic. Added explicit text colors and
  inheritance to `.consensus` (mirrors the existing `.decision-brief` rules).
- **Fixed:** the report's "Judge" field was hard-coded to "Grok fixed" regardless
  of mode. It now reflects the actual judge — "Local LLM (fixed)" in local mode,
  "Grok (fixed)" in external mode.

### Documentation and known limitations

- Intent triage and agent reasoning depend on the judge/agent model's
  instruction-following ability. The triage classifier receives only the scenario
  `query`, `agent_set`, and `risk_level` — not the agent prompts.
- Capable models (e.g., Grok) honor research/mock framing and classify scope
  correctly (e.g., recognizing that pharmaceutical R&D is **not** EU AI Act
  Annex III high-risk). Weaker local models (e.g., Qwen 2.5 14B) may ignore the
  mock marker, over-escalate at triage, or refuse to answer at the agent stage.
- This is a model-capability limitation, not a system defect, and is **not**
  worked around by forcing routing outcomes in code. Selecting a sufficiently
  capable judge/agent model is an operator responsibility. Where local resources
  allow, agents may run locally while triage/judge uses a more capable model.

### Deployment notes (local LLM)

- `.env` changes require recreating the container — `docker restart` does **not**
  re-read `env_file`. Use `docker compose up -d --force-recreate backend`, then
  verify with `docker exec 4cm_backend printenv | grep INTERNAL_LLM`.
- `INTERNAL_LLM_BASE_URL`: use `http://host.docker.internal:11434/v1` on Docker
  Desktop, `http://172.17.0.1:11434/v1` on plain Linux Docker.
- The host's Ollama must bind `0.0.0.0` (`OLLAMA_HOST=0.0.0.0:11434`); the default
  `127.0.0.1` is unreachable from inside the container.
- Set `OLLAMA_KEEP_ALIVE=-1` to keep the model resident and avoid per-call cold
  starts across the triage → agents → judge sequence.

### Tested with

- Ollama on remote EC2 (`g4dn`, Tesla T4 16 GB) via SSH tunnel, and local Mac.
- Model: `qwen2.5:14b` (Q4_K_M). All four agents converged at Round 1 on the
  Whistleblower scenario.
- Sample reports (both Grok and local-LLM runs, across Light / Monokai /
  Dark Classic themes and all convergence states) are kept under
  `docs/sample-reports/`.

## [2.0.4] — 2026-06-13

### Security — memory-only API key storage

- API keys are no longer written to disk (`.env`).
- Keys entered via the UI persist in memory only and are cleared on container
  restart. For persistent keys, use your organization's secret manager or
  `env_file`.
- Added `.env.example` for deployment guidance.

## [2.0.3] — 2026-06-10

### Added — local / internal LLM support

4CM supports any OpenAI-compatible LLM endpoint as an alternative to the external
Claude + Grok APIs, enabling air-gapped, private-cloud, or enterprise deployments
where external API calls are restricted.

- `All Local LLM` provider-routing mode in the sidebar.
- Each of the four agents can be assigned a different LLM endpoint (URL + API key).
- Per-agent extra payload fields (temperature, top_p, model, etc.).
- `Use External API` toggle in the API Keys modal — syncs with the backend
  `USE_EXTERNAL_API` env var at runtime.
- All endpoint and payload settings saved in browser localStorage.

**Backend**

- `GET/POST /fourCM/config` — runtime toggle of `USE_EXTERNAL_API` without
  container restart.
- `is_external_api()` — dynamic runtime check (replaces a module-load-time
  constant).
- `call_internal_llm()` — OpenAI-compatible `/v1/chat/completions` caller with
  retry logic.
- Per-agent payload merge via the `INTERNAL_LLM_EXTRA_PAYLOAD` env var.
- `use_external_api`, `local_payloads`, `local_endpoints` added to
  `FourCMRequest`. Local LLM mode skips the external API key requirement.

## [2.0.2] — prior release

See git commit history.
