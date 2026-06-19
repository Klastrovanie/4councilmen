# Changelog

## [2.1.0] — 2026-06-19

### Changed — Theory Alignment: Constants Restored to Dissertation Values

This release aligns the implementation constants with the original 2011 PhD
dissertation. **Behavior and outputs are unchanged** — the singularity ratio
still resolves to ≈ 1.62 and all demo results reproduce identically. This is a
provenance/correctness pass, not a functional change.

**Agent placement coordinates → (±0.8, ±0.8)**
- Restored the four orthogonal agent positions from `±0.85` to the dissertation's
  canonical `±0.8` across all scenario definitions
  (`orthogonal_agents.py`, `torus_math.py`, `fourCM_router.py`).
- `±0.8` is not an arbitrary tuning value: it encodes how far each agent sits at
  its extreme. The four extremes are what force the consensus ring to exist.
- Max neutrality drift bound aligned to `0.8` for consistency.

**Singularity peak constant → (1/4)**(1/8)**
- The on-axis peak coordinate was previously hard-coded as an approximation
  (`0.83`, later `0.841`). It is now the exact analytic value `(1/4)**(1/8) ≈ 0.8409`.
- Derivation: along the diagonal, `f(x,x) = 2·x⁴·e^(−2x⁸)`; solving `df/dx = 0`
  gives `x⁸ = 1/4`, i.e. `x = (1/4)**(1/8)`.
- The constant is now derived rather than a magic number, matching the
  dissertation's figure exactly.

**Docs/comments**
- Updated inline comments to state the analytic origin of the peak and to remove
  an inaccurate ring-radius note.

### Notes
- No change to the gate logic, thresholds, convergence behavior, or API.
- Singularity ratio: `f(peak,peak) / threshold ≈ 0.6065 / 0.3743 ≈ 1.62` (unchanged).
- Existing reports and demo screenshots remain valid.

## [2.0.4] — 2026-06-13

### Security: Memory-Only API Key Storage

- API keys are no longer written to disk (.env file)
- Keys entered via UI persist in memory only; cleared on container restart
- For persistent keys, use your organization's secret manager or env_file
- Added .env.example for deployment guidance

## [2.0.3] — 2026-06-10

### Added

**Local / Internal LLM Support**

4CM now supports any OpenAI-compatible LLM endpoint as an alternative to the external Claude + Grok APIs.
This enables deployment in air-gapped environments, private cloud infrastructure, or pharmaceutical/enterprise networks
where external API calls are restricted.

- `All Local LLM` mode — new provider routing option in the sidebar
- Each of the four agents can be assigned a different LLM endpoint (URL + API key)
- Per-agent extra payload fields (temperature, top_p, model, etc.) configurable per agent
- `Use External API` toggle in the API Keys modal — syncs with backend `USE_EXTERNAL_API` env var at runtime
- All endpoint and payload settings saved in browser localStorage

**Backend**

- `GET/POST /fourCM/config` — runtime toggle of `USE_EXTERNAL_API` without container restart
- `is_external_api()` — dynamic runtime check (replaces module-load-time constant)
- `call_internal_llm()` — OpenAI-compatible `/v1/chat/completions` caller with retry logic
- Per-agent payload merge via `INTERNAL_LLM_EXTRA_PAYLOAD` environment variable
- `use_external_api`, `local_payloads`, `local_endpoints` fields added to `FourCMRequest`
- Local LLM mode skips external API key requirement

**Screenshots**
- `custom_LLM_screenshots/` — LLM Endpoints, Payloads, and Local LLM mode in action

**Tested with**

- Ollama (local Mac, remote EC2 g4dn.2xlarge via SSH tunnel)
- Model: `qwen2.5:14b` (Q4_K_M)
- All four agents converged at Round 1 on the Whistleblower scenario

---

## [2.0.2] — prior release

See git commit history.
