# Changelog

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
