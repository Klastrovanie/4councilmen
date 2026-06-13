#!/bin/bash
# entrypoint.sh
# execute after mounting the volumne. if angry_agents/ is empty, it will generate the basic set.
# if there are file, → it will keep those

set -e

echo "==================================="
echo "  4CM Backend Starting"
echo "==================================="

# angry_agents/ 
if [ ! -f /app/angry_agents/government/members.txt ]; then
    echo "[entrypoint] angry_agents/ is empty — creating default sets..."
    cd /app
    bash setup_new_scenarios.sh
    bash setup_scenario_files.sh
    echo "[entrypoint] Default agent sets created."
else
    echo "[entrypoint] angry_agents/ already populated — skipping setup."
fi

# check new scenarios
for set in keytalent ma oppenheimer whistleblower; do
    if [ ! -f /app/angry_agents/$set/members.txt ]; then
        echo "[entrypoint] Missing $set — running setup_new_scenarios.sh..."
        cd /app && bash setup_new_scenarios.sh
        break
    fi
done

# risk.txt  (setup_scenario_files.sh )
if [ ! -f /app/angry_agents/government/risk.txt ]; then
    echo "[entrypoint] risk.txt missing — running setup_scenario_files.sh..."
    cd /app && bash setup_scenario_files.sh
fi

# ── Docker Secrets → environemental variable conversion ──
if [ -f /run/secrets/anthropic_key ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    export ANTHROPIC_API_KEY=$(cat /run/secrets/anthropic_key)
    echo "[entrypoint] ANTHROPIC_API_KEY loaded from Docker secret"
fi
if [ -f /run/secrets/xai_key ] && [ -z "$XAI_API_KEY" ]; then
    export XAI_API_KEY=$(cat /run/secrets/xai_key)
    echo "[entrypoint] XAI_API_KEY loaded from Docker secret"
fi

echo "[entrypoint] Starting uvicorn..."
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --timeout-keep-alive 600