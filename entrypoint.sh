#!/bin/bash
# entrypoint.sh
# 볼륨 마운트 후 실행 — angry_agents/가 비어있으면 기본 세트 자동 생성
# 파일이 이미 있으면 스킵 → 사용자 수정 내용 유지

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

echo "[entrypoint] Starting uvicorn..."
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --timeout-keep-alive 600