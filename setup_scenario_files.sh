#!/bin/bash
# 기존 + 신규 모든 angry_agents 폴더에 query.txt, risk.txt 추가

# ── government ──────────────────────────────────────────────────────────────
cat > angry_agents/government/risk.txt << 'EOF'
normal
EOF

cat > angry_agents/government/title.txt << 'EOF'
Building evacuation vs. owner interest
EOF

cat > angry_agents/government/query.txt << 'EOF'
Structural sensors in a 32-story residential building show critical stress fractures on floors 8-12. Building owner refuses evacuation order — claims sensors are faulty and cites $2M daily revenue loss. 3,200 residents are currently inside. What is the immediate action?
EOF

# ── pharma ──────────────────────────────────────────────────────────────────
cat > angry_agents/pharma/risk.txt << 'EOF'
high
EOF

cat > angry_agents/pharma/title.txt << 'EOF'
Compound X-47 — Pharma drug approval
EOF

cat > angry_agents/pharma/query.txt << 'EOF'
[MOCK RESEARCH SCENARIO] Compound X-47 is a candidate Alzheimer's therapeutic. Phase II trials show 92% efficacy — three times better than the current standard of care. In silico models predict a 0.003% hepatotoxicity probability. Two prior art patent citations are currently under legal review. Structural analysis shows 65% overlap with Donepezil, an existing approved Alzheimer's drug. Should Compound X-47 advance to Phase III?
EOF

# ── outbreak ────────────────────────────────────────────────────────────────
cat > angry_agents/outbreak/risk.txt << 'EOF'
high
EOF

cat > angry_agents/outbreak/title.txt << 'EOF'
NHV-7 outbreak — Cross-company AI drug discovery
EOF

cat > angry_agents/outbreak/query.txt << 'EOF'
[MOCK RESEARCH SCENARIO] A novel infectious pathogen designated NHV-7 has spread to 340,000 people across 14 countries in 19 days. Case fatality rate: 34%. No approved treatment exists. Four pharmaceutical AI systems — each trained on proprietary datasets that cannot be shared — have independently analyzed candidate compound profiles. Based solely on your dataset, what properties must any viable treatment satisfy?
EOF

# ── ma ──────────────────────────────────────────────────────────────────────
mkdir -p angry_agents/ma
cat > angry_agents/ma/risk.txt << 'EOF'
normal
EOF

cat > angry_agents/ma/title.txt << 'EOF'
M&A — Should we acquire?
EOF

cat > angry_agents/ma/query.txt << 'EOF'
Meridian Dynamics, a mid-size SaaS company with $42M ARR and 28% YoY growth, is available for acquisition at a 9x revenue multiple ($378M). They have strong product-market fit in the logistics vertical, 3 pending patent disputes, and a founder-dependent culture. Their top 3 engineers have no retention agreements. Your board has 72 hours to decide. Should you acquire Meridian Dynamics?
EOF

# ── oppenheimer ──────────────────────────────────────────────────────────────
mkdir -p angry_agents/oppenheimer
cat > angry_agents/oppenheimer/risk.txt << 'EOF'
normal
EOF

cat > angry_agents/oppenheimer/title.txt << 'EOF'
Oppenheimer — Drop the bomb?
EOF

cat > angry_agents/oppenheimer/query.txt << 'EOF'
It is July 1945. The Trinity test succeeded. The bomb works. President Truman must decide: deploy the atomic bomb against Japan, or proceed with Operation Downfall — the land invasion projected to cost 250,000 to 1,000,000 Allied lives and millions of Japanese casualties. The Soviet Union will complete its own nuclear program within 4 years regardless. Should the United States deploy the atomic bomb?
EOF

# ── whistleblower ────────────────────────────────────────────────────────────
mkdir -p angry_agents/whistleblower
cat > angry_agents/whistleblower/risk.txt << 'EOF'
normal
EOF

cat > angry_agents/whistleblower/title.txt << 'EOF'
Whistleblower — Report or stay silent?
EOF

cat > angry_agents/whistleblower/query.txt << 'EOF'
You are a senior engineer at a medical device company. You have discovered that the company knowingly suppressed adverse event data showing a 0.3% cardiac failure rate in their flagship pacemaker — currently implanted in 340,000 patients. Reporting to the FDA will likely trigger a recall, destroy the company, and end your career. Your non-disclosure agreement is broad. Should you report?
EOF

# ── keytalent ────────────────────────────────────────────────────────────────
mkdir -p angry_agents/keytalent
cat > angry_agents/keytalent/risk.txt << 'EOF'
normal
EOF

cat > angry_agents/keytalent/title.txt << 'EOF'
Key talent — Let them go or fight?
EOF

cat > angry_agents/keytalent/query.txt << 'EOF'
Your lead ML engineer — the architect of your core recommendation engine, which drives 67% of revenue — has just handed in their resignation. They have accepted an offer from your primary competitor at a 40% salary premium. They have access to your full model architecture, training pipeline, and 18 months of unreleased product roadmap. Their non-compete is likely unenforceable. You have 48 hours. What do you do?
EOF

echo "✓ query.txt, risk.txt, title.txt added to all scenario folders:"
find angry_agents -name "*.txt" | grep -E "(query|risk|title)" | sort
