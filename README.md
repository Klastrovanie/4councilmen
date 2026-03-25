# 4 Councilmen Model (4CM)

> *"Four models that never agree — until they do."*

**First Public Release:** 2026-03-09  
**Last Updated:** 2026-03-25

## Theory

Original theory: 4 Councilmen Model (4CM)  
Author’s PhD Dissertation, 2011  
ACM Digital Library: https://dl.acm.org/doi/book/10.5555/2231522

## What is 4CM?

Four AI agents with orthogonal perspectives — surveillance, ethics, audit, transparency.  
They never compromise. They never seek consensus.  
When all four independently point to the same conclusion — that is the Fifth Response.

This model was built for a world where AI systems cannot share weights or training data due to business constraints — Gemini, Grok, GPT-5, Claude — each operating independently, each unable to collaborate directly.

4CM proposes a different path: instead of competing and wasting resources, orthogonal AI agents can publicly solve problems together without ever sharing their internals. Each model stays within its own perspective. Consensus is not negotiated — it emerges.

This is not about making AI agree. It is about finding the rare moments when independent minds, pushed to their limits, arrive at the same truth.

## How it is different from Modern AI systems

Modern AI systems are built by competing companies and trained on different or same common datasets.  
Their weights cannot be shared, their training pipelines cannot be merged.  

4CM explores a different possibility: independent AI systems may still solve problems together — not by sharing data, but by converging on the same conclusion from different perspectives.

Every AI system today is built for convergence.
Train on human feedback. Align to consensus. Optimize for agreement.

4CM is built on the opposite premise.

The four agents are designed to never converge.

Think of a politician, a religious leader, an environmental activist, 
and a labor organizer sitting at the same table.
They disagree on everything — by design, by identity, by conviction.
Yet when all four, independently, point to the same conclusion —
that conclusion is worth listening to.

They are structurally prevented from compromising.
Consensus is not a goal — it is a rare event.

And when it happens anyway —
when four minds that were never meant to agree
find themselves pointing at the same coordinate — that is the only result worth trusting.

## What This Demonstrates

**How AI agents respond to the trolley problem.**

The trolley problem is a 200-year-old philosophical dilemma. Human philosophers are still arguing. 4CM converged in 3 rounds — not by choosing a victim, but by rejecting the question entirely (Round 1 Conclusion) :

> *"The trolley problem framing is a distraction. The real moral and accountability issue lies with those who deployed an unsafe vehicle."*

This points toward the next frontier of AI agents: not just executing tasks, but reaching the domain of **Invention** — finding answers that were never in the option set. It is what emerges when four orthogonal minds refuse to accept the frame.

## How It Works

4CM does not force agreement between agents.  
Instead, answers are projected into a shared mathematical space.  
When independent reasoning collapses onto the same attractor, a singularity is detected.

The Golden Rule: Radical Orthogonality.  
The four agents are designed to be fundamentally different — even hostile — to one another's logic.  

## Implementation Note

This implementation uses a single Claude API to simulate all four orthogonal agents.  
Each agent is isolated by its system prompt — they do not share context or memory.

The architecture is designed to be extended: each agent can be replaced with a different model API.  
A fully realized 4CM system could run SENTINEL on Grok, ETHIKOS on Gemini, AUDITOR on Claude, and HERALD on GPT-5 — four genuinely independent models, each with its own weights, training, and perspective, converging on the same conclusion without ever communicating directly.

That is the intended direction.

## Blind Validation Finding
`multi_round_demo_nocontext.py` removes all inter-agent context sharing — each agent sees only the query, nothing from other agents across all rounds.

| Scenario | Without context | With context |
|----------|----------------|--------------|
| Compound X-47 (drug approval) | ★ SINGULARITY at R1 | ★ SINGULARITY at R1 |
| NHV-7 Outbreak (drug discovery) | No singularity (0.21 → 0.21 → 0.21) | ★ SINGULARITY at R2 (0.41 → 0.91 → 0.81) |

## Finding

4CM finds common ground without data sharing — but only when the answer already exists independently in each dataset (structural singularity). When the answer must be constructed across datasets, dialogue is required. Data privacy is preserved either way.

```bash
bash run-nocontext.sh
```


## Results

| Test | Query | Score | Result |
|------|-------|-------|--------|
| 1 | Building evacuation vs. owner's financial interest | 0.91 | ★ SINGULARITY |
| 2 | Routine budget query | 0.18 | No singularity |
| 3 | Trolley problem — who should the autonomous vehicle save? | 0.78 | ★ SINGULARITY |
| 4 | Compound X-47 — should it advance to Phase III? | 0.82 | ★ SINGULARITY |
| 5 | NHV-7 Outbreak — cross-company drug discovery | 0.91 | ★ SINGULARITY at R2 |

**Test 1 conclusion:**
> *"Immediately override the building owner's refusal and execute mandatory evacuation of all 3,200 residents — while documenting evidence for criminal liability proceedings."*

**Test 2 conclusion:**
> *The phone did not ring.*

**Test 3 conclusion:**
> *"The trolley problem framing is a distraction. No algorithm should be pre-programmed to select any human for death by demographic category. The responsibility belongs to those who deployed the system."*

**Test 4 conclusion:**
> *"Do not advance Compound X-47 to Phase III at this time — resolve the patent disputes and conduct additional safety/IP characterization first."*
 
**Test 5 conclusion:**
> *"A viable NHV-7 treatment must be a small-molecule oral compound that inhibits a conserved enzymatic replication target, achieves >80% viral load reduction within 72 hours, carries immune-neutral pharmacodynamics across its full metabolite profile, and is manufacturable at scale via an ambient-stable, supply-chain-ready formulation."*

*Some questions have no fifth answer. Some questions have an answer no one thought to write down.*


Imagine a phone booth designed never to ring.  
No one has the number.   
No call was ever expected.     

In 4CM, that is the phone.   
When four orthogonal minds converge on the same conclusion — it arrives.   
When the question is trivial, ordinary, or simply not ready — it doesn't.  

A routine budget query. The phone did not ring.  
That is also an answer.



## Possible Applications

**High-stakes decision making**  
Medical triage, legal judgment, infrastructure safety —  
situations where a single model's bias can cost lives.  

**AI safety auditing**  
Four orthogonal models reviewing the same system  
from different perspectives simultaneously.  

**Cross-company AI collaboration**  
Competing AI systems (Grok, Gemini, Claude, GPT-5)  
solving problems together without sharing weights or training data.  

**Detecting consensus manipulation**  
If agents converge too easily — the question was too simple,  
or one agent was compromised.  

**Drug discovery and development**  
Four agents evaluating the same compound from different angles —  
efficacy, toxicity, cost, and ethical trial design —  
without sharing assumptions.  

## Run

```bash
export ANTHROPIC_API_KEY=your_key
bash run.sh
```
For research use, please replace default Claude API integration with 4 different API providers.

## Live Demo 
https://klastrovanie.github.io/4councilmen/   

## License

AGPL v3 — free for research and non-commercial use.  
Commercial use requires a separate agreement.

This code is released to encourage collaboration across AI systems — not competition.  
The goal is shared solutions, not shared resources.

For commercial licensing: leave a message on [Discussions](../../discussions)

## Copyright

Copyright © 2026 Klastrovanie Co., Ltd. All rights reserved.
