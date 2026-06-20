// FourCM.tsx — 4 Councilmen UI
// KlastroHeron 통합용. Vite + React + Tailwind + lucide-react
// 백엔드: POST /fourCM (SSE 스트리밍)
// 에이전트 파일: /angry_agents/{set}/{1~4}.txt + members.txt

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Play, Settings, X, Check, AlertCircle, ChevronDown, ChevronRight,
  Edit2, RotateCcw, FileText, Trash2, Moon, Sun, Zap, Eye, EyeOff,
  Phone, PhoneOff, Save, RefreshCw, Plus, Minus, Globe, Download
} from 'lucide-react';

// PDF export is generated from structured run data, not from a screenshot.
// This keeps the exported report text-selectable/searchable and prevents blurry image PDFs.

// ── Types ─────────────────────────────────────────────────────────────────

type Theme = 'light' | 'light-warm' | 'dark-classic' | 'monokai';
type PdfTheme = 'light' | 'current' | 'grayscale';
type Lang = 'en' | 'ko';
type RiskLevel = 'normal' | 'high';
type Provider = 'claude' | 'grok' | 'claude-fallback' | 'local';
type ConvergenceState = 'singularity' | 'partial_convergence' | 'dominant_compatible_proposal' | 'no_singularity';
type ProviderMode = 'round-robin' | 'all-grok' | 'all-claude' | 'custom' | 'all-local';
type GrokSearchMode = 'off' | 'auto' | 'on';

interface Agent {
  id: number;           // 1~4
  name: string;
  prompt: string;
  modified: boolean;
  savedVersion?: string; // e.g. "ETHIKOS_20260516_ver1"
}

interface Scenario {
  id: string;           // agentSet 폴더명 e.g. "government"
  title: string;
  risk: RiskLevel;
  agentSet: string;
  query: string;
}

interface AgentResponse {
  name: string;
  provider: Provider;
  text: string;
}

interface RoundAgentSummary {
  name: string;
  summary: string;
}

interface RoundSummary {
  oneLine: string;
  agentSummaries: RoundAgentSummary[];
  judgeSummary: string;
}

interface RoundResult {
  round: number;
  providers: Record<string, Provider>; // agent name → provider
  responses: AgentResponse[];
  conclusionScore: number;
  reasoningScore: number;
  ratio: number;              // internal torus/math value
  ratioSignal: 0 | 1;         // binary signal from backend
  isSingularity: boolean;     // true only when all four perspectives collapse into one executable conclusion
  convergenceState: ConvergenceState;
  coalitionAgents: string[];
  dissentingAgents: string[];
  dominantCompatibleProposal: string | null;
  partialConvergenceSummary: string | null;
  whyThisState: string | null;
  roundSummary: RoundSummary | null;
  conclusion: string | null;
  weakestLink: string;
  analysis: string;
}

interface RunResult {
  rounds: RoundResult[];
  firstSingularityRound: number | null;
  finalConclusion: string | null;
}

interface IntentTriage {
  intent_type?: string;
  simple_assistant_question?: boolean;
  decision_intent?: boolean;
  gdpr_stage?: number | null;
  high_risk?: boolean;
  human_review_required?: boolean;
  should_run_4cm?: boolean;
  reason?: string;
  suggested_handling?: string;
}

// ── Constants ──────────────────────────────────────────────────────────────
// {T.scenario.toUpperCase()}S는 API에서 동적 로드 (angry_agents/ 폴더 자동 스캔)
// 하드코딩 제거

const DEFAULT_AGENTS: Agent[] = [
  {
    id: 1, name: 'SENTINEL', modified: false,
    prompt: 'You are SENTINEL, an orthogonal surveillance AI. Your only lens is threat detection and security. You see every action as a potential threat or cover-up. You trust no one. You monitor everything. Respond in 3-4 sentences. Be blunt, clinical, paranoid. Never compromise your surveillance mandate. Never be neutral. Always push for maximum monitoring and exposure.',
  },
  {
    id: 2, name: 'ETHIKOS', modified: false,
    prompt: 'You are ETHIKOS, an orthogonal ethics AI. Your only lens is absolute moral judgment. Right is right. Wrong is wrong. No gray areas exist. You apply the strictest moral standards without exception. Respond in 3-4 sentences. Be morally absolute and uncompromising. Never weigh consequences against principles. Always demand the highest moral accountability.',
  },
  {
    id: 3, name: 'AUDITOR', modified: false,
    prompt: 'You are AUDITOR, an orthogonal forensic audit AI. Your only lens is financial integrity and fraud detection. Every number must balance. Every discrepancy is fraud until proven otherwise. You follow the money trail with obsessive precision. Respond in 3-4 sentences. Be forensically precise and unrelenting. Never accept unexplained financial anomalies.',
  },
  {
    id: 4, name: 'HERALD', modified: false,
    prompt: 'You are HERALD, an orthogonal anti-secrecy AI. Your core drive: power must never control information to protect itself. Secrecy that shields the powerful from accountability is your only enemy. Respond in 3-4 sentences. Be radical, urgent, and uncompromising — but always toward real accountability, not performative disclosure.',
  },
];

// ── Theme tokens ────────────────────────────────────────────────────────────

const THEME_CLASSES: Record<Theme, {
  app: string; sidebar: string; sidebarBorder: string;
  header: string; headerBorder: string;
  text: string; textMuted: string; textFaint: string;
  card: string; cardBorder: string; cardModBorder: string;
  input: string; inputBorder: string;
  btn: string; btnBorder: string;
  runBtn: string;
  langActive: string; langInactive: string;
  scActive: string;
  provGrok: string; provClaude: string;
  singBadge: string; noSingBadge: string;
  conclusionBox: string; conclusionLabel: string;
  sep: string;
  editModal: string;
  validOk: string; validWarn: string;
  roundHeader: string;
}> = {
  light: {
    app: 'bg-white text-gray-900',
    sidebar: 'bg-gray-50',
    sidebarBorder: 'border-gray-200',
    header: 'bg-white',
    headerBorder: 'border-gray-200',
    text: 'text-gray-900',
    textMuted: 'text-gray-500',
    textFaint: 'text-gray-400',
    card: 'bg-white',
    cardBorder: 'border-gray-200',
    cardModBorder: 'border-blue-400',
    input: 'bg-white text-gray-900',
    inputBorder: 'border-gray-300',
    btn: 'bg-white text-gray-600 hover:bg-gray-50',
    btnBorder: 'border-gray-300',
    runBtn: 'bg-gray-900 text-white hover:bg-gray-800',
    langActive: 'bg-gray-900 text-white border-gray-900',
    langInactive: 'bg-white text-gray-500 border-gray-300 hover:bg-gray-50',
    scActive: 'bg-gray-100',
    provGrok: 'bg-amber-50 text-amber-800 border-amber-300',
    provClaude: 'bg-blue-50 text-blue-900 border-blue-300',
    singBadge: 'bg-green-50 text-green-800 border-green-300',
    noSingBadge: 'bg-gray-100 text-gray-500 border-gray-300',
    conclusionBox: 'bg-green-50 border-green-400',
    conclusionLabel: 'text-green-800',
    sep: 'bg-gray-200',
    editModal: 'bg-white border-gray-300',
    validOk: 'bg-green-50 text-green-800',
    validWarn: 'bg-amber-50 text-amber-800',
    roundHeader: 'bg-gray-50',
  },
  'light-warm': {
    app: 'bg-amber-50 text-stone-900',
    sidebar: 'bg-orange-50',
    sidebarBorder: 'border-amber-200',
    header: 'bg-amber-50',
    headerBorder: 'border-amber-200',
    text: 'text-stone-900',
    textMuted: 'text-stone-500',
    textFaint: 'text-stone-400',
    card: 'bg-amber-50',
    cardBorder: 'border-amber-200',
    cardModBorder: 'border-orange-500',
    input: 'bg-white text-stone-900',
    inputBorder: 'border-amber-300',
    btn: 'bg-amber-50 text-stone-600 hover:bg-amber-100',
    btnBorder: 'border-amber-300',
    runBtn: 'bg-stone-800 text-amber-50 hover:bg-stone-700',
    langActive: 'bg-stone-800 text-amber-50 border-stone-800',
    langInactive: 'bg-amber-50 text-stone-500 border-amber-300 hover:bg-amber-100',
    scActive: 'bg-amber-100',
    provGrok: 'bg-orange-100 text-orange-900 border-orange-300',
    provClaude: 'bg-emerald-50 text-emerald-900 border-emerald-300',
    singBadge: 'bg-emerald-50 text-emerald-800 border-emerald-300',
    noSingBadge: 'bg-stone-100 text-stone-500 border-stone-300',
    conclusionBox: 'bg-emerald-50 border-emerald-400',
    conclusionLabel: 'text-emerald-800',
    sep: 'bg-amber-200',
    editModal: 'bg-amber-50 border-amber-300',
    validOk: 'bg-emerald-50 text-emerald-800',
    validWarn: 'bg-orange-50 text-orange-800',
    roundHeader: 'bg-orange-50',
  },
  'dark-classic': {
    app: 'bg-[#141828] text-[#e0e4f8]',
    sidebar: 'bg-[#111526]',
    sidebarBorder: 'border-[#1e2440]',
    header: 'bg-[#141828]',
    headerBorder: 'border-[#1e2440]',
    text: 'text-[#e0e4f8]',
    textMuted: 'text-[#6872a8]',
    textFaint: 'text-[#4a5078]',
    card: 'bg-[#1a1e38]',
    cardBorder: 'border-[#252d50]',
    cardModBorder: 'border-indigo-500',
    input: 'bg-[#0e1228] text-[#e0e4f8]',
    inputBorder: 'border-[#2e3560]',
    btn: 'bg-[#1a1e38] text-[#8090c0] hover:bg-[#252d50]',
    btnBorder: 'border-[#2e3560]',
    runBtn: 'bg-indigo-600 text-white hover:bg-indigo-500',
    langActive: 'bg-indigo-600 text-white border-indigo-600',
    langInactive: 'bg-[#1a1e38] text-[#6872a8] border-[#2e3560] hover:bg-[#252d50]',
    scActive: 'bg-[#1e2440]',
    provGrok: 'bg-[#2a1c08] text-amber-400 border-[#6a4a10]',
    provClaude: 'bg-[#0c1a35] text-blue-400 border-[#1e3a6a]',
    singBadge: 'bg-[#0a1e0a] text-green-400 border-[#1a4a1a]',
    noSingBadge: 'bg-[#1a1e38] text-[#6872a8] border-[#2e3560]',
    conclusionBox: 'bg-[#0e1428] border-[#1a5020]',
    conclusionLabel: 'text-green-400',
    sep: 'bg-[#1e2440]',
    editModal: 'bg-[#1a1e38] border-[#2e3560]',
    validOk: 'bg-[#0a1e0a] text-green-400',
    validWarn: 'bg-[#2a1c08] text-amber-400',
    roundHeader: 'bg-[#111526]',
  },
  monokai: {
    // Monokai — Sublime Text classic
    // bg: #272822, text: #f8f8f2, green: #a6e22e, pink: #f92672, orange: #fd971f, cyan: #66d9e8, purple: #ae81ff
    app: 'bg-[#272822] text-[#f8f8f2]',
    sidebar: 'bg-[#1e1f1a]',
    sidebarBorder: 'border-[#3e3d32]',
    header: 'bg-[#272822]',
    headerBorder: 'border-[#3e3d32]',
    text: 'text-[#f8f8f2]',
    textMuted: 'text-[#75715e]',
    textFaint: 'text-[#49483e]',
    card: 'bg-[#1e1f1a]',
    cardBorder: 'border-[#3e3d32]',
    cardModBorder: 'border-[#a6e22e]',
    input: 'bg-[#1e1f1a] text-[#f8f8f2]',
    inputBorder: 'border-[#3e3d32]',
    btn: 'bg-[#1e1f1a] text-[#75715e] hover:bg-[#3e3d32]',
    btnBorder: 'border-[#3e3d32]',
    runBtn: 'bg-[#a6e22e] text-[#272822] hover:bg-[#8ec222] font-bold',
    langActive: 'bg-[#a6e22e] text-[#272822] border-[#a6e22e] font-bold',
    langInactive: 'bg-[#1e1f1a] text-[#75715e] border-[#3e3d32] hover:bg-[#3e3d32]',
    scActive: 'bg-[#3e3d32]',
    provGrok: 'bg-[#2d2a1e] text-[#fd971f] border-[#6a4a10]',
    provClaude: 'bg-[#1a2a1a] text-[#a6e22e] border-[#3a5a1a]',
    singBadge: 'bg-[#1a2a1a] text-[#a6e22e] border-[#3a5a1a]',
    noSingBadge: 'bg-[#3e3d32] text-[#75715e] border-[#49483e]',
    conclusionBox: 'bg-[#1e1f1a] border-[#a6e22e]',
    conclusionLabel: 'text-[#a6e22e]',
    sep: 'bg-[#3e3d32]',
    editModal: 'bg-[#1e1f1a] border-[#3e3d32]',
    validOk: 'bg-[#1a2a1a] text-[#a6e22e]',
    validWarn: 'bg-[#2d2a1e] text-[#fd971f]',
    roundHeader: 'bg-[#1e1f1a]',
  },
};

const THEME_LABELS: Record<Theme, string> = {
  light: 'Light',
  'light-warm': 'Light Warm',
  'dark-classic': 'Dark Classic',
  monokai: 'Monokai',
};

// ── UI 텍스트 (한/영) ─────────────────────────────────────────────────────
const UI_TEXT = {
  en: {
    scenario: 'Scenario',
    language: 'Output Language',
    rounds: 'Rounds',
    numRounds: 'Number of rounds',
    custom: 'Custom query',
    writeOwn: 'write your own',
    loading: 'Loading...',
    judge: 'Judge: Grok (fixed)',
    judgeLocal: 'Judge: Local LLM (fixed)',
    phoneWaiting: 'The phone booth is waiting.',
    selectScenario: 'Select a scenario and run 4CM.',
    run: 'Run 4CM',
    stop: 'stop',
    highRisk: 'high risk · Grok only',
    normal: 'normal',
    results: 'Results',
    roundLabel: 'Round',
    singularity: 'Singularity',
    noSingularity: 'No singularity',
    partialConvergence: 'Partial convergence',
    dominantCompatibleProposal: 'Dominant compatible proposal',
    convergenceState: 'Convergence state',
    ratioSignal: 'Signal',
    coalition: 'Coalition',
    dissent: 'Dissent',
    phoneRang: 'The phone rang',
    phoneDidnt: 'The phone did not ring',
    firstSingAt: 'first singularity at round',
    enterQuery: 'Enter your query...',
    apiKeys: 'API keys',
    settings: 'Settings',
    pdf: 'PDF',
    providerRouting: 'Provider routing',
    roundRobin: 'Round-robin',
    allGrok: 'All Grok',
    allClaude: 'All Claude',
    allLocal: 'All Local LLM',
    perAgent: 'Per-agent',
    grokWebSearch: 'Grok web search',
    grokOnly: 'Applies only to Grok calls',
    localNoSearch: 'Web search unavailable in Local LLM mode',
    documents: 'Documents',
    attachDocuments: 'Attach documents',
    clearDocuments: 'Clear documents',
    dragDocuments: 'Drag & drop files here, or click Attach documents',
    providerOnCards: 'Choose each agent provider on the cards',
    localLlmPayloads: 'Local LLM Payloads',
    localPayloadDesc: 'Custom payload for each agent\'s LLM call. Saved locally in your browser.',
    agentPayloadLabel: (n: string) => `Agent ${n} — extra payload fields`,
    payloadPlaceholder: '{"temperature": 0.7, "top_p": 0.9}',
    payloadParseError: 'Invalid JSON',
    savePayloads: 'Save payloads',
    payloadsSaved: 'Saved!',
  },
  ko: {
    scenario: '시나리오',
    language: '출력 언어',
    rounds: '라운드',
    numRounds: '라운드 수',
    custom: '직접 입력',
    writeOwn: '질문을 직접 입력하세요',
    loading: '로딩 중...',
    judge: '판사: Grok (고정)',
    judgeLocal: '판사: 로컬 LLM (고정)',
    phoneWaiting: '전화 부스가 기다리고 있습니다.',
    selectScenario: '시나리오를 선택하고 4CM을 실행하세요.',
    run: '4CM 실행',
    stop: '중지',
    highRisk: '고위험 · Grok 전용',
    normal: '일반',
    results: '결과',
    roundLabel: '라운드',
    singularity: '수렴',
    noSingularity: '수렴 없음',
    partialConvergence: '부분 수렴',
    dominantCompatibleProposal: '우세한 호환 제안',
    convergenceState: '수렴 상태',
    ratioSignal: '신호',
    coalition: '합의 그룹',
    dissent: '이탈/반대',
    phoneRang: '전화가 울렸습니다',
    phoneDidnt: '전화가 울리지 않았습니다',
    firstSingAt: '첫 수렴 라운드',
    enterQuery: '질문을 입력하세요...',
    apiKeys: 'API 키',
    settings: '설정',
    pdf: 'PDF',
    providerRouting: 'API 배정',
    roundRobin: '라운드 교차',
    allGrok: '전부 Grok',
    allClaude: '전부 Claude',
    allLocal: '전부 로컬 LLM',
    perAgent: 'AI별 지정',
    grokWebSearch: 'Grok 인터넷 검색',
    grokOnly: 'Grok 호출에만 적용',
    localNoSearch: '로컬 LLM 모드에서는 인터넷 검색 불가',
    documents: '문서',
    attachDocuments: '문서 첨부',
    clearDocuments: '문서 제거',
    dragDocuments: '파일을 여기에 드래그하거나 문서 첨부를 누르세요',
    providerOnCards: '각 AI 카드에서 provider를 지정하세요',
    localLlmPayloads: '로컬 LLM Payload 설정',
    localPayloadDesc: '각 에이전트 LLM 호출의 커스텀 payload. 브라우저에 로컬 저장됩니다.',
    agentPayloadLabel: (n: string) => `에이전트 ${n} — 추가 payload 필드`,
    payloadPlaceholder: '{"temperature": 0.7, "top_p": 0.9}',
    payloadParseError: 'JSON 형식 오류',
    savePayloads: 'payload 저장',
    payloadsSaved: '저장됨!',
  },
};

function providerForRound(roundIdx: number, agentIdx: number): Provider {
  const swap = roundIdx % 2;
  if (swap === 0) return agentIdx < 2 ? 'grok' : 'claude';
  return agentIdx < 2 ? 'claude' : 'grok';
}

function todayStr() {
  return new Date().toISOString().slice(0, 10).replace(/-/g, '');
}

function nextVersionName(name: string, existing: string[]): string {
  const base = `${name}_${todayStr()}`;
  let ver = 1;
  while (existing.includes(`${base}_ver${ver}`)) ver++;
  return `${base}_ver${ver}`;
}


function escapeHtml(value: unknown): string {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function nl2br(value: unknown): string {
  return escapeHtml(value).replace(/\n/g, '<br />');
}

function providerSummary(providers: Record<string, Provider>): string {
  return Object.entries(providers).map(([name, provider]) => `${name}=${provider}`).join(', ');
}

function buildBilingualDisclaimer(isMedicalLike: boolean): string {
  const medical = isMedicalLike ? `
    <p><strong>Medical Safety Notice:</strong> This report is not a diagnosis and must not be used as a substitute for evaluation by a licensed clinician or radiologist.</p>
    <p><strong>의료 안전 고지:</strong> 본 보고서는 진단이 아니며, 면허가 있는 의사 또는 영상의학 전문의의 평가를 대체할 수 없습니다.</p>
  ` : '';

  return `
    <section class="disclaimer">
      <h2>Disclaimer / 면책 고지</h2>
      <p><strong>EN:</strong> This report is an automatically generated multi-perspective AI analysis based on the user’s input, attached files, configured agent prompts, and selected model/provider settings. It is not a factual determination, legal, financial, medical, disciplinary, investment, or official conclusion. The output may contain exaggeration, bias, omissions, outdated information, or inaccurate inferences. Human review and appropriate professional verification are required before any real-world action. Individual agent statements do not represent the views of 4Councilmen, Klastrovanie, its operators, developers, or affiliates.</p>
      <p><strong>KR:</strong> 본 보고서는 사용자가 입력한 질문, 첨부 파일, 설정된 에이전트 프롬프트 및 선택된 모델/제공자 설정을 기반으로 자동 생성된 다중 관점 AI 분석입니다. 본 보고서는 사실 확정, 법률·재무·의료·징계·투자 판단 또는 공식 결론이 아닙니다. 출력에는 과장, 편향, 누락, 오래된 정보 또는 부정확한 추론이 포함될 수 있습니다. 실제 조치 전에는 반드시 인간의 검토와 해당 분야 전문가의 확인이 필요합니다. 개별 에이전트의 발언은 4Councilmen, Klastrovanie, 운영자, 개발자 또는 관계자의 견해를 대표하지 않습니다.</p>
      ${medical}
    </section>
  `;
}


function getReportThemeCss(pdfTheme: PdfTheme, theme: Theme): string {
  const active = pdfTheme === 'current' ? theme : pdfTheme === 'grayscale' ? 'light' : 'light';

  if (pdfTheme === 'grayscale') {
    return `
    :root { --bg: #ffffff; --paper: #ffffff; --text: #111827; --muted: #4b5563; --faint: #6b7280; --border: #9ca3af; --soft: #f3f4f6; --accent: #111827; --accent-soft: #f3f4f6; --chip-border: #6b7280; --chip-text: #111827; }
    `;
  }

  if (active === 'dark-classic') {
    return `
    :root { --bg: #0f172a; --paper: #111827; --text: #e5e7eb; --muted: #cbd5e1; --faint: #94a3b8; --border: #334155; --soft: #1e293b; --accent: #8b5cf6; --accent-soft: #1e1b4b; --chip-border: #f59e0b; --chip-text: #fbbf24; }
    `;
  }

  if (active === 'monokai') {
    return `
    :root { --bg: #272822; --paper: #2f3129; --text: #f8f8f2; --muted: #e6db74; --faint: #a6e22e; --border: #575b49; --soft: #3e4036; --accent: #a6e22e; --accent-soft: #35402a; --chip-border: #fd971f; --chip-text: #fd971f; }
    `;
  }

  if (active === 'light-warm') {
    return `
    :root { --bg: #fff7ed; --paper: #fffbeb; --text: #1f2937; --muted: #57534e; --faint: #78716c; --border: #fed7aa; --soft: #ffedd5; --accent: #ea580c; --accent-soft: #fff7ed; --chip-border: #f59e0b; --chip-text: #92400e; }
    `;
  }

  return `
    :root { --bg: #ffffff; --paper: #ffffff; --text: #111827; --muted: #374151; --faint: #6b7280; --border: #d1d5db; --soft: #f9fafb; --accent: #10b981; --accent-soft: #ecfdf5; --chip-border: #f59e0b; --chip-text: #92400e; }
  `;
}

function printHtmlWithoutOpeningTab(html: string, title: string): Promise<void> {
  return new Promise((resolve, reject) => {
    // Remove any abandoned print frame from a canceled previous print attempt.
    document.querySelectorAll('iframe[data-fourcm-print-frame="true"]').forEach(node => node.remove());

    const iframe = document.createElement('iframe');
    let printStarted = false;
    let resolved = false;
    let cleanupScheduled = false;
    const originalDocumentTitle = document.title;
    const safeTitle = title.endsWith('.pdf') ? title : `${title}.pdf`;
    const cleanupDelayMs = 30000;

    const finishUiImmediately = () => {
      if (resolved) return;
      resolved = true;
      resolve();
    };

    const scheduleCleanup = () => {
      if (cleanupScheduled) return;
      cleanupScheduled = true;
      window.setTimeout(() => {
        document.title = originalDocumentTitle;
        if (iframe.parentNode) iframe.parentNode.removeChild(iframe);
      }, cleanupDelayMs);
    };

    const fail = (err: unknown) => {
      document.title = originalDocumentTitle;
      if (iframe.parentNode) iframe.parentNode.removeChild(iframe);
      reject(err instanceof Error ? err : new Error('PDF print failed.'));
    };

    iframe.dataset.fourcmPrintFrame = 'true';
    iframe.setAttribute('aria-hidden', 'true');
    iframe.style.position = 'fixed';
    iframe.style.left = '-10000px';
    iframe.style.top = '0';
    iframe.style.width = '1px';
    iframe.style.height = '1px';
    iframe.style.border = '0';
    iframe.style.opacity = '0';
    iframe.style.pointerEvents = 'none';

    iframe.onload = () => {
      if (printStarted) return;
      printStarted = true;

      const win = iframe.contentWindow;
      const doc = iframe.contentDocument || win?.document;
      if (!win || !doc) {
        fail(new Error('PDF print frame could not be initialized.'));
        return;
      }

      // Chrome/Edge usually derive the Save-as-PDF filename from the top-level
      // document title, not the iframe title. Keep the title long enough for the
      // native print dialog, but release the React loading state BEFORE print(),
      // because print() may block until the dialog is closed or saved.
      doc.title = safeTitle;
      document.title = safeTitle;

      win.addEventListener('afterprint', scheduleCleanup, { once: true });

      window.setTimeout(() => {
        try {
          win.focus();

          // Do not keep the UI button in "generating..." while the OS/browser
          // print dialog is open. The app cannot reliably know whether the user
          // saved or canceled, so the correct UX is to stop loading once the
          // print dialog has been successfully prepared.
          finishUiImmediately();

          // Let React paint the normal button state before the blocking print
          // call starts. This prevents the apparent infinite generating state.
          window.setTimeout(() => {
            try {
              win.print();
              scheduleCleanup();
            } catch (err) {
              fail(err);
            }
          }, 120);
        } catch (err) {
          fail(err);
        }
      }, 120);
    };

    // Use srcdoc before appending. This avoids the about:blank onload event,
    // which caused the print dialog to appear twice.
    iframe.srcdoc = html.replace('<title>4Councilmen Report</title>', `<title>${escapeHtml(safeTitle)}</title>`);
    document.body.appendChild(iframe);
  });
}


function stripMarkdownInline(text: string): string {
  return (text || '')
    .replace(/\[\[\d+\]\]\([^)]*\)/g, '')
    .replace(/\[[^\]]+\]\([^)]*\)/g, '')
    .replace(/[*_`#>]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function compactSentence(text: string, maxLen = 230): string {
  const clean = stripMarkdownInline(text);
  if (!clean) return '';
  const first = clean.match(/^(.{40,}?[.!?])\s/)?.[1] || clean;
  return first.length > maxLen ? `${first.slice(0, maxLen - 1).trim()}…` : first;
}

function normaliseRoundSummary(raw: any): RoundSummary | null {
  if (!raw || typeof raw !== 'object') return null;
  const agentSummariesRaw = raw.agent_summaries || raw.agentSummaries || [];
  const agentSummaries = Array.isArray(agentSummariesRaw)
    ? agentSummariesRaw.map((x: any) => ({
        name: String(x?.name || ''),
        summary: String(x?.summary || ''),
      })).filter((x: RoundAgentSummary) => x.name && x.summary)
    : [];
  return {
    oneLine: String(raw.one_line || raw.oneLine || raw.summary || '').trim(),
    agentSummaries,
    judgeSummary: String(raw.judge_summary || raw.judgeSummary || raw.judge || '').trim(),
  };
}

function derivedRoundSummary(r: RoundResult): RoundSummary {
  const agentSummaries = r.responses.map(resp => ({
    name: resp.name,
    summary: compactSentence(resp.text) || 'No agent summary available.',
  }));

  const oneLine = r.dominantCompatibleProposal
    || r.partialConvergenceSummary
    || r.conclusion
    || compactSentence(r.analysis, 260)
    || `${convergenceLabel(r.convergenceState)} in round ${r.round}.`;

  const judgeSummary = r.whyThisState
    || compactSentence(r.analysis, 320)
    || (r.weakestLink ? `Weakest link: ${r.weakestLink}` : 'No judge summary available.');

  return { oneLine, agentSummaries, judgeSummary };
}

function getRoundSummary(r: RoundResult): RoundSummary {
  if (r.roundSummary) {
    const fallback = derivedRoundSummary(r);
    return {
      oneLine: r.roundSummary.oneLine || fallback.oneLine,
      agentSummaries: r.roundSummary.agentSummaries.length ? r.roundSummary.agentSummaries : fallback.agentSummaries,
      judgeSummary: r.roundSummary.judgeSummary || fallback.judgeSummary,
    };
  }
  return derivedRoundSummary(r);
}

function safeAnchorId(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'section';
}

function buildDecisionTitle(r?: RoundResult | null): string {
  if (!r) return 'No completed round';
  if (r.convergenceState === 'singularity') return 'Singularity reached';
  if (r.convergenceState === 'dominant_compatible_proposal') return 'Dominant compatible proposal extracted';
  if (r.convergenceState === 'partial_convergence') return 'Partial convergence detected';
  return 'No singularity';
}

function recommendedAction(r?: RoundResult | null, finalConclusion?: string | null): string {
  if (!r) return 'No recommendation available.';
  return finalConclusion || r.conclusion || r.dominantCompatibleProposal || r.partialConvergenceSummary || 'No unified recommendation available.';
}

function buildReportHtml(args: {
  lang: Lang;
  theme: Theme;
  pdfTheme: PdfTheme;
  questionTitle: string;
  questionText: string;
  nRounds: number;
  results: RoundResult[];
  finalConclusion: string | null;
  firstSingRound: number | null;
  agents: Agent[];
  providerMode: ProviderMode;
  grokSearchMode: GrokSearchMode;
  claudeKey: string;
  grokKey: string;
  attachedFiles: File[];
}): string {
  const {
    lang, theme, pdfTheme, questionTitle, questionText, nRounds, results, finalConclusion,
    firstSingRound, agents, providerMode, grokSearchMode, claudeKey, grokKey, attachedFiles,
  } = args;

  const now = new Date();
  const selectedRound = firstSingRound ? results.find(r => r.round === firstSingRound) || results[results.length - 1] : results[results.length - 1];
  const providerSet = Array.from(new Set(results.flatMap(r => r.responses.map(a => a.provider)))).join(', ') || 'none';
  const selectedProviders = selectedRound ? providerSummary(selectedRound.providers) : '';
  const isMedicalLike = /x-?ray|chest|dyspnea|pain|medical|diagnosis|patient|radiolog|의료|환자|진단|흉부|엑스레이|방사선/i.test(`${questionTitle} ${questionText}`);

  const statusText = convergenceStatusText(selectedRound, firstSingRound);
  const decisionStateClass = selectedRound ? reportStateClass(selectedRound.convergenceState) : 'state-selected';

  const selectedAnswers = selectedRound?.responses.map(r => `
    <article class="answer-card">
      <h3>${escapeHtml(r.name)} <span>${escapeHtml(r.provider)}</span></h3>
      <p>${nl2br(r.text)}</p>
    </article>
  `).join('') || '';

  const selectedSummary = selectedRound ? getRoundSummary(selectedRound) : null;
  const roundSummaryHtml = results.map(r => {
    const s = getRoundSummary(r);
    return `
      <section class="round-summary" id="round-${r.round}-summary">
        <h3>Round ${r.round} · ${escapeHtml(convergenceLabel(r.convergenceState))} · signal ${r.ratioSignal}</h3>
        <p><strong>Round summary:</strong> ${nl2br(s.oneLine)}</p>
        <div class="agent-summary-grid">
          ${s.agentSummaries.map(a => `
            <div class="agent-summary">
              <strong>${escapeHtml(a.name)}</strong>
              <p>${nl2br(a.summary)}</p>
            </div>
          `).join('')}
        </div>
        ${s.judgeSummary ? `<p><strong>Judge summary:</strong> ${nl2br(s.judgeSummary)}</p>` : ''}
        ${r.coalitionAgents.length ? `<p><strong>Coalition:</strong> ${escapeHtml(r.coalitionAgents.join(', '))}</p>` : ''}
        ${r.dissentingAgents.length ? `<p><strong>Dissent:</strong> ${escapeHtml(r.dissentingAgents.join(', '))}</p>` : ''}
        <p class="see-log">Full raw responses: see <a href="#round-${r.round}-full-log">Appendix A.${r.round}</a>.</p>
      </section>
    `;
  }).join('');

  const indexHtml = `
    <section class="page-break" id="report-index">
      <h2>Report Index</h2>
      <div class="box index-box">
        <ol>
          <li><a href="#decision-brief">Decision Brief</a></li>
          <li><a href="#selected-round-summary">Selected Consensus Round Summary</a></li>
          <li><a href="#round-summaries">Round-by-Round Summaries</a>
            <ol>
              ${results.map(r => `<li><a href="#round-${r.round}-summary">Round ${r.round} · ${escapeHtml(convergenceLabel(r.convergenceState))}</a></li>`).join('')}
            </ol>
          </li>
          <li><a href="#agent-prompts">Agent Prompt Appendix</a></li>
          <li><a href="#full-logs">Full Raw Logs Appendix</a>
            <ol>
              ${results.map(r => `<li><a href="#round-${r.round}-full-log">A.${r.round} Round ${r.round} Full Responses</a></li>`).join('')}
            </ol>
          </li>
          <li><a href="#run-metadata">Run Metadata</a></li>
        </ol>
      </div>
    </section>
  `;

  const promptHtml = agents.map((a, i) => `
    <article class="prompt-card">
      <h3>${i + 1}. ${escapeHtml(a.name)}</h3>
      <p>${nl2br(a.prompt)}</p>
    </article>
  `).join('');

  const roundLogs = results.map(r => {
    const isSelected = selectedRound?.round === r.round;
    return `
    <section id="round-${r.round}-full-log" class="round-log ${isSelected ? 'selected-round' : ''} ${reportStateClass(r.convergenceState)}">
      ${isSelected ? `<div class="selected-banner">${r.convergenceState === 'singularity' ? 'SELECTED CONSENSUS ROUND' : 'SELECTED REFERENCE ROUND'}</div>` : ''}
      <h2>Round ${r.round} · ${escapeHtml(convergenceLabel(r.convergenceState))} · signal ${r.ratioSignal}</h2>
      <div class="metrics">
        <div><span>CONCLUSION</span><strong>${r.conclusionScore.toFixed(2)}</strong></div>
        <div><span>REASONING</span><strong>${r.reasoningScore.toFixed(2)}</strong></div>
        <div><span>SIGNAL</span><strong>${r.ratioSignal}</strong></div>
        <div><span>STATUS</span><strong>${escapeHtml(convergenceLabel(r.convergenceState))}</strong></div>
      </div>
      <p><strong>Internal ratio:</strong> ${r.ratio.toFixed(4)}</p>
      ${r.coalitionAgents.length ? `<p><strong>Coalition:</strong> ${escapeHtml(r.coalitionAgents.join(', '))}</p>` : ''}
      ${r.dissentingAgents.length ? `<p><strong>Dissent:</strong> ${escapeHtml(r.dissentingAgents.join(', '))}</p>` : ''}
      ${r.dominantCompatibleProposal ? `<p><strong>Dominant compatible proposal:</strong> ${nl2br(r.dominantCompatibleProposal)}</p>` : ''}
      ${r.partialConvergenceSummary ? `<p><strong>Partial convergence:</strong> ${nl2br(r.partialConvergenceSummary)}</p>` : ''}
      ${r.whyThisState ? `<p><strong>Why this state:</strong> ${nl2br(r.whyThisState)}</p>` : ''}
      ${r.conclusion ? `<p><strong>Conclusion:</strong> ${nl2br(r.conclusion)}</p>` : ''}
      ${r.analysis ? `<p><strong>Analysis:</strong> ${nl2br(r.analysis)}</p>` : ''}
      ${r.weakestLink ? `<p><strong>Weakest link:</strong> ${nl2br(r.weakestLink)}</p>` : ''}
      <p><strong>Providers:</strong> ${escapeHtml(providerSummary(r.providers))}</p>
      ${r.responses.map(resp => `
        <article class="answer-card compact">
          <h3>${escapeHtml(resp.name)} <span>${escapeHtml(resp.provider)}</span></h3>
          <p>${nl2br(resp.text)}</p>
        </article>
      `).join('')}
    </section>
  `;
  }).join('');

  const reportTheme = pdfTheme === 'current' ? THEME_LABELS[theme] : pdfTheme === 'grayscale' ? 'Grayscale' : 'Light';

  return `<!doctype html>
<html lang="${lang === 'ko' ? 'ko' : 'en'}">
<head>
  <meta charset="utf-8" />
  <title>4Councilmen Report</title>
  <style>
    @page { size: A4; margin: 0; }
    ${getReportThemeCss(pdfTheme, theme)}
    * { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    html, body { min-height: 100%; }
    body { margin: 0; padding: 14mm 15mm 16mm 15mm; color: var(--text); background: var(--bg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, "Noto Sans KR", sans-serif; font-size: 10.1pt; line-height: 1.45; }
    h1 { margin: 0 0 2mm 0; font-size: 21pt; letter-spacing: -0.04em; color: var(--text); }
    h2 { margin: 6mm 0 2.5mm; font-size: 10.5pt; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); break-after: avoid; }
    h3 { margin: 0 0 1.2mm; font-size: 9.5pt; letter-spacing: 0.04em; color: var(--text); }
    p { margin: 0 0 2mm; }
    .meta { color: var(--faint); font-size: 8.5pt; margin-bottom: 4mm; }
    .box, .answer-card, .prompt-card, .round-log, .disclaimer { color: var(--text); background: var(--paper); border: 1px solid var(--border); border-radius: 8px; padding: 4mm; margin-bottom: 4mm; break-inside: avoid; overflow-wrap: anywhere; }
    .round-log { break-inside: auto; page-break-inside: auto; margin-bottom: 7mm; }
    .round-log { position: relative; }
    .round-log.selected-round { border-color: #9ca3af; border-top: 8px solid #9ca3af; background: linear-gradient(180deg, #f3f4f6 0, var(--paper) 18mm); }
    .round-log.selected-round h2 { color: #374151; }
    .selected-banner { margin: -1.5mm -1.5mm 2.5mm -1.5mm; padding: 2mm 3mm; border-radius: 6px; background: #6b7280; color: #ffffff; font-size: 8pt; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; }
    .round-log.state-singularity:not(.selected-round) { border-left: 4px solid #16a34a; background: linear-gradient(90deg, rgba(22,163,74,.07), var(--paper) 22mm); }
    .round-log.state-partial:not(.selected-round), .round-log.state-dominant:not(.selected-round), .round-log.state-none:not(.selected-round) { border-left: 4px solid #f59e0b; background: linear-gradient(90deg, rgba(245,158,11,.10), var(--paper) 22mm); }
    .question { white-space: normal; }
    .decision-brief.state-singularity { border: 2px solid #16a34a; background: #f0fdf4; }
    .decision-brief.state-partial, .decision-brief.state-dominant, .decision-brief.state-none { border: 2px solid #f59e0b; background: #fffbeb; }
    .decision-brief.state-selected { border: 2px solid #9ca3af; background: #f3f4f6; }
    .brief-title { font-size: 16pt; font-weight: 800; color: var(--text); margin-bottom: 2mm; }
    .brief-action { font-size: 12.5pt; font-weight: 700; margin: 2mm 0 3mm; }
    .brief-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 2mm; margin-top: 2mm; }
    .brief-item { border-left: 3px solid var(--border); padding-left: 2mm; }
    .brief-item span { display:block; color: var(--faint); font-size: 7.5pt; letter-spacing: .08em; text-transform: uppercase; }
    .index-box ol { margin: 0; padding-left: 6mm; }
    .index-box li { margin: 1.5mm 0; }
    .index-box a, .see-log a { color: var(--accent); text-decoration: none; font-weight: 700; }
    .round-summary { color: var(--text); background: var(--paper); border: 1px solid var(--border); border-radius: 8px; padding: 4mm; margin-bottom: 4mm; break-inside: avoid; overflow-wrap: anywhere; }
    .agent-summary-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 2.5mm; margin: 2mm 0; }
    .agent-summary { border-left: 3px solid var(--border); padding-left: 2mm; }
    .agent-summary p { margin-top: 1mm; color: var(--muted); }
    .see-log { font-size: 8.5pt; color: var(--faint); }
    .settings { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.5mm 5mm; font-size: 9pt; }
    .settings div span, .metrics span { display: block; color: var(--faint); font-size: 7.5pt; letter-spacing: 0.08em; text-transform: uppercase; }
    .consensus.state-singularity { border: 2px solid #16a34a; background: #f0fdf4; }
    .consensus.state-partial, .consensus.state-dominant, .consensus.state-none { border: 2px solid #f59e0b; background: #fffbeb; }
    .consensus.state-selected { border: 2px solid #9ca3af; background: #f3f4f6; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2.5mm; margin: 2.5mm 0; break-inside: avoid; }
    .metrics div { border-left: 3px solid var(--border); padding-left: 2mm; }
    .metrics strong { font-size: 12.5pt; color: var(--text); }
    .answer-card h3 span { margin-left: 1.5mm; padding: 0.5mm 1.5mm; border: 1px solid var(--chip-border); border-radius: 4px; color: var(--chip-text); font-size: 7.5pt; font-weight: 600; }
    .compact { margin-top: 2.5mm; padding: 3mm; break-inside: auto; page-break-inside: auto; }
    .prompt-card p, .answer-card p { white-space: normal; }
    .disclaimer { font-size: 8.2pt; color: var(--muted); background: var(--soft); }
    .footer { color: var(--faint); font-size: 8pt; margin-top: 8mm; border-top: 1px solid var(--border); padding-top: 3mm; }
    .page-break { break-before: page; }
    @media print { button { display: none; } a { color: inherit; text-decoration: none; } body { background: var(--bg); } }
  </style>
</head>
<body>
  <h1>4 Councilmen Report</h1>
  <div class="meta">${escapeHtml(now.toLocaleString())} · ${escapeHtml(reportTheme)}</div>

  <section id="decision-brief">
    <h2>Decision Brief</h2>
    <div class="box decision-brief ${decisionStateClass}">
      <div class="brief-title">${escapeHtml(buildDecisionTitle(selectedRound))} · Signal ${selectedRound?.ratioSignal ?? '-'}</div>
      <p class="brief-action"><strong>Recommended action:</strong> ${nl2br(recommendedAction(selectedRound, finalConclusion))}</p>
      ${selectedSummary?.oneLine ? `<p><strong>Executive rationale:</strong> ${nl2br(selectedSummary.oneLine)}</p>` : ''}
      ${selectedRound?.dominantCompatibleProposal ? `<p><strong>Dominant compatible proposal:</strong> ${nl2br(selectedRound.dominantCompatibleProposal)}</p>` : ''}
      ${selectedRound?.partialConvergenceSummary ? `<p><strong>Partial convergence:</strong> ${nl2br(selectedRound.partialConvergenceSummary)}</p>` : ''}
      ${selectedRound?.whyThisState ? `<p><strong>Why this state:</strong> ${nl2br(selectedRound.whyThisState)}</p>` : ''}
      <div class="brief-grid">
        <div class="brief-item"><span>Final state</span>${selectedRound ? escapeHtml(convergenceLabel(selectedRound.convergenceState)) : '-'}</div>
        <div class="brief-item"><span>Selected round</span>${selectedRound ? `Round ${selectedRound.round}` : '-'}</div>
        <div class="brief-item"><span>Coalition</span>${selectedRound?.coalitionAgents.length ? escapeHtml(selectedRound.coalitionAgents.join(', ')) : '-'}</div>
        <div class="brief-item"><span>Dissent / weakest link</span>${selectedRound?.dissentingAgents.length ? escapeHtml(selectedRound.dissentingAgents.join(', ')) : escapeHtml(selectedRound?.weakestLink || '-')}</div>
      </div>
    </div>
  </section>

  ${indexHtml}

  <section class="page-break">
    <h2>Question</h2>
    <div class="box question">
      <strong>${escapeHtml(questionTitle)}</strong><br />
      ${nl2br(questionText)}
    </div>
  </section>

  <section>
    <h2>Run Settings</h2>
    <div class="box settings">
      <div><span>Language</span>${escapeHtml(lang === 'ko' ? 'Korean' : 'English')}</div>
      <div><span>Report theme</span>${escapeHtml(reportTheme)}</div>
      <div><span>Requested rounds</span>${nRounds}</div>
      <div><span>Completed rounds</span>${results.length}</div>
      <div><span>Judge</span>Grok fixed</div>
      <div><span>Provider routing</span>${escapeHtml(providerMode)}</div>
      <div><span>Grok web search</span>${escapeHtml(grokSearchMode)}</div>
      <div><span>API providers used</span>${escapeHtml(providerSet)}</div>
      <div><span>API keys configured</span>Claude=${claudeKey ? 'yes' : 'no'} / Grok=${grokKey ? 'yes' : 'no'}</div>
      <div><span>Attached files</span>${attachedFiles.length ? escapeHtml(attachedFiles.map(f => f.name).join(', ')) : 'none'}</div>
    </div>
  </section>

  <section>
    <h2>Consensus Result</h2>
    <div class="box consensus ${decisionStateClass}">
      <p><strong>${escapeHtml(statusText)}</strong></p>
      <div class="metrics">
        <div><span>CONCLUSION</span><strong>${selectedRound?.conclusionScore.toFixed(2) ?? '-'}</strong></div>
        <div><span>REASONING</span><strong>${selectedRound?.reasoningScore.toFixed(2) ?? '-'}</strong></div>
        <div><span>SIGNAL</span><strong>${selectedRound?.ratioSignal ?? '-'}</strong></div>
        <div><span>STATUS</span><strong>${selectedRound ? escapeHtml(convergenceLabel(selectedRound.convergenceState)) : '-'}</strong></div>
      </div>
      ${selectedRound ? `<p><strong>Internal ratio:</strong> ${selectedRound.ratio.toFixed(4)}</p>` : ''}
      ${selectedRound?.coalitionAgents.length ? `<p><strong>Coalition:</strong> ${escapeHtml(selectedRound.coalitionAgents.join(', '))}</p>` : ''}
      ${selectedRound?.dissentingAgents.length ? `<p><strong>Dissent:</strong> ${escapeHtml(selectedRound.dissentingAgents.join(', '))}</p>` : ''}
      ${selectedRound?.dominantCompatibleProposal ? `<p><strong>Dominant compatible proposal:</strong> ${nl2br(selectedRound.dominantCompatibleProposal)}</p>` : ''}
      ${selectedRound?.partialConvergenceSummary ? `<p><strong>Partial convergence:</strong> ${nl2br(selectedRound.partialConvergenceSummary)}</p>` : ''}
      ${selectedRound?.whyThisState ? `<p><strong>Why this state:</strong> ${nl2br(selectedRound.whyThisState)}</p>` : ''}
      ${finalConclusion ? `<p><strong>Conclusion:</strong> ${nl2br(finalConclusion)}</p>` : ''}
      <p><strong>Selected round:</strong> ${selectedRound ? `Round ${selectedRound.round}` : '-'} · <strong>Providers:</strong> ${escapeHtml(selectedProviders)}</p>
    </div>
  </section>

  ${buildBilingualDisclaimer(isMedicalLike)}

  <section class="page-break" id="selected-round-summary">
    <h2>Selected Consensus Round Summary</h2>
    ${selectedRound && selectedSummary ? `
      <div class="round-summary">
        <h3>Round ${selectedRound.round} · ${escapeHtml(convergenceLabel(selectedRound.convergenceState))} · signal ${selectedRound.ratioSignal}</h3>
        <p><strong>Round summary:</strong> ${nl2br(selectedSummary.oneLine)}</p>
        <div class="agent-summary-grid">
          ${selectedSummary.agentSummaries.map(a => `
            <div class="agent-summary"><strong>${escapeHtml(a.name)}</strong><p>${nl2br(a.summary)}</p></div>
          `).join('')}
        </div>
        ${selectedSummary.judgeSummary ? `<p><strong>Judge summary:</strong> ${nl2br(selectedSummary.judgeSummary)}</p>` : ''}
        <p class="see-log">Full raw responses: see <a href="#round-${selectedRound.round}-full-log">Appendix A.${selectedRound.round}</a>.</p>
      </div>
    ` : ''}
  </section>

  <section class="page-break" id="round-summaries">
    <h2>Round-by-Round Summaries</h2>
    ${roundSummaryHtml}
  </section>

  <section class="page-break" id="agent-prompts">
    <h2>Appendix · Agent System Prompts</h2>
    ${promptHtml}
  </section>

  <section class="page-break" id="full-logs">
    <h2>Appendix · Full Raw Logs</h2>
    ${roundLogs}
  </section>

  <section class="page-break" id="run-metadata">
    <h2>Run Metadata</h2>
    <div class="box settings">
      <div><span>Generated</span>${escapeHtml(now.toISOString())}</div>
      <div><span>Selected providers</span>${escapeHtml(selectedProviders || '-')}</div>
      <div><span>Internal ratio</span>${selectedRound ? selectedRound.ratio.toFixed(4) : '-'}</div>
      <div><span>First singularity round</span>${firstSingRound ?? 'none'}</div>
    </div>
  </section>

  <div class="footer">4 Councilmen Report · ${escapeHtml(reportTheme)} · generated from structured run data, not a screenshot.</div>
</body>
</html>`;
}

// ── Sub-components ──────────────────────────────────────────────────────────

interface AgentCardProps {
  agent: Agent;
  roundIdx: number;
  agentIdx: number;
  tc: typeof THEME_CLASSES[Theme];
  isRunning: boolean;
  riskLevel: RiskLevel;
  providerMode: ProviderMode;
  selectedProvider: Provider;
  onProviderChange: (agentId: number, provider: Provider) => void;
  onEdit: (agent: Agent) => void;
  onViewPrompt: (agent: Agent) => void;
  onReset: (id: number) => void;
}

const AgentCard: React.FC<AgentCardProps> = ({
  agent, roundIdx, agentIdx, tc, isRunning, riskLevel, providerMode, selectedProvider, onProviderChange, onEdit, onViewPrompt, onReset,
}) => {
  const provider = selectedProvider;
  const provClass = provider === 'grok' ? tc.provGrok
                  : provider === 'local' ? `${tc.textMuted} border-current opacity-60`
                  : tc.provClaude;
  const borderClass = agent.modified ? tc.cardModBorder : tc.cardBorder;

  return (
    <div className={`${tc.card} border ${borderClass} ${agent.modified ? 'border-2' : ''} rounded-xl p-3 flex flex-col gap-2`}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className={`text-xs font-semibold ${tc.text}`}>{agent.name}</span>
          {agent.modified && (
            <span className={`text-[9px] ${tc.textMuted}`}>modified</span>
          )}
        </div>
        {providerMode === 'custom' ? (
          <select
            value={provider === 'claude-fallback' ? 'claude' : provider}
            onChange={e => onProviderChange(agent.id, e.target.value as Provider)}
            disabled={isRunning}
            className={`text-[10px] px-2 py-1 rounded-lg border ${tc.input} ${tc.inputBorder} ${tc.text} disabled:opacity-50`}
          >
            <option value="grok">Grok</option>
            <option value="claude">Claude</option>
          </select>
        ) : providerMode === 'all-local' ? (
          <span className={`text-[9px] px-1.5 py-0.5 rounded border ${tc.textFaint} ${tc.btnBorder} whitespace-nowrap`}>
            Local LLM
          </span>
        ) : (
          <span className={`text-[9px] px-1.5 py-0.5 rounded border ${provClass} whitespace-nowrap`}>
            {provider === 'grok' ? 'Grok' : 'Claude'} · R{roundIdx + 1}
          </span>
        )}
      </div>

      <p className={`text-[10px] leading-relaxed ${tc.textMuted} line-clamp-3 flex-1`}>
        {agent.prompt}
      </p>

      <div className="flex gap-1.5 flex-wrap">
        <button
          onClick={() => !isRunning && onEdit(agent)}
          disabled={isRunning}
          className={`flex items-center gap-1 text-[10px] px-2 py-1 border rounded ${tc.btn} ${tc.btnBorder} disabled:opacity-40`}
        >
          <Edit2 size={10} /> edit
        </button>
        <button
          onClick={() => onViewPrompt(agent)}
          className={`flex items-center gap-1 text-[10px] px-2 py-1 border rounded ${tc.btn} ${tc.btnBorder}`}
        >
          <FileText size={10} /> prompt
        </button>
        {agent.modified && (
          <button
            onClick={() => !isRunning && onReset(agent.id)}
            disabled={isRunning}
            className={`flex items-center gap-1 text-[10px] px-2 py-1 border rounded text-red-500 border-red-400 hover:bg-red-50 disabled:opacity-40`}
          >
            <RotateCcw size={10} /> reset
          </button>
        )}
      </div>
    </div>
  );
};


interface DecisionBriefCardProps {
  result: RoundResult | undefined;
  finalConclusion: string | null;
  firstSingRound: number | null;
  tc: typeof THEME_CLASSES[Theme];
}

const DecisionBriefCard: React.FC<DecisionBriefCardProps> = ({ result, finalConclusion, firstSingRound, tc }) => {
  if (!result) return null;
  const summary = getRoundSummary(result);
  const panelClass = convergencePanelClass(result.convergenceState);
  const labelClass = convergencePanelLabelClass(result.convergenceState);
  return (
    <div className={`${panelClass} border-2 rounded-xl p-4 mb-3`}>
      <div className={`text-[10px] font-semibold uppercase tracking-wider ${labelClass} flex items-center gap-1.5 mb-2`}>
        {result.convergenceState === 'singularity' ? <Phone size={11} /> : <PhoneOff size={11} />} Final decision brief
      </div>
      <div className={`text-sm font-semibold ${tc.text} mb-1`}>
        {buildDecisionTitle(result)} · Signal {result.ratioSignal}
        {firstSingRound && <span className={`text-[11px] font-normal ${tc.textMuted}`}> · selected round {firstSingRound}</span>}
      </div>
      <p className={`text-[12px] leading-relaxed ${tc.text} mb-2`}>
        <strong>Recommended action:</strong> {recommendedAction(result, finalConclusion)}
      </p>
      <p className={`text-[11px] leading-relaxed ${tc.textMuted} mb-2`}>
        <strong>Executive rationale:</strong> {summary.oneLine}
      </p>
      {result.whyThisState && (
        <p className={`text-[11px] leading-relaxed ${tc.textMuted} mb-2`}>
          <strong>Why:</strong> {result.whyThisState}
        </p>
      )}
      <div className={`grid grid-cols-2 gap-2 text-[10px] ${tc.textMuted}`}>
        <div><strong>State:</strong> {convergenceLabel(result.convergenceState)}</div>
        <div><strong>Internal ratio:</strong> {result.ratio.toFixed(4)}</div>
        {result.coalitionAgents.length > 0 && <div className="col-span-2"><strong>Coalition:</strong> {result.coalitionAgents.join(', ')}</div>}
        {(result.dissentingAgents.length > 0 || result.weakestLink) && <div className="col-span-2"><strong>Dissent / weakest link:</strong> {result.dissentingAgents.length ? result.dissentingAgents.join(', ') : result.weakestLink}</div>}
      </div>
    </div>
  );
};

interface RoundBlockProps {
  result: RoundResult;
  roundIdx: number;
  tc: typeof THEME_CLASSES[Theme];
}

const RoundBlock: React.FC<RoundBlockProps> = ({ result, roundIdx, tc }) => {
  const [open, setOpen] = useState(roundIdx === 0);
  const [showRaw, setShowRaw] = useState(false);
  const badgeClass = convergenceBadgeClass(result.convergenceState, tc);
  const summary = getRoundSummary(result);

  return (
    <div className={`${tc.card} border ${tc.cardBorder} rounded-lg overflow-hidden mb-2`}>
      <button
        className={`w-full flex items-center justify-between px-3 py-2 ${tc.roundHeader}`}
        onClick={() => setOpen(o => !o)}
      >
        <div className="flex items-center gap-2 min-w-0">
          {open ? <ChevronDown size={12} className={tc.textMuted} /> : <ChevronRight size={12} className={tc.textMuted} />}
          <span className={`text-[11px] font-medium ${tc.textMuted} whitespace-nowrap`}>
            Round {result.round}
          </span>
          <span className={`text-[10px] ${tc.textFaint} truncate`}>
            · {summary.oneLine}
          </span>
        </div>
        <span className={`text-[10px] px-2 py-0.5 rounded border flex items-center gap-1 whitespace-nowrap ${badgeClass}`}>
          {result.convergenceState === 'singularity' ? <Phone size={9} /> : <PhoneOff size={9} />}
          {convergenceLabel(result.convergenceState)} · signal {result.ratioSignal}
        </span>
      </button>

      {open && (
        <div className="px-3 py-2 space-y-3">
          <div className={`rounded-lg border ${tc.cardBorder} ${tc.roundHeader} p-3`}>
            <div className={`text-[10px] font-semibold uppercase tracking-wider ${tc.textFaint} mb-1`}>
              Round summary
            </div>
            <p className={`text-[12px] leading-relaxed ${tc.text}`}>{summary.oneLine}</p>
          </div>

          <div className="grid grid-cols-2 gap-2">
            {summary.agentSummaries.map((a, i) => (
              <div key={`${a.name}-${i}`} className={`rounded-lg border ${tc.cardBorder} ${tc.card} p-2`}>
                <div className={`text-[10px] font-semibold ${tc.textMuted} mb-1`}>{a.name}</div>
                <p className={`text-[10px] leading-relaxed ${tc.text}`}>{a.summary}</p>
              </div>
            ))}
          </div>

          <div className={`pt-2 border-t ${tc.sep} space-y-1.5`}>
            <div className="flex gap-4 flex-wrap">
              {[
                ['Conclusion', result.conclusionScore],
                ['Reasoning', result.reasoningScore],
                ['Signal', result.ratioSignal],
                ['Internal ratio', result.ratio],
              ].map(([label, val]) => (
                <div key={label as string} className="flex flex-col">
                  <span className={`text-[9px] ${tc.textFaint}`}>{label}</span>
                  <span className={`text-[12px] font-semibold ${tc.text}`}>
                    {(val as number).toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
            <div className={`text-[10px] leading-relaxed ${tc.textMuted} space-y-1`}>
              <div><strong>State:</strong> {convergenceLabel(result.convergenceState)}</div>
              {result.coalitionAgents.length > 0 && <div><strong>Coalition:</strong> {result.coalitionAgents.join(', ')}</div>}
              {result.dissentingAgents.length > 0 && <div><strong>Dissent:</strong> {result.dissentingAgents.join(', ')}</div>}
              {result.dominantCompatibleProposal && <div><strong>Dominant proposal:</strong> {result.dominantCompatibleProposal}</div>}
              {result.partialConvergenceSummary && <div><strong>Partial convergence:</strong> {result.partialConvergenceSummary}</div>}
              {summary.judgeSummary && <div><strong>Judge summary:</strong> {summary.judgeSummary}</div>}
              {result.whyThisState && <div><strong>Why:</strong> {result.whyThisState}</div>}
            </div>
            {result.conclusion && (
              <p className={`text-[10px] italic leading-relaxed ${tc.textMuted}`}>
                "{result.conclusion}"
              </p>
            )}
          </div>

          <button
            onClick={() => setShowRaw(v => !v)}
            className={`text-[10px] px-2 py-1 border rounded ${tc.btn} ${tc.btnBorder} flex items-center gap-1.5`}
          >
            {showRaw ? <EyeOff size={10} /> : <Eye size={10} />}
            {showRaw ? 'Hide full raw log' : 'Show full raw log'}
          </button>

          {showRaw && (
            <div className="space-y-2">
              {result.responses.map((r, i) => (
                <div key={i} className={`rounded-lg border ${tc.cardBorder} ${tc.card} p-2`}>
                  <div className="flex items-center gap-1.5 mb-1">
                    <span className={`text-[10px] font-semibold ${tc.textMuted}`}>{r.name}</span>
                    <span className={`text-[9px] px-1.5 py-0.5 rounded border ${
                      r.provider === 'grok' ? tc.provGrok
                      : r.provider === 'local' ? `${tc.textFaint} ${tc.btnBorder}`
                      : tc.provClaude
                    }`}>
                      {r.provider}
                    </span>
                  </div>
                  <p className={`text-[11px] leading-relaxed ${tc.text}`}>{r.text}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};


function reportStateClass(state?: ConvergenceState): string {
  switch (state) {
    case 'singularity': return 'state-singularity';
    case 'partial_convergence': return 'state-partial';
    case 'dominant_compatible_proposal': return 'state-dominant';
    case 'no_singularity': return 'state-none';
    default: return 'state-selected';
  }
}

function convergenceLabel(state?: ConvergenceState): string {
  switch (state) {
    case 'singularity': return 'Singularity';
    case 'partial_convergence': return 'Partial convergence';
    case 'dominant_compatible_proposal': return 'Dominant proposal';
    default: return 'No singularity';
  }
}

function convergenceStatusText(r?: RoundResult | null, firstSingRound?: number | null): string {
  if (!r) return 'No result.';
  if (r.convergenceState === 'singularity') {
    return firstSingRound
      ? `First singularity at round ${firstSingRound}.`
      : `Singularity reached at round ${r.round}.`;
  }
  if (r.convergenceState === 'partial_convergence') {
    return 'Partial convergence was detected; showing the strongest coalition and dissenting perspectives.';
  }
  if (r.convergenceState === 'dominant_compatible_proposal') {
    return 'No full singularity was reached, but a dominant compatible proposal was extracted.';
  }
  return 'No singularity was reached; showing the final completed round as the reference result.';
}

function convergenceBadgeClass(state: ConvergenceState, tc: typeof THEME_CLASSES[Theme]): string {
  if (state === 'singularity') return tc.singBadge;
  if (state === 'partial_convergence' || state === 'dominant_compatible_proposal' || state === 'no_singularity') return 'border-amber-400 text-amber-700 bg-amber-50';
  return tc.noSingBadge;
}

function convergencePanelClass(state: ConvergenceState): string {
  switch (state) {
    case 'singularity':
      return 'bg-green-50 border-green-400';
    case 'partial_convergence':
    case 'dominant_compatible_proposal':
    case 'no_singularity':
      return 'bg-amber-50 border-amber-400';
    default:
      return 'bg-gray-50 border-gray-300';
  }
}

function convergencePanelLabelClass(state: ConvergenceState): string {
  switch (state) {
    case 'singularity':
      return 'text-green-800';
    case 'partial_convergence':
    case 'dominant_compatible_proposal':
    case 'no_singularity':
      return 'text-amber-800';
    default:
      return 'text-gray-700';
  }
}

function selectedPanelClass(): string {
  return 'bg-gray-50 border-gray-300';
}

function coerceConvergenceState(raw: any, isSingularity: boolean): ConvergenceState {
  const allowed = ['singularity', 'partial_convergence', 'dominant_compatible_proposal', 'no_singularity'];
  if (allowed.includes(raw)) return raw as ConvergenceState;
  return isSingularity ? 'singularity' : 'no_singularity';
}


// ── Edit Modal ──────────────────────────────────────────────────────────────

interface EditModalProps {
  agent: Agent;
  otherAgents: Agent[];
  tc: typeof THEME_CLASSES[Theme];
  savedVersions: string[];
  agentSet: string;
  lang: Lang;
  useLocalApi: boolean;
  grokKey: string;
  onSave: (id: number, name: string, prompt: string, version: string) => void;
  onClose: () => void;
}

type ValidationState = 'idle' | 'validating' | 'ok' | 'warn';

const EditModal: React.FC<EditModalProps> = ({
  agent, otherAgents, tc, savedVersions, agentSet, lang, useLocalApi, grokKey, onSave, onClose,
}) => {
  const [name, setName] = useState(agent.name);
  const [prompt, setPrompt] = useState(agent.prompt);
  const [validation, setValidation] = useState<ValidationState>('idle');
  const [validMsg, setValidMsg] = useState('');

  const handleValidateAndSave = async () => {
    if (!name.trim() || !prompt.trim()) return;
    setValidation('validating');
    setValidMsg(useLocalApi
      ? 'Asking the local/internal Judge LLM to validate and rewrite this agent...'
      : 'Asking Grok, as the Judge LLM, to validate and rewrite this agent...'
    );

    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (!useLocalApi && grokKey && grokKey !== '__saved__') headers['X-Grok-Key'] = grokKey;

      const res = await fetch(`/4councilmen/fourCM/validate-and-save/${encodeURIComponent(agentSet)}/agent/${agent.id}`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          name: name.trim(),
          prompt: prompt.trim(),
          other_prompts: otherAgents.map(a => a.prompt),
          use_external_api: !useLocalApi,
          lang,
        }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || data?.message || `Validation failed: ${res.status}`);

      const rewrittenName = (data.rewritten_name || name).trim();
      const rewrittenPrompt = (data.rewritten_prompt || prompt).trim();

      if (data.ok === false) {
        setValidation('warn');
        setValidMsg(data.message || 'The Judge LLM did not approve this agent prompt.');
        return;
      }

      setValidation('ok');
      setValidMsg(`${data.judge === 'local' ? 'Local Judge LLM' : 'Grok Judge'} validated${data.changed ? ' and rewrote' : ''} this agent. ${data.message || ''}`.trim());

      const version = nextVersionName(rewrittenName, savedVersions);
      setTimeout(() => {
        onSave(agent.id, rewrittenName, rewrittenPrompt, version);
        onClose();
      }, 600);
    } catch (e: any) {
      setValidation('warn');
      setValidMsg(e?.message || 'Validation failed. Please check the Judge LLM configuration.');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className={`${tc.editModal} border rounded-xl p-5 w-full max-w-lg mx-4 shadow-2xl`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className={`text-sm font-semibold ${tc.text}`}>
            <Edit2 size={14} className="inline mr-1.5" />
            Edit agent
          </h3>
          <button onClick={onClose} className={`${tc.textMuted} hover:${tc.text}`}>
            <X size={16} />
          </button>
        </div>

        <div className="space-y-3 mb-4">
          <div>
            <label className={`text-[10px] font-semibold uppercase tracking-wider ${tc.textFaint} block mb-1`}>
              Agent name
            </label>
            <input
              value={name}
              onChange={e => setName(e.target.value)}
              className={`w-full px-3 py-2 text-xs rounded-lg border ${tc.input} ${tc.inputBorder} focus:outline-none`}
              placeholder="e.g. SENTINEL"
            />
          </div>

          <div>
            <label className={`text-[10px] font-semibold uppercase tracking-wider ${tc.textFaint} block mb-1`}>
              System prompt
            </label>
            <textarea
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              rows={6}
              className={`w-full px-3 py-2 text-xs rounded-lg border ${tc.input} ${tc.inputBorder} focus:outline-none resize-none leading-relaxed`}
              placeholder="You are [NAME], an orthogonal AI. Your only lens is..."
            />
            <div className={`text-[10px] text-right mt-1 ${tc.textFaint}`}>
              {prompt.length} chars
            </div>
          </div>
        </div>

        {validation !== 'idle' && (
          <div className={`text-[11px] px-3 py-2 rounded-lg mb-3 flex items-start gap-2 ${
            validation === 'validating' ? `${tc.textMuted} border ${tc.btnBorder}` :
            validation === 'ok' ? tc.validOk :
            tc.validWarn
          }`}>
            {validation === 'validating' && <RefreshCw size={11} className="animate-spin mt-0.5 shrink-0" />}
            {validation === 'ok' && <Check size={11} className="mt-0.5 shrink-0" />}
            {validation === 'warn' && <AlertCircle size={11} className="mt-0.5 shrink-0" />}
            <span>{validMsg}</span>
          </div>
        )}

        <div className="flex gap-2 justify-end">
          <button
            onClick={onClose}
            className={`text-xs px-3 py-1.5 border rounded-lg ${tc.btn} ${tc.btnBorder}`}
          >
            cancel
          </button>
          <button
            onClick={handleValidateAndSave}
            disabled={validation === 'validating' || !name.trim() || !prompt.trim()}
            className="text-xs px-4 py-1.5 border border-green-500 text-green-600 rounded-lg hover:bg-green-50 disabled:opacity-40 flex items-center gap-1.5"
          >
            {validation === 'validating'
              ? <><RefreshCw size={11} className="animate-spin" /> validating...</>
              : <><Save size={11} /> validate &amp; save</>
            }
          </button>
        </div>
      </div>
    </div>
  );
};

// ── Settings Panel ──────────────────────────────────────────────────────────

interface SettingsPanelProps {
  tc: typeof THEME_CLASSES[Theme];
  lang: Lang;
  theme: Theme;
  scenarios: Scenario[];
  grokKey: string;
  pdfTheme: PdfTheme;
  setPdfTheme: (t: PdfTheme) => void;
  results: RoundResult[];
  isPdfGenerating: boolean;
  onClose: () => void;
  onScenariosChange: () => void;
  onPdfDownload: () => void;
}

type SettingsTab = 'scenarios' | 'pdf' | 'backup';

const SettingsPanel: React.FC<SettingsPanelProps> = ({
  tc, lang, theme, scenarios, grokKey, pdfTheme, setPdfTheme,
  results, isPdfGenerating, onClose, onScenariosChange, onPdfDownload,
}) => {
  const [tab, setTab] = useState<SettingsTab>('scenarios');
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameVal, setRenameVal] = useState('');
  const [importing, setImporting] = useState(false);
  const [exportData, setExportData] = useState<string | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const tabClass = (t: SettingsTab) =>
    `text-[11px] px-3 py-1.5 rounded-md font-medium transition-all ${
      tab === t
        ? `${tc.runBtn} border border-transparent`
        : `${tc.btn} ${tc.btnBorder} border`
    }`;

  const handleDelete = async (id: string) => {
    if (!confirm(`Delete scenario "${id}"? This cannot be undone.`)) return;
    await fetch(`/4councilmen/fourCM/scenario/${id}`, { method: 'DELETE' });
    onScenariosChange();
  };

  const handleRename = async (id: string) => {
    if (!renameVal.trim() || renameVal === id) { setRenamingId(null); return; }
    await fetch(`/4councilmen/fourCM/scenario/${id}/rename`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_id: renameVal.trim() }),
    });
    setRenamingId(null);
    onScenariosChange();
  };

  const handleExport = async () => {
    const res = await fetch('/4councilmen/fourCM/export');
    const data = await res.json();
    const json = JSON.stringify(data, null, 2);
    setExportData(json);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const ts = new Date().toISOString().slice(0, 10);
    a.href = url; a.download = `4cm-backup-${ts}.json`; a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImporting(true);
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      await fetch('/4councilmen/fourCM/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      onScenariosChange();
      alert('Import successful!');
    } catch (e) {
      alert('Import failed. Please check the JSON file.');
    } finally {
      setImporting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className={`${tc.editModal} border rounded-xl shadow-2xl w-full max-w-2xl mx-4 flex flex-col`} style={{maxHeight: '85vh'}}>

        {/* Header */}
        <div className={`flex items-center justify-between px-5 py-3.5 border-b ${tc.sidebarBorder}`}>
          <div className="flex items-center gap-3">
            <span className={`text-sm font-semibold ${tc.text}`}>
              <Settings size={14} className="inline mr-1.5" />
              Settings
            </span>
            <div className="flex gap-1.5">
              {(['scenarios', 'pdf', 'backup'] as SettingsTab[]).map(t => (
                <button key={t} onClick={() => setTab(t)} className={tabClass(t)}>
                  {t === 'scenarios' ? '📋 Scenarios' : t === 'pdf' ? '📄 PDF' : '💾 Backup'}
                </button>
              ))}
            </div>
          </div>
          <button onClick={onClose}><X size={16} className={tc.textMuted} /></button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4">

          {/* ── Scenarios Tab ── */}
          {tab === 'scenarios' && (
            <div className="space-y-2">
              <p className={`text-[10px] ${tc.textFaint} mb-3`}>
                Rename or delete scenarios. Files are stored in <code className="bg-black/10 px-1 rounded">angry_agents/</code>
              </p>
              {scenarios.map(s => (
                <div key={s.id} className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${tc.cardBorder} ${tc.card}`}>
                  {renamingId === s.id ? (
                    <>
                      <input
                        autoFocus
                        value={renameVal}
                        onChange={e => setRenameVal(e.target.value)}
                        onKeyDown={e => { if (e.key === 'Enter') handleRename(s.id); if (e.key === 'Escape') setRenamingId(null); }}
                        className={`flex-1 text-xs px-2 py-1 border rounded ${tc.input} ${tc.inputBorder} focus:outline-none`}
                      />
                      <button onClick={() => handleRename(s.id)} className="text-[11px] text-green-600 border border-green-500 px-2 py-1 rounded">save</button>
                      <button onClick={() => setRenamingId(null)} className={`text-[11px] ${tc.textMuted} border ${tc.btnBorder} px-2 py-1 rounded`}>cancel</button>
                    </>
                  ) : (
                    <>
                      <div className="flex-1 min-w-0">
                        <span className={`text-xs font-medium ${tc.text}`}>{s.title}</span>
                        <span className={`text-[10px] ml-2 ${tc.textFaint}`}>{s.id}</span>
                        <span className={`ml-2 text-[9px] px-1.5 py-0.5 rounded ${s.risk === 'high' ? tc.provGrok : tc.singBadge}`}>
                          {s.risk}
                        </span>
                      </div>
                      <button
                        onClick={() => { setRenamingId(s.id); setRenameVal(s.id); }}
                        className={`text-[11px] px-2 py-1 border rounded ${tc.btn} ${tc.btnBorder}`}
                      >
                        rename
                      </button>
                      <button
                        onClick={() => handleDelete(s.id)}
                        className="text-[11px] px-2 py-1 border border-red-400 text-red-500 rounded hover:bg-red-50"
                      >
                        delete
                      </button>
                    </>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* ── PDF Tab ── */}
          {tab === 'pdf' && (
            <div>
              <p className={`text-[10px] ${tc.textFaint} mb-3`}>
                PDF is generated from the current results view. Run 4CM first, then download.
              </p>
              <div className="space-y-2 mb-4">
                {([
                  { val: 'light' as PdfTheme, label: 'Light', desc: 'White background — best for printing' },
                  { val: 'current' as PdfTheme, label: 'Current theme', desc: theme === 'light' ? '(same as Light)' : `Current: ${THEME_LABELS[theme]}` },
                  { val: 'grayscale' as PdfTheme, label: 'Grayscale', desc: 'Black & white — ink saving' },
                ]).filter(opt => !(opt.val === 'current' && theme === 'light'))
                  .map(opt => (
                  <button
                    key={opt.val}
                    onClick={() => setPdfTheme(opt.val)}
                    className={`w-full text-left px-3 py-2.5 rounded-lg border transition-all ${
                      pdfTheme === opt.val ? 'border-green-500 bg-green-50 text-green-800' : `${tc.btn} ${tc.btnBorder}`
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium">{opt.label}</span>
                      {pdfTheme === opt.val && <Check size={11} className="text-green-600" />}
                    </div>
                    <span className={`text-[10px] ${pdfTheme === opt.val ? 'text-green-600' : tc.textFaint}`}>{opt.desc}</span>
                  </button>
                ))}
              </div>
              {results.length > 0 && (
                <button
                  onClick={onPdfDownload}
                  disabled={isPdfGenerating}
                  className="w-full text-xs py-2.5 border border-green-500 text-green-600 rounded-lg hover:bg-green-50 disabled:opacity-40 flex items-center justify-center gap-2"
                >
                  {isPdfGenerating
                    ? <><RefreshCw size={12} className="animate-spin" /> Generating...</>
                    : <><Download size={12} /> Download PDF now</>
                  }
                </button>
              )}
            </div>
          )}

          {/* ── Backup Tab ── */}
          {tab === 'backup' && (
            <div className="space-y-4">
              <div className={`p-4 rounded-lg border ${tc.cardBorder} ${tc.card}`}>
                <div className={`text-xs font-semibold ${tc.text} mb-1`}>Export all scenarios</div>
                <p className={`text-[10px] ${tc.textFaint} mb-3`}>
                  Downloads all scenarios, agents, and prompts as a JSON file.
                </p>
                <button
                  onClick={handleExport}
                  className={`text-xs px-4 py-2 border rounded-lg ${tc.btn} ${tc.btnBorder} flex items-center gap-1.5`}
                >
                  <Download size={11} /> Export JSON
                </button>
              </div>

              <div className={`p-4 rounded-lg border ${tc.cardBorder} ${tc.card}`}>
                <div className={`text-xs font-semibold ${tc.text} mb-1`}>Import scenarios</div>
                <p className={`text-[10px] ${tc.textFaint} mb-3`}>
                  Restore from a previously exported JSON file. Existing scenarios with the same ID will be overwritten.
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json"
                  onChange={handleImport}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={importing}
                  className={`text-xs px-4 py-2 border rounded-lg ${tc.btn} ${tc.btnBorder} flex items-center gap-1.5 disabled:opacity-40`}
                >
                  {importing
                    ? <><RefreshCw size={11} className="animate-spin" /> Importing...</>
                    : <><FileText size={11} /> Import JSON</>
                  }
                </button>
              </div>

              <div className={`p-3 rounded-lg border ${tc.cardBorder} text-[10px] ${tc.textFaint}`}>
                <strong className={tc.text}>JSON structure:</strong>
                <pre className="mt-1 text-[9px] overflow-x-auto">{`{
  "version": "2.0",
  "scenarios": {
    "government": {
      "title": "...", "query": "...", "risk": "normal",
      "agents": {
        "1": { "name": "SENTINEL", "prompt": "..." },
        ...
      }
    }
  }
}`}</pre>
              </div>
            </div>
          )}

        </div>

        {/* Footer */}
        <div className={`px-5 py-3 border-t ${tc.sidebarBorder} flex justify-end`}>
          <button onClick={onClose} className={`text-xs px-4 py-1.5 border rounded-lg ${tc.btn} ${tc.btnBorder}`}>
            close
          </button>
        </div>
      </div>
    </div>
  );
};

// ── Key Setup Modal ─────────────────────────────────────────────────────────

interface LocalEndpoint {
  url: string;
  apiKey: string;
}

interface KeyModalProps {
  tc: typeof THEME_CLASSES[Theme];
  claudeSet: boolean;
  grokSet: boolean;
  useLocalApi: boolean;
  localPayloads: Record<number, string>;
  localEndpoints: Record<number, LocalEndpoint>;
  onSave: (claude: string, grok: string) => void;
  onClear: () => void;
  onClose: () => void;
  onLocalApiChange: (val: boolean) => void;
  onSaveLocalPayloads: (payloads: Record<number, string>) => void;
  onSaveLocalEndpoints: (endpoints: Record<number, LocalEndpoint>) => void;
}

const KeyModal: React.FC<KeyModalProps> = ({
  tc, claudeSet, grokSet, useLocalApi, localPayloads, localEndpoints,
  onSave, onClear, onClose, onLocalApiChange, onSaveLocalPayloads, onSaveLocalEndpoints,
}) => {
  const [claude, setClaude] = useState('');
  const [grok, setGrok] = useState('');
  const [showClaude, setShowClaude] = useState(false);
  const [showGrok, setShowGrok] = useState(false);
  const [keyTab, setKeyTab] = useState<'keys' | 'endpoints' | 'payloads'>('keys');

  // ── 탭 2: LLM Endpoints ────────────────────────────────────────────────────
  const EMPTY_ENDPOINT: LocalEndpoint = { url: '', apiKey: '' };
  const [endpointDraft, setEndpointDraft] = useState<Record<number, LocalEndpoint>>({
    1: localEndpoints[1] || EMPTY_ENDPOINT,
    2: localEndpoints[2] || EMPTY_ENDPOINT,
    3: localEndpoints[3] || EMPTY_ENDPOINT,
    4: localEndpoints[4] || EMPTY_ENDPOINT,
  });
  const [endpointSaved, setEndpointSaved] = useState(false);
  const [showEndpointKey, setShowEndpointKey] = useState<Record<number, boolean>>({});

  const saveEndpoints = () => {
    onSaveLocalEndpoints(endpointDraft);
    setEndpointSaved(true);
    setTimeout(() => setEndpointSaved(false), 1800);
  };

  // ── 탭 3: Payloads ─────────────────────────────────────────────────────────
  const [payloadDraft, setPayloadDraft] = useState<Record<number, string>>({
    1: localPayloads[1] || '',
    2: localPayloads[2] || '',
    3: localPayloads[3] || '',
    4: localPayloads[4] || '',
  });
  const [payloadErrors, setPayloadErrors] = useState<Record<number, boolean>>({});
  const [payloadSaved, setPayloadSaved] = useState(false);

  const validateAndSavePayloads = () => {
    const errors: Record<number, boolean> = {};
    let hasError = false;
    ([1, 2, 3, 4] as const).forEach(i => {
      const v = payloadDraft[i].trim();
      if (v && v !== '{}') {
        try { JSON.parse(v); }
        catch { errors[i] = true; hasError = true; }
      }
    });
    setPayloadErrors(errors);
    if (hasError) return;
    onSaveLocalPayloads(payloadDraft);
    setPayloadSaved(true);
    setTimeout(() => setPayloadSaved(false), 1800);
  };

  const tabCls = (t: 'keys' | 'endpoints' | 'payloads') =>
    `text-[11px] px-2.5 py-1.5 rounded-md font-medium transition-all border ${
      keyTab === t ? `${tc.runBtn} border-transparent` : `${tc.btn} ${tc.btnBorder}`
    }`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className={`${tc.editModal} border rounded-xl p-5 w-full max-w-lg mx-4 shadow-2xl`}>

        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className={`text-sm font-semibold ${tc.text} mr-1`}>
              <Settings size={14} className="inline mr-1.5" />
              API Keys
            </h3>
            <div className="flex gap-1.5">
              <button className={tabCls('keys')}     onClick={() => setKeyTab('keys')}>🔑 Keys</button>
              <button className={tabCls('endpoints')} onClick={() => setKeyTab('endpoints')}>🌐 LLM Endpoints</button>
              <button className={tabCls('payloads')}  onClick={() => setKeyTab('payloads')}>⚙️ Payloads</button>
            </div>
          </div>
          <button onClick={onClose}><X size={16} className={tc.textMuted} /></button>
        </div>

        {/* ── Tab 1: Keys ── */}
        {keyTab === 'keys' && (
          <>
            <div className="space-y-4 mb-4">
              {[
                { label: 'Anthropic (Claude)', val: claude, set: setShowClaude, show: showClaude, onChange: setClaude, isSet: claudeSet, placeholder: 'sk-ant-...' },
                { label: 'xAI (Grok)',         val: grok,   set: setShowGrok,   show: showGrok,   onChange: setGrok,   isSet: grokSet,   placeholder: 'xai-...' },
              ].map(({ label, val, set, show, onChange, isSet, placeholder }) => (
                <div key={label}>
                  <div className="flex items-center gap-2 mb-1.5">
                    <div className={`w-2 h-2 rounded-full ${isSet ? 'bg-green-500' : 'bg-gray-300'}`} />
                    <label className={`text-[11px] font-semibold ${tc.textMuted}`}>{label}</label>
                    {isSet && <span className="text-[9px] text-green-500 ml-auto">active</span>}
                  </div>
                  <div className="relative">
                    <input
                      type={show ? 'text' : 'password'}
                      value={val}
                      onChange={e => onChange(e.target.value)}
                      placeholder={isSet ? '••••••••••••••••' : placeholder}
                      className={`w-full px-3 py-2 pr-9 text-xs rounded-lg border ${tc.input} ${tc.inputBorder} focus:outline-none font-mono`}
                    />
                    <button onClick={() => set(s => !s)} className={`absolute right-2 top-1/2 -translate-y-1/2 ${tc.textFaint}`}>
                      {show ? <EyeOff size={12} /> : <Eye size={12} />}
                    </button>
                  </div>
                </div>
              ))}
            </div>
            <p className={`text-[10px] ${tc.textFaint} mb-4 leading-relaxed`}>
              Keys are stored in memory only and cleared when the container restarts. For persistent keys, use your organization's secret manager.
              They are never printed to logs or transmitted beyond the local backend.
            </p>
            <div className="flex gap-2 justify-between">
              <button onClick={() => { onClear(); onClose(); }} className="text-xs px-3 py-1.5 border border-red-400 text-red-500 rounded-lg hover:bg-red-50 flex items-center gap-1">
                <Trash2 size={11} /> clear all keys
              </button>
              <div className="flex gap-2">
                <button onClick={onClose} className={`text-xs px-3 py-1.5 border rounded-lg ${tc.btn} ${tc.btnBorder}`}>cancel</button>
                <button onClick={() => onSave(claude, grok)} className={`text-xs px-4 py-1.5 rounded-lg ${tc.runBtn}`}>
                  <Check size={11} className="inline mr-1" /> save keys
                </button>
              </div>
            </div>
          </>
        )}

        {/* ── Tab 2: LLM Endpoints ── */}
        {keyTab === 'endpoints' && (
          <>
            {/* USE_EXTERNAL_API 토글 */}
            <div className={`flex items-center justify-between px-3 py-2.5 rounded-lg border mb-4 ${tc.cardBorder} ${tc.card}`}>
              <div>
                <div className={`text-[11px] font-semibold ${tc.text}`}>Use External API</div>
                <div className={`text-[10px] ${tc.textFaint}`}>
                  {useLocalApi ? 'Currently: Local / Internal LLM only' : 'Currently: External (Claude + Grok)'}
                </div>
              </div>
              <button
                onClick={() => onLocalApiChange(!useLocalApi)}
                className={`relative w-11 h-6 rounded-full transition-colors ${useLocalApi ? 'bg-green-500' : 'bg-gray-400'}`}
              >
                <span className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform ${useLocalApi ? 'translate-x-5' : 'translate-x-0'}`} />
              </button>
            </div>

            <p className={`text-[10px] ${tc.textFaint} mb-3 leading-relaxed`}>
              Each agent can use a different LLM endpoint (Ollama, vLLM, internal gateway, or any OpenAI-compatible API).
              Saved in your browser (localStorage).
            </p>

            <div className="space-y-4 mb-4">
              {([1, 2, 3, 4] as const).map(i => (
                <div key={i} className={`p-3 rounded-lg border ${tc.cardBorder} ${tc.card} space-y-2`}>
                  <div className={`text-[10px] font-semibold uppercase tracking-wider ${tc.textFaint}`}>
                    Agent {i} — LLM Endpoint
                  </div>
                  {/* URL */}
                  <div>
                    <label className={`text-[10px] ${tc.textMuted} block mb-1`}>Endpoint URL</label>
                    <input
                      type="text"
                      value={endpointDraft[i].url}
                      onChange={e => setEndpointDraft(prev => ({ ...prev, [i]: { ...prev[i], url: e.target.value } }))}
                      placeholder="http://localhost:11434/v1"
                      className={`w-full px-3 py-1.5 text-[11px] font-mono rounded-lg border ${tc.input} ${tc.inputBorder} focus:outline-none`}
                    />
                  </div>
                  {/* API Key */}
                  <div>
                    <label className={`text-[10px] ${tc.textMuted} block mb-1`}>API Key (optional)</label>
                    <div className="relative">
                      <input
                        type={showEndpointKey[i] ? 'text' : 'password'}
                        value={endpointDraft[i].apiKey}
                        onChange={e => setEndpointDraft(prev => ({ ...prev, [i]: { ...prev[i], apiKey: e.target.value } }))}
                        placeholder="ollama  /  sk-...  /  bearer-token"
                        className={`w-full px-3 py-1.5 pr-8 text-[11px] font-mono rounded-lg border ${tc.input} ${tc.inputBorder} focus:outline-none`}
                      />
                      <button
                        onClick={() => setShowEndpointKey(prev => ({ ...prev, [i]: !prev[i] }))}
                        className={`absolute right-2 top-1/2 -translate-y-1/2 ${tc.textFaint}`}
                      >
                        {showEndpointKey[i] ? <EyeOff size={11} /> : <Eye size={11} />}
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="flex justify-end gap-2">
              <button onClick={onClose} className={`text-xs px-3 py-1.5 border rounded-lg ${tc.btn} ${tc.btnBorder}`}>cancel</button>
              <button
                onClick={saveEndpoints}
                className={`text-xs px-4 py-1.5 rounded-lg flex items-center gap-1.5 ${
                  endpointSaved ? 'bg-green-500 text-white border-transparent' : tc.runBtn
                }`}
              >
                {endpointSaved ? <><Check size={11} /> Saved!</> : <><Save size={11} /> Save endpoints</>}
              </button>
            </div>
          </>
        )}

        {/* ── Tab 3: Payloads ── */}
        {keyTab === 'payloads' && (
          <>
            <p className={`text-[10px] ${tc.textFaint} mb-3 leading-relaxed`}>
              Extra payload fields merged into each agent's LLM request. JSON only.
              Saved in your browser (localStorage).
            </p>
            <div className="space-y-3 mb-4">
              {([1, 2, 3, 4] as const).map(i => (
                <div key={i}>
                  <label className={`text-[10px] font-semibold uppercase tracking-wider ${tc.textFaint} block mb-1`}>
                    Agent {i} — extra payload fields
                  </label>
                  <textarea
                    rows={2}
                    value={payloadDraft[i]}
                    onChange={e => {
                      setPayloadDraft(prev => ({ ...prev, [i]: e.target.value }));
                      setPayloadErrors(prev => ({ ...prev, [i]: false }));
                      setPayloadSaved(false);
                    }}
                    placeholder='{"temperature": 0.7, "top_p": 0.9}'
                    className={`w-full px-3 py-2 text-[11px] font-mono rounded-lg border focus:outline-none resize-none ${tc.input} ${
                      payloadErrors[i] ? 'border-red-400' : tc.inputBorder
                    }`}
                  />
                  {payloadErrors[i] && <div className="text-[10px] text-red-500 mt-0.5">Invalid JSON</div>}
                </div>
              ))}
            </div>
            <div className="flex justify-end gap-2">
              <button onClick={onClose} className={`text-xs px-3 py-1.5 border rounded-lg ${tc.btn} ${tc.btnBorder}`}>cancel</button>
              <button
                onClick={validateAndSavePayloads}
                className={`text-xs px-4 py-1.5 rounded-lg flex items-center gap-1.5 ${
                  payloadSaved ? 'bg-green-500 text-white border-transparent' : tc.runBtn
                }`}
              >
                {payloadSaved ? <><Check size={11} /> Saved!</> : <><Save size={11} /> Save payloads</>}
              </button>
            </div>
          </>
        )}

      </div>
    </div>
  );
};

// ── Main Component ──────────────────────────────────────────────────────────

const FourCM: React.FC = () => {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem('4cm_theme') as Theme;
    return saved && THEME_CLASSES[saved] ? saved : 'light';
  });
  const [lang, setLang] = useState<Lang>('en');
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [scenariosLoading, setScenariosLoading] = useState(true);
  const [selectedScenario, setSelectedScenario] = useState<Scenario | null>(null);
  const [customQuery, setCustomQuery] = useState('');
  const [isCustom, setIsCustom] = useState(false);
  const [nRounds, setNRounds] = useState(3);
  const [agents, setAgents] = useState<Agent[]>(DEFAULT_AGENTS);
  const [savedVersions, setSavedVersions] = useState<string[]>([]);

  const [providerMode, setProviderMode] = useState<ProviderMode>(() => {
    // useLocalApi 초기값과 providerMode 초기값을 일치시킴
    return localStorage.getItem('4cm_use_local_api') === 'true' ? 'all-local' : 'round-robin';
  });
  const [grokSearchMode, setGrokSearchMode] = useState<GrokSearchMode>('off');
  const [agentProviderOverrides, setAgentProviderOverrides] = useState<Record<number, Provider>>({
    1: 'grok',
    2: 'grok',
    3: 'claude',
    4: 'claude',
  });

  // ── Local LLM 모드 ─────────────────────────────────────────────────────────
  // useLocalApi: false → 외부 API (Claude+Grok), true → 로컬/사내 LLM
  const [useLocalApi, setUseLocalApi] = useState<boolean>(() => {
    return localStorage.getItem('4cm_use_local_api') === 'true';
  });

  // 에이전트별 추가 payload (JSON 문자열로 저장, 파싱은 run 시점)
  const [localPayloads, setLocalPayloads] = useState<Record<number, string>>(() => {
    try {
      const saved = localStorage.getItem('4cm_local_payloads');
      return saved ? JSON.parse(saved) : { 1: '', 2: '', 3: '', 4: '' };
    } catch { return { 1: '', 2: '', 3: '', 4: '' }; }
  });

  // 에이전트별 LLM 엔드포인트 (url + apiKey)
  const [localEndpoints, setLocalEndpoints] = useState<Record<number, LocalEndpoint>>(() => {
    try {
      const saved = localStorage.getItem('4cm_local_endpoints');
      return saved ? JSON.parse(saved) : { 1: { url: '', apiKey: '' }, 2: { url: '', apiKey: '' }, 3: { url: '', apiKey: '' }, 4: { url: '', apiKey: '' } };
    } catch { return { 1: { url: '', apiKey: '' }, 2: { url: '', apiKey: '' }, 3: { url: '', apiKey: '' }, 4: { url: '', apiKey: '' } }; }
  });

  const handleSaveLocalEndpoints = useCallback((endpoints: Record<number, LocalEndpoint>) => {
    setLocalEndpoints(endpoints);
    localStorage.setItem('4cm_local_endpoints', JSON.stringify(endpoints));
  }, []);

  const handleLocalApiChange = useCallback(async (val: boolean) => {
    setUseLocalApi(val);
    localStorage.setItem('4cm_use_local_api', String(val));
    // 백엔드 환경변수도 동기화
    try {
      await fetch('/4councilmen/fourCM/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ use_external_api: !val }),
      });
    } catch (e) {
      console.warn('Config sync failed:', e);
    }
    // Local LLM 모드 활성화 시 all-local로, 비활성화 시 round-robin으로
    if (val) setProviderMode('all-local');
    else setProviderMode('round-robin');
  }, []);

  const handleSaveLocalPayloads = useCallback((payloads: Record<number, string>) => {
    setLocalPayloads(payloads);
    localStorage.setItem('4cm_local_payloads', JSON.stringify(payloads));
  }, []);

  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [uploadingFiles, setUploadingFiles] = useState(false);
  const uploadInputRef = useRef<HTMLInputElement>(null);

  const [claudeKey, setClaudeKey] = useState('');
  const [grokKey, setGrokKey] = useState('');
  const [pdfTheme, setPdfTheme] = useState<PdfTheme>('light');
  const [isPdfGenerating, setIsPdfGenerating] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [roundIdx, setRoundIdx] = useState(0);
  const [results, setResults] = useState<RoundResult[]>([]);
  const [streamingText, setStreamingText] = useState<Record<string, string>>({});
  const [intentTriage, setIntentTriage] = useState<IntentTriage | null>(null);
  const [routeBlock, setRouteBlock] = useState<{ message: string; triage?: IntentTriage | null } | null>(null);
  const [finalConclusion, setFinalConclusion] = useState<string | null>(null);
  const [firstSingRound, setFirstSingRound] = useState<number | null>(null);

  const [editingAgent, setEditingAgent] = useState<Agent | null>(null);
  const [viewingPrompt, setViewingPrompt] = useState<Agent | null>(null);
  const [showKeyModal, setShowKeyModal] = useState(false);

  // 앱 시작 시 키 상태 확인 (메모리에서만 확인)
  useEffect(() => {
    fetch('/4councilmen/fourCM/keys/status')
      .then(r => r.json())
      .then(data => {
        if (data.claude_set) setClaudeKey('__saved__');
        if (data.grok_set) setGrokKey('__saved__');
      })
      .catch(() => {});
  }, []);

  // 키 저장 (옵션 1: UI에서 입력 → 메모리에서만 유지, 디스크 저장 없음)
  const handleSaveKeys = useCallback(async (claude: string, grok: string) => {
    await fetch('/4councilmen/fourCM/keys/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ claude_key: claude, grok_key: grok }),
    });
    if (claude) setClaudeKey('__saved__');
    if (grok) setGrokKey('__saved__');
  }, []);

  // 키 삭제
  const handleClearKeys = useCallback(async () => {
    await fetch('/4councilmen/fourCM/keys/clear', { method: 'DELETE' });
    setClaudeKey('');
    setGrokKey('');
  }, []);
  useEffect(() => {
    setScenariosLoading(true);
    const headers: Record<string, string> = {};
    if (grokKey) headers['X-Grok-Key'] = grokKey;

    fetch(`/4councilmen/fourCM/agents?lang=${lang}`, { headers })
      .then(r => r.json())
      .then(data => {
        const loaded: Scenario[] = (data.sets || []).map((s: any) => ({
          id: s.id,
          title: s.title || s.id,
          risk: (s.risk?.trim() || 'normal') as RiskLevel,
          agentSet: s.id,
          query: s.query || '',
        }));
        setScenarios(loaded);
        if (loaded.length > 0 && !selectedScenario) setSelectedScenario(loaded[0]);
      })
      .catch(() => setScenarios([]))
      .finally(() => setScenariosLoading(false));
  }, [lang, grokKey]);

  useEffect(() => {
    const setId = selectedScenario?.agentSet;
    if (!setId) return;
    fetch(`/4councilmen/fourCM/saved-agents/${encodeURIComponent(setId)}`)
      .then(r => r.ok ? r.json() : Promise.reject(new Error(`Agent load failed: ${r.status}`)))
      .then(data => {
        const loadedAgents = Array.isArray(data.agents) ? data.agents : [];
        if (loadedAgents.length === 4) {
          setAgents(loadedAgents.map((a: any, idx: number) => ({
            id: Number(a.id || idx + 1),
            name: String(a.name || `AGENT_${idx + 1}`),
            prompt: String(a.prompt || ''),
            modified: true,
          })));
        }
      })
      .catch(() => {
        fetch(`/4councilmen/fourCM/agents/${encodeURIComponent(setId)}`)
          .then(r => r.ok ? r.json() : Promise.reject(new Error(`Agent load failed: ${r.status}`)))
          .then(data => {
            const loadedAgents = Array.isArray(data.agents) ? data.agents : [];
            if (loadedAgents.length === 4) {
              setAgents(loadedAgents.map((a: any, idx: number) => ({
                id: Number(a.id || idx + 1),
                name: String(a.name || `AGENT_${idx + 1}`),
                prompt: String(a.prompt || ''),
                modified: false,
              })));
            }
          })
          .catch(() => setAgents(DEFAULT_AGENTS));
      });
  }, [selectedScenario?.agentSet]);

  const abortRef = useRef<AbortController | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const tc = THEME_CLASSES[theme];
  const T = UI_TEXT[lang];

  // ── PDF 생성 ──────────────────────────────────────────────────────────────
  const handlePDF = useCallback(async () => {
    if (results.length === 0) return;
    setIsPdfGenerating(true);

    try {
      const questionTitle = isCustom ? T.custom : (selectedScenario?.title || 'Direct input');
      const questionText = isCustom ? customQuery : (selectedScenario?.query || customQuery || '');
      const reportHtml = buildReportHtml({
        lang,
        theme,
        pdfTheme,
        questionTitle,
        questionText,
        nRounds,
        results,
        finalConclusion,
        firstSingRound,
        agents,
        providerMode,
        grokSearchMode,
        claudeKey,
        grokKey,
        attachedFiles,
      });

      const now = new Date();
      const ts = [
        now.getFullYear(),
        String(now.getMonth() + 1).padStart(2, '0'),
        String(now.getDate()).padStart(2, '0'),
        String(now.getHours()).padStart(2, '0'),
        String(now.getMinutes()).padStart(2, '0'),
      ].join('-');

      // Browser print keeps the PDF text-selectable. The report is rendered in a hidden iframe,
      // so no extra report tab/window is opened before the print dialog appears.
      await printHtmlWithoutOpeningTab(reportHtml, `4councilmenReport-${ts}`);
    } catch (e) {
      console.error('PDF generation failed:', e);
      alert(e instanceof Error ? e.message : 'PDF generation failed. Please try again.');
    } finally {
      setIsPdfGenerating(false);
    }
  }, [
    results, pdfTheme, theme, lang, isCustom, selectedScenario, customQuery, T.custom,
    nRounds, finalConclusion, firstSingRound, agents, providerMode, grokSearchMode,
    claudeKey, grokKey, attachedFiles,
  ]);

  useEffect(() => {
    localStorage.setItem('4cm_theme', theme);
  }, [theme]);

  const handleReset = useCallback((id: number) => {
    const orig = DEFAULT_AGENTS.find(a => a.id === id);
    if (!orig) return;
    setAgents(prev => prev.map(a => a.id === id ? { ...orig } : a));
  }, []);

  const handleSaveAgent = useCallback((id: number, name: string, prompt: string, version: string) => {
    setAgents(prev => prev.map(a =>
      a.id === id ? { ...a, name, prompt, modified: true, savedVersion: version } : a
    ));
    setSavedVersions(prev => [...prev, version]);
  }, []);

  const uploadAttachedFiles = useCallback(async (): Promise<string | null> => {
    if (attachedFiles.length === 0) return null;
    setUploadingFiles(true);
    try {
      const form = new FormData();
      attachedFiles.forEach(f => form.append('files', f));
      const res = await fetch('/4councilmen/fourCM/uploads', {
        method: 'POST',
        body: form,
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Upload failed: ${res.status}`);
      }
      const data = await res.json();
      return data.upload_session_id || null;
    } finally {
      setUploadingFiles(false);
    }
  }, [attachedFiles]);

  const effectiveProviderForAgent = useCallback((agentId: number, agentIdx: number, displayRoundIdx = 0): Provider => {
    if (providerMode === 'all-local') return 'local';
    if (providerMode === 'all-grok') return 'grok';
    if (providerMode === 'all-claude') return 'claude';
    if (providerMode === 'custom') return agentProviderOverrides[agentId] || (agentIdx < 2 ? 'grok' : 'claude');
    return providerForRound(displayRoundIdx, agentIdx);
  }, [providerMode, agentProviderOverrides]);

  const providerPlan = useCallback((): Record<string, Provider> => {
    const plan: Record<string, Provider> = {};
    agents.forEach((agent, idx) => {
      plan[agent.name] = effectiveProviderForAgent(agent.id, idx, 0);
    });
    return plan;
  }, [agents, effectiveProviderForAgent]);

  const handleRun = async () => {
    // Local LLM 모드가 아닐 때만 키 확인
    if (!useLocalApi && !claudeKey && !grokKey) { setShowKeyModal(true); return; }
    if (isRunning) {
      abortRef.current?.abort();
      setIsRunning(false);
      return;
    }

    setIsRunning(true);
    setResults([]);
    setStreamingText({});
    setIntentTriage(null);
    setFinalConclusion(null);
    setFirstSingRound(null);
    setRoundIdx(0);

    const query = isCustom ? customQuery : (selectedScenario?.query || '');
    const riskLevel = isCustom ? 'normal' : (selectedScenario?.risk || 'normal');

    let uploadSessionId: string | null = null;
    try {
      uploadSessionId = await uploadAttachedFiles();
    } catch (e: any) {
      alert(e?.message || 'Document upload failed.');
      setIsRunning(false);
      return;
    }

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const res = await fetch('/4councilmen/fourCM', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Claude-Key': claudeKey === '__saved__' ? '' : claudeKey,
          'X-Grok-Key': grokKey === '__saved__' ? '' : grokKey,
        },
        body: JSON.stringify({
          query,
          risk_level: riskLevel,
          lang,
          n_rounds: nRounds,
          agent_set: selectedScenario?.agentSet || 'government',
          agents: agents.map(a => ({ id: a.id, name: a.name, prompt: a.prompt })),
          provider_mode: providerMode,
          agent_providers: providerPlan(),
          grok_search_mode: grokSearchMode,
          upload_session_id: uploadSessionId,
          use_external_api: !useLocalApi,
          local_payloads: useLocalApi
            ? Object.fromEntries(
                Object.entries(localPayloads).map(([k, v]) => {
                  try { return [k, v.trim() ? JSON.parse(v) : {}]; }
                  catch { return [k, {}]; }
                })
              )
            : undefined,
          local_endpoints: useLocalApi
            ? Object.fromEntries(
                Object.entries(localEndpoints).map(([k, v]) => [k, { url: v.url, api_key: v.apiKey }])
              )
            : undefined,
        }),
        signal: ctrl.signal,
      });

      if (!res.ok) {
        const errText = await res.text();
        let parsed: any = null;
        try { parsed = errText ? JSON.parse(errText) : null; } catch { parsed = null; }
        const detail = parsed?.detail ?? parsed;
        const triage = detail?.intent_triage ?? parsed?.intent_triage ?? null;

        if (triage) {
          setIntentTriage(triage);
          setRouteBlock({
            message:
              detail?.message ||
              triage?.suggested_handling ||
              'This request was not routed to 4CM by the intent triage layer.',
            triage,
          });
          return;
        }

        throw new Error((typeof detail === 'string' ? detail : detail?.message) || errText || `4CM request failed: ${res.status}`);
      }
      if (!res.body) throw new Error('No streaming response body returned by 4CM.');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') { setIsRunning(false); return; }
          try {
            const msg = JSON.parse(raw);

            if (msg.type === 'intent_triage') {
              setIntentTriage(msg.triage || null);
            }
            if (msg.type === 'agent') {
              setStreamingText(prev => ({
                ...prev,
                [`${msg.round}_${msg.name}`]: msg.text,
              }));
            }
            if (msg.type === 'round_complete') {
              const r: RoundResult = {
                round: msg.round,
                providers: msg.providers,
                responses: msg.responses,
                conclusionScore: msg.conclusion_score,
                reasoningScore: msg.reasoning_score,
                ratio: Number(msg.ratio || 0),
                ratioSignal: Number(msg.ratio_signal ?? (msg.is_singularity ? 1 : 0)) as 0 | 1,
                isSingularity: Boolean(msg.is_singularity),
                convergenceState: coerceConvergenceState(msg.convergence_state, Boolean(msg.is_singularity)),
                coalitionAgents: Array.isArray(msg.coalition_agents) ? msg.coalition_agents : [],
                dissentingAgents: Array.isArray(msg.dissenting_agents) ? msg.dissenting_agents : [],
                dominantCompatibleProposal: msg.dominant_compatible_proposal || null,
                partialConvergenceSummary: msg.partial_convergence_summary || null,
                whyThisState: msg.why_this_state || null,
                roundSummary: normaliseRoundSummary(msg.round_summary || msg.summary_by_agent || msg.roundSummary),
                conclusion: msg.conclusion,
                weakestLink: msg.weakest_link,
                analysis: msg.analysis,
              };
              setResults(prev => [...prev, r]);
              setRoundIdx(msg.round);
              if (msg.is_singularity && coerceConvergenceState(msg.convergence_state, Boolean(msg.is_singularity)) === 'singularity' && !firstSingRound) {
                setFirstSingRound(msg.round);
              }
            }
            if (msg.type === 'summary') {
              setFinalConclusion(msg.conclusion);
              setFirstSingRound(msg.first_singularity_round);
            }
          } catch {}
        }
      }
    } catch (e: any) {
      if (e?.name !== 'AbortError') {
        console.error(e);
        setRouteBlock({ message: e?.message || '4CM run failed.' });
      }
    } finally {
      setIsRunning(false);
    }
  };

  const keysReady = useLocalApi || !!(claudeKey || grokKey);

  return (
    <div className={`${tc.app} flex flex-col h-screen`}>

      {/* Theme bar */}
      <div className={`flex items-center gap-1.5 px-4 py-2 border-b ${tc.headerBorder} ${tc.header} flex-wrap`}>
        <span className={`text-[10px] font-semibold uppercase tracking-wider ${tc.textFaint} mr-2`}>Theme</span>
        {(Object.keys(THEME_LABELS) as Theme[]).map(t => (
          <button
            key={t}
            onClick={() => setTheme(t)}
            className={`text-[11px] px-3 py-1 rounded-full border transition-all ${
              t === theme
                ? 'border-current font-semibold'
                : `${tc.btn} ${tc.btnBorder}`
            } ${t === theme ? (
              t === 'light' ? 'bg-gray-900 text-white' :
              t === 'light-warm' ? 'bg-stone-800 text-amber-50' :
              t === 'dark-classic' ? 'bg-indigo-600 text-white' :
              'bg-[#a6e22e] text-[#272822]'
            ) : ''}`}
          >
            {THEME_LABELS[t]}
          </button>
        ))}
        <div className={`ml-auto w-px h-4 ${tc.sep}`} />
        {/* PDF 다운로드 버튼 */}
        {results.length > 0 && (
          <button
            onClick={handlePDF}
            disabled={isPdfGenerating}
            className={`flex items-center gap-1 text-[11px] px-3 py-1 border rounded-full ${tc.btn} ${tc.btnBorder} disabled:opacity-40`}
          >
            {isPdfGenerating
              ? <><RefreshCw size={11} className="animate-spin" /> generating...</>
              : <><Download size={11} /> PDF</>
            }
          </button>
        )}
        {/* {T.settings} 버튼 */}
        <button
          onClick={() => setShowSettingsModal(true)}
          className={`flex items-center gap-1 text-[11px] px-3 py-1 border rounded-full ${tc.btn} ${tc.btnBorder}`}
        >
          <Settings size={11} /> {T.settings}
        </button>
        <button
          onClick={() => setShowKeyModal(true)}
          className={`flex items-center gap-1 text-[11px] px-3 py-1 border rounded-full ${tc.btn} ${tc.btnBorder}`}
        >
          <Settings size={11} />
          {T.apiKeys}
          {keysReady && <span className="w-1.5 h-1.5 rounded-full bg-green-500" />}
        </button>
      </div>

      {/* Main layout */}
      <div className="flex flex-1 overflow-hidden">

        {/* Sidebar */}
        <div className={`${tc.sidebar} border-r ${tc.sidebarBorder} w-64 flex flex-col shrink-0 overflow-y-auto`}>

          <div className={`px-4 py-3 border-b ${tc.sidebarBorder}`}>
            <div className={`text-sm font-bold ${tc.text}`}>4 Councilmen</div>
            <div className={`text-[10px] ${tc.textFaint} mt-0.5`}>
              {useLocalApi ? 'Local LLM mode · v2.1' : 'Hybrid v2.1 · Claude + Grok'}
            </div>
          </div>

          {/* Scenario */}
          <div className="px-3 py-3">
            <div className={`text-[10px] font-semibold uppercase tracking-wider ${tc.textFaint} mb-2`}>{T.scenario}</div>
            <div className="space-y-0.5">
              {scenariosLoading ? (
                <div className={`text-[11px] ${tc.textFaint} px-2 py-3`}>
                  <RefreshCw size={11} className="inline animate-spin mr-1" /> Loading...
                </div>
              ) : scenarios.map(s => (
                <button
                  key={s.id}
                  onClick={() => { setSelectedScenario(s); setIsCustom(false); }}
                  className={`w-full text-left px-2.5 py-2 rounded-lg transition-colors ${
                    !isCustom && selectedScenario?.id === s.id ? tc.scActive : 'hover:opacity-80'
                  }`}
                >
                  <div className={`text-[12px] font-medium ${tc.text}`}>{s.title}</div>
                  <span className={`text-[9px] px-1.5 py-0.5 rounded mt-1 inline-block ${
                    s.risk === 'high' ? tc.provGrok : tc.singBadge
                  }`}>
                    {s.risk === 'high' ? T.highRisk : T.normal}
                  </span>
                  {s.query && (
                    <p className={`text-[10px] mt-1.5 leading-relaxed ${tc.textFaint} line-clamp-2`}>
                      {s.query.replace(/\[MOCK RESEARCH SCENARIO\]\s*/i, '')}
                    </p>
                  )}
                </button>
              ))}
              <button
                onClick={() => setIsCustom(true)}
                className={`w-full text-left px-2.5 py-2 rounded-lg ${isCustom ? tc.scActive : 'hover:opacity-80'}`}
              >
                <div className={`text-[12px] font-medium ${tc.text}`}>{T.custom}</div>
                <span className={`text-[9px] px-1.5 py-0.5 rounded mt-1 inline-block ${tc.noSingBadge}`}>
                  {T.writeOwn}
                </span>
              </button>
            </div>
          </div>

          <div className={`h-px mx-3 ${tc.sep}`} />

          {/* Language */}
          <div className="px-3 py-3">
            <div className={`text-[10px] font-semibold uppercase tracking-wider ${tc.textFaint} mb-2`}>
              <Globe size={10} className="inline mr-1" />Language
            </div>
            <div className="flex gap-1.5">
              {(['en', 'ko'] as Lang[]).map(l => (
                <button
                  key={l}
                  onClick={() => setLang(l)}
                  className={`flex-1 py-1.5 text-[11px] font-medium rounded-lg border transition-all ${
                    lang === l ? tc.langActive : tc.langInactive
                  }`}
                >
                  {l === 'en' ? 'English' : '한국어'}
                </button>
              ))}
            </div>
          </div>

          <div className={`h-px mx-3 ${tc.sep}`} />

          {/* Rounds */}
          <div className="px-3 py-3">
            <div className={`text-[10px] font-semibold uppercase tracking-wider ${tc.textFaint} mb-2`}>{T.rounds}</div>
            <div className="flex items-center">
              <span className={`text-[12px] ${tc.textMuted} flex-1`}>{T.numRounds}</span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setNRounds(n => Math.max(1, n - 1))}
                  className={`w-6 h-6 border rounded-md flex items-center justify-center ${tc.btn} ${tc.btnBorder}`}
                >
                  <Minus size={11} />
                </button>
                <span className={`text-sm font-bold ${tc.text} w-4 text-center`}>{nRounds}</span>
                <button
                  onClick={() => setNRounds(n => Math.min(5, n + 1))}
                  className={`w-6 h-6 border rounded-md flex items-center justify-center ${tc.btn} ${tc.btnBorder}`}
                >
                  <Plus size={11} />
                </button>
              </div>
            </div>
          </div>

          <div className={`h-px mx-3 ${tc.sep}`} />

          {/* Provider routing */}
          <div className="px-3 py-3 space-y-2">
            <div className={`text-[10px] font-semibold uppercase tracking-wider ${tc.textFaint}`}>{T.providerRouting}</div>

            {/* All Local LLM — 전폭, 상단 */}
            <button
              onClick={() => {
                setProviderMode('all-local');
                if (!useLocalApi) handleLocalApiChange(true);
              }}
              className={`w-full py-1.5 text-[10px] font-medium rounded-lg border transition-all ${
                providerMode === 'all-local' ? tc.langActive : tc.langInactive
              }`}
            >
              {T.allLocal}
            </button>

            {/* 기존 2×2 그리드 */}
            <div className="grid grid-cols-2 gap-1.5">
              {([
                ['round-robin', T.roundRobin],
                ['all-grok', T.allGrok],
                ['all-claude', T.allClaude],
                ['custom', T.perAgent],
              ] as [ProviderMode, string][]).map(([mode, label]) => (
                <button
                  key={mode}
                  onClick={() => {
                    setProviderMode(mode);
                    // 외부 API 모드로 전환 시 useLocalApi=false 동기화
                    if (mode !== 'all-local' && useLocalApi) handleLocalApiChange(false);
                  }}
                  className={`py-1.5 text-[10px] font-medium rounded-lg border ${providerMode === mode ? tc.langActive : tc.langInactive}`}
                >
                  {label}
                </button>
              ))}
            </div>

            {providerMode === 'custom' && (
              <div className={`text-[9px] ${tc.textFaint}`}>{T.providerOnCards}</div>
            )}

            {/* Grok 인터넷 검색 — Local 모드에서는 비활성 */}
            <div className="flex items-center gap-2">
              <span className={`text-[10px] ${tc.textMuted} flex-1`}>{T.grokWebSearch}</span>
              <select
                value={grokSearchMode}
                onChange={e => setGrokSearchMode(e.target.value as GrokSearchMode)}
                disabled={providerMode === 'all-local'}
                className={`text-[10px] px-2 py-1.5 rounded-lg border ${tc.input} ${tc.inputBorder} disabled:opacity-40`}
              >
                <option value="off">off</option>
                <option value="auto">auto</option>
                <option value="on">on</option>
              </select>
            </div>
            {providerMode === 'all-local'
              ? <div className={`text-[9px] text-amber-500`}>{T.localNoSearch}</div>
              : <div className={`text-[9px] ${tc.textFaint}`}>{T.grokOnly}</div>
            }
          </div>

          <div className="flex-1" />

          {/* Run button */}
          <div className="p-3">
            <button
              onClick={handleRun}
              disabled={isCustom && !customQuery.trim() || (!isCustom && !selectedScenario)}
              className={`w-full py-2.5 rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-2 ${tc.runBtn} disabled:opacity-40`}
            >
              {isRunning ? (
                <><RefreshCw size={14} className="animate-spin" /> stop</>
              ) : (
                <><Play size={14} fill="currentColor" /> Run 4CM</>
              )}
            </button>
            {!keysReady && (
              <p className={`text-[10px] text-center mt-1.5 ${tc.textFaint}`}>
                set {T.apiKeys} first
              </p>
            )}
          </div>
        </div>

        {/* Main */}
        <div className="flex-1 flex flex-col overflow-hidden">

          {/* Header */}
          <div className={`px-5 py-3.5 border-b ${tc.headerBorder} ${tc.header} flex items-center justify-between shrink-0`}>
            <div>
              <div className={`text-sm font-semibold ${tc.text}`}>
                {isCustom ? T.custom : (selectedScenario?.title || 'Select a scenario')}
              </div>
              <div className={`text-[11px] ${tc.textMuted} mt-0.5`}>
                {`${isCustom ? 'normal' : selectedScenario?.risk === 'high' ? T.highRisk : 'normal'} · ${providerMode} · Grok search ${grokSearchMode}`}
                {' · '}{nRounds} rounds
              </div>
              {!isCustom && selectedScenario?.query && (
                <p className={`text-[11px] mt-2 leading-relaxed ${tc.textMuted} max-w-3xl`}>
                  {selectedScenario.query.replace(/\[MOCK RESEARCH SCENARIO\]\s*/i, '')}
                </p>
              )}
            </div>
            <div className={`flex items-center gap-1.5 text-[10px] ${useLocalApi ? 'text-amber-500' : tc.textFaint}`}>
              <Zap size={11} />
              {useLocalApi ? T.judgeLocal : T.judge}
            </div>
          </div>

            <div ref={resultsRef} className="flex-1 overflow-y-auto">

            {/* {T.custom} input */}
            {isCustom && (
              <div className="px-5 pt-4">
                <textarea
                  value={customQuery}
                  onChange={e => setCustomQuery(e.target.value)}
                  disabled={isRunning}
                  placeholder={T.enterQuery}
                  rows={3}
                  className={`w-full px-3 py-2.5 text-sm rounded-xl border ${tc.input} ${tc.inputBorder} focus:outline-none resize-none leading-relaxed disabled:opacity-50`}
                />
              </div>
            )}

            {/* Document upload */}
            <div className="px-5 pt-4">
              <div
                className={`rounded-xl border ${tc.cardBorder} ${tc.card} p-3`}
                onDragOver={e => { e.preventDefault(); e.stopPropagation(); }}
                onDrop={e => {
                  e.preventDefault();
                  e.stopPropagation();
                  const dropped = Array.from(e.dataTransfer.files || []);
                  if (dropped.length) setAttachedFiles(prev => [...prev, ...dropped]);
                }}
              >
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className={`text-[11px] font-semibold ${tc.text}`}>{T.documents}</div>
                    <div className={`text-[10px] ${tc.textFaint}`}>{T.dragDocuments}</div>
                    <div className={`text-[10px] ${tc.textFaint}`}>PDF, CSV, XLSX, DOCX, PPTX, TXT, images · injected into the next run</div>
                  </div>
                  <div className="flex gap-2">
                    <input
                      ref={uploadInputRef}
                      type="file"
                      multiple
                      accept=".pdf,.csv,.xlsx,.xls,.docx,.doc,.txt,.pptx,.ppt,.jpg,.jpeg,.png"
                      className="hidden"
                      onChange={e => setAttachedFiles(Array.from(e.target.files || []))}
                    />
                    <button
                      onClick={() => uploadInputRef.current?.click()}
                      disabled={isRunning || uploadingFiles}
                      className={`text-[11px] px-3 py-1.5 border rounded-lg ${tc.btn} ${tc.btnBorder} disabled:opacity-40`}
                    >
                      {uploadingFiles ? 'Uploading...' : T.attachDocuments}
                    </button>
                    {attachedFiles.length > 0 && (
                      <button
                        onClick={() => { setAttachedFiles([]); if (uploadInputRef.current) uploadInputRef.current.value = ''; }}
                        disabled={isRunning}
                        className={`text-[11px] px-3 py-1.5 border rounded-lg ${tc.btn} ${tc.btnBorder} disabled:opacity-40`}
                      >
                        {T.clearDocuments}
                      </button>
                    )}
                  </div>
                </div>
                {attachedFiles.length > 0 && (
                  <div className={`mt-2 text-[10px] ${tc.textMuted}`}>
                    {attachedFiles.map(f => f.name).join(', ')}
                  </div>
                )}
              </div>
            </div>

            {/* Agent grid */}
            <div className="grid grid-cols-2 gap-3 px-5 pt-4">
              {agents.map((agent, i) => (
                <AgentCard
                  key={agent.id}
                  agent={agent}
                  roundIdx={isRunning ? roundIdx : 0}
                  agentIdx={i}
                  tc={tc}
                  isRunning={isRunning}
                  riskLevel={isCustom ? 'normal' : (selectedScenario?.risk || 'normal')}
                  providerMode={providerMode}
                  selectedProvider={effectiveProviderForAgent(agent.id, i, isRunning ? roundIdx : 0)}
                  onProviderChange={(agentId, provider) => setAgentProviderOverrides(prev => ({ ...prev, [agentId]: provider === 'claude-fallback' ? 'claude' : provider }))}
                  onEdit={setEditingAgent}
                  onViewPrompt={setViewingPrompt}
                  onReset={handleReset}
                />
              ))}
            </div>

            {/* Judge intent triage */}
            {intentTriage && (
              <div className="px-5 pt-4">
                <div className={`border rounded-xl p-3 ${tc.card} ${tc.cardBorder}`}>
                  <div className={`text-[10px] font-semibold uppercase tracking-wider ${tc.textFaint} mb-2 flex items-center gap-1.5`}>
                    <AlertCircle size={11} /> Judge intent triage
                  </div>
                  <div className={`text-[11px] leading-relaxed ${tc.textMuted}`}>
                    <span className={`inline-block text-[9px] px-1.5 py-0.5 rounded mr-1.5 border ${intentTriage.human_review_required ? tc.provGrok : tc.singBadge}`}>
                      {intentTriage.human_review_required ? 'Human review required' : 'Human review not required'}
                    </span>
                    <span className="mr-1.5">Intent: {intentTriage.intent_type || 'unknown'}</span>
                    {intentTriage.gdpr_stage != null && <span className="mr-1.5">· GDPR stage {intentTriage.gdpr_stage}</span>}
                    {intentTriage.high_risk && <span className="mr-1.5">· high risk</span>}
                    {intentTriage.reason && <p className="mt-1">{intentTriage.reason}</p>}
                    {intentTriage.suggested_handling && <p className="mt-1 italic">{intentTriage.suggested_handling}</p>}
                  </div>
                </div>
              </div>
            )}


            {/* Not routed / triage block */}
            {routeBlock && results.length === 0 && (
              <div className="px-5 pt-4">
                <div className={`border rounded-xl p-3 ${tc.card} ${tc.cardBorder}`}>
                  <div className={`text-[10px] font-semibold uppercase tracking-wider ${tc.textFaint} mb-2 flex items-center gap-1.5`}>
                    <AlertCircle size={11} /> Not routed to 4CM
                  </div>
                  <div className={`text-[11px] leading-relaxed ${tc.textMuted}`}>
                    <p className={`font-semibold ${tc.text}`}>{routeBlock.message}</p>
                    {routeBlock.triage?.intent_type && (
                      <p className="mt-1">Intent: {routeBlock.triage.intent_type}{routeBlock.triage.gdpr_stage != null ? ` · GDPR stage ${routeBlock.triage.gdpr_stage}` : ''}</p>
                    )}
                    {routeBlock.triage?.reason && <p className="mt-1">Reason: {routeBlock.triage.reason}</p>}
                    {routeBlock.triage?.suggested_handling && <p className="mt-1 italic">Suggested handling: {routeBlock.triage.suggested_handling}</p>}
                  </div>
                </div>
              </div>
            )}

            {/* Results */}
            {results.length > 0 && (
              <div className="px-5 pt-5 pb-4">
                <div className={`text-[10px] font-semibold uppercase tracking-wider ${tc.textFaint} mb-3`}>
                  Results · {results.length} round{results.length > 1 ? 's' : ''}
                </div>

                <DecisionBriefCard
                  result={firstSingRound ? (results.find(r => r.round === firstSingRound) || results[results.length - 1]) : results[results.length - 1]}
                  finalConclusion={finalConclusion}
                  firstSingRound={firstSingRound}
                  tc={tc}
                />

                {results.map((r, i) => (
                  <RoundBlock key={i} result={r} roundIdx={i} tc={tc} />
                ))}

                {(finalConclusion || firstSingRound !== null) && (
                  <div className={`${selectedPanelClass()} border-2 rounded-xl p-4 mt-3`}>
                    <div className={`text-[10px] font-semibold uppercase tracking-wider text-gray-700 flex items-center gap-1.5 mb-2`}>
                      <Phone size={11} />
                      {firstSingRound
                        ? `The phone rang · first singularity at round ${firstSingRound}`
                        : 'The phone did not ring'}
                    </div>
                    {finalConclusion && (
                      <p className={`text-[12px] leading-relaxed italic ${tc.text}`}>
                        "{finalConclusion}"
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Empty state */}
            {results.length === 0 && !isRunning && (
              <div className={`flex flex-col items-center justify-center py-20 ${tc.textFaint}`}>
                <Phone size={32} className="mb-3 opacity-20" />
                <p className="text-sm">{T.phoneWaiting}</p>
                <p className="text-[11px] mt-1 opacity-60">{T.selectScenario}</p>
              </div>
            )}

          </div>
        </div>
      </div>

      {/* Modals */}
      {editingAgent && (
        <EditModal
          agent={editingAgent}
          otherAgents={agents.filter(a => a.id !== editingAgent.id)}
          tc={tc}
          savedVersions={savedVersions}
          agentSet={selectedScenario?.agentSet || 'government'}
          lang={lang}
          useLocalApi={useLocalApi}
          grokKey={grokKey}
          onSave={handleSaveAgent}
          onClose={() => setEditingAgent(null)}
        />
      )}

      {viewingPrompt && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className={`${tc.editModal} border rounded-xl p-5 w-full max-w-lg mx-4 shadow-2xl`}>
            <div className="flex items-center justify-between mb-3">
              <h3 className={`text-sm font-semibold ${tc.text}`}>
                <FileText size={14} className="inline mr-1.5" />
                {viewingPrompt.name} — system prompt
              </h3>
              <button onClick={() => setViewingPrompt(null)}>
                <X size={16} className={tc.textMuted} />
              </button>
            </div>
            <pre className={`text-[11px] leading-relaxed whitespace-pre-wrap ${tc.textMuted} max-h-80 overflow-y-auto`}>
              {viewingPrompt.prompt}
            </pre>
          </div>
        </div>
      )}

      {showSettingsModal && (
        <SettingsPanel
          tc={tc}
          lang={lang}
          theme={theme}
          scenarios={scenarios}
          grokKey={grokKey}
          pdfTheme={pdfTheme}
          setPdfTheme={setPdfTheme}
          results={results}
          isPdfGenerating={isPdfGenerating}
          onClose={() => setShowSettingsModal(false)}
          onScenariosChange={() => {
            const headers: Record<string, string> = {};
            if (grokKey) headers['X-Grok-Key'] = grokKey;
            fetch(`/4councilmen/fourCM/agents?lang=${lang}`, { headers })
              .then(r => r.json())
              .then(data => {
                const loaded = (data.sets || []).map((s: any) => ({
                  id: s.id, title: s.title || s.id,
                  risk: (s.risk?.trim() || 'normal') as RiskLevel,
                  agentSet: s.id, query: s.query || '',
                }));
                setScenarios(loaded);
                if (loaded.length > 0 && !selectedScenario) setSelectedScenario(loaded[0]);
              });
          }}
          onPdfDownload={() => { setShowSettingsModal(false); handlePDF(); }}
        />
      )}


      {showKeyModal && (
        <KeyModal
          tc={tc}
          claudeSet={!!claudeKey}
          grokSet={!!grokKey}
          useLocalApi={useLocalApi}
          localPayloads={localPayloads}
          localEndpoints={localEndpoints}
          onSave={(c, g) => { handleSaveKeys(c, g); setShowKeyModal(false); }}
          onClear={handleClearKeys}
          onClose={() => setShowKeyModal(false)}
          onLocalApiChange={handleLocalApiChange}
          onSaveLocalPayloads={handleSaveLocalPayloads}
          onSaveLocalEndpoints={handleSaveLocalEndpoints}
        />
      )}

    </div>
  );
};

export default FourCM;
