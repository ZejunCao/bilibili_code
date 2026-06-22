import { DEMO_QUERY, DEMO_SCRIPT, MOCK_LLM_RESPONSE } from '../data/script'
import {
  MOCK_CONTINUITY,
  MOCK_KNOWLEDGE,
  MOCK_META_INFO,
  MOCK_MULTI_SUMMARY,
  MOCK_PROFILE,
  MOCK_RETRIEVE_ASST_K,
  MOCK_RETRIEVE_MID,
  MOCK_RETRIEVE_USER_K,
} from '../data/fixtures'
import type {
  DemoSnapshot,
  MidTermPage,
  MidTermSession,
  ShortTermQA,
  StepLog,
} from '../types/memory'
import { getStepGuide } from '../data/stepGuides'
import {
  buildContinuityCall,
  buildGenerateResponseCall,
  buildKnowledgeExtractionCall,
  buildMetaInfoCall,
  buildMultiSummaryCall,
  buildProfileAnalysisCall,
  collectUnanalyzedPages,
} from './llmPromptBuilders'
import { READ_STEPS, WRITE_STEPS } from './steps'
import { EMBEDDING_OMITTED, projectDiskFiles } from './diskFiles'
import {
  buildSegmentHeatFormulaText,
  computeSegmentHeat,
} from './segmentHeatFormula'
import { getRetrieveFlowGuide } from './retrieveFlowGuides'
import type { LlmCallPreview } from '../types/memory'

let pageCounter = 0
function genPageId() {
  pageCounter += 1
  return `page_${String(pageCounter).padStart(3, '0')}`
}

let sessionCounter = 0
function genSessionId() {
  sessionCounter += 1
  return `session_${String(sessionCounter).padStart(3, '0')}`
}

function ts(offsetMin = 0) {
  const d = new Date(2026, 5, 2, 10, 0, 0)
  d.setMinutes(d.getMinutes() + offsetMin)
  return d.toISOString().slice(0, 19).replace('T', ' ')
}

function clone<T>(v: T): T {
  return JSON.parse(JSON.stringify(v)) as T
}

/** 预生成快照时跳过会阻塞步进的前置校验 */
let buildingPresets = false

function emptySnapshot(): DemoSnapshot {
  return {
    config: {
      userId: 'demo_user',
      assistantId: 'demo_assistant',
      shortTermCapacity: 3,
      midTermHeatThreshold: 5.0,
    },
    shortTerm: [],
    midTermSessions: {},
    heapTop: null,
    longTermUser: {
      user_profiles: {
        demo_user: { data: 'None', last_updated: '' },
      },
      knowledge_base: [],
    },
    longTermAssistant: { assistant_knowledge: [] },
    pendingEvictedQas: [],
    pendingPages: [],
    pendingMultiSummary: null,
    lastEvictedPageId: null,
    lastEvictedPageForContinuity: null,
    writeStepIndex: 0,
    readStepIndex: 0,
    writeLogs: WRITE_STEPS.map((s) => ({
      id: s.id,
      label: s.label,
      fn: s.fn,
      description: getStepGuide(s.id).detail,
      before: null,
      after: null,
      done: false,
    })),
    readLogs: READ_STEPS.map((s) => ({
      id: s.id,
      label: s.label,
      fn: s.fn,
      description: getStepGuide(s.id).detail,
      before: null,
      after: null,
      done: false,
    })),
    query: DEMO_QUERY,
    lastRetrieval: null,
    lastPrompt: null,
  }
}

function computeHeat(s: MidTermSession): number {
  return computeSegmentHeat({
    N_visit: s.N_visit,
    L_interaction: s.L_interaction,
    R_recency: s.R_recency,
  })
}

function rebuildHeap(sessions: Record<string, MidTermSession>) {
  let top: { sessionId: string; heat: number } | null = null
  for (const [id, s] of Object.entries(sessions)) {
    s.H_segment = computeHeat(s)
    if (!top || s.H_segment > top.heat) top = { sessionId: id, heat: s.H_segment }
  }
  return top
}

/** 对齐 insert_pages_into_session：新 QA 同时写入短期，并合并进已有中期 session */
function appendQaToMidSession(state: DemoSnapshot, qa: ShortTermQA) {
  const sid = Object.keys(state.midTermSessions)[0]
  if (!sid) return null
  const session = state.midTermSessions[sid]
  const page: MidTermPage = {
    page_id: genPageId(),
    user_input: qa.user_input,
    agent_response: qa.agent_response,
    timestamp: qa.timestamp,
    pre_page: session.details.at(-1)?.page_id ?? null,
    next_page: null,
    meta_info: null,
    analyzed: false,
  }
  const prev = session.details.at(-1)
  if (prev) prev.next_page = page.page_id
  session.details.push(page)
  session.L_interaction = session.details.length
  session.last_visit_time = qa.timestamp
  session.H_segment = computeHeat(session)
  state.heapTop = rebuildHeap(state.midTermSessions)
  return {
    session_id: sid,
    page_id: page.page_id,
    details_count: session.details.length,
    L_interaction: session.L_interaction,
    H_segment: session.H_segment,
  }
}

export class DemoEngine {
  /** 存储流运行时状态 */
  state: DemoSnapshot = emptySnapshot()
  /** 检索流运行时状态（与存储流独立，从预跑完成存储后的快照起步） */
  readState: DemoSnapshot | null = null
  writeSnapshots: DemoSnapshot[] = []
  presetWriteLogs: StepLog[] = []
  readSnapshots: DemoSnapshot[] = []
  presetReadLogs: StepLog[] = []

  private freshReadLogs(): StepLog[] {
    return READ_STEPS.map((s) => ({
      id: s.id,
      label: s.label,
      fn: s.fn,
      description: getStepGuide(s.id).detail,
      before: null,
      after: null,
      done: false,
    }))
  }

  /** 检索流初始数据：存储流全部跑完后的快照，不依赖用户是否亲手点过存储流 */
  ensureReadState(): DemoSnapshot {
    if (this.readState) return this.readState
    const base = this.readSnapshots[0]
    if (!base) {
      this.readState = emptySnapshot()
      return this.readState
    }
    this.readState = clone(base)
    this.readState.readStepIndex = 0
    this.readState.readLogs = this.freshReadLogs()
    this.readState.lastRetrieval = null
    this.readState.lastPrompt = null
    return this.readState
  }

  /** 预跑全部步骤，生成每步完成后的快照（演示脚本固定，可任意跳转） */
  private rebuildPresets() {
    buildingPresets = true
    try {
      pageCounter = 0
      sessionCounter = 0
      const temp = new DemoEngine()
      temp.state = emptySnapshot()

      this.writeSnapshots = [clone(temp.state)]
      this.presetWriteLogs = []
      for (let n = 0; n < WRITE_STEPS.length; n++) {
        if (!temp.getWriteStep()) break
        const res = temp.executeWriteStep()
        if (!res.ok) {
          console.warn('[MemoryOS demo] 预设存储流步骤失败:', temp.getWriteStep()?.id ?? n, res.message)
          break
        }
        const i = temp.state.writeStepIndex - 1
        this.presetWriteLogs.push(clone(temp.state.writeLogs[i]) as StepLog)
        this.writeSnapshots.push(clone(temp.state))
      }

      this.readSnapshots = [clone(temp.state)]
      this.presetReadLogs = []
      for (let n = 0; n < READ_STEPS.length; n++) {
        if (!temp.getReadStep()) break
        const res = temp.executeReadStep()
        if (!res.ok) {
          console.warn('[MemoryOS demo] 预设检索流步骤失败:', temp.getReadStep()?.id ?? n, res.message)
          break
        }
        const i = temp.state.readStepIndex - 1
        this.presetReadLogs.push(clone(temp.state.readLogs[i]) as StepLog)
        this.readSnapshots.push(clone(temp.state))
      }
    } finally {
      buildingPresets = false
    }
  }

  getPresetSnapshot(flow: 'write' | 'read', stepIndex: number): DemoSnapshot {
    const snaps = flow === 'write' ? this.writeSnapshots : this.readSnapshots
    if (!snaps.length) return this.state
    return snaps[stepIndex + 1] ?? snaps[snaps.length - 1]
  }

  getPresetLog(flow: 'write' | 'read', stepIndex: number): StepLog | null {
    const logs = flow === 'write' ? this.presetWriteLogs : this.presetReadLogs
    return logs[stepIndex] ?? null
  }

  reset() {
    pageCounter = 0
    sessionCounter = 0
    this.rebuildPresets()
    this.state = emptySnapshot()
    this.readState = null
  }

  getWriteStep() {
    return WRITE_STEPS[this.state.writeStepIndex] ?? null
  }

  getReadStep() {
    const idx = this.readState?.readStepIndex ?? 0
    return READ_STEPS[idx] ?? null
  }

  writeComplete() {
    return this.state.writeStepIndex >= WRITE_STEPS.length
  }

  readComplete() {
    const idx = this.readState?.readStepIndex ?? 0
    return idx >= READ_STEPS.length
  }

  executeWriteStep(): { ok: boolean; message?: string } {
    const step = this.getWriteStep()
    if (!step) return { ok: false, message: '存储流已全部完成' }

    const before = clone(this.state)
    let artifacts: unknown
    let llmCalls: LlmCallPreview[] | undefined
    let sidePanel: { title: string; content: string } | undefined

    switch (step.id) {
      case 'w_add_1':
      case 'w_add_2':
      case 'w_add_3':
      case 'w_add_4':
      case 'w_add_5': {
        const idx = { w_add_1: 0, w_add_2: 1, w_add_3: 2, w_add_4: 3, w_add_5: 4 }[step.id]!
        if (
          this.state.shortTerm.length >= this.state.config.shortTermCapacity &&
          Object.keys(this.state.midTermSessions).length === 0
        ) {
          return { ok: false, message: '短期已满，请按顺序完成 W4a–W4f 晋升步骤' }
        }
        const turn = DEMO_SCRIPT[idx]
        const qa: ShortTermQA = {
          ...turn,
          timestamp: ts(idx),
        }
        this.state.shortTerm.push(qa)
        if (this.state.shortTerm.length > this.state.config.shortTermCapacity) {
          this.state.shortTerm.shift()
        }
        let midUpdate: ReturnType<typeof appendQaToMidSession> = null
        if (step.id === 'w_add_4' || step.id === 'w_add_5') {
          if (!Object.keys(this.state.midTermSessions).length) {
            return { ok: false, message: '请先完成「短期 → 中期」写入' }
          }
          midUpdate = appendQaToMidSession(this.state, qa)
        }
        artifacts = {
          qa_pair: qa,
          max_capacity: this.state.config.shortTermCapacity,
          mid_session_update: midUpdate,
        }
        break
      }
      case 'w4a': {
        if (this.state.shortTerm.length < this.state.config.shortTermCapacity) {
          return { ok: false, message: '短期未满，请先添加 3 轮对话' }
        }
        const evicted: ShortTermQA[] = []
        while (this.state.shortTerm.length >= this.state.config.shortTermCapacity) {
          const qa = this.state.shortTerm.shift()
          if (qa?.user_input && qa?.agent_response) evicted.push(qa)
        }
        if (!evicted.length) return { ok: false, message: '短期无 QA 可弹出' }
        this.state.pendingEvictedQas = evicted
        artifacts = {
          evicted_qas: evicted,
          short_term_len: this.state.shortTerm.length,
        }
        break
      }
      case 'w4b': {
        if (!this.state.pendingEvictedQas.length) {
          return { ok: false, message: '请先执行 W4a' }
        }
        this.state.pendingPages = this.state.pendingEvictedQas.map((qa) => ({
          page_id: genPageId(),
          user_input: qa.user_input,
          agent_response: qa.agent_response,
          timestamp: qa.timestamp,
          pre_page: null,
          next_page: null,
          meta_info: null,
          analyzed: false,
          preloaded: false,
        }))
        artifacts = { current_batch_pages: this.state.pendingPages }
        break
      }
      case 'w4c': {
        if (!this.state.pendingPages.length) return { ok: false, message: '请先执行 W4b' }
        // 对齐 updater.py：每一页都调用 check_conversation_continuity；
        // temp_last 可为 null（首屏或跨批首条），utils 内上一页 User/Assistant 为空字符串。
        llmCalls = []
        let tempLast: MidTermPage | null = this.state.lastEvictedPageForContinuity
        const continuityResults: { page_id: string; is_continuous: boolean }[] = []

        for (const p of this.state.pendingPages) {
          const mockResponse =
            tempLast === null ? 'false' : MOCK_CONTINUITY.page2.continuous ? 'true' : 'false'
          llmCalls.push(buildContinuityCall(tempLast, p, mockResponse))
          const is_continuous = mockResponse === 'true'

          if (is_continuous && tempLast) {
            p.pre_page = tempLast.page_id
            const prevInBatch = this.state.pendingPages.find(
              (x) => x.page_id === tempLast!.page_id,
            )
            if (prevInBatch) {
              prevInBatch.next_page = p.page_id
            } else {
              tempLast.next_page = p.page_id
            }
          }

          continuityResults.push({ page_id: p.page_id, is_continuous })
          tempLast = p
        }

        artifacts = {
          current_batch_pages: this.state.pendingPages,
          is_continuous: continuityResults.at(-1)?.is_continuous ?? false,
        }
        break
      }
      case 'w4d': {
        if (!this.state.pendingPages.length) return { ok: false, message: '请先执行 W4c' }
        llmCalls = []
        let lastMeta: string | null = null
        this.state.pendingPages.forEach((p, i) => {
          const key = `page_${i + 1}` as keyof typeof MOCK_META_INFO
          const mock = MOCK_META_INFO[key] ?? `page ${i + 1} 摘要`
          llmCalls!.push(buildMetaInfoCall(lastMeta, p, mock))
          p.meta_info = mock
          lastMeta = mock
        })
        artifacts = { current_batch_pages: this.state.pendingPages }
        break
      }
      case 'w4e': {
        if (!this.state.pendingPages.length) return { ok: false, message: '请先执行 W4d' }
        this.state.pendingMultiSummary = MOCK_MULTI_SUMMARY.summaries
        const mockJson = JSON.stringify(MOCK_MULTI_SUMMARY.summaries, null, 2)
        llmCalls = [buildMultiSummaryCall(this.state.pendingPages, mockJson)]
        artifacts = { multi_summary_result: MOCK_MULTI_SUMMARY }
        break
      }
      case 'w4f': {
        if (!this.state.pendingPages.length || !this.state.pendingMultiSummary) {
          return { ok: false, message: '请先执行 W4e' }
        }
        const sid = genSessionId()
        const summary = this.state.pendingMultiSummary[0]
        const pages_to_insert = this.state.pendingPages.map((p) => ({ ...p }))
        const summary_keywords = summary.keywords
        const currentTs = ts(10)

        // 对齐 mid_term.add_session：先 processed_details，再 session_obj
        const processed_details = pages_to_insert.map((p) => ({
          ...p,
          page_keywords: p.page_keywords?.length ? p.page_keywords : [],
          page_embedding: EMBEDDING_OMITTED,
        }))

        const N_visit = 0
        const L_interaction = processed_details.length
        const R_recency = 1.0

        const session_obj = {
          id: sid,
          summary: summary.content,
          summary_keywords,
          summary_embedding: EMBEDDING_OMITTED,
          details: processed_details,
          L_interaction,
          R_recency,
          N_visit,
          H_segment: 0.0,
          timestamp: currentTs,
          last_visit_time: currentTs,
          access_count_lfu: 0,
        }

        session_obj.H_segment = computeSegmentHeat({
          N_visit,
          L_interaction,
          R_recency,
        })

        const sessionForState: MidTermSession = {
          id: sid,
          summary: session_obj.summary,
          summary_keywords: session_obj.summary_keywords,
          H_segment: session_obj.H_segment,
          N_visit,
          L_interaction,
          R_recency,
          last_visit_time: currentTs,
          details: pages_to_insert,
        }

        this.state.midTermSessions[sid] = sessionForState
        this.state.heapTop = rebuildHeap(this.state.midTermSessions)
        const lastPage = pages_to_insert.at(-1) ?? null
        this.state.lastEvictedPageId = lastPage?.page_id ?? null
        this.state.lastEvictedPageForContinuity = lastPage ? { ...lastPage } : null
        this.state.pendingEvictedQas = []
        this.state.pendingPages = []
        this.state.pendingMultiSummary = null
        artifacts = { processed_details, session_obj }
        sidePanel = {
          title: 'H_segment',
          content: buildSegmentHeatFormulaText({
            N_visit,
            L_interaction,
            R_recency,
            H_segment: session_obj.H_segment,
          }),
        }
        break
      }
      case 'w_ltm_check': {
        if (!this.state.midTermSessions || !Object.keys(this.state.midTermSessions).length) {
          return { ok: false, message: '中期尚无 session，请先完成「短期 → 中期」' }
        }
        const top = this.state.heapTop
        if (!top) return { ok: false, message: '无中期 session' }
        const pass = top.heat >= this.state.config.midTermHeatThreshold
        const unanalyzed = Object.values(this.state.midTermSessions)
          .flatMap((s) => s.details.filter((p) => !p.analyzed))
        artifacts = {
          heap_top: top,
          H_segment: top.heat,
          mid_term_heat_threshold: this.state.config.midTermHeatThreshold,
          triggered: pass,
          unanalyzed_pages_count: unanalyzed.length,
        }
        sidePanel = {
          title: '检查中期条件',
          content:
            '若存在中期记忆，则取 H_segment 最高的 session，检查其 H_segment 是否不低于 mid_term_heat_threshold；' +
            '满足则对其中尚未分析的 page 做画像与知识提取。（演示不拦截未达标情况，可继续后续长期更新步骤。）',
        }
        break
      }
      case 'w_ltm_profile': {
        const uid = this.state.config.userId
        const pages = collectUnanalyzedPages(this.state)
        const existing =
          this.state.longTermUser.user_profiles[uid]?.data ?? 'None'
        llmCalls = [buildProfileAnalysisCall(pages, existing, MOCK_PROFILE)]
        this.state.longTermUser.user_profiles[uid] = {
          data: MOCK_PROFILE,
          last_updated: ts(20),
        }
        artifacts = { updated_user_profile: MOCK_PROFILE }
        break
      }
      case 'w_ltm_knowledge': {
        const pages = collectUnanalyzedPages(this.state)
        const mockRaw = `【User Private Data】\n${MOCK_KNOWLEDGE.private}\n\n【Assistant Knowledge】\n${MOCK_KNOWLEDGE.assistant_knowledge}`
        llmCalls = [buildKnowledgeExtractionCall(pages, mockRaw)]
        const lines = MOCK_KNOWLEDGE.private.split('\n').filter(Boolean)
        const now = ts(21)
        for (const line of lines) {
          this.state.longTermUser.knowledge_base.push({ knowledge: line, timestamp: now })
        }
        const asst = MOCK_KNOWLEDGE.assistant_knowledge.split('\n').filter(Boolean)
        for (const line of asst) {
          this.state.longTermAssistant.assistant_knowledge.push({ knowledge: line, timestamp: now })
        }
        artifacts = {
          new_user_private_knowledge: lines,
          new_assistant_knowledge: asst,
        }
        break
      }
      case 'w_ltm_finalize': {
        for (const s of Object.values(this.state.midTermSessions)) {
          s.details.forEach((p) => {
            p.analyzed = true
          })
          s.N_visit = 0
          s.L_interaction = 0
          s.H_segment = computeHeat(s)
        }
        this.state.heapTop = rebuildHeap(this.state.midTermSessions)
        artifacts = { message: '所有 page.analyzed=true，N_visit 与 L_interaction 已重置，H_segment 已重算' }
        sidePanel = {
          title: 'H_segment 重置',
          content:
            '把「已经消化进长期」的 session 降温：N_visit 与 L_interaction 清零，' +
            'R_recency 按 last_visit_time 重算；page 仍在中期供检索，但标记为已分析，避免同一批对话反复写长期。',
        }
        break
      }
      default:
        return { ok: false, message: '未知步骤' }
    }

    const after = clone(this.state)
    this.logWrite(
      step.id,
      step.label,
      step.fn,
      getStepGuide(step.id).detail,
      before,
      after,
      artifacts,
      llmCalls,
      sidePanel,
    )
    this.state.writeStepIndex += 1
    return { ok: true }
  }

  executeReadStep(): { ok: boolean; message?: string } {
    // 预跑快照时仍用单份 state（temp 实例尚无 readSnapshots）
    const rs = buildingPresets ? this.state : this.ensureReadState()
    const step = buildingPresets
      ? READ_STEPS[this.state.readStepIndex] ?? null
      : this.getReadStep()
    if (!step) return { ok: false, message: '检索流已全部完成' }

    const before = clone(rs)
    let artifacts: unknown
    let llmCalls: LlmCallPreview[] | undefined
    const sidePanel = getRetrieveFlowGuide(step.id)

    switch (step.id) {
      case 'r_mid': {
        const session = Object.values(rs.midTermSessions)[0]
        if (!session) return { ok: false, message: '无中期数据' }
        const pages: DemoSnapshot['lastRetrieval'] extends null ? never : NonNullable<DemoSnapshot['lastRetrieval']>['pages'] = []
        for (const m of MOCK_RETRIEVE_MID) {
          const p = session.details[m.pageIndex]
          if (p) pages.push({ ...p, score: m.score, session_id: session.id })
        }
        rs.lastRetrieval = {
          pages,
          userKnowledge: [],
          assistantKnowledge: [],
        }
        session.N_visit += 1
        session.H_segment = computeHeat(session)
        rs.heapTop = rebuildHeap(rs.midTermSessions)
        artifacts = {
          query: rs.query,
          retrieved_pages: pages,
          N_visit: session.N_visit,
          H_segment: session.H_segment,
        }
        break
      }
      case 'r_user_k': {
        if (!rs.lastRetrieval) return { ok: false, message: '请先执行中期检索' }
        const kb = rs.longTermUser.knowledge_base
        rs.lastRetrieval.userKnowledge = MOCK_RETRIEVE_USER_K.filter((m) => kb[m.index]).map((m) => ({
          ...kb[m.index],
          score: m.score,
        }))
        artifacts = { retrieved_user_knowledge: rs.lastRetrieval.userKnowledge }
        break
      }
      case 'r_asst_k': {
        if (!rs.lastRetrieval) return { ok: false, message: '请先执行中期检索' }
        const kb = rs.longTermAssistant.assistant_knowledge
        rs.lastRetrieval.assistantKnowledge = MOCK_RETRIEVE_ASST_K.filter((m) => kb[m.index]).map((m) => ({
          ...kb[m.index],
          score: m.score,
        }))
        artifacts = {
          retrieved_assistant_knowledge: rs.lastRetrieval.assistantKnowledge,
          parallel_with: 'r_user_k',
        }
        break
      }
      case 'r_short': {
        artifacts = {
          short_term_memory: rs.shortTerm,
          history_text: rs.shortTerm
            .map((qa) => `User: ${qa.user_input}\nAssistant: ${qa.agent_response}`)
            .join('\n'),
        }
        break
      }
      case 'r_profile': {
        const p = rs.longTermUser.user_profiles[rs.config.userId]
        artifacts = { user_profile: p?.data ?? 'None' }
        break
      }
      case 'r_prompt': {
        const call = buildGenerateResponseCall(rs)
        llmCalls = [call]
        rs.lastPrompt = {
          system: call.messages[0].content,
          user: call.messages[1].content,
        }
        artifacts = { prompt_ready: true }
        break
      }
      case 'r_generate': {
        llmCalls = [buildGenerateResponseCall(rs, MOCK_LLM_RESPONSE)]
        artifacts = { response_content: MOCK_LLM_RESPONSE }
        break
      }
      case 'r_writeback': {
        const qa: ShortTermQA = {
          user_input: rs.query,
          agent_response: MOCK_LLM_RESPONSE,
          timestamp: ts(30),
        }
        rs.shortTerm.push(qa)
        artifacts = { qa_pair: qa }
        break
      }
      default:
        return { ok: false, message: '未知步骤' }
    }

    const after = clone(rs)
    this.logRead(
      step.id,
      step.label,
      step.fn,
      getStepGuide(step.id).detail,
      before,
      after,
      artifacts,
      llmCalls,
      sidePanel,
    )
    if (buildingPresets) {
      this.state.readStepIndex += 1
    } else {
      rs.readStepIndex += 1
    }
    return { ok: true }
  }

  private logWrite(
    id: string,
    _label: string,
    _fn: string,
    _description: string,
    before: DemoSnapshot,
    after: DemoSnapshot,
    artifacts?: unknown,
    llmCalls?: LlmCallPreview[],
    sidePanel?: { title: string; content: string },
  ) {
    const log = this.state.writeLogs.find((l) => l.id === id)
    if (log) {
      log.done = true
      log.before = projectDiskFiles(before)
      log.after = projectDiskFiles(after)
      log.artifacts = artifacts
      log.llmCalls = llmCalls
      log.sidePanel = sidePanel
    }
  }

  private logRead(
    id: string,
    _label: string,
    _fn: string,
    _description: string,
    before: DemoSnapshot,
    after: DemoSnapshot,
    artifacts?: unknown,
    llmCalls?: LlmCallPreview[],
    sidePanel?: { title: string; content: string },
  ) {
    const logs = buildingPresets ? this.state.readLogs : this.ensureReadState().readLogs
    const log = logs.find((l) => l.id === id)
    if (log) {
      log.done = true
      log.before = projectDiskFiles(before)
      log.after = projectDiskFiles(after)
      log.artifacts = artifacts
      log.llmCalls = llmCalls
      log.sidePanel = sidePanel
    }
  }
}

export const engine = new DemoEngine()
