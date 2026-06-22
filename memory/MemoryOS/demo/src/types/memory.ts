/** 对齐 memoryos-pypi 核心数据结构（演示用简化） */

export interface ShortTermQA {
  user_input: string
  agent_response: string
  timestamp: string
}

export interface MidTermPage {
  page_id: string
  user_input: string
  agent_response: string
  timestamp: string
  pre_page: string | null
  next_page: string | null
  meta_info: string | null
  analyzed: boolean
  preloaded?: boolean
  page_keywords?: string[]
}

export interface MidTermSession {
  id: string
  summary: string
  summary_keywords: string[]
  H_segment: number
  N_visit: number
  L_interaction: number
  R_recency: number
  last_visit_time: string
  details: MidTermPage[]
}

export interface KnowledgeEntry {
  knowledge: string
  timestamp: string
}

export interface UserProfile {
  data: string
  last_updated: string
}

export interface LongTermUser {
  user_profiles: Record<string, UserProfile>
  knowledge_base: KnowledgeEntry[]
}

export interface LongTermAssistant {
  assistant_knowledge: KnowledgeEntry[]
}

export interface RetrievedPage extends MidTermPage {
  score: number
  session_id: string
}

/** memoryos-pypi simple_demo_data 落盘文件集合（与 engine/diskFiles 一致） */
export type { DiskFileSnapshot } from '../engine/diskFiles'
import type { DiskFileSnapshot } from '../engine/diskFiles'

export type LlmMessageRole = 'system' | 'user' | 'assistant'

export interface LlmMessage {
  role: LlmMessageRole
  content: string
}

/** 演示用：对齐 utils.py 中 chat_completion 的 messages 入参 */
export interface LlmCallPreview {
  label: string
  messages: LlmMessage[]
  /** 演示脚本中的模拟模型输出（非 messages 数组的一部分） */
  mockResponse?: string
}

export interface StepLog {
  id: string
  label: string
  fn: string
  description: string
  /** 本步执行前的磁盘 JSON 快照 */
  before: DiskFileSnapshot | null
  /** 本步执行后的磁盘 JSON 快照 */
  after: DiskFileSnapshot | null
  artifacts?: unknown
  /** 本步触发的大模型调用（中文 prompt，来自 prompt_zh.py） */
  llmCalls?: LlmCallPreview[]
  /** 非 LLM 步骤时右侧说明（如 w4f 的 H_segment 公式） */
  sidePanel?: { title: string; content: string }
  done: boolean
}

export interface DemoConfig {
  userId: string
  assistantId: string
  shortTermCapacity: number
  midTermHeatThreshold: number
}

export interface DemoSnapshot {
  config: DemoConfig
  shortTerm: ShortTermQA[]
  midTermSessions: Record<string, MidTermSession>
  heapTop: { sessionId: string; heat: number } | null
  longTermUser: LongTermUser
  longTermAssistant: LongTermAssistant
  /** 晋升管道中的临时数据 */
  pendingEvictedQas: ShortTermQA[]
  pendingPages: MidTermPage[]
  pendingMultiSummary: { theme: string; content: string; keywords: string[] }[] | null
  lastEvictedPageId: string | null
  /** 上一批晋升留下的最后一页，供下一批连续性判断（对齐 updater.last_evicted_page_for_continuity） */
  lastEvictedPageForContinuity: MidTermPage | null
  writeStepIndex: number
  readStepIndex: number
  writeLogs: StepLog[]
  readLogs: StepLog[]
  query: string
  lastRetrieval: {
    pages: RetrievedPage[]
    userKnowledge: (KnowledgeEntry & { score: number })[]
    assistantKnowledge: (KnowledgeEntry & { score: number })[]
  } | null
  lastPrompt: { system: string; user: string } | null
}
