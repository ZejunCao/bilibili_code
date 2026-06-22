import type { DemoSnapshot, KnowledgeEntry, MidTermPage, MidTermSession } from '../types/memory'

/** 与 memoryos-pypi 落盘 JSON 对齐；向量在演示中省略 */
export const EMBEDDING_OMITTED = 'embedding_vector'

export const DISK_FILE_ORDER = [
  'simple_demo_data/users/demo_user/short_term.json',
  'simple_demo_data/users/demo_user/mid_term.json',
  'simple_demo_data/users/demo_user/long_term_user.json',
  'simple_demo_data/assistants/demo_assistant/long_term_assistant.json',
] as const

export type DiskFilePath = (typeof DISK_FILE_ORDER)[number]
export type DiskFileSnapshot = Record<DiskFilePath, unknown>

function projectPage(p: MidTermPage) {
  return {
    page_id: p.page_id,
    user_input: p.user_input,
    agent_response: p.agent_response,
    timestamp: p.timestamp,
    preloaded: p.preloaded ?? false,
    analyzed: p.analyzed,
    pre_page: p.pre_page,
    next_page: p.next_page,
    meta_info: p.meta_info,
    page_keywords: p.page_keywords ?? [],
    page_embedding: EMBEDDING_OMITTED,
  }
}

function projectSession(s: MidTermSession) {
  return {
    id: s.id,
    summary: s.summary,
    summary_keywords: s.summary_keywords,
    summary_embedding: EMBEDDING_OMITTED,
    details: s.details.map(projectPage),
    L_interaction: s.L_interaction,
    R_recency: s.R_recency,
    N_visit: s.N_visit,
    H_segment: s.H_segment,
    timestamp: s.last_visit_time,
    last_visit_time: s.last_visit_time,
    access_count_lfu: 0,
  }
}

function projectKnowledge(k: KnowledgeEntry) {
  return {
    knowledge: k.knowledge,
    timestamp: k.timestamp,
    knowledge_embedding: EMBEDDING_OMITTED,
  }
}

/** 将演示内存状态投影为 memoryos-pypi 磁盘 JSON 结构 */
export function projectDiskFiles(state: DemoSnapshot): DiskFileSnapshot {
  const sessions = Object.fromEntries(
    Object.entries(state.midTermSessions).map(([id, s]) => [id, projectSession(s)]),
  )
  const access_frequency = Object.fromEntries(
    Object.keys(state.midTermSessions).map((id) => [id, 0]),
  )

  return {
    'simple_demo_data/users/demo_user/short_term.json': state.shortTerm.map((qa) => ({
      user_input: qa.user_input,
      agent_response: qa.agent_response,
      timestamp: qa.timestamp,
    })),
    'simple_demo_data/users/demo_user/mid_term.json': {
      sessions,
      access_frequency,
    },
    'simple_demo_data/users/demo_user/long_term_user.json': {
      user_profiles: state.longTermUser.user_profiles,
      knowledge_base: state.longTermUser.knowledge_base.map(projectKnowledge),
      assistant_knowledge: [],
    },
    'simple_demo_data/assistants/demo_assistant/long_term_assistant.json': {
      user_profiles: {},
      knowledge_base: [],
      assistant_knowledge: state.longTermAssistant.assistant_knowledge.map(projectKnowledge),
    },
  }
}

export function diskFileChanged(before: unknown, after: unknown): boolean {
  return JSON.stringify(before) !== JSON.stringify(after)
}
