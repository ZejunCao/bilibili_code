import {
  CONTINUITY_CHECK_SYSTEM_PROMPT,
  CONTINUITY_CHECK_USER_PROMPT,
  GENERATE_SYSTEM_RESPONSE_SYSTEM_PROMPT,
  GENERATE_SYSTEM_RESPONSE_USER_PROMPT,
  KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT,
  KNOWLEDGE_EXTRACTION_USER_PROMPT,
  META_INFO_SYSTEM_PROMPT,
  META_INFO_USER_PROMPT,
  MULTI_SUMMARY_SYSTEM_PROMPT,
  MULTI_SUMMARY_USER_PROMPT,
  PERSONALITY_ANALYSIS_SYSTEM_PROMPT,
  PERSONALITY_ANALYSIS_USER_PROMPT,
} from '../data/promptZh'
import type { DemoSnapshot, LlmCallPreview, LlmMessage, MidTermPage } from '../types/memory'

function fill(template: string, vars: Record<string, string>): string {
  return template.replace(/\{(\w+)\}/g, (_, key: string) => vars[key] ?? `{${key}}`)
}

export function toMessages(system: string, user: string): LlmMessage[] {
  return [
    { role: 'system', content: system },
    { role: 'user', content: user },
  ]
}

function pageDialogue(page: MidTermPage): string {
  return `User: ${page.user_input}\nAssistant: ${page.agent_response}`
}

function pagesConversation(pages: MidTermPage[]): string {
  return pages
    .map(
      (p) =>
        `User: ${p.user_input} (Timestamp: ${p.timestamp})\nAssistant: ${p.agent_response} (Timestamp: ${p.timestamp})`,
    )
    .join('\n')
}

/** 与 utils.check_conversation_continuity 一致：previous_page 为 null 时上一页文本为空字符串 */
export function buildContinuityCall(
  prev: MidTermPage | null,
  curr: MidTermPage,
  mockResponse: string,
): LlmCallPreview {
  const user = fill(CONTINUITY_CHECK_USER_PROMPT, {
    prev_user: prev?.user_input ?? '',
    prev_agent: prev?.agent_response ?? '',
    curr_user: curr.user_input,
    curr_agent: curr.agent_response,
  })
  const label = prev
    ? `对话连续性判断 · check_conversation_continuity（${prev.page_id} → ${curr.page_id}）`
    : `对话连续性判断 · check_conversation_continuity（无上一 page → ${curr.page_id}）`
  return {
    label,
    messages: toMessages(CONTINUITY_CHECK_SYSTEM_PROMPT, user),
    mockResponse,
  }
}

export function buildMetaInfoCall(
  lastMeta: string | null,
  page: MidTermPage,
  mockResponse: string,
): LlmCallPreview {
  const user = fill(META_INFO_USER_PROMPT, {
    last_meta: lastMeta ?? 'None',
    new_dialogue: pageDialogue(page),
  })
  return {
    label: `page 主题摘要 · generate_page_meta_info（${page.page_id}）`,
    messages: toMessages(META_INFO_SYSTEM_PROMPT, user),
    mockResponse,
  }
}

export function buildMultiSummaryCall(
  pages: MidTermPage[],
  mockResponse: string,
): LlmCallPreview {
  const text = pages
    .map((p) => `User: ${p.user_input}\nAssistant: ${p.agent_response}`)
    .join('\n')
  const user = fill(MULTI_SUMMARY_USER_PROMPT, { text })
  return {
    label: 'session 多主题归纳 · gpt_generate_multi_summary',
    messages: toMessages(MULTI_SUMMARY_SYSTEM_PROMPT, user),
    mockResponse,
  }
}

export function buildProfileAnalysisCall(
  pages: MidTermPage[],
  existingProfile: string,
  mockResponse: string,
): LlmCallPreview {
  const user = fill(PERSONALITY_ANALYSIS_USER_PROMPT, {
    conversation: pagesConversation(pages),
    existing_user_profile: existingProfile || 'None',
  })
  return {
    label: '用户画像分析 · gpt_user_profile_analysis',
    messages: toMessages(PERSONALITY_ANALYSIS_SYSTEM_PROMPT, user),
    mockResponse,
  }
}

export function buildKnowledgeExtractionCall(
  pages: MidTermPage[],
  mockResponse: string,
): LlmCallPreview {
  const user = fill(KNOWLEDGE_EXTRACTION_USER_PROMPT, {
    conversation: pagesConversation(pages),
  })
  return {
    label: '长期知识抽取 · gpt_knowledge_extraction',
    messages: toMessages(KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT, user),
    mockResponse,
  }
}

export function buildGenerateResponseCall(
  state: DemoSnapshot,
  mockResponse?: string,
): LlmCallPreview {
  const r = state.lastRetrieval
  const profile =
    state.longTermUser.user_profiles[state.config.userId]?.data ?? 'None'
  const history =
    state.shortTerm
      .map((qa) => `User: ${qa.user_input}\nAssistant: ${qa.agent_response}`)
      .join('\n') || '(空)'
  const retrieval =
    r?.pages
      .map(
        (p) =>
          `【Historical Memory】\nUser: ${p.user_input}\nAssistant: ${p.agent_response}\nTime: ${p.timestamp}\nConversation chain overview: ${p.meta_info ?? 'N/A'}`,
      )
      .join('\n\n') ?? '（无中期命中）'
  const assistantKnowledgeText =
    r?.assistantKnowledge.length ?
      '【Assistant Knowledge Base】\n' +
      r.assistantKnowledge
        .map((k) => `- ${k.knowledge} (Recorded: ${k.timestamp})`)
        .join('\n')
    : '【Assistant Knowledge Base】\n- No relevant assistant knowledge found for this query.\n'
  const metaDataText = '【Current Conversation Metadata】\nNone provided for this turn.'
  const relationship = 'friend'
  const system = fill(GENERATE_SYSTEM_RESPONSE_SYSTEM_PROMPT, {
    relationship,
    assistant_knowledge_text: assistantKnowledgeText,
    meta_data_text: metaDataText,
  })
  const user = fill(GENERATE_SYSTEM_RESPONSE_USER_PROMPT, {
    history_text: history,
    retrieval_text: retrieval,
    background: profile,
    relationship,
    query: state.query,
  })
  return {
    label: '生成回复 · OpenAIClient.chat_completion',
    messages: toMessages(system, user),
    mockResponse,
  }
}

export function collectUnanalyzedPages(state: DemoSnapshot): MidTermPage[] {
  return Object.values(state.midTermSessions).flatMap((s) =>
    s.details.filter((p) => !p.analyzed),
  )
}
