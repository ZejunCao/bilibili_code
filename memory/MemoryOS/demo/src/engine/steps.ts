/** 存储流 / 检索流步骤定义 */

export interface StepDef {
  id: string
  label: string
  fn: string
  level: 1 | 2
  group?: string
  groupLabel?: string
}

export const WRITE_STEPS: StepDef[] = [
  { id: 'w_add_1', label: '第 1 轮对话', fn: 'ShortTermMemory.add_qa_pair', level: 1 },
  { id: 'w_add_2', label: '第 2 轮对话', fn: 'ShortTermMemory.add_qa_pair', level: 1 },
  { id: 'w_add_3', label: '第 3 轮对话', fn: 'ShortTermMemory.add_qa_pair', level: 1 },
  {
    id: 'w4a',
    label: '挤出短期对话',
    fn: 'Updater.process_short_term_to_mid_term',
    level: 2,
    group: 'promote',
    groupLabel: '短期 → 中期',
  },
  {
    id: 'w4b',
    label: '整理为 page',
    fn: 'Updater → page',
    level: 2,
    group: 'promote',
    groupLabel: '短期 → 中期',
  },
  {
    id: 'w4c',
    label: '判断连续性',
    fn: 'check_conversation_continuity',
    level: 2,
    group: 'promote',
    groupLabel: '短期 → 中期',
  },
  {
    id: 'w4d',
    label: 'page 主题摘要',
    fn: 'generate_page_meta_info',
    level: 2,
    group: 'promote',
    groupLabel: '短期 → 中期',
  },
  {
    id: 'w4e',
    label: 'session 归纳',
    fn: 'gpt_generate_multi_summary',
    level: 2,
    group: 'promote',
    groupLabel: '短期 → 中期',
  },
  {
    id: 'w4f',
    label: '写入中期',
    fn: 'MidTermMemory.insert_pages_into_session',
    level: 2,
    group: 'promote',
    groupLabel: '短期 → 中期',
  },
  { id: 'w_add_4', label: '第 4 轮对话', fn: 'ShortTermMemory.add_qa_pair', level: 1 },
  { id: 'w_add_5', label: '第 5 轮对话', fn: 'ShortTermMemory.add_qa_pair', level: 1 },
  {
    id: 'w_ltm_check',
    label: '检查长期条件',
    fn: 'Memoryos._trigger_profile_and_knowledge_update_if_needed',
    level: 2,
    group: 'longterm',
    groupLabel: '中期 → 长期',
  },
  {
    id: 'w_ltm_profile',
    label: '更新用户画像',
    fn: 'gpt_user_profile_analysis',
    level: 2,
    group: 'longterm',
    groupLabel: '中期 → 长期',
  },
  {
    id: 'w_ltm_knowledge',
    label: '提取长期知识',
    fn: 'gpt_knowledge_extraction',
    level: 2,
    group: 'longterm',
    groupLabel: '中期 → 长期',
  },
  {
    id: 'w_ltm_finalize',
    label: '完成长期更新',
    fn: 'session finalize',
    level: 2,
    group: 'longterm',
    groupLabel: '中期 → 长期',
  },
]

export const READ_STEPS: StepDef[] = [
  {
    id: 'r_mid',
    label: '检索中期',
    fn: 'Retriever._retrieve_mid_term_context',
    level: 2,
    group: 'retrieve',
    groupLabel: '并行检索',
  },
  {
    id: 'r_user_k',
    label: '检索用户知识',
    fn: 'Retriever._retrieve_user_knowledge',
    level: 2,
    group: 'retrieve',
    groupLabel: '并行检索',
  },
  {
    id: 'r_asst_k',
    label: '检索助手知识',
    fn: 'Retriever._retrieve_assistant_knowledge',
    level: 2,
    group: 'retrieve',
    groupLabel: '并行检索',
  },
  {
    id: 'r_short',
    label: '读短期历史',
    fn: 'ShortTermMemory.get_all',
    level: 2,
    group: 'context',
    groupLabel: '读取上下文',
  },
  {
    id: 'r_profile',
    label: '读用户画像',
    fn: 'LongTermMemory.get_raw_user_profile',
    level: 2,
    group: 'context',
    groupLabel: '读取上下文',
  },
  {
    id: 'r_prompt',
    label: '拼装提示',
    fn: 'prompts.GENERATE_SYSTEM_RESPONSE_*',
    level: 2,
    group: 'generate',
    groupLabel: '生成回复',
  },
  {
    id: 'r_generate',
    label: '生成回复',
    fn: 'OpenAIClient.chat_completion',
    level: 2,
    group: 'generate',
    groupLabel: '生成回复',
  },
  {
    id: 'r_writeback',
    label: '写回短期',
    fn: 'memoryos.add_memory',
    level: 2,
    group: 'generate',
    groupLabel: '生成回复',
  },
]
