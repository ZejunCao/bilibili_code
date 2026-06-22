/** 预设「LLM 输出」— 对应 updater / retriever 中的 gpt_* 调用 */

export const MOCK_CONTINUITY = {
  page2: { continuous: true, reason: '同一用户继续聊跑步地点，话题连贯' },
  page3: { continuous: true, reason: '从跑步自然过渡到宠物，同一对话流' },
}

export const MOCK_META_INFO: Record<string, string> = {
  page_1: '用户汤姆是一位居住在旧金山的数据科学家，他刚向助手介绍了自己的背景，而助手正询问他具体的专业领域。',
  page_2: '技术栈：Python 机器学习（TensorFlow / PyTorch）',
  page_3: '爱好：周末跑步',
}

export const MOCK_MULTI_SUMMARY = {
  summaries: [
    {
      theme: '自我介绍',
      content: '汤姆介绍居住地与职业',
      keywords: ['旧金山', '数据科学家'],
    },
  ],
}

export const MOCK_PROFILE = `【用户画像】
- 姓名：Tom
- 职业：数据科学家（旧金山）
- 技能：Python、TensorFlow、PyTorch、机器学习
- 爱好：周末跑步（常去金门大桥附近）
- 宠物：金毛犬 Max，3 岁，性格黏人`

export const MOCK_KNOWLEDGE = {
  private: `Tom 在旧金山担任数据科学家
Tom 使用 Python 做机器学习
Tom 周末喜欢跑步
Tom 常在金门大桥附近跑步
Tom 养了一只叫 Max 的三岁金毛犬`,
  assistant_knowledge: `金毛犬通常性格温顺，适合作为陪伴犬`,
}

/** 检索演示：与 query 相关的命中（写死分数便于展示） */
export const MOCK_RETRIEVE_MID = [
  { pageIndex: 0, score: 0.82 },
  { pageIndex: 1, score: 0.76 },
  { pageIndex: 2, score: 0.71 },
]

export const MOCK_RETRIEVE_USER_K = [
  { index: 0, score: 0.88 },
  { index: 1, score: 0.85 },
  { index: 3, score: 0.79 },
  { index: 4, score: 0.77 },
]

export const MOCK_RETRIEVE_ASST_K = [{ index: 0, score: 0.65 }]
