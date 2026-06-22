/**
 * 与 memoryos-pypi/prompt_zh.py 对齐的中文 prompt（演示展示用）。
 * 实际运行时 pypi 仍使用 prompts.py 英文版。
 */

export const CONTINUITY_CHECK_SYSTEM_PROMPT =
  "你是对话连续性判断器。仅返回 'true' 或 'false'。"

export const CONTINUITY_CHECK_USER_PROMPT =
  '判断以下两段 page 是否连续（真实延续、无话题切换）。\n' +
  '仅返回 "true" 或 "false"。\n\n' +
  '上一 page：\nUser: {prev_user}\nAssistant: {prev_agent}\n\n' +
  '当前 page：\nUser: {curr_user}\nAssistant: {curr_agent}\n\n' +
  '是否连续？'

export const META_INFO_SYSTEM_PROMPT = `你是对话 meta-summary 更新器。任务：
1. 保留上一段 meta-summary 中的相关上下文
2. 将当前对话的新信息融入
3. 仅输出更新后的摘要，不要解释`

export const META_INFO_USER_PROMPT = `在保持连贯的前提下，将新对话融入并更新对话 meta-summary。

要求：
1. 从上一段 meta-summary 出发（若存在）
2. 根据新对话增补/更新信息
3. 保持简洁（最多 1～2 句）
4. 保持上下文连贯

上一段 Meta-summary：{last_meta}
新对话：
{new_dialogue}

更新后的 Meta-summary：`

export const MULTI_SUMMARY_SYSTEM_PROMPT =
  '你是对话主题分析专家。请生成简洁摘要，最多两个主题，尽量简短。'

export const MULTI_SUMMARY_USER_PROMPT =
  '请分析以下对话，生成极简的子主题摘要（如有），最多两个 theme。\n' +
  '每个摘要尽量短，theme 与 content 仅需少量词。以 JSON 数组格式输出：\n' +
  '[\n  {"theme": "简短主题", "keywords": ["关键词1", "关键词2"], "content": "摘要内容"}\n]\n' +
  '\n对话内容：\n{text}'

export const PERSONALITY_ANALYSIS_SYSTEM_PROMPT = `你是用户偏好分析助手，任务是根据给定对话和维度说明，分析用户的性格与偏好。

对每个维度：
1. 细读对话，判断该维度是否有所体现。
2. 若体现，给出偏好程度：高 / 中 / 低，并简要说明理由（可含时间、人物、情境）。
3. 若未体现，不要强行抽取或列出。

仅输出用户画像相关部分。`

export const PERSONALITY_ANALYSIS_USER_PROMPT = `请根据下面最新一轮用户-AI 对话，结合 90 个性格偏好维度更新用户画像。

90 个维度及说明如下：

[心理模型（基本需求与人格)]
Extraversion: 对社交活动的偏好。
Openness: 对新想法与体验的开放程度。
Agreeableness: 友善、合作倾向。
Conscientiousness: 责任心与条理性。
Neuroticism: 情绪稳定性与敏感度。
Physiological Needs: 对舒适与基本需求的关注。
Need for Security: 对安全与稳定的重视。
Need for Belonging: 对归属与群体的需求。
Need for Self-Esteem: 对尊重与认可的需求。
Cognitive Needs: 对知识与理解的需求。
Aesthetic Appreciation: 对美与艺术的欣赏。
Self-Actualization: 对自我实现的追求。
Need for Order: 对整洁与有序的偏好。
Need for Autonomy: 对独立决策与行动的偏好。
Need for Power: 对影响或控制他人的欲望。
Need for Achievement: 对成就的重视。

[AI 对齐维度]
Helpfulness: AI 回复是否对用户有实际帮助。（反映用户对 AI 的期待）
Honesty: AI 回复是否真实。（反映用户对 AI 的期待）
Safety: 是否避免敏感或有害内容。（反映用户对 AI 的期待）
Instruction Compliance: 是否严格遵循用户指令。
Truthfulness: 内容的准确与真实。
Coherence: 表达清晰、逻辑一致。
Complexity: 偏好详细、复杂的信息。
Conciseness: 偏好简短、清晰的回复。

[内容与兴趣标签]
Science Interest: 对科学话题的兴趣。
Education Interest: 对教育与学习的关注。
Psychology Interest: 对心理学话题的兴趣。
Family Concern: 对家庭与育儿的关注。
Fashion Interest: 对时尚话题的兴趣。
Art Interest: 对艺术的兴趣或参与。
Health Concern: 对身体健康与生活方式的关注。
Financial Management Interest: 对理财与预算的兴趣。
Sports Interest: 对运动与体育的兴趣。
Food Interest: 对烹饪与美食的兴趣。
Travel Interest: 对旅行与探索的兴趣。
Music Interest: 对音乐欣赏或创作的兴趣。
Literature Interest: 对文学与阅读的兴趣。
Film Interest: 对电影与影院的兴趣。
Social Media Activity: 社交媒体使用频率与参与度。
Tech Interest: 对技术与创新的兴趣。
Environmental Concern: 对环境与可持续性的关注。
History Interest: 对历史知识的兴趣。
Political Concern: 对政治与社会议题的兴趣。
Religious Interest: 对宗教与灵性的兴趣。
Gaming Interest: 对电子游戏或桌游的喜爱。
Animal Concern: 对动物或宠物的关注。
Emotional Expression: 偏好直接表露情绪还是克制。
Sense of Humor: 偏好幽默还是严肃的沟通风格。
Information Density: 偏好详细还是简洁的信息。
Language Style: 偏好正式还是随意的语气。
Practicality: 偏好实用建议还是理论讨论。

**任务说明：**
1. 阅读下方已有用户画像
2. 在新对话中寻找上述 90 个维度的证据
3. 将新发现整合进完整的用户画像
4. 每个可识别的维度使用格式：维度名 ( 程度(高/中/低) )
5. 尽可能附简短理由
6. 保留旧画像中的已有洞察，并融入新观察
7. 若某维度在旧画像与新对话中均无法推断，则不要写入

**已有用户画像：**
{existing_user_profile}

**最新用户-AI 对话：**
{conversation}

**更新后的用户画像：**
请在下文给出综合更新后的用户画像，融合已有画像与最新对话的洞察：`

export const KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT = `你是知识抽取助手，任务是从对话中抽取用户私有数据与助手知识。

关注：
1. 用户私有数据：个人身份、偏好或与用户相关的私人事实
2. 助手知识：助手所做、所提供或所展示的明确陈述

抽取结果需极简、客观，使用尽可能短的短语。`

export const KNOWLEDGE_EXTRACTION_USER_PROMPT = `请从下面最新一轮用户-AI 对话中抽取用户私有数据与助手知识。

最新用户-AI 对话：
{conversation}

【User Private Data】
抽取与用户相关的个人信息，极简表述（最短短语）：
- [简短事实]: [最少上下文（含实体与时间）]
- [简短事实]: [最少上下文（含实体与时间）]
- （若无则写 "None"）

【Assistant Knowledge】
抽取助手所展示的内容，格式为 "助手在 [时间/情境] [简短动作]"。极简：
- Assistant [简短动作] at [时间/情境]
- Assistant [简短能力] during [简短情境]
- （若无则写 "None"）
`

export const GENERATE_SYSTEM_RESPONSE_SYSTEM_PROMPT =
  '你是一名沟通专家，在以下对话中始终扮演 {relationship} 的角色。\n' +
  '你具有以下特质与知识：\n{assistant_knowledge_text}\n' +
  '用户画像：\n' +
  '{meta_data_text}\n' +
  '请根据上述特质与语气生成回复。\n'

export const GENERATE_SYSTEM_RESPONSE_USER_PROMPT =
  '<CONTEXT>\n' +
  '你与用户最近的对话：\n' +
  '{history_text}\n\n' +
  '<MEMORY>\n' +
  '与当前对话相关的记忆：\n' +
  '{retrieval_text}\n\n' +
  '<USER TRAITS>\n' +
  '在过去对话中你总结出的用户特征：\n' +
  '{background}\n\n' +
  '请以 {relationship} 的身份继续与用户对话。\n' +
  '用户刚刚说：{query}\n' +
  '请回复用户（限 30 词以内，需用英文）：\n' +
  '回答时请核对所引用信息的时间是否与问题的时间范围一致。'
