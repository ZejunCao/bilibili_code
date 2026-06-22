/** 预设对话剧本 — 对应 memoryos.add_memory 的输入 */

export interface ScriptTurn {
  user_input: string
  agent_response: string
}

export const DEMO_SCRIPT: ScriptTurn[] = [
  {
    user_input: '你好！我叫 Tom，在旧金山做数据科学家。',
    agent_response: '你好 Tom！数据科学很有前景，你主要用什么技术栈？',
  },
  {
    user_input: '主要用 Python 做机器学习，TensorFlow 和 PyTorch 都会。',
    agent_response: 'Python 生态很适合 ML。你平时有别的爱好吗？',
  },
  {
    user_input: '周末喜欢跑步，能放松脑子。',
    agent_response: '跑步很不错！你一般在哪儿跑？',
  },
  {
    user_input: '经常在金门大桥附近跑。我还养了只狗叫 Max。',
    agent_response: '金门大桥风景好，Max 一定很可爱！',
  },
  {
    user_input: 'Max 是金毛，三岁了，特别黏人。',
    agent_response: '金毛性格温顺，很适合当陪伴犬。',
  },
]

export const DEMO_QUERY =
  '你还记得我叫什么、做什么工作、住在哪、有什么爱好和宠物吗？'

export const MOCK_LLM_RESPONSE =
  '记得的：你叫 Tom，在旧金山做数据科学家，主要用 Python 做机器学习；' +
  '爱好是周末跑步，常在金门大桥附近；宠物是三岁金毛 Max，很黏人。'
