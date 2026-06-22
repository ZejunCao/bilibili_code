/** 每步说明 + artifacts 最外层字段注释（字段名对齐源码变量） */

export interface StepGuide {
  detail: string
  artifactFieldComments: Record<string, string>
}

export const STEP_GUIDES: Record<string, StepGuide> = {
  w_add_1: {
    detail: '记录第一轮用户与助手的对话，写入短期记忆。',
    artifactFieldComments: {
      qa_pair: '本轮写入短期记忆的一问一答',
      max_capacity: '短期记忆最多保留几条',
    },
  },
  w_add_2: {
    detail: '再记一轮对话，短期里共有两条最近记录。',
    artifactFieldComments: {
      qa_pair: '本轮新写入的一问一答',
      max_capacity: '短期记忆容量上限',
    },
  },
  w_add_3: {
    detail:
      '再记一轮对话，短期已满。对齐 add_memory：下次写入前会先 process_short_term_to_mid_term（本演示紧接「短期→中期」）。',
    artifactFieldComments: {
      qa_pair: '本轮新写入的一问一答',
      max_capacity: '短期记忆容量上限',
    },
  },
  w4a: {
    detail:
      '短期已满时，while is_full(): pop_oldest()，将全部 QA 弹出准备晋升中期（对齐 updater）。',
    artifactFieldComments: {
      evicted_qas: '从短期弹出的全部一问一答',
      short_term_len: '弹出后短期剩余条数（通常为 0）',
    },
  },
  w4b: {
    detail: '把弹出的对话整理成中期里的 page，加入本批 page 列表。',
    artifactFieldComments: {
      current_batch_pages: '本批待写入中期的 page 列表',
    },
  },
  w4c: {
    detail:
      '对每个 page 做连续性判断，供后续生成 meta_info 时决定是否承接前文。',
    artifactFieldComments: {
      current_batch_pages: '已处理连续性后的 page 列表',
      is_continuous: '本批最后一个 page 的连续性判定结果',
    },
  },
  w4d: {
    detail: '为每个 page 生成主题摘要，写入 meta_info。',
    artifactFieldComments: {
      current_batch_pages: '已写好 meta_info 的 page 列表',
    },
  },
  w4e: {
    detail: '对本批 page 做归纳，得到 session 级摘要和关键词。',
    artifactFieldComments: {
      multi_summary_result: '多主题归纳的完整结果',
    },
  },
  w4f: {
    detail: '把摘要、page 和关键词写入中期 session，并计算 H_segment。',
    artifactFieldComments: {
      processed_details: 'add_session 里逐 page 补全 embedding 等后的列表',
      session_obj: '写入 sessions 的对象；details 即上方的 processed_details',
      details: '与 processed_details 同一列表',
    },
  },
  w_add_4: {
    detail:
      '记录第四轮对话：写入短期，并合并进已有中期 session（insert_pages_into_session），L_interaction 与 H_segment 随之更新。',
    artifactFieldComments: {
      qa_pair: '本轮一问一答',
      max_capacity: '短期记忆容量上限',
      mid_session_update: '中期 session 新增 page 及更新后的 H_segment',
    },
  },
  w_add_5: {
    detail:
      '记录第五轮对话：同样写入短期并合并进中期 session，page 增至 5 条，H_segment 继续升高。',
    artifactFieldComments: {
      qa_pair: '本轮一问一答',
      max_capacity: '短期记忆容量上限',
      mid_session_update: '中期 session 新增 page 及更新后的 H_segment',
    },
  },
  w_ltm_check: {
    detail:
      'add_memory 末尾逻辑：有中期记忆则看堆顶 session 的 H_segment 是否达 mid_term_heat_threshold；达标则分析未处理 page。',
    artifactFieldComments: {
      heap_top: 'H_segment 最高的 session',
      H_segment: '堆顶 session 的 H_segment',
      mid_term_heat_threshold: '长期更新门槛（默认 5）',
      triggered: '是否触发长期更新',
      unanalyzed_pages_count: '待分析 page 数',
    },
  },
  w_ltm_profile: {
    detail: '从对话中提炼用户画像，写入长期记忆。',
    artifactFieldComments: {
      updated_user_profile: '更新后的用户画像全文',
    },
  },
  w_ltm_knowledge: {
    detail: '从对话中提炼用户与助手的长期知识点，分别写入对应存储。',
    artifactFieldComments: {
      new_user_private_knowledge: '新提取的用户私有知识（多条）',
      new_assistant_knowledge: '新提取的助手知识（多条）',
    },
  },
  w_ltm_finalize: {
    detail:
      '画像与知识写入长期后收尾：page 标记 analyzed，N_visit / L_interaction 清零并重算 H_segment（见右侧说明）。',
    artifactFieldComments: {
      message: '本步收尾操作说明',
    },
  },
  r_mid: {
    detail:
      '根据用户问题检索中期 session；命中时 N_visit++ 并重算 H_segment（对齐 mid_term.search_sessions 的加热）。',
    artifactFieldComments: {
      query: '用户当前提问',
      retrieved_pages: '命中的相关 page 列表',
      N_visit: '检索命中后更新的 N_visit',
      H_segment: '重算后的 H_segment',
    },
  },
  r_user_k: {
    detail: '在长期记忆中查找与用户相关的私有知识。',
    artifactFieldComments: {
      retrieved_user_knowledge: '命中的用户知识列表',
    },
  },
  r_asst_k: {
    detail: '在长期记忆中查找与助手相关的知识（与用户知识检索同时进行）。',
    artifactFieldComments: {
      retrieved_assistant_knowledge: '命中的助手知识列表',
      parallel_with: '演示中与「检索用户知识」同步进行',
    },
  },
  r_short: {
    detail: '读取短期记忆中的最近几轮对话，供生成回答时参考。',
    artifactFieldComments: {
      short_term_memory: '短期记忆中的问答列表',
      history_text: '整理成文字的近期对话',
    },
  },
  r_profile: {
    detail: '读取已保存的用户画像，作为回答时的背景信息。',
    artifactFieldComments: {
      user_profile: '用户画像全文',
    },
  },
  r_prompt: {
    detail: '把近期对话、检索结果、画像与知识等拼成发给模型的提示。',
    artifactFieldComments: {
      system: '系统提示部分',
      user: '用户提示部分（含历史、检索、画像、问题等）',
    },
  },
  r_generate: {
    detail: '根据提示生成回答（演示中使用预设回复）。',
    artifactFieldComments: {
      response_content: '模型生成的回答正文',
    },
  },
  r_writeback: {
    detail: '把本轮问答记回短期记忆，供后续对话继续使用。',
    artifactFieldComments: {
      qa_pair: '写回短期的一问一答',
    },
  },
}

export function getStepGuide(stepId: string): StepGuide {
  return (
    STEP_GUIDES[stepId] ?? {
      detail: '（暂无说明）',
      artifactFieldComments: {},
    }
  )
}
