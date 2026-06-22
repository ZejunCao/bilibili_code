/** 对齐 memoryos-pypi/mid_term.py 中 compute_segment_heat */

export const HEAT_ALPHA = 1.0
export const HEAT_BETA = 1.0
export const HEAT_GAMMA = 1
export const RECENCY_TAU_HOURS = 24

export function computeSegmentHeat(session: {
  N_visit: number
  L_interaction: number
  R_recency: number
}): number {
  return (
    HEAT_ALPHA * session.N_visit +
    HEAT_BETA * session.L_interaction +
    HEAT_GAMMA * session.R_recency
  )
}

export function buildSegmentHeatFormulaText(params: {
  N_visit: number
  L_interaction: number
  R_recency: number
  H_segment: number
}): string {
  const { N_visit, L_interaction, R_recency, H_segment } = params
  const a = HEAT_ALPHA
  const b = HEAT_BETA
  const g = HEAT_GAMMA

  const nWhy =
    N_visit === 0
      ? 'session 刚写入中期，还没有被检索命中过，计数为 0。'
      : `该 session 已被检索命中 ${N_visit} 次。`

  const lWhy =
    L_interaction === 1
      ? '第一次写入，新建了一个 session，里面只有这一个 page。'
      : `短期满时弹出的 ${L_interaction} 轮 QA 各成 1 个 page，写入本 session 的 details。`

  const rWhy =
    R_recency >= 0.99
      ? 'last_visit_time 设为写入时刻，距现在 Δt≈0，exp(−Δt/τ)≈1。'
      : `按 last_visit_time 与当前时间计算，得到 R_recency=${R_recency}。`

  return `公式：
  H_segment = α·N_visit + β·L_interaction + γ·R_recency

系数：
  α = ${a}    β = ${b}    γ = ${g}

── N_visit ────────────────────────────────
  中期检索命中该 session 时 N_visit 加 1；越大说明越常被用到。
  本步取值 ${N_visit}：${nWhy}

── L_interaction ──────────────────────────
  该 session 的 details 中有多少个 page。
  本步取值 ${L_interaction}：${lWhy}

── R_recency ──────────────────────────────
  由 last_visit_time 与当前时间计算：R_recency = exp(−Δt / τ)，τ=${RECENCY_TAU_HOURS}。
  本步取值 ${R_recency}：${rWhy}

代入：
  H_segment = ${a}×${N_visit} + ${b}×${L_interaction} + ${g}×${R_recency} = ${H_segment}`
}
