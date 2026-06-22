import type { StepDef } from './steps'

export type SidebarBlock =
  | { type: 'primary'; step: StepDef; index: number }
  | {
      type: 'group'
      groupId: string
      groupLabel: string
      steps: { step: StepDef; index: number }[]
    }

/** 将扁平步骤列表转为侧栏层级块：L1 主步骤 + L2 分组（带 tab 标题） */
export function buildSidebarBlocks(steps: StepDef[]): SidebarBlock[] {
  const blocks: SidebarBlock[] = []
  let i = 0
  while (i < steps.length) {
    const s = steps[i]
    if (s.level === 1) {
      blocks.push({ type: 'primary', step: s, index: i })
      i += 1
      continue
    }
    const groupId = s.group ?? 'default'
    const groupLabel = s.groupLabel ?? '子流程'
    const groupSteps: { step: StepDef; index: number }[] = []
    while (i < steps.length && steps[i].level === 2 && (steps[i].group ?? 'default') === groupId) {
      groupSteps.push({ step: steps[i], index: i })
      i += 1
    }
    blocks.push({ type: 'group', groupId, groupLabel, steps: groupSteps })
  }
  return blocks
}

export function stepStatus(
  index: number,
  panelStepIndex: number,
  execStepIndex: number,
): 'done' | 'current' | 'pending' {
  if (index < execStepIndex) return 'done'
  if (index === panelStepIndex || index === execStepIndex) return 'current'
  return 'pending'
}
