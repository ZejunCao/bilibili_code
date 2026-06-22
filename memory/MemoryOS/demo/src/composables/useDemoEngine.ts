import { computed, inject, provide, ref, watch, type InjectionKey, type Ref } from 'vue'
import { engine } from '../engine/demoEngine'
import type { DemoSnapshot, StepLog } from '../types/memory'

const tick = ref(0)
/** -1 = 跟随当前进度；>=0 = 查看预设的第 n+1 步结果 */
const viewStepIndex = ref(-1)

function bump() {
  tick.value += 1
}

function activeSnapshot(flow: 'write' | 'read') {
  return flow === 'read' ? engine.ensureReadState() : engine.state
}

function getExecIndex(flow: 'write' | 'read') {
  return flow === 'write'
    ? engine.state.writeStepIndex
    : engine.ensureReadState().readStepIndex
}

export type DemoEngineApi = ReturnType<typeof createDemoEngine>

export const demoEngineKey: InjectionKey<DemoEngineApi> = Symbol('demoEngine')

function createDemoEngine(activeFlow: Ref<'write' | 'read'>) {
  watch(activeFlow, (flow) => {
    viewStepIndex.value = -1
    if (flow === 'read') engine.ensureReadState()
  })

  const state = computed(() => {
    tick.value
    return activeSnapshot(activeFlow.value)
  })

  const execStepIndex = computed(() => {
    tick.value
    return getExecIndex(activeFlow.value)
  })

  /**
   * 与正文「第 N 步」卡片一致的步骤下标（0-based）。
   * 未回看历史时：首步为 0；每执行完一步后正文仍展示该步，直到再点执行才进入下一步说明。
   */
  const panelStepIndex = computed(() => {
    tick.value
    if (viewStepIndex.value >= 0) return viewStepIndex.value
    const exec = getExecIndex(activeFlow.value)
    if (exec <= 0) return 0
    return exec - 1
  })

  const isViewingHistory = computed(() => viewStepIndex.value >= 0)

  const displayState = computed((): DemoSnapshot => {
    tick.value
    if (viewStepIndex.value >= 0) {
      return engine.getPresetSnapshot(activeFlow.value, viewStepIndex.value)
    }
    return activeSnapshot(activeFlow.value)
  })

  const selectedLog = computed((): StepLog | null => {
    tick.value
    if (viewStepIndex.value >= 0) {
      return engine.getPresetLog(activeFlow.value, viewStepIndex.value)
    }
    const exec = getExecIndex(activeFlow.value)
    if (exec <= 0) return null
    const logs =
      activeFlow.value === 'write'
        ? engine.state.writeLogs
        : engine.ensureReadState().readLogs
    return logs[exec - 1] ?? null
  })

  const writeStep = computed(() => {
    tick.value
    return engine.getWriteStep()
  })

  const readStep = computed(() => {
    tick.value
    return engine.getReadStep()
  })

  function goLatestStep() {
    viewStepIndex.value = -1
  }

  function selectStep(i: number) {
    viewStepIndex.value = i
  }

  function runWriteStep() {
    const res = engine.executeWriteStep()
    bump()
    return res
  }

  function runReadStep() {
    const res = engine.executeReadStep()
    bump()
    return res
  }

  function reset() {
    engine.reset()
    viewStepIndex.value = -1
    bump()
  }

  return {
    state,
    displayState,
    writeStep,
    readStep,
    execStepIndex,
    panelStepIndex,
    viewStepIndex,
    isViewingHistory,
    selectedLog,
    goLatestStep,
    selectStep,
    runWriteStep,
    runReadStep,
    reset,
    tick,
  }
}

export function provideDemoEngine(activeFlow: Ref<'write' | 'read'>) {
  const api = createDemoEngine(activeFlow)
  provide(demoEngineKey, api)
  return api
}

export function useDemoEngine() {
  const api = inject(demoEngineKey)
  if (!api) throw new Error('useDemoEngine() 需在 provideDemoEngine 之后调用')
  return api
}
