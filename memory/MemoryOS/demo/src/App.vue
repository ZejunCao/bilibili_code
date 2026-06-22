<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import MemoryBoard from './components/MemoryBoard.vue'
import DiskFileInspector from './components/DiskFileInspector.vue'
import LlmMessagesPanel from './components/LlmMessagesPanel.vue'
import { getStepGuide } from './data/stepGuides'
import { formatArtifactsWithComments } from './engine/formatArtifacts'
import { engine } from './engine/demoEngine'
import { READ_STEPS, WRITE_STEPS, type StepDef } from './engine/steps'
import { provideDemoEngine } from './composables/useDemoEngine'
import StepSidebar from './components/StepSidebar.vue'

const ready = ref(false)
const initError = ref<string | null>(null)

onMounted(() => {
  try {
    engine.reset()
    ready.value = true
  } catch (e) {
    console.error(e)
    initError.value = e instanceof Error ? e.message : String(e)
    ready.value = true
  }
})

const tab = ref<'write' | 'read'>('write')
const message = ref<{ type: 'error' | 'ok'; text: string } | null>(null)

const {
  displayState,
  writeStep,
  readStep,
  selectedLog,
  execStepIndex,
  panelStepIndex,
  isViewingHistory,
  goLatestStep,
  selectStep,
  runWriteStep,
  runReadStep,
  reset,
} = provideDemoEngine(tab)

const steps = computed(() => (tab.value === 'write' ? WRITE_STEPS : READ_STEPS))

const canRun = computed(() =>
  tab.value === 'write' ? !!writeStep.value : !!readStep.value,
)

const activeStep = computed((): StepDef | null => {
  return steps.value[panelStepIndex.value] ?? null
})

const stepGuide = computed(() =>
  activeStep.value ? getStepGuide(activeStep.value.id) : null,
)

const stepCardMode = computed((): 'pending' | 'viewed' | 'completed' => {
  if (isViewingHistory.value) return 'viewed'
  if (execStepIndex.value > panelStepIndex.value) return 'completed'
  return 'pending'
})

const stepArtifacts = computed(() => {
  if (stepCardMode.value === 'pending') return undefined
  return selectedLog.value?.artifacts
})

function runStep() {
  if (isViewingHistory.value) return
  message.value = null
  const res = tab.value === 'write' ? runWriteStep() : runReadStep()
  if (!res.ok) {
    message.value = { type: 'error', text: res.message ?? '执行失败' }
  }
}

function onReset() {
  reset()
  message.value = null
}
</script>

<template>
  <div v-if="!ready" style="padding: 2rem; text-align: center; color: var(--muted)">
    正在准备演示数据…
  </div>
  <div v-else-if="initError" style="padding: 2rem; color: var(--danger)">
    <p>演示初始化失败：{{ initError }}</p>
    <p style="color: var(--muted); font-size: 0.85rem">请打开控制台查看详情，或强制刷新后重试。</p>
  </div>
  <template v-else>
    <header class="app-header">
      <div>
        <h1>MemoryOS 记忆流向演示</h1>
      </div>
      <span class="badge">short=3 · heat≥5</span>
    </header>

    <nav class="tabs">
      <button class="tab" :class="{ active: tab === 'write' }" @click="tab = 'write'">
        存储流
      </button>
      <button class="tab" :class="{ active: tab === 'read' }" @click="tab = 'read'">
        检索流
      </button>
    </nav>

    <div class="layout">
      <aside class="sidebar">
        <h4 style="margin: 0 0 0.75rem; font-size: 0.85rem; color: var(--muted)">
          步骤 {{ execStepIndex }} / {{ steps.length }}
        </h4>
        <StepSidebar
          :steps="steps"
          :panel-step-index="panelStepIndex"
          :exec-step-index="execStepIndex"
          @select="selectStep"
        />
      </aside>

      <main class="main">
        <div class="toolbar">
          <button
            class="btn btn-primary"
            :disabled="!canRun || isViewingHistory"
            @click="runStep"
          >
            ▶ {{ activeStep?.label ?? '已完成' }}
          </button>
          <button class="btn" :disabled="!isViewingHistory" @click="goLatestStep">
            回到最新
          </button>
          <button class="btn" @click="onReset">重置</button>
        </div>

        <div v-if="message" class="msg" :class="message.type">{{ message.text }}</div>

        <div v-if="activeStep && stepGuide" class="explain step-card">
          <h4 class="step-card-title">第 {{ panelStepIndex + 1 }} 步</h4>
          <p class="step-detail">{{ stepGuide.detail }}</p>
          <div v-if="stepCardMode !== 'pending'" class="step-output-grid">
            <section class="step-output-col">
              <h5 class="step-section-title">本步产出</h5>
              <pre
                v-if="stepArtifacts"
                class="json-box step-output-pre"
              >{{ formatArtifactsWithComments(activeStep.id, stepArtifacts) }}</pre>
              <p v-else class="step-output-empty">
                无单独产出对象，见记忆面板或下方磁盘对比
              </p>
            </section>
            <section class="step-output-col">
              <h5 class="step-section-title">
                {{ selectedLog?.sidePanel?.title ?? (selectedLog?.llmCalls?.length ? '大模型输入' : '说明') }}
              </h5>
              <pre
                v-if="selectedLog?.sidePanel"
                class="json-box step-output-pre step-formula-pre"
              >{{ selectedLog.sidePanel.content }}</pre>
              <div
                v-if="selectedLog?.llmCalls?.length"
                class="llm-panel-after-flow"
              >
                <LlmMessagesPanel :calls="selectedLog.llmCalls" />
              </div>
              <p
                v-else-if="!selectedLog?.sidePanel"
                class="step-output-empty"
              >本步未调用大模型</p>
            </section>
          </div>
          <section v-else class="step-section step-section-muted">
            <p class="step-pending-hint">点击上方「▶ 执行」后，此处将展示本步结果。</p>
          </section>
        </div>

        <MemoryBoard :state="displayState" />

        <DiskFileInspector
          v-if="selectedLog?.before != null && selectedLog?.after != null"
          :before="selectedLog.before"
          :after="selectedLog.after"
          style="margin-top: 1rem"
        />
      </main>
    </div>
  </template>
</template>
