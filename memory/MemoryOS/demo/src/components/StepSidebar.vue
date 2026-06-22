<script setup lang="ts">
import { computed } from 'vue'
import { buildSidebarBlocks, stepStatus } from '../engine/stepSidebar'
import type { StepDef } from '../engine/steps'

const props = defineProps<{
  steps: StepDef[]
  panelStepIndex: number
  execStepIndex: number
}>()

const emit = defineEmits<{
  select: [index: number]
}>()

const blocks = computed(() => buildSidebarBlocks(props.steps))

function statusClass(index: number) {
  return stepStatus(index, props.panelStepIndex, props.execStepIndex)
}

function groupHasCurrent(block: { steps: { index: number }[] }) {
  return block.steps.some((s) => s.index === props.panelStepIndex)
}
</script>

<template>
  <div class="step-sidebar">
    <template v-for="(block, bi) in blocks" :key="bi">
      <button
        v-if="block.type === 'primary'"
        type="button"
        class="step-l1"
        :class="statusClass(block.index)"
        @click="emit('select', block.index)"
      >
        <span class="step-l1-rail" />
        <span class="step-l1-body">
          <span class="step-l1-label">{{ block.step.label }}</span>
          <code class="step-fn">{{ block.step.fn }}</code>
        </span>
        <span class="step-l1-dot" />
      </button>

      <div
        v-else
        class="step-group"
        :class="{ 'step-group-active': groupHasCurrent(block) }"
      >
        <div class="step-group-head">
          <span class="step-group-tab">{{ block.groupLabel }}</span>
        </div>
        <ul class="step-sublist">
          <li
            v-for="{ step, index } in block.steps"
            :key="step.id"
            class="step-l2"
            :class="statusClass(index)"
            @click="emit('select', index)"
          >
            <span class="step-dot" />
            <span>
              <span class="step-l2-label">{{ step.label }}</span>
              <code class="step-fn">{{ step.fn }}</code>
            </span>
          </li>
        </ul>
      </div>
    </template>
  </div>
</template>
