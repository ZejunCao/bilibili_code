<script setup lang="ts">
import { formatDisplayText } from '../utils/formatDisplayText'
import type { LlmCallPreview } from '../types/memory'

defineProps<{
  calls: LlmCallPreview[]
}>()

const roleLabel: Record<string, string> = {
  system: 'system',
  user: 'user',
  assistant: 'assistant',
}

function displayContent(text: string) {
  return formatDisplayText(text)
}
</script>

<template>
  <div class="llm-messages-panel">
    <article v-for="(call, idx) in calls" :key="idx" class="llm-call">
      <h6 v-if="calls.length > 1" class="llm-call-title">{{ call.label }}</h6>
      <div
        v-for="(msg, mi) in call.messages"
        :key="mi"
        class="llm-msg"
        :class="`llm-msg--${msg.role}`"
      >
        <div class="llm-msg-head">
          <span class="llm-role-badge">{{ roleLabel[msg.role] ?? msg.role }}</span>
        </div>
        <pre class="llm-msg-content">{{ displayContent(msg.content) }}</pre>
      </div>
      <div v-if="call.mockResponse" class="llm-mock-response">
        <span class="llm-role-badge llm-role-badge--assistant">assistant</span>
        <pre class="llm-msg-content">{{ displayContent(call.mockResponse) }}</pre>
      </div>
    </article>
  </div>
</template>
