<script setup lang="ts">
import { computed } from 'vue'
import type { DemoSnapshot } from '../types/memory'

const props = defineProps<{ state: DemoSnapshot }>()

const threshold = computed(() => props.state.config.midTermHeatThreshold)
const profile = computed(
  () => props.state.longTermUser.user_profiles[props.state.config.userId]?.data ?? 'None',
)
const sessions = computed(() => Object.values(props.state.midTermSessions))
</script>

<template>
  <div class="memory-grid">
    <div class="mem-col short">
      <h3>
        短期
        <span class="badge">{{ state.shortTerm.length }}/{{ state.config.shortTermCapacity }}</span>
      </h3>
      <div class="mem-body">
        <p v-if="state.pendingEvictedQas.length" class="session-meta">
          ⏳ 待晋升 {{ state.pendingEvictedQas.length }} QA
        </p>
        <p v-if="state.pendingPages.length" class="session-meta">
          ⏳ {{ state.pendingPages.length }} Page 待写入中期
        </p>
        <div
          v-if="!state.shortTerm.length && !state.pendingPages.length"
          class="empty"
        >
          空
        </div>
        <div v-for="(qa, i) in state.shortTerm" :key="i" class="qa-card">
          <div class="role">#{{ i + 1 }}</div>
          <div><strong>U:</strong> {{ qa.user_input }}</div>
          <div><strong>A:</strong> {{ qa.agent_response }}</div>
        </div>
      </div>
    </div>

    <div class="mem-col mid">
      <h3>
        中期
        <span class="badge">{{ sessions.length }} sessions</span>
      </h3>
      <div class="mem-body">
        <p v-if="state.heapTop" class="session-meta">
          🔥 {{ state.heapTop.sessionId }} H={{ state.heapTop.heat.toFixed(1) }}/{{ threshold }}
        </p>
        <div
          v-for="s in sessions"
          :key="s.id"
          class="qa-card"
          style="border-left-color: var(--mid)"
        >
          <div class="role">{{ s.id }}</div>
          <div>{{ s.summary }}</div>
          <div class="session-meta">
            L={{ s.L_interaction }} · N_visit={{ s.N_visit }} · H={{ s.H_segment.toFixed(1) }}
          </div>
          <div class="heat-bar">
            <div
              class="heat-fill"
              :style="{ width: `${Math.min(100, (s.H_segment / threshold) * 100)}%` }"
            />
          </div>
          <div
            v-for="p in s.details"
            :key="p.page_id"
            class="page-card"
            :class="{ analyzed: p.analyzed }"
          >
            <div class="role">{{ p.page_id }} {{ p.analyzed ? '✓' : '○' }}</div>
            <div v-if="p.meta_info" class="session-meta">{{ p.meta_info }}</div>
            <div><strong>U:</strong> {{ p.user_input }}</div>
          </div>
        </div>
      </div>
    </div>

    <div class="mem-col long">
      <h3>长期</h3>
      <div class="mem-body">
        <div class="k-card profile-text">{{ profile }}</div>
        <div
          v-for="(k, i) in state.longTermUser.knowledge_base"
          :key="'u' + i"
          class="k-card"
        >
          {{ k.knowledge }}
        </div>
        <div
          v-for="(k, i) in state.longTermAssistant.assistant_knowledge"
          :key="'a' + i"
          class="k-card"
        >
          {{ k.knowledge }}
        </div>
      </div>
    </div>
  </div>
</template>
