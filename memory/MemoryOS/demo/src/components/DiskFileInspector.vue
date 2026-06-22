<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import {
  DISK_FILE_ORDER,
  diskFileChanged,
  type DiskFilePath,
  type DiskFileSnapshot,
} from '../engine/diskFiles'

const props = defineProps<{
  before: DiskFileSnapshot
  after: DiskFileSnapshot
}>()

const activePath = ref<DiskFilePath>(DISK_FILE_ORDER[0])

const changedPaths = computed(() =>
  DISK_FILE_ORDER.filter((p) => diskFileChanged(props.before[p], props.after[p])),
)

const activeChanged = computed(() =>
  diskFileChanged(props.before[activePath.value], props.after[activePath.value]),
)

watch(
  changedPaths,
  (paths) => {
    if (paths.length && !paths.includes(activePath.value)) {
      activePath.value = paths[0]
    }
  },
  { immediate: true },
)

function shortName(path: DiskFilePath) {
  return path.split('/').pop() ?? path
}
</script>

<template>
  <div class="json-panel snapshot-panel disk-file-panel">
    <div class="disk-file-tabs">
      <button
        v-for="path in DISK_FILE_ORDER"
        :key="path"
        type="button"
        class="disk-file-tab"
        :class="{
          active: activePath === path,
          changed: diskFileChanged(before[path], after[path]),
        }"
        @click="activePath = path"
      >
        {{ shortName(path) }}
        <span v-if="diskFileChanged(before[path], after[path])" class="disk-dot" />
      </button>
    </div>

    <div class="json-panel-col">
      <h5 style="color: var(--muted); margin: 0 0 0.5rem">Before</h5>
      <pre class="json-box">{{ JSON.stringify(before[activePath], null, 2) }}</pre>
    </div>
    <div class="json-panel-col">
      <h5 style="color: var(--muted); margin: 0 0 0.5rem">After</h5>
      <pre class="json-box" :class="{ 'json-box-changed': activeChanged }">{{
        JSON.stringify(after[activePath], null, 2)
      }}</pre>
    </div>
  </div>
</template>
