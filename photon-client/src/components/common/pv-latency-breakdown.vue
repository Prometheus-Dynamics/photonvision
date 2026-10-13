<script setup lang="ts">
import { computed } from "vue";
import type { PipelineResult } from "@/types/PhotonTrackingTypes";

// Where a Styx frame's latency goes. The backend sends each stage in ms after the sensor's start of
// frame; the reported latency also counts half the exposure (the capture time is the middle of the
// exposure), so what comes before the start of frame is latency minus the published stage.
const props = defineProps<{ result?: PipelineResult }>();

interface Segment {
  label: string;
  ms: number;
  color: string;
  note: string;
}

const segments = computed<Segment[] | undefined>(() => {
  const r = props.result;
  const s = r?.stages;
  if (!r || !s || s.dequeued === undefined || s.published === undefined) return undefined;
  const at = (key: string, fallback: number) => s[key] ?? fallback;
  const dequeued = s.dequeued;
  const ispDone = at("ispDone", dequeued);
  const taken = at("taken", ispDone);
  const detectStart = at("detectStart", taken);
  const detectEnd = at("detectEnd", detectStart);
  const published = s.published;
  const span = (a: number, b: number) => Math.max(0, b - a);
  return [
    { label: "Exposure (half)", ms: Math.max(0, r.latency - published), color: "#7e57c2", note: "sensor" },
    { label: "Sensor readout", ms: dequeued, color: "#5c6bc0", note: "sensor" },
    { label: "ISP", ms: span(dequeued, ispDone), color: "#26a69a", note: "hardware" },
    { label: "Queue", ms: span(ispDone, detectStart), color: "#8d6e63", note: "software" },
    { label: "Detection", ms: span(detectStart, detectEnd), color: "#ffca28", note: "software" },
    { label: "PhotonVision + NT", ms: span(detectEnd, published), color: "#ef5350", note: "software" }
  ];
});

const total = computed(() => segments.value?.reduce((a, b) => a + b.ms, 0) ?? 0);
// From the moment the whole frame has arrived to the result on NetworkTables: the part the
// camera's processing decides, as opposed to the sensor's exposure and readout.
const processing = computed(() => {
  const s = props.result?.stages;
  return s?.dequeued !== undefined && s.published !== undefined ? Math.max(0, s.published - s.dequeued) : undefined;
});
</script>

<template>
  <div v-if="segments" class="latency-breakdown">
    <div class="bar">
      <div
        v-for="seg in segments"
        :key="seg.label"
        class="bar-seg"
        :style="{ width: `${total > 0 ? (100 * seg.ms) / total : 0}%`, background: seg.color }"
      />
    </div>
    <table>
      <tbody>
        <tr v-for="seg in segments" :key="seg.label">
          <td><span class="swatch" :style="{ background: seg.color }" />{{ seg.label }}</td>
          <td class="num">{{ seg.ms.toFixed(seg.ms < 1 ? 2 : 1) }} ms</td>
          <td class="note">{{ seg.note }}</td>
        </tr>
        <tr class="sum">
          <td>Total (capture to NetworkTables)</td>
          <td class="num">{{ (result?.latency ?? total).toFixed(1) }} ms</td>
          <td />
        </tr>
        <tr v-if="processing !== undefined" class="sum">
          <td>Processing (frame received to NetworkTables)</td>
          <td class="num">{{ processing.toFixed(2) }} ms</td>
          <td />
        </tr>
      </tbody>
    </table>
  </div>
  <div v-else class="latency-breakdown">No stage breakdown for this camera.</div>
</template>

<style scoped>
.latency-breakdown {
  min-width: 300px;
  font-size: 0.85rem;
}
.bar {
  display: flex;
  height: 10px;
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 6px;
}
.bar-seg {
  height: 100%;
}
table {
  width: 100%;
  border-collapse: collapse;
}
td {
  padding: 1px 4px;
}
.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.note {
  opacity: 0.6;
}
.sum td {
  border-top: 1px solid rgba(255, 255, 255, 0.25);
  font-weight: 600;
}
.swatch {
  display: inline-block;
  width: 9px;
  height: 9px;
  border-radius: 2px;
  margin-right: 6px;
}
</style>
