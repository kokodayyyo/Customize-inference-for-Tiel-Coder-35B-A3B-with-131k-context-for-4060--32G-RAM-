/**
 * 控制台页面的集成自检（需要 Node.js 18+，项目运行期不需要它）。
 *
 * 为什么需要：`node --check` 只能查语法，查不出 renderDash() 里
 * "某个字段是 null / 类型不对" 这类**运行时**错误 —— 那会让页面白屏。
 * 本脚本把 index.html 里的 <script> 拿出来，配一套最小 DOM 桩在 Node 里
 * 真跑一遍，并用**真实服务**返回的数据喂给它。
 *
 * 用法（服务需已启动）：
 *     node scripts/test_ui.mjs                       // 默认 127.0.0.1:8000
 *     node scripts/test_ui.mjs http://127.0.0.1:8000
 *     node scripts/test_ui.mjs --offline             // 用内置样例数据，不需要服务
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const HTML = join(HERE, "..", "src", "ornith_server", "web", "index.html");

const args = process.argv.slice(2);
const OFFLINE = args.includes("--offline");
const BASE = (args.find((a) => a.startsWith("http")) || "http://127.0.0.1:8000").replace(/\/$/, "");

let passed = 0;
const failed = [];
function check(name, ok, detail = "") {
  if (ok) {
    passed++;
    console.log(`  [通过] ${name}${detail ? "  — " + detail : ""}`);
  } else {
    failed.push(name);
    console.log(`  [失败] ${name}${detail ? "  — " + detail : ""}`);
  }
}

/* ---------------- 最小 DOM 桩 ---------------- */
const elements = new Map();
function makeEl(id) {
  return {
    id,
    _html: "",
    textContent: "",
    className: "",
    style: {},
    dataset: {},
    clientWidth: 620,
    width: 0,
    height: 0,
    disabled: false,
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    appendChild() {},
    addEventListener() {},
    set innerHTML(v) {
      this._html = String(v);
    },
    get innerHTML() {
      return this._html;
    },
    getContext() {
      return new Proxy(
        {},
        {
          get: (_t, prop) => {
            if (prop === "canvas") return this;
            return () => {};
          },
          set: () => true,
        }
      );
    },
  };
}
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeEl(id));
    return elements.get(id);
  },
  querySelectorAll() {
    return [];
  },
  createElement(tag) {
    return makeEl(tag);
  },
  addEventListener() {},
};
const localStorage = {
  _m: new Map(),
  getItem(k) {
    return this._m.has(k) ? this._m.get(k) : null;
  },
  setItem(k, v) {
    this._m.set(k, String(v));
  },
  removeItem(k) {
    this._m.delete(k);
  },
};

/* ---------------- 离线样例数据（字段与真实接口一致） ---------------- */
const SAMPLE = {
  ts: Date.now() / 1000,
  system: {
    memory: { total_gib: 31.18, avail_gib: 6.4, used_gib: 24.78, percent: 79.5,
              commit_total_gib: 33.7, commit_avail_gib: 8.2 },
    cpu_percent: 61.2,
    cpu_count: 32,
    gpus: [{ index: "0", name: "NVIDIA GeForce RTX 4060 Laptop GPU", total_gib: 8,
             used_gib: 6.31, free_gib: 1.69, used_percent: 78.9, util_percent: 49,
             mem_util_percent: 37, temp_c: 52, power_w: 62.4 }],
    process: { working_set_gib: 13.76, peak_working_set_gib: 14.2, commit_gib: 20.6,
               cpu_seconds: 812.3, cpu_percent: 532.8, cpu_percent_of_total: 16.7 },
  },
  backend: {
    running: true, state: "running", pid: 30868,
    live: { prefill_tps: 131.9, decode_tps: 37.22 },
    avg: { prefill_tps: 138.0, decode_tps: 37.22 },
    prompt_tokens_total: 202, predicted_tokens_total: 700,
    prompt_tokens_cached_total: 0, prompt_seconds_total: 1.5,
    predicted_seconds_total: 23.4, n_decode_total: 902, n_tokens_max: 1107,
    requests_processing: 0, requests_deferred: 0, n_busy_slots_per_decode: 1,
    kv_cache_usage_ratio: 0.0044, kv_cache_tokens: 901, context_total: 204800,
    slots_total: 1,
    slots: [{ id: 0, state: "空闲", n_ctx: 204800, prompt_tokens: 901,
              processed: 0, cached: 0, n_remain: -1, n_decoded: 700 }],
    model_path: "D:\\models\\x.gguf", model_alias: "tile-35b-a3b-apex",
    model_ftype: "Q4_K - Medium", modalities: { vision: false },
    is_sleeping: false, build_info: "b1-1fb7ef3", chat_template: true,
  },
  gateway: {
    uptime_seconds: 120.5, requests_total: 1, requests_active: 0, requests_failed: 0,
    requests_rejected: 0, prompt_tokens_total: 202, completion_tokens_total: 700,
    avg_completion_tokens_per_second: 5.8, latency_p50_seconds: 20.3,
    latency_p95_seconds: 20.4, requests_by_model: {}, queue_waiting: 0,
  },
  history: [
    { t: 1, prefill_tps: null, decode_tps: null, kv_ratio: 0, gpu_util: 3, cpu: 4 },
    { t: 2, prefill_tps: 131.9, decode_tps: null, kv_ratio: 0.0019, gpu_util: 35, cpu: 69 },
    { t: 3, prefill_tps: null, decode_tps: 37.22, kv_ratio: 0.0044, gpu_util: 3, cpu: 532 },
  ],
};

/* ---------------- 取出并执行页面脚本 ---------------- */
const html = readFileSync(HTML, "utf8");
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) {
  console.error("index.html 里找不到 <script> 段");
  process.exit(1);
}
const js = m[1];
console.log(`页面 ${html.length} 字符，脚本 ${js.length} 字符\n`);

const realFetch = globalThis.fetch;
const sandbox = {
  document,
  window: { devicePixelRatio: 1 }, // drawChart 里会读它
  localStorage,
  navigator: { clipboard: { writeText: async () => {} } },
  location: { origin: BASE },
  confirm: () => true,
  prompt: () => null,
  alert: () => {},
  setTimeout,
  clearTimeout,
  setInterval: () => 0, // 不启动轮询，避免脚本跑完后进程不退出
  clearInterval: () => {},
  fetch: async (url, opts) => {
    if (OFFLINE) {
      const path = String(url);
      if (path.includes("/admin/metrics")) {
        return { status: 200, ok: true, json: async () => SAMPLE };
      }
      if (path.includes("/admin/models")) {
        return { status: 200, ok: true, json: async () => ({ search_roots: ["D:/models"], models: [], projectors: [] }) };
      }
      return { status: 200, ok: true, json: async () => ({}) };
    }
    return realFetch(url.startsWith("http") ? url : BASE + url, opts);
  },
  console,
};

/* ---------------- 执行 ---------------- */
console.log(OFFLINE ? "=== 离线模式（内置样例数据）===" : `=== 在线模式（真实服务 ${BASE}）===`);
let runError = null;
try {
  const fn = new Function(...Object.keys(sandbox), `${js}\n;return {renderDash, drawChart, renderModels, renderStatus, fmtUptime, n};`);
  var api = fn(...Object.values(sandbox));
} catch (e) {
  runError = e;
}
check("脚本可执行", !runError, runError ? String(runError) : "");

if (!runError) {
  // 工具函数
  check("n(null) 显示为占位符", api.n(null) === "—");
  check("n(3.14159,2) 格式化正确", api.n(3.14159, 2) === "3.14");
  check("fmtUptime 正常", api.fmtUptime(3725).includes("1h"));

  // 真正的渲染
  let renderError = null;
  try {
    api.renderDash(SAMPLE);
  } catch (e) {
    renderError = e;
  }
  check("renderDash 不抛异常", !renderError, renderError ? String(renderError) : "");

  const dash = elements.get("dash")?.innerHTML || "";
  check("渲染出了仪表盘内容", dash.length > 500, `${dash.length} 字符`);
  // 注意断言的是**界面实际显示的形式**（有格式化），不是原始字段值
  const want = [
    ["decode 速度", "37.22"],
    ["prefill 速度", "131.9"],
    ["显存（GiB）", "6.31"],
    ["内存占用", "24.8"],           // n(mem.used_gib, 1)
    ["KV 占用率", "0.44"],
    ["上下文 token 数", "204,800"],  // toLocaleString 带千分位
    ["上下文简写", "200K"],
    ["CPU 单核口径", "532.8"],
    ["网关 P95", "20.4"],
    ["累计 token", "700"],
    ["槽位状态", "空闲"],
    ["量化类型", "Q4_K"],
  ];
  for (const [label, needle] of want) {
    check(`仪表盘含${label}`, dash.includes(needle), needle);
  }
  check("画布已创建", dash.includes("<canvas"));

  // 空数据不应崩
  let nullOk = true;
  try {
    api.renderDash({
      ts: 0, system: {}, backend: { live: {}, avg: {}, slots: [] }, gateway: {}, history: [],
    });
  } catch (e) {
    nullOk = false;
    renderError = e;
  }
  check("字段全空也不崩", nullOk, nullOk ? "" : String(renderError));

  // 后端未运行时（后端字段为空 dict）也不应崩
  let offlineOk = true;
  try {
    api.renderDash({ ts: 0, system: { gpus: [] }, backend: {}, gateway: {}, history: [] });
  } catch (e) {
    offlineOk = false;
    renderError = e;
  }
  check("后端未运行也不崩", offlineOk, offlineOk ? "" : String(renderError));
}

console.log(`\n${"=".repeat(60)}`);
console.log(`通过 ${passed} 项，失败 ${failed.length} 项`);
failed.forEach((f) => console.log(`  失败: ${f}`));
console.log("=".repeat(60));
process.exit(failed.length ? 1 : 0);
