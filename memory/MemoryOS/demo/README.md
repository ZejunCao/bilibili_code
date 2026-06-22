# MemoryOS 记忆流向演示

纯前端分步演示 **MemoryOS** 的短 / 中 / 长期记忆**存储**与**检索**流程。不连接后端、不调用真实 LLM，对话与模型输出均为预设数据，便于对照源码理解数据如何流动。

## 环境要求

- Node.js 18+
- npm 9+（或 pnpm / yarn）

## 快速开始

```bash
cd demo
npm install
npm run dev
```

浏览器打开 **http://localhost:5174**。

国内安装依赖较慢时可使用镜像：

```bash
npm install --registry=https://registry.npmmirror.com
```

## 界面说明

顶部分为两个 Tab，**彼此独立**，无需先跑完存储流再跑检索流：

| Tab | 内容 |
|-----|------|
| **存储流** | 模拟 `add_memory`：短期写入 → 满容晋升中期（page / session / meta_info）→ 热度达标后写长期画像与知识 |
| **检索流** | 模拟 `get_response`：中期向量检索 → 长期知识检索 → 读短期与画像 → 拼装 Prompt → 生成回复 → 写回短期 |

- 左侧：步骤列表，可点击回看任一步的结果  
- 中间：当前步说明、产出 JSON、流程文字说明（及拼装/生成步的大模型消息预览）  
- 下方：短期 / 中期 / 长期三层记忆看板  

## 存储流建议顺序

1. **第 1～3 轮对话** — 写入短期（容量 3）  
2. **短期 → 中期** — 挤出最旧 QA → 整理为 page → 连续性判断 → 主题摘要 → session 归纳 → 写入中期并计算 `H_segment`  
3. **第 4、5 轮对话** — 同时写短期并合并进已有中期 session，`L_interaction` 与 `H_segment` 升高  
4. **中期 → 长期** — 检查堆顶 session 的 `H_segment` 是否达阈值 → 画像分析 → 知识提取 → 收尾（`N_visit` / `L_interaction` 清零，重算 `H_segment`）  

## 检索流

切换到「检索流」即可逐步执行，初始数据为存储流跑完后的预设快照。各步右侧有自然语言流程说明（如问题转向量、在中期 session / page 中检索等）。

## 生产构建（可选）

```bash
npm run build
```

将 `dist/` 用任意静态服务器托管，例如：

```bash
cd dist && python3 -m http.server 8765 --bind 0.0.0.0
```

或本地预览：`npm run preview`。

## 与源码对照

| 演示步骤 | memoryos-pypi |
|---------|----------------|
| 短期写入 | `short_term.py` → `add_qa_pair` |
| 短期满晋升 | `updater.py` → `process_short_term_to_mid_term` |
| 中期热度 | `mid_term.py` → `compute_segment_heat` |
| 长期更新 | `memoryos.py` → `_trigger_profile_and_knowledge_update_if_needed` |
| 检索 | `retriever.py` → `retrieve_context` |
| 生成回复 | `memoryos.py` → `get_response` |

## 技术栈

Vue 3 + TypeScript + Vite，无其他运行时依赖。
