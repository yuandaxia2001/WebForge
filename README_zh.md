<h1 align="center">WebForge</h1>

<h3 align="center">打破浏览器智能体基准测试的<br>真实性-可复现性-可扩展性三难困境</h3>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-即将发布-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/yuandaxia/WebForge"><img src="https://img.shields.io/badge/🤗%20HuggingFace-WebForge-yellow.svg" alt="HuggingFace"></a>
  <a href="./README.md"><img src="https://img.shields.io/badge/English-README-blue.svg" alt="English"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>

<p align="center">
  <b>WebForge</b> 是首个全自动化框架，用于构建具有多维难度控制的真实、可复现、可扩展的浏览器智能体基准测试。
</p>

---

## 基准测试三难困境

现有浏览器智能体基准测试无法同时做到**真实**、**可复现**和**可扩展**：

| | 真实网站（如 WebVoyager） | 受控环境（如 WebArena） | **WebForge（本文）** |
|---|---|---|---|
| 真实性 | ✅ 真实内容 | ❌ 无弹窗/噪声 | ✅ 真实数据 + 注入噪声 |
| 可复现性 | ❌ 内容漂移 | ✅ 自包含 | ✅ 自包含静态网站 |
| 可扩展性 | ❌ 人工标注 | ❌ 人工标注 | ✅ 全自动流水线 |

WebForge 通过**四智能体流水线**解决这一困境，端到端生成交互式、自包含的 Web 环境，无需任何人工标注。

---

## 流水线概览

<p align="center">
  <img src="assets/overall_pipeline.jpg" width="100%" alt="WebForge 流水线概览">
</p>

WebForge 通过四个阶段构建基准测试：**计划 → 生成 → 优化 → 验证**。

### 计划智能体（Plan Agent）

将目标领域和难度等级转化为结构化任务蓝图。采用双 LLM 流程：高温采样（T=2.0）产生创意草案，低温精炼（T=1.0）进行约束验证。

<details>
<summary>📐 计划智能体工作流</summary>
<p align="center"><img src="assets/plan_agent.jpg" width="100%" alt="Plan Agent"></p>
</details>

### 生成智能体（Generation Agent）

构建功能完整的网站，包含真实数据、防作弊机制和基于 `localStorage` 的有状态交互。

<details>
<summary>🏗️ 生成智能体工作流</summary>
<p align="center"><img src="assets/generation_agent.jpg" width="100%" alt="Generation Agent"></p>
</details>

### 优化智能体（Refinement Agent）

注入真实网络噪声（弹窗广告、Cookie 对话框、网络延迟），修复死链、表单错误等质量问题。

<details>
<summary>🔧 优化智能体工作流</summary>
<p align="center"><img src="assets/refinement_agent.jpg" width="100%" alt="Refinement Agent"></p>
</details>

### 验证智能体（Validation Agent）

在真实 Chromium 浏览器中重放解题路径，验证每个任务在 50 步内可解。

<details>
<summary>✅ 验证智能体工作流</summary>
<p align="center"><img src="assets/validation_agent.jpg" width="100%" alt="Validation Agent"></p>
</details>

---

## WebForge-Bench

**934 个任务** · **7 个领域** · **3 个难度等级** · **七维难度控制**

基准测试数据集托管在 [🤗 HuggingFace](https://huggingface.co/datasets/yuandaxia/WebForge)。本仓库提供评测智能体代码。

### 与现有基准的比较

| 基准测试 | 类型 | 任务数 | 领域数 | 难度控制 | 噪声 | 自动 | 可复现 | 评估方式 |
|----------|------|--------|--------|----------|------|------|--------|----------|
| Mind2Web | 真实 | 2,350 | 137† | ✗ | ✗ᵃ | ✗ | ✗ᵇ | 逐步预测 |
| WebVoyager | 真实 | 643 | 15† | ✗ | 被动 | ✗ | ✗ | LMM 评判 |
| MMInA | 真实 | 1,050 | 14† | ✗ | 被动 | ✗ | ✗ | Hop SR |
| WebArena | 受控 | 812 | 4 | ✗ | ✗ | ✗ | ✓ | 程序化 |
| VisualWebArena | 受控 | 910 | 3 | 2维ᶜ | ✗ | ✗ | ✓ | 手工构造 |
| WorkArena++ | 受控 | 682 | 1 | 1维ᵈ | ✗ | ✗ | ✓ | Oracle 函数 |
| EntWorld | 受控 | 1,756 | 6 | 事后ᵉ | ✗ | 部分ᶠ | ✓ | SQL 验证 |
| TheAgentCompany | 受控 | 175 | 1ᵍ | ✗ | 部分 | ✗ | ✓ | 检查点 |
| **WebForge（本文）** | **自动** | **934** | **7** | **7维×3级** | **✓** | **✓** | **✓** | **最终状态** |

<sub>†网站数量，非主题领域数。ᵃ标注协议明确排除弹窗和验证码。ᵇ约50%的任务在两年内过期。ᶜ动作难度+视觉难度，事后标注。ᵈ仅控制指令明确程度。ᵉ基于5个SQL结构维度的加权分数，事后计算。ᶠ任务实例化自动化，环境部署手动。ᵍ单一模拟公司，含7个岗位类别。</sub>

---

## 主要结果

14 个模型配置在 WebForge-Bench（934 个任务）上的准确率 (%)。

### 表 1：主要结果 — 难度等级 & 跨领域

| | | 难度等级 | | | | 跨领域 | | | | | |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **模型** | **L1** | **L2** | **L3** | **总计** | **D1** | **D2** | **D3** | **D4** | **D5** | **D6** | **D7** |
| *（a）多模态（截图 + DOM）* | | | | | | | | | | | |
| Gemini-3-Pro | **86.4** | **82.1** | **58.0** | **75.9** | **72.2** | 67.2 | **82.4** | **79.4** | 71.0 | **76.6** | **80.9** |
| Gemini-3-Flash | 82.4 | 73.5 | 44.0 | 67.1 | 65.2 | 61.6 | 66.4 | 62.5 | **74.0** | 66.0 | 74.8 |
| Gemini-2.5-Flash-Lite | 58.5 | 33.5 | 12.6 | 35.0 | 34.8 | 28.8 | 26.7 | 41.9 | 38.2 | 33.3 | 39.7 |
| Claude-4.5-Sonnet | 85.7 | 74.7 | 48.1 | 69.9 | 58.3 | **70.4** | 71.8 | 73.8 | 69.5 | 67.4 | 76.3 |
| GPT-5.2 | 80.1 | 65.9 | 31.1 | 59.5 | 48.7 | 58.4 | 51.1 | 64.4 | 57.3 | 63.1 | 71.0 |
| GPT-5-Mini | 82.4 | 68.2 | 28.7 | 60.4 | 51.3 | 56.8 | 50.4 | 73.8 | 60.3 | 58.2 | 67.9 |
| GPT-5-Nano | 61.8 | 25.9 | 6.1 | 31.3 | 20.9 | 29.6 | 29.0 | 43.8 | 31.3 | 29.8 | 30.5 |
| Kimi-K2.5 | 84.4 | 73.8 | 39.2 | 66.4 | 60.0 | 61.6 | 65.6 | 75.6 | 62.6 | 61.7 | 74.8 |
| Qwen3-VL-235B | 73.4 | 50.3 | 20.1 | 48.3 | 37.4 | 40.8 | 46.6 | 58.8 | 51.1 | 48.2 | 51.1 |
| Qwen3-Omni-30B | 26.9 | 9.1 | 2.4 | 12.7 | 6.1 | 9.6 | 7.6 | 26.2 | 10.7 | 12.1 | 13.0 |
| *（b）纯文本（仅 DOM）* | | | | | | | | | | | |
| DeepSeek-V3.2 | 77.1 | 47.4 | 21.5 | 48.8 | 54.8 | 46.4 | 48.9 | 45.6 | 49.6 | 48.2 | 49.6 |
| GLM-4.7 | 76.4 | 49.4 | 24.2 | 50.2 | 50.4 | 43.2 | 55.7 | 48.8 | 52.7 | 48.9 | 51.9 |
| Gemini-3-Pro (T) | 80.1 | 61.8 | 34.8 | 59.2 | 61.7 | 56.0 | 61.1 | 57.5 | 59.5 | 56.7 | 62.6 |
| Gemini-3-Flash (T) | 78.7 | 50.9 | 23.2 | 51.2 | 54.8 | 45.6 | 52.7 | 43.8 | 55.0 | 51.8 | 56.5 |
| **平均** | **73.9** | **54.8** | **28.1** | **52.6** | **48.3** | **48.3** | **51.1** | **56.9** | **53.1** | **51.6** | **57.2** |

> D1: 消费交易, D2: 内容审核, D3: 企业流程, D4: 信息检索, D5: 平台管理, D6: 工具使用, D7: 内容创作。(T) = 纯文本模式（仅 DOM，无截图）。

### 表 2：运行时效率（每任务平均）

| | Level 1 | | | | Level 2 | | | | Level 3 | | | |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **模型** | **轮次** | **操作** | **输入** | **输出** | **轮次** | **操作** | **输入** | **输出** | **轮次** | **操作** | **输入** | **输出** |
| Gemini-3-Pro | 7.9 | 12.2 | 133K | 4.2K | 13.8 | 21.6 | 307K | 5.9K | 26.9 | 44.6 | 1036K | 11.2K |
| Gemini-3-Flash | 8.0 | 12.3 | 159K | 5.5K | 13.1 | 19.3 | 304K | 6.5K | 25.3 | 39.1 | 962K | 15.3K |
| Gemini-2.5-Flash-Lite† | 12.0 | 6.6 | 224K | 4.6K | 16.5 | 11.5 | 254K | 3.4K | 26.1 | 21.9 | 520K | 5.6K |
| Claude-4.5-Sonnet | 11.0 | 12.3 | 260K | 3.8K | 18.7 | 20.7 | 591K | 6.9K | 33.8 | 37.4 | 1608K | 12.6K |
| GPT-5.2† | 8.8 | 8.5 | 80K | 0.4K | 15.6 | 16.1 | 236K | 0.6K | 26.1 | 27.7 | 656K | 1.0K |
| GPT-5-Mini† | 11.5 | 10.5 | 150K | 2.2K | 20.7 | 19.7 | 421K | 4.2K | 36.7 | 36.0 | 1164K | 9.7K |
| GPT-5-Nano† | 18.1 | 13.7 | 277K | 9.4K | 29.3 | 23.3 | 590K | 19.5K | 38.4 | 30.8 | 892K | 31.3K |
| Kimi-K2.5 | 13.3 | 11.1 | 176K | 3.2K | 21.1 | 19.8 | 385K | 5.8K | 36.2 | 34.6 | 904K | 10.5K |
| Qwen3-VL-235B | 9.0 | 9.2 | 135K | 1.9K | 16.2 | 17.4 | 363K | 3.7K | 28.7 | 32.4 | 845K | 6.9K |
| Qwen3-Omni-30B† | 34.3 | 6.9 | 463K | 4.4K | 43.2 | 6.8 | 641K | 6.6K | 46.8 | 8.0 | 740K | 7.1K |
| DeepSeek-V3.2 | 12.4 | 11.7 | 165K | 3.5K | 22.7 | 24.2 | 420K | 6.6K | 36.3 | 40.9 | 920K | 10.5K |
| GLM-4.7 | 11.6 | 12.8 | 138K | 3.7K | 22.7 | 25.6 | 376K | 7.5K | 34.4 | 40.2 | 761K | 11.5K |
| Gemini-3-Pro (T) | 10.6 | 16.8 | 144K | 5.4K | 21.6 | 33.9 | 412K | 8.9K | 33.7 | 57.7 | 875K | 13.2K |
| Gemini-3-Flash (T) | 10.5 | 15.4 | 213K | 7.5K | 29.8 | 47.1 | 854K | 26.1K | 41.4 | 65.5 | 1328K | 29.9K |

> 轮次 = LLM 对话轮数；操作 = 浏览器动作数；输入/输出 = token 数量。标 † 的模型不支持逐步日志记录模式，token 统计偏低。

### 表 3：各维度准确率 (%)

| | 跳转深度 | | | 跳转广度 | | | 页面交互 | | | 视觉复杂度 | | | 信息复杂度 | | | 推理/计算 | | | 风险因子 | | |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **模型** | L1 | L2 | L3 | L1 | L2 | L3 | L1 | L2 | L3 | L1 | L2 | L3 | L1 | L2 | L3 | L1 | L2 | L3 | L1 | L2 | L3 |
| Gemini-3-Pro | **86.5** | **78.9** | **60.2** | 84.8 | **79.9** | **51.2** | **84.0** | **74.9** | **65.0** | **90.8** | **78.9** | **55.8** | **84.7** | **75.7** | **53.2** | **91.4** | **74.6** | **58.3** | **80.6** | **70.3** | 23.1 |
| Claude-4.5-Sonnet | 85.8 | 71.8 | 50.0 | **85.9** | 70.7 | 48.1 | 81.7 | 69.2 | 49.0 | 86.5 | 69.0 | 51.5 | 81.2 | 66.9 | 48.9 | 87.4 | 70.4 | 46.8 | 76.4 | 60.9 | 30.8 |
| Gemini-3-Flash | 82.3 | 71.1 | 45.1 | 83.8 | 67.6 | 45.7 | 74.6 | 67.8 | 47.0 | 83.1 | 69.0 | 46.8 | 81.2 | 64.0 | 39.0 | 84.7 | 68.3 | 42.6 | 72.2 | 60.0 | **38.5** |
| Gemini-2.5-Flash-Lite | 57.3 | 33.2 | 13.5 | 56.0 | 34.3 | 13.0 | 52.1 | 33.3 | 9.0 | 54.7 | 34.2 | 13.0 | 50.4 | 28.6 | 13.5 | 56.8 | 31.7 | 12.8 | 42.7 | 23.7 | 0.0 |
| Claude-4.5-Sonnet | 85.8 | 71.8 | 50.0 | **85.9** | 70.7 | 48.1 | 81.7 | 69.2 | 49.0 | 86.5 | 69.0 | 51.5 | 81.2 | 66.9 | 48.9 | 87.4 | 70.4 | 46.8 | 76.4 | 60.9 | 30.8 |
| GPT-5.2 | 79.2 | 62.9 | 33.5 | 76.4 | 62.8 | 27.8 | 71.8 | 58.1 | 42.0 | 84.5 | 58.1 | 31.9 | 74.0 | 58.1 | 25.5 | 86.0 | 59.0 | 26.4 | 67.3 | 48.6 | 15.4 |
| GPT-5-Mini | 81.2 | 66.1 | 29.7 | 82.2 | 63.0 | 25.3 | 80.8 | 59.4 | 23.0 | 83.7 | 62.7 | 31.2 | 77.2 | 56.4 | 27.7 | 84.7 | 61.8 | 26.8 | 71.1 | 44.3 | 23.1 |
| GPT-5-Nano | 61.8 | 26.1 | 5.6 | 59.2 | 28.7 | 7.4 | 61.5 | 25.4 | 3.0 | 50.1 | 27.8 | 12.6 | 47.2 | 24.3 | 9.9 | 51.2 | 30.9 | 6.4 | 40.3 | 17.7 | 0.0 |
| Kimi-K2.5 | 84.7 | 70.3 | 41.0 | 83.8 | 70.1 | 32.7 | 81.2 | 65.1 | 43.0 | 84.2 | 71.5 | 40.9 | 79.9 | 62.6 | 41.8 | 86.4 | 67.3 | 39.1 | 75.0 | 54.3 | 15.4 |
| Qwen3-VL-235B | 72.2 | 48.9 | 21.4 | 70.7 | 49.1 | 19.1 | 69.0 | 46.1 | 18.0 | 73.9 | 44.7 | 21.9 | 63.0 | 45.0 | 19.1 | 75.1 | 45.5 | 18.7 | 58.7 | 32.3 | 23.1 |
| Qwen3-Omni-30B | 27.1 | 8.9 | 2.6 | 23.0 | 11.9 | 3.7 | 27.2 | 9.7 | 1.0 | 24.1 | 10.2 | 2.0 | 17.2 | 11.9 | 3.5 | 24.3 | 9.8 | 3.0 | 18.4 | 4.0 | 0.0 |
| *（b）纯文本（仅 DOM）* | | | | | | | | | | | | | | | | | | | | | |
| DeepSeek-V3.2 | 76.4 | 45.8 | 23.3 | 71.7 | 48.9 | 21.6 | 58.2 | 51.2 | 14.0 | 81.7 | 39.8 | 19.3 | 67.3 | 42.4 | 19.1 | 79.4 | 43.0 | 19.6 | 56.2 | 38.0 | 15.4 |
| GLM-4.7 | 75.7 | 47.4 | 26.7 | 72.3 | 51.6 | 19.1 | 58.7 | 51.4 | 25.0 | 84.2 | 39.8 | 20.6 | 66.8 | 44.5 | 23.4 | 81.7 | 43.2 | 21.7 | 56.6 | 41.4 | 7.7 |
| Gemini-3-Pro (T) | 79.5 | 59.7 | 36.5 | 77.5 | 61.4 | 29.6 | 66.2 | 60.2 | 38.0 | 87.4 | 56.7 | 28.9 | 74.0 | 55.2 | 31.9 | 87.7 | 52.0 | 34.9 | 64.6 | 52.0 | 15.4 |
| Gemini-3-Flash (T) | 78.1 | 48.9 | 25.2 | 73.3 | 52.0 | 22.2 | 57.3 | 52.5 | 30.0 | 86.0 | 42.6 | 18.9 | 69.2 | 45.0 | 22.0 | 82.7 | 45.5 | 20.4 | 58.0 | 41.7 | 7.7 |

> (T) = 纯文本模式（仅 DOM，无截图）。

---

## 快速开始

### 1. 配置环境

```bash
git clone https://github.com/yuandaxia2001/WebForge.git
cd WebForge

conda create -n webforge python=3.11 -y
conda activate webforge

pip install -r requirements.txt
playwright install chromium
```

### 2. 配置

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml，填入你的 API Key
```

唯一必须修改的是 API Key：

```yaml
llm:
  model: "gemini-2.5-flash"
  base_url: "https://..."
  api_key: "你的API密钥"      # <-- 替换这里
```

完整参数说明见 [`config.example.yaml`](./config.example.yaml)。

### 3. 下载 WebForge-Bench

```bash
huggingface-cli download yuandaxia/WebForge --repo-type dataset --local-dir ./benchmark_data
```

### 4. 运行评测

```bash
# 测试单个任务
python run_eval.py \
    --config config.yaml \
    --task-file benchmark_data/tasks.jsonl \
    --task-id 0bb8a4f7e6919eca \
    --website-dir benchmark_data/websites

# 运行全部 934 个任务
python run_eval.py \
    --config config.yaml \
    --task-file benchmark_data/tasks.jsonl \
    --website-dir benchmark_data/websites
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | YAML 配置文件 | （必填） |
| `--task-file` | `tasks.jsonl` 路径 | （必填） |
| `--website-dir` | `websites/` 目录 | （必填） |
| `--task-id` | 按 ID 运行特定任务 | 全部任务 |
| `--output-dir` | 输出目录 | `./output` |
| `--port` | HTTP 服务器端口 | `8000` |

### 5. 评判结果

评测完成后，运行 judge 对 agent 的回答进行评分：

```bash
# 评判所有已完成任务并输出准确率
python run_judge.py \
    --config config.yaml \
    --output-dir ./output

# 导出逐任务 CSV
python run_judge.py \
    --config config.yaml \
    --output-dir ./output \
    --save-csv accuracy.csv
```

Judge LLM 将每个 agent 的 `answer` 与 `ground_truth` 进行比较，输出总体准确率。在 `config.yaml` 中配置 `judge_llm`（推荐使用快速便宜的模型如 `gemini-2.0-flash`）。

---

## 智能体变体

在 config 中设置 `agent.type`：

| 类型 | 工具 | 推荐场景 |
|------|------|----------|
| `full` | `record_step` + `browser_use` + `terminate` | 前沿模型（Gemini、Claude、Kimi）——每步记录观察/推理/行动 |
| `simple` | `browser_use` + `terminate` | 较小模型（GPT-5-Mini/Nano、Qwen-Omni）——无需多工具调用 |

## 引用

```bibtex
@article{yuan2026webforge,
  title={WebForge: Breaking the Realism-Reproducibility-Scalability Trilemma in Browser Agent Benchmark},
  author={Yuan, Peng and Yin, Yuyang and Cai, Yuxuan and Wei, Zheng},
  year={2026}
}
```

## 许可证

[Apache License 2.0](./LICENSE)

## 致谢

基于 [browser-use](https://github.com/browser-use/browser-use)、[OpenManus](https://github.com/FoundationAgents/OpenManus) 和 [Playwright](https://playwright.dev/) 构建。
