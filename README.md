# vLLM 0.2 for 2x RTX 2080 Ti 22GB NVLink

> **先读这份。** 下面是本机在 2× RTX 2080 Ti 22GB + NVLink 上的实测结论和完整交叉对照。  
> 官方原文在 [docs/upstream/](docs/upstream/)，我们的踩坑原文在 [docs/ours/](docs/ours/)，原始测速/评测文件在 [test_results/](test_results/)。

姊妹仓：[FastLLM for 2x RTX 2080 Ti 22GB NVLink](https://github.com/mercury666cn/FastLLM-2x2080Ti-22GB-NVLink)。上游源码：[weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) 的 `v0.2.1-pre3`。

**建议 Docker 部署。** 不要在 CUDA 12.1 / SM75 上现场编译 FastLLM master。两张卡同时只能跑一套。

## 1. 结论（先看这个）

同一台对照机、同一套 Docker，**FP8 和 NVFP4 都测过**。2026-09-14 日常已切到姊妹仓 FastLLM（NVFP4 + FP16 KV + 240K + 关思考，**带图像**）。本仓仍是 **v0.2 部署说明**。

**若部署 v0.2：用 NVFP4 + FP8 KV + MTP3 + 240K 文本档（`text-only`）。** 长输出仍比 FastLLM 稳，但这档不带图像。

| 路线 | 本机结果 |
|---|---|
| 官方 FP8 + 128K + FP16-KV | 空启动失败。KV 要约 4.64 GiB，实际只剩约 3.5–3.7 GiB |
| v0.2 NVFP4 + FP8 KV + 240K **文本档** | `text-only`，不带图像；实跑过 245600 输入，200K 中文 decode 仍约 50 tok/s |
| FastLLM NVFP4 + FP16 KV + 240K（现网日常，见姊妹仓） | 完整权重，**带图像**；关思考 849 题 64.15%，NIAH 15/15；235K 会 OOM |

不要把 v0.2 的文本档和 FastLLM 现网混成一回事。不要把 FP8 128K 改短再宣称「官方 128K 通过」。

质量不要和 9/12 的 48.14% / 49.15% 横比（开思考、打分只读 `content`）。9/14 关思考后 FastLLM 两套平手，找针 15/15。详见 [test_results/FP8FP8-vs-NVFP4FP16-20260914.md](test_results/FP8FP8-vs-NVFP4FP16-20260914.md)。

v0.2 固定英文尺子上 NVFP4 decode 会比 FP8 慢（MTP 接受率更低）。建议 NVFP4 的主因是 **上下文能开到 240K 文本**，以及长 decode 比 FastLLM 稳，不是每把短尺子都更快。

**不要部署 v0.1。** NVFP4 加载失败（`lm_head.weight_scale`）；官方 128K FP16-KV 空启动失败；仓库宣传 100 tok/s 是参考机单核更强的短英文，本机 E5-2686 v4 同口径大约 **69.6**。详见 [docs/ours/runtime-v0.1.md](docs/ours/runtime-v0.1.md)。

## 2. 对照机（本机实测配置）

| 项 | 值 |
|---|---|
| 主板 | RD450X |
| 内存 | DDR4 2133 ECC，16G × 8，4 通道（约 125 GiB） |
| CPU | 2× Xeon E5-2686 v4，72 线程 |
| GPU | 2× RTX 2080 Ti **22GB 魔改**，SM75，NVLink，**PCIe x4** |
| 并发 | 1（`max_batch=1` / `MAX_NUM_SEQS=1`） |
| 对照机例子 | 局域网 IP:8000，模型根 `$MODEL_ROOT` |

22G 魔改 + x4 不是官方 11G / x16。别人机器容量和带宽会不一样。

| 方案 | 运行时 | CUDA / Python | 本机镜像 |
|---|---|---|---|
| FastLLM | 官方 `ftllm[all]` 0.1.8.2 wheel | 12.1 / 3.10 | `native/fastllm:official-wheel` |
| v0.2 | fork `v0.2.1-pre3`，vLLM 0.27.1 | 13.0 / 3.12 | `native/vllm-2080ti:v0.2.1-pre3-sm75` |
| v0.1（不要部署） | fork 0.1.17，vLLM 0.1.15 | 12.8 / 3.11 | `native/vllm-2080ti:v0.1.17` |

## 3. 交叉速度对照（完整表）

口径写在各结果文件里。旧测速数字默认 **关思考**，不要和开思考混排。

### 3.1 固定英文主尺子（4096 in / 128 out，temperature=0，MTP3，三次中位数）

| 运行时 / 模型 | 模式 | Prefill tok/s | Decode tok/s | TTFT |
|---|---:|---:|---:|---:|
| v0.1 FP8 | MTP3 | **1680.57** | 48.41 | 2.437s |
| v0.1 NVFP4 | MTP3 | 不支持 | 不支持 | 权重加载失败 |
| v0.2 FP8 | MTP3 | 1344.61 | 52.45 | 3.046s |
| v0.2 NVFP4 | MTP3 | 1207.49 | 34.56 | 3.392s |
| FastLLM FP8 | MTP3 | 1356.45 | 66.31 | 3.020s |
| FastLLM NVFP4 | MTP3 | **1535.18** | **74.91** | **2.668s** |
| FastLLM FP8 | DFlash7 | 1393.14 | 63.31 | 2.940s |

- v0.2 FP8 对 v0.1 FP8：prefill **-20.0%**，decode **+8.3%**
- FastLLM NVFP4 对 FastLLM FP8：prefill **+13.2%**，decode **+13.0%**
- FastLLM DFlash7 对 FastLLM FP8 MTP3：decode **-4.5%**（本机 DFlash 不是更快）

### 3.2 项目英文尺子（随机词袋，temperature=0.7，4096 点）

| 运行时 / 模型 | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| v0.1 FP8 MTP3 | 1692.79 | 61.81 |
| v0.2 FP8 MTP3 | 1359.79 | 76.19 |
| v0.2 NVFP4 MTP3 | 1212.01 | 40.94 |
| FastLLM FP8 MTP3 | 1337.49 | 74.37 |
| FastLLM NVFP4 MTP3 | 1537.38 | 78.79 |
| FastLLM FP8 DFlash7 | 1381.83 | 63.62 |

v0.2 MTP 接受率随提示词暴涨暴跌：FP8 约 53.3%–98.9%，NVFP4 约 42.3%–84.4%。不要拿一条提示词定终身。

### 3.3 长上下文真实请求（冒烟，出 32 token）

| 方案 | 输入 / 输出 | 结果 | TTFT | Prefill |
|---|---:|---|---:|---:|
| v0.1 FP8 官方 128K | 启动 | 容量不足 | - | - |
| v0.2 FP8 官方 128K | 启动 | 容量不足 | - | - |
| v0.2 NVFP4 MTP3 官方 240K 文本 | 245600 / 32 | **通过** | 378.316s | 649.19 |
| FastLLM FP8 MTP3 128K | 130900 / 32 | **通过** | 167.969s | 779.31 |
| FastLLM NVFP4 MTP3 128K | 130900 / 32 | **通过** | 157.263s | 832.36 |
| FastLLM FP8 DFlash7 128K | 130900 / 32 | **通过** | 171.383s | 763.79 |

### 3.4 中文长上下文（精确 64K / 128K / 200K，出 128，关思考）

| 运行时 | 上下文 | Prefill | Decode | TTFT | 总耗时 |
|---|---:|---:|---:|---:|---:|
| v0.2 NVFP4 MTP3 | 64K | 563.52 | **55.07** | 116.3s | 118.6s |
| FastLLM NVFP4 MTP3 | 64K | **1077.14** | 51.50 | **60.8s** | **63.3s** |
| FastLLM FP8 DFlash3 | 64K | 964.89 | 45.90 | 67.9s | 70.7s |
| FastLLM FP8 DFlash5 | 64K | 959.35 | 53.88 | 68.3s | 70.7s |
| FastLLM FP8 DFlash7 | 64K | 967.13 | 53.62 | 67.8s | 70.1s |
| v0.2 NVFP4 MTP3 | 128K | 780.38 | 47.24 | 168.0s | 170.6s |
| FastLLM NVFP4 MTP3 | 128K | **822.70** | 36.53 | **159.3s** | **162.8s** |
| FastLLM FP8 DFlash3 | 128K | 752.73 | 37.98 | 174.1s | 177.5s |
| FastLLM FP8 DFlash5 | 128K | 755.54 | 45.78 | 173.5s | 176.3s |
| FastLLM FP8 DFlash7 | 128K | 754.39 | **47.70** | 173.7s | 176.4s |
| v0.2 NVFP4 MTP3 | 200K | 623.05 | **50.01** | 328.7s | 331.2s |
| FastLLM NVFP4 MTP3 | 200K | **667.76** | 27.62 | **306.7s** | **311.3s** |
| FastLLM FP8 DFlash3 | 200K | 629.16 | 32.42 | 325.5s | 329.4s |
| FastLLM FP8 DFlash5 | 200K | 629.01 | 30.74 | 325.6s | 329.7s |
| FastLLM FP8 DFlash7 | 200K | 627.60 | 35.18 | 326.3s | 329.9s |

长输入端到端：FastLLM NVFP4 总耗时相对 v0.2 少约 46.6% / 4.6% / 6.0%。  
长输出：v0.2 在 200K 仍约 50 tok/s，FastLLM 掉到约 28。  
DFlash 三档用的是 **FP8 主模型**，不能写成 NVFP4+DFlash。

### 3.5 短答总耗时（出 128 token，50K 档）

| 输入 | FastLLM 中文 | v0.2 中文 | FastLLM 英文 | v0.2 英文 |
|---:|---:|---:|---:|---:|
| 0.5K | **2.4** | 2.7 | 6.0（5K） | **2.2** |
| 50K | **25.0** | 26.8 | **48.7** | 48.8 |
| 100K | **58.7** | 63.0 | **113.4** | 122.9 |
| 150K | **99.0** | 108.5 | **195.7** | 214.1 |
| 180K | **125.9** | 137.3 | **255.6** | 277.8 |

50K 起 FastLLM 总耗时更短约 7–9%。中文大约比英文快一倍（词表，不是某一套引擎独有）。v0.2 的优势在长 decode，不在短答总时间。

### 3.6 仓库宣传 100 tok/s

本机同口径 v0.1 大约 **69.6**，不是 100。参考机单核更强。DFlash 英文短续写 + draft 5/7 可以到 147 / 201，中文思考仍是 50–64。日常中文对话不会变成 180。见 [test_results/对照-仓库100与官方尺子.md](test_results/对照-仓库100与官方尺子.md)。

## 4. 质量交叉对照（FastLLM FP8 vs NVFP4，849 题）

同一镜像、MTP3、FP8 KV、210K、temperature=0、顺序单并发。不转权重、不打补丁。

### 4.1 去重后（生产门禁通过）

| 类别 | FP8 | NVFP4 | 差值 |
|---|---:|---:|---:|
| 中文知识 | 33.50% | **42.00%** | +8.50pp |
| 数学推理 | 69.50% | **70.50%** | +1.00pp |
| 代码 HumanEval | 65.24% | 65.24% | 0 |
| 指令遵循 IFEval | 74.89% | 74.22% | -0.67pp |
| 长上下文 | 45.72% | 42.94% | -2.79pp |
| 多轮 JSON 约束 | 0.00% | 0.00% | 0 |
| **等权综合** | **48.14%** | **49.15%** | **+1.01pp** |

比值 102.1%。849 题两组都无接口错误。长上下文差距主要来自 NIAH（FP8 12/15，NVFP4 8/15）。多轮两边都是 0：模型爱写中文键 `预算`，题面要 `budget`。

### 4.2 首轮（未去重，门禁未过，一并保留）

综合 100.44%，但长上下文 -6.78pp。NIAH 首轮 7/15，复跑 4/15。这份说明「容量能装下 ≠ 检索一样稳」。原始文件都在 `test_results/`。

全量评测时 CPU 约 6.2% / 72 核，GPU 约 94%。**单并发别换 CPU。**

v0.2 **没有**做同尺 849 题质量赛。质量结论只对 FastLLM 的 FP8 / NVFP4 负责。

### 4.3 2026-09-14 关思考对打（FastLLM 日常留下 B）

同一 849 题、关思考。A：FP8 权 + FP8 KV。B：NVFP4 权 + FP16 KV。综合 64.18% / 64.15%，NIAH 15/15。不要和 9/12 开思考的 48%/49% 横比。报告：[test_results/FP8FP8-vs-NVFP4FP16-20260914.md](test_results/FP8FP8-vs-NVFP4FP16-20260914.md)。

## 5. 建议怎么部署（Docker）

镜像：`native/vllm-2080ti:v0.2.1-pre3-sm75`。权重：`$MODEL_ROOT/Qwen3.8-27B-NVFP4`。

```text
--gpus all --network host --ipc host
挂载: $MODEL_ROOT(只读) + $VLLM_CACHE + $TRITON_CACHE + $RUN_LOGS
ln -sfn /opt/venv /workspace/.venv
./launcher.sh --non-interactive \
  --set PROFILE=qwen27b/w4a16/normal/fp8kv-240K-mtp3-text-only.env \
  --set MODEL_DIR=$MODEL_ROOT/Qwen3.8-27B-NVFP4 \
  --set PORT=8000 \
  --set MAX_MODEL_LEN=210000 \
  --set SERVED_NAME=qwen38-v02-nvfp4-mtp3-210k \
  --set MAX_NUM_SEQS=1 \
  --set SERVICE_SCOPE=lan
然后 wait 住 PID，不要 exec 完就结束
```

必做：烤进 `libxcb1`；两处缓存一起挂；`SERVICE_SCOPE=lan`；launcher 冒烟后 wait PID。  
对照机脚本见 [scripts/start-v02.sh](scripts/start-v02.sh)。


不要给 CUDA / 权重大包走高倍率代理。Chatbox：API Host 填 `http://<IP>:8000/v1`，**API Path 留空**。

## 6. 目录

| 路径 | 内容 |
|---|---|
| [README.md](README.md) | **我们的说明（本页）** |
| [docs/ours/](docs/ours/) | 本机踩坑原文（FastLLM / v0.2 / v0.1） |
| [docs/upstream/](docs/upstream/) | 上游官方 README 原文，未改 |
| [test_results/](test_results/) | 完整测速 JSON/CSV、质量 jsonl、失败栈、交叉对照 md |
| [quality_eval/](quality_eval/) | 题集清单、基线（完整 12MB 题面 `suite.jsonl` 太大，不进仓） |
| [scripts/](scripts/) | 对照机互斥启动脚本附录 |

交叉对照 md：

- [test_results/原生Docker三方案对照-20260912.md](test_results/原生Docker三方案对照-20260912.md)
- [test_results/NVFP4生产验收-20260912.md](test_results/NVFP4生产验收-20260912.md)
- [test_results/NVFP4整体能力验收-20260912.md](test_results/NVFP4整体能力验收-20260912.md)
- [test_results/NVFP4中文长上下文对照-20260912.md](test_results/NVFP4中文长上下文对照-20260912.md)
- [test_results/中英文总耗时对照-50k-20260912.md](test_results/中英文总耗时对照-50k-20260912.md)
- [test_results/对照-仓库100与官方尺子.md](test_results/对照-仓库100与官方尺子.md)
- [test_results/FP8FP8-vs-NVFP4FP16-20260914.md](test_results/FP8FP8-vs-NVFP4FP16-20260914.md)
- [test_results/并发1-2-4-1k30k-20260914.md](test_results/并发1-2-4-1k30k-20260914.md)

原始逐点文件包括：`native_*_standard_en_chat_4k128.json`、`native_*_128k_real.json`、`native_v02_nvfp4_240k_real.json`、`native_cn_*`、`quality_model_a_fp8.jsonl`、`quality_model_b_nvfp4*.jsonl`、`quality_gate_20260912.json`、失败栈 `native_v01_*.log` / `native_v02_fp8_128k_native_failure.log`。

## 7. 我们改了什么 / 没改什么

不改推理源码，不转权重，不打兼容补丁。修的是能稳定对外服务的 Docker 跑法：FastLLM 只用官方 wheel；v0.2 烤 `libxcb1`、双缓存、`SERVICE_SCOPE=lan`、launcher 冒烟后 wait PID。详见 [docs/ours/](docs/ours/)。

---

下面两节是原文附件，**以第 1 节本机结论为准**，不要用上游海报上的 100 tok/s 覆盖本机数字。

## 原文附件：README

<!-- markdownlint-disable MD001 MD041 -->
# vLLM 2080 Ti Definitive Edition

![vLLM 2080 Ti Definitive Edition cover](docs/assets/vllm-2080ti-cover.jpg)

Hardware-focused vLLM fork for dual RTX 2080 Ti 22 GB / SM75 serving. This is the `vllm-2080ti-definitive-0.2.x` maintenance branch: the public `0.2.x` prerelease line rebased on upstream vLLM `v0.27.1`, CUDA 13.0, and PyTorch 2.13. It is not the stable production default; the maintained `0.1.x` line remains the project's primary stable release line for CUDA 12.8 and PyTorch 2.11.

This project preserves the SM75-specific source changes, launcher profiles,
and benchmark evidence needed to reproduce the dual-2080-Ti TP=2 stack. It is
based on upstream vLLM; retain both the upstream license and attribution to
`github.com/weicj` when redistributing a derivative.

Language: English | [Simplified Chinese](README.zh-CN.md)

![Live single-request throughput demo](docs/assets/vllmspeed.gif)

Fork release: `0.2.1-pre3`
Base vLLM: `0.27.1`

Branch: [`vllm-2080ti-definitive-0.2.x`](https://github.com/weicj/vLLM-2080Ti-Definitive/tree/vllm-2080ti-definitive-0.2.x)
Prerelease snapshot: [v0.2.1-pre3](https://github.com/weicj/vLLM-2080Ti-Definitive/releases/tag/v0.2.1-pre3)
Changes since pre2: [CHANGELOG.md](CHANGELOG.md)

## Why RTX 2080 Ti For LLM Inference?

The project is built around a practical cost/performance premise: two 22 GB
RTX 2080 Ti cards joined by NVLink provide 44 GB of VRAM, substantial memory
bandwidth, and 136 Turing SMs. With an SM75-aware vLLM route, that is enough
for serious local 27B and 35B-class serving rather than only small-model use.

| Metric | 2x RTX 2080 Ti 22 GB + NVLink | RTX 3090 Ti 24 GB baseline | Ratio |
| --- | ---: | ---: | ---: |
| Physical CUDA cores | 8,704 | 5,376 | 1.62x |
| SM count | 136 | 84 | 1.62x |
| Physical Tensor Cores | 1,088 | 336 | 3.24x |
| Dense FP16 matrix throughput | 228 TFLOPS | 160 TFLOPS | 1.43x |
| Total memory bandwidth | 1,232 GB/s | 1,008 GB/s | 1.22x |
| Total VRAM | 44 GB | 24 GB | 1.83x |

The fork turns those hardware properties into a usable serving stack through
Marlin, FlashInfer/FlashQLA, TurboQuant/INT8 KV, MTP, and CUDA Graph support.

## Status

The `0.2.x` target is Ubuntu 26.04 or later, Linux kernel 7 or later, GCC/G++ 15, CUDA 13.0, and PyTorch 2.13. The maintained `0.1.x` line remains the compatibility route for CUDA 12.8, PyTorch 2.11, older kernels, and GCC 12/13/14.

The dual-2080-Ti CUDA Graph validation is recorded in
[the migration report](docs/2080ti-0.2.1-pre-validation.md). It includes the
exact host selection rule, build gate, correctness regressions, and 4K/128
benchmark method. Do not treat a checkpoint that merely loads as a promoted
deployment route.

The disposition of the previously merged SM75 PRs is recorded in [the 0.2.x PR migration audit](docs/0.2.x-pr-migration-audit.md).

The launcher supports selecting tensor parallelism (`TP_SIZE`) and pipeline
parallelism (`PP_SIZE`), including mixed TP/PP inference layouts when the
visible GPU count matches the requested topology. The primary validated
deployment remains dual RTX 2080 Ti with TP=2 and PP=1; other layouts are
available for engineering tests and require separate validation.

## Tested Model Checkpoints

Current model and weight routes. Individual serving presets and measurements
are listed in the [Profile Guide](profiles/README.md).

| Model route | Weight route | Model card | Recommended use |
| --- | --- | --- | --- |
| Qwen3.8 27B | FP8 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | High-precision single-request inference |
| Qwen3.8 27B | NVFP4 | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | Long-context concurrent inference |
| Qwen3.x 35B | FP8 | [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | Fast personal inference |

## Build And Launch

```bash
git clone https://github.com/weicj/vLLM-2080Ti-Definitive.git
cd vLLM-2080Ti-Definitive
git switch --track origin/vllm-2080ti-definitive-0.2.x
./build.sh
```

```bash
MODEL_DIR=/path/to/checkpoint \
PROFILE=qwen27b/w8a16/fast/tqk8v4-256K-mtp3-text-only.env \
MODE=fast GPU_DEVICES=4,5 TP_SIZE=2 \
NON_INTERACTIVE=1 ./launcher.sh
```

Use `./launcher.sh` for interactive setup or `./launcher.sh --print-config` to
preview a route. See the [Profile Guide](profiles/README.md) for available
profiles.

## Profiles

Start with [the Profile Guide](profiles/README.md). Profiles use the layout
`profiles/<model>/<weight>/<mode>/<route>.env`; for example,
`qwen27b/w8a16/normal/fp16kv-128K-mtp3-text-only.env`,
`qwen35b/w8a16/normal/fp16kv-256K-nomtp-text-only.env`, and
`qwen35b/w8a16/normal/fp16kv-136K-nomtp-text-image.env`.

Available modes:

- `normal`: stable daily deployment mode.
- `fast`: higher-performance mode, to be used only with a validated route.
- `aggressive`: highest-performance mode with increased quality risk.
- `safe`: conservative fallback for troubleshooting and compatibility.

The profile selects only route parameters. The launcher owns GPU selection,
port, model path, chat template, and reasoning defaults.

## MTP And KV Precision

Use a shipped profile before hand-tuning MTP and KV settings. Choose KV by
intent: FP16/default KV for output quality, INT8 KV for balanced long-context
service, and TurboQuant K8V4 for compressed fast routes. MTP gains depend on
acceptance rate, so a synthetic peak must be checked against real output and
the profile's quality probe.

For the current migration, use the exact method and measurements in
[the validation report](docs/2080ti-0.2.1-pre-validation.md), especially for
TurboQuant and MTP3. Historical profile capacities are not cu130 evidence.
INT6/AutoRound checkpoints require `humming-kernels[cu13]==0.1.13`, which is
the version pinned by this branch; the full Minachist route remains unverified.

## Hardware Target

- Two RTX 2080 Ti 22 GB GPUs connected by NVLink
- NVIDIA Turing / SM75, tensor parallel size 2
- `0.2.1-pre3` target: CUDA 13.0, PyTorch 2.13, Python 3.12
- Target host: Ubuntu 26.04 or later, Linux kernel 7 or later, GCC/G++ 15

Other Turing cards need independent validation for VRAM capacity, PCIe/NVLink
topology, model head dimensions, KV-cache dtype, and CUDA Graph behavior.

## Hardware Q&A

**What GPU interconnect is required?**

NVLink is recommended. PCIe P2P is the baseline requirement, but narrow PCIe
links without NVLink are not a proven substitute for the validated topology.
Confirm P2P and benchmark the actual host topology before treating it as a
deployment route.

**Does the host need a strong CPU or a lot of RAM?**

A high-end CPU is not required, but modern single-core performance and low
platform latency matter. More RAM mainly helps builds, downloads, and compile
cache. Very old CPU platforms can lower decode throughput even when the GPUs
are unchanged.

**Can 11 GB and 22 GB Turing cards be mixed?**

Not for the documented 27B/35B TP=2 routes. Tensor parallelism is effectively
limited by the smaller rank. Better alternatives are paired high-VRAM TU102
cards, such as TITAN RTX, Quadro RTX 6000, or Quadro RTX 8000, with NVLink or
confirmed PCIe P2P and a separate profile validation.

**Which CUDA and PyTorch versions apply?**

The `0.2.1-pre3` target is CUDA 13.0 with PyTorch 2.13. The older CUDA 12.8 /
PyTorch 2.11 stack remains a separate `v0.1.x` compatibility line. Keep the
PyTorch CUDA build, toolkit, FlashInfer/FlashQLA build, and selected profile
aligned; they are not interchangeable runtime combinations.

**What other hardware risks matter?**

Cooling, stable power delivery, and enough SSD capacity for weights and compile
caches. Thermal throttling can look like a software performance regression,
particularly during long prefill and repeated CUDA Graph/AOT compilation.

## Related Project

- [2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox): companion
  toolbox for dual-2080-Ti model routes, benchmark summaries, model notes, and
  operational guidance. This repository focuses on the patched vLLM runtime.

## Credits And Upstream Projects

This repository is a hardware-focused fork of
[vLLM](https://github.com/vllm-project/vllm), licensed under Apache-2.0. It
keeps the upstream project structure and adds local SM75 runtime patches,
launch profiles, and dual-2080-Ti validation notes.

Acceleration components used or integrated by this runtime include:

- [vLLM](https://github.com/vllm-project/vllm): base inference engine and
  serving stack.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer): attention,
  sampling, and quantized kernel paths used by vLLM.
- [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA): upstream Gated
  DeltaNet / Qwen hybrid linear-attention implementation.
- [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75):
  SM70/SM75 adaptation used by the validated Qwen prefill route.
- TurboQuant, Marlin, CUTLASS, Triton, and related vLLM kernels.

Upstream updates are re-evaluated within the SM75-specific scope of this fork.

## 原文附件：README.zh-CN

<!-- markdownlint-disable MD001 MD041 -->
# vLLM 2080 Ti Definitive Edition

![vLLM 2080 Ti Definitive Edition 题图](docs/assets/vllm-2080ti-cover.jpg)

面向双 RTX 2080 Ti 22 GB / SM75 推理的硬件定向 vLLM fork。本分支是 `vllm-2080ti-definitive-0.2.x` 维护线：基于上游 vLLM `v0.27.1`、CUDA 13.0 和 PyTorch 2.13 的公开 `0.2.x` 预发布线。它不是稳定生产默认版本；持续维护的 `0.1.x` 线仍是项目当前的稳定发布主线，对应 CUDA 12.8 与 PyTorch 2.11。

项目保留复现双 2080 Ti TP=2 栈所需的 SM75 专用源码修改、launcher profile 和
测试证据。它基于上游 vLLM；再发布派生版本时必须保留上游许可证、上游署名以及
`github.com/weicj` 的项目署名。

语言：[English](README.md) | 简体中文

![单请求实时测速演示](docs/assets/vllmspeed.gif)

Fork 版本：`0.2.1-pre3`
基础 vLLM：`0.27.1`

分支：[`vllm-2080ti-definitive-0.2.x`](https://github.com/weicj/vLLM-2080Ti-Definitive/tree/vllm-2080ti-definitive-0.2.x)
预发布快照：[v0.2.1-pre3](https://github.com/weicj/vLLM-2080Ti-Definitive/releases/tag/v0.2.1-pre3)
相比 pre2 的变化：[CHANGELOG.md](CHANGELOG.md)

## 为什么用 RTX 2080 Ti 做 LLM 推理？

这个项目的判断很实际：两张通过 NVLink 连接的 22 GB RTX 2080 Ti 提供 44 GB
显存、较高的显存带宽和 136 个 Turing SM。经过针对 SM75 的 vLLM 适配后，这套
硬件不只是运行小模型，也足以承载严肃的本地 27B 与 35B 级模型服务。

| 指标 | 2x RTX 2080 Ti 22 GB + NVLink | RTX 3090 Ti 24 GB 基线 | 倍率 |
| --- | ---: | ---: | ---: |
| 物理 CUDA core | 8,704 | 5,376 | 1.62x |
| SM 数量 | 136 | 84 | 1.62x |
| 物理 Tensor Core | 1,088 | 336 | 3.24x |
| Dense FP16 矩阵吞吐 | 228 TFLOPS | 160 TFLOPS | 1.43x |
| 总显存带宽 | 1,232 GB/s | 1,008 GB/s | 1.22x |
| 总显存 | 44 GB | 24 GB | 1.83x |

本 fork 通过 Marlin、FlashInfer/FlashQLA、TurboQuant/INT8 KV、MTP 和 CUDA
Graph，把这些硬件资源转成可用的 serving 栈。

## 当前状态

`0.2.x` 的目标环境是 Ubuntu 26.04 及以上、Linux kernel 7 及以上、GCC/G++ 15、CUDA 13.0 与 PyTorch 2.13。持续维护的 `0.1.x` 线仍是 CUDA 12.8、PyTorch 2.11、较早 kernel 以及 GCC 12/13/14 的兼容路线。

双 2080 Ti 的 CUDA Graph 验证记录在
[迁移验证报告](docs/2080ti-0.2.1-pre-validation.md)：其中包含实际显卡选择规则、
构建门槛、正确性回归以及 4K/128 测试口径。仅能加载的 checkpoint 不应被视为
已提升为部署路线。

此前已合并 SM75 PR 的迁移判断见 [0.2.x PR 迁移审计](docs/0.2.x-pr-migration-audit.md)。

Launcher 支持选择 tensor parallel（`TP_SIZE`）和 pipeline parallel（`PP_SIZE`），
当可见 GPU 数量与拓扑要求匹配时可以启动 TP/PP 混合推理。当前主要验证部署仍是双
RTX 2080 Ti、TP=2、PP=1；其他并行布局可用于工程测试，但需要单独完成验证。

## 已测试模型权重

当前模型和权重路线。具体服务预设和性能数据见
[Profile 导引](profiles/README.zh-CN.md)。

| 模型路线 | 权重路线 | 模型卡 | 推荐场景 |
| --- | --- | --- | --- |
| Qwen3.8 27B | FP8 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | 高精度单并发 |
| Qwen3.8 27B | NVFP4 | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | 长上下文多并发 |
| Qwen3.x 35B | FP8 | [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | 个人快速推理 |

## 构建与启动

```bash
git clone https://github.com/weicj/vLLM-2080Ti-Definitive.git
cd vLLM-2080Ti-Definitive
git switch --track origin/vllm-2080ti-definitive-0.2.x
./build.sh
```

```bash
MODEL_DIR=/path/to/checkpoint \
PROFILE=qwen27b/w8a16/fast/tqk8v4-256K-mtp3-text-only.env \
MODE=fast GPU_DEVICES=4,5 TP_SIZE=2 \
NON_INTERACTIVE=1 ./launcher.sh
```

使用 `./launcher.sh` 进入交互式配置，或使用 `./launcher.sh --print-config` 预览路线。
可用 profile 见 [Profile 导引](profiles/README.zh-CN.md)。

## Profile 与推荐路线

从 [Profile 导引](profiles/README.zh-CN.md) 开始选。Profile 按
`profiles/<model>/<weight>/<mode>/<route>.env` 组织，例如
`qwen27b/w8a16/normal/fp16kv-128K-mtp3-text-only.env`、
`qwen35b/w8a16/normal/fp16kv-256K-nomtp-text-only.env` 和
`qwen35b/w8a16/normal/fp16kv-136K-nomtp-text-image.env`。

可用模式：

- `normal`：稳定的日常部署模式。
- `fast`：更高性能模式，只用于已验证路线。
- `aggressive`：性能最高但质量风险也最高。
- `safe`：用于排障和兼容性的保守回退模式。

Profile 只选择路线参数。GPU、端口、模型路径、chat template 和 reasoning 默认值
由 launcher 统一管理。

## MTP 与 KV 精度

优先使用项目自带 profile，不要一开始手动调 MTP 和 KV。KV 先按目标选择：
FP16/default KV 追求输出质量，INT8 KV 用于平衡型长上下文服务，TurboQuant K8V4
用于压缩 fast 路线。MTP 收益取决于接受率，合成峰值必须再用真实输出和质量探针
检查。

当前迁移请以[验证报告](docs/2080ti-0.2.1-pre-validation.md)中的精确方法和数据
为准，尤其是 TurboQuant 与 MTP3。历史 profile 容量不是 cu130 证据。
INT6/AutoRound checkpoint 需要 `humming-kernels[cu13]==0.1.13`，这是本分支锁定的版本；
完整 Minachist 路线仍未验证。

## 目标硬件

- 两张经 NVLink 连接的 RTX 2080 Ti 22 GB
- NVIDIA Turing / SM75，tensor parallel size 2
- `0.2.1-pre3` 目标：CUDA 13.0、PyTorch 2.13、Python 3.12
- 目标主机：Ubuntu 26.04 及以上、Linux kernel 7 及以上、GCC/G++ 15

其它 Turing 显卡仍需针对显存容量、PCIe/NVLink 拓扑、模型 head dimension、
KV cache dtype 和 CUDA Graph 行为独立验证。

## 硬件 Q&A

**需要什么样的卡间互联？**

推荐 NVLink。PCIe P2P 是底线，但没有 NVLink 时不能把窄 PCIe 链路直接视为已验证
替代方案；应先确认 P2P，再按实际主机拓扑测试。

**需要很强的 CPU 或很多内存吗？**

不需要高端 CPU，但现代单核性能和较低的平台延迟很重要。更多内存主要帮助构建、
下载和 compile cache；即使 GPU 相同，老旧 CPU 平台也可能降低 decode 吞吐。

**可以混用 11 GB 和 22 GB Turing 卡吗？**

不建议用于文档中的 27B/35B TP=2 路线。TP 会受到较小 rank 显存的限制。更好的
候选是成对的高显存 TU102 卡，例如 TITAN RTX、Quadro RTX 6000 或 Quadro RTX
8000，并且要有 NVLink 或确认可用的 PCIe P2P，之后仍需独立验证 profile。

**应该使用哪些 CUDA 和 PyTorch 版本？**

`0.2.1-pre3` 目标是 CUDA 13.0 + PyTorch 2.13。旧的 CUDA 12.8 + PyTorch 2.11
仍作为独立的 `v0.1.x` 兼容路线维护。PyTorch CUDA 构建、toolkit、FlashInfer/
FlashQLA 构建和启动 profile 必须保持一致，不能混用运行时假设。

**还有哪些硬件风险？**

注意散热、供电稳定性，以及模型和编译缓存所需的 SSD 空间。长 prefill 或反复
CUDA Graph/AOT 编译时降频很容易被误判为软件性能回退。

## 相关项目

- [2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox)：双 2080 Ti
  模型路线、benchmark 汇总、模型记录和运行建议的配套工具箱。本仓库聚焦于补丁后
  的 vLLM runtime。

## 致谢 / 上游项目

本仓库是基于上游 [vLLM](https://github.com/vllm-project/vllm) 的硬件定向 fork，
遵循 Apache-2.0 license，保留上游项目结构，并加入面向双 2080 Ti 的 SM75 runtime
补丁、启动 profile 和验证记录。

当前使用或集成的加速组件包括：

- [vLLM](https://github.com/vllm-project/vllm)：基础推理引擎和 serving 框架。
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)：attention、sampling
  和量化 kernel 路线。
- [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA)：上游 Gated DeltaNet /
  Qwen hybrid linear-attention 实现。
- [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75)：
  SM70/SM75 适配版本，用于已验证的 Qwen prefill 路线。
- TurboQuant、Marlin、CUTLASS、Triton 以及 vLLM 相关 kernel。

上游更新合入后，仍会在本 fork 的 SM75 范围内重新验证。

