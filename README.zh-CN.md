# vLLM-2080Ti-Definitive 0.2 for 2x RTX 2080 Ti 22GB NVLink

语言：[English](README.md) | 简体中文

> **先读这份。** 下面是本机在 2× RTX 2080 Ti 22GB + NVLink 上的实测结论和完整交叉对照。  
> 官方原文在 [docs/upstream/](docs/upstream/)，我们的踩坑原文在 [docs/ours/](docs/ours/)，原始测速/评测文件在 [test_results/](test_results/)。

**评测模型：Qwen3.8-27B 的多个权重版本，不是别的尺寸。** 文中 FP8 / NVFP4 / DFlash 都是这一套 27B：

| 权重 | 目录名 | 用途 |
|---|---|---|
| FP8 | `Qwen3.8-27B-FP8` | 对照、DFlash 三档、9/12 质量 A |
| NVFP4 | `Qwen3.8-27B-NVFP4` | 日常 FastLLM、Definitive 0.2 240K 文本档、9/14 质量 B |

不要看成 Qwen3-8B、Qwen3.8-Flash-Next 或其他参数量。

**叫法：** 文中 **Definitive 0.2 / Definitive 0.1** 都是 [vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) 的发布线，**不是** 官方 vLLM 的版本号。本机 0.2 钉标签 `v0.2.1-pre3`（引擎是上游 vLLM 0.27.1）；0.1 钉 `0.1.17`，不要部署。脚本 `start-v02`、镜像 `native/vllm-2080ti:v0.2.1-pre3-sm75` 都指向这条 0.2 线。

姊妹仓：[FastLLM for 2x RTX 2080 Ti 22GB NVLink](https://github.com/mercury666cn/FastLLM-2x2080Ti-22GB-NVLink)。上游源码：[weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) 的 `v0.2.1-pre3`。

**建议 Docker 部署。** 不要在 CUDA 12.1 / SM75 上现场编译 FastLLM master。两张卡同时只能跑一套。

## 1. 结论（先看这个）

同一台对照机、同一套 Docker，**Qwen3.8-27B 的 FP8 和 NVFP4 都测过**。2026-09-14 日常已切到姊妹仓 FastLLM（NVFP4 + FP16 KV + 240K + 关思考，**带图像**）。本仓仍是 **vLLM-2080Ti-Definitive 0.2 部署说明**。

**若部署 Definitive 0.2：用 NVFP4 + FP8 KV + MTP3 + 240K 文本档（`text-only`）。** 长输出仍比 FastLLM 稳，但这档不带图像。

| 路线 | 本机结果 |
|---|---|
| 官方 FP8 + 128K + FP16-KV | 空启动失败。KV 要约 4.64 GiB，实际只剩约 3.5–3.7 GiB |
| Definitive 0.2 NVFP4 + FP8 KV + 240K **文本档** | `text-only`，不带图像；实跑过 245600 输入，200K 中文 decode 仍约 50 tok/s |
| FastLLM NVFP4 + FP16 KV + 240K（现网日常，见姊妹仓） | 完整权重，**带图像**；关思考 849 题 64.15%，NIAH 15/15；235K 会 OOM |

不要把 Definitive 0.2 的文本档和 FastLLM 现网混成一回事。不要把 FP8 128K 改短再宣称「官方 128K 通过」。

质量不要和 9/12 的 48.14% / 49.15% 横比（开思考、打分只读 `content`）。9/14 关思考后 FastLLM 两套平手，找针 15/15。详见 [test_results/FP8FP8-vs-NVFP4FP16-20260914.md](test_results/FP8FP8-vs-NVFP4FP16-20260914.md)。

Definitive 0.2 固定英文尺子上 NVFP4 decode 会比 FP8 慢（MTP 接受率更低）。建议 NVFP4 的主因是 **上下文能开到 240K 文本**，以及长 decode 比 FastLLM 稳，不是每把短尺子都更快。

**不要部署 Definitive 0.1。** NVFP4 加载失败（`lm_head.weight_scale`）；官方 128K FP16-KV 空启动失败；仓库宣传 100 tok/s 是参考机单核更强的短英文，本机 E5-2686 v4 同口径大约 **69.6**。详见 [docs/ours/runtime-v0.1.md](docs/ours/runtime-v0.1.md)。

## 2. 对照机（本机实测配置）

| 项 | 值 |
|---|---|
| 主板 | RD450X |
| 内存 | DDR4 2133 ECC，16G × 8，4 通道（约 125 GiB） |
| CPU | 2× Xeon E5-2686 v4，72 线程 |
| GPU | 2× RTX 2080 Ti **22GB 魔改**，SM75，NVLink，**PCIe x4** |
| 并发 | 1（`max_batch=1` / `MAX_NUM_SEQS=1`） |
| 评测模型 | **Qwen3.8-27B**（FP8 与 NVFP4 两套权重） |
| 对照机例子 | 局域网 IP:8000，模型根 `$MODEL_ROOT` |

22G 魔改 + x4 不是官方 11G / x16。别人机器容量和带宽会不一样。

| 方案 | 运行时 | CUDA / Python | 本机镜像 |
|---|---|---|---|
| FastLLM | 官方 `ftllm[all]` 0.1.8.2 wheel | 12.1 / 3.10 | `native/fastllm:official-wheel` |
| Definitive 0.2 | fork `v0.2.1-pre3`，vLLM 0.27.1 | 13.0 / 3.12 | `native/vllm-2080ti:v0.2.1-pre3-sm75` |
| Definitive 0.1（不要部署） | fork 0.1.17，vLLM 0.1.15 | 12.8 / 3.11 | `native/vllm-2080ti:v0.1.17` |

## 3. 交叉速度对照（完整表）

口径写在各结果文件里。旧测速数字默认 **关思考**，不要和开思考混排。

### 3.1 固定英文主尺子（4096 in / 128 out，temperature=0，MTP3，三次中位数）

| 运行时 / 模型 | 模式 | Prefill tok/s | Decode tok/s | TTFT |
|---|---:|---:|---:|---:|
| Definitive 0.1 FP8 | MTP3 | **1680.57** | 48.41 | 2.437s |
| Definitive 0.1 NVFP4 | MTP3 | 不支持 | 不支持 | 权重加载失败 |
| Definitive 0.2 FP8 | MTP3 | 1344.61 | 52.45 | 3.046s |
| Definitive 0.2 NVFP4 | MTP3 | 1207.49 | 34.56 | 3.392s |
| FastLLM FP8 | MTP3 | 1356.45 | 66.31 | 3.020s |
| FastLLM NVFP4 | MTP3 | **1535.18** | **74.91** | **2.668s** |
| FastLLM FP8 | DFlash7 | 1393.14 | 63.31 | 2.940s |

- Definitive 0.2 FP8 对 Definitive 0.1 FP8：prefill **-20.0%**，decode **+8.3%**
- FastLLM NVFP4 对 FastLLM FP8：prefill **+13.2%**，decode **+13.0%**
- FastLLM DFlash7 对 FastLLM FP8 MTP3：decode **-4.5%**（本机 DFlash 不是更快）

### 3.2 项目英文尺子（随机词袋，temperature=0.7，4096 点）

| 运行时 / 模型 | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Definitive 0.1 FP8 MTP3 | 1692.79 | 61.81 |
| Definitive 0.2 FP8 MTP3 | 1359.79 | 76.19 |
| Definitive 0.2 NVFP4 MTP3 | 1212.01 | 40.94 |
| FastLLM FP8 MTP3 | 1337.49 | 74.37 |
| FastLLM NVFP4 MTP3 | 1537.38 | 78.79 |
| FastLLM FP8 DFlash7 | 1381.83 | 63.62 |

Definitive 0.2 MTP 接受率随提示词暴涨暴跌：FP8 约 53.3%–98.9%，NVFP4 约 42.3%–84.4%。不要拿一条提示词定终身。

### 3.3 长上下文真实请求（冒烟，出 32 token）

| 方案 | 输入 / 输出 | 结果 | TTFT | Prefill |
|---|---:|---|---:|---:|
| Definitive 0.1 FP8 官方 128K | 启动 | 容量不足 | - | - |
| Definitive 0.2 FP8 官方 128K | 启动 | 容量不足 | - | - |
| Definitive 0.2 NVFP4 MTP3 官方 240K 文本 | 245600 / 32 | **通过** | 378.316s | 649.19 |
| FastLLM FP8 MTP3 128K | 130900 / 32 | **通过** | 167.969s | 779.31 |
| FastLLM NVFP4 MTP3 128K | 130900 / 32 | **通过** | 157.263s | 832.36 |
| FastLLM FP8 DFlash7 128K | 130900 / 32 | **通过** | 171.383s | 763.79 |

### 3.4 中文长上下文（精确 64K / 128K / 200K，出 128，关思考）

| 运行时 | 上下文 | Prefill | Decode | TTFT | 总耗时 |
|---|---:|---:|---:|---:|---:|
| Definitive 0.2 NVFP4 MTP3 | 64K | 563.52 | **55.07** | 116.3s | 118.6s |
| FastLLM NVFP4 MTP3 | 64K | **1077.14** | 51.50 | **60.8s** | **63.3s** |
| FastLLM FP8 DFlash3 | 64K | 964.89 | 45.90 | 67.9s | 70.7s |
| FastLLM FP8 DFlash5 | 64K | 959.35 | 53.88 | 68.3s | 70.7s |
| FastLLM FP8 DFlash7 | 64K | 967.13 | 53.62 | 67.8s | 70.1s |
| Definitive 0.2 NVFP4 MTP3 | 128K | 780.38 | 47.24 | 168.0s | 170.6s |
| FastLLM NVFP4 MTP3 | 128K | **822.70** | 36.53 | **159.3s** | **162.8s** |
| FastLLM FP8 DFlash3 | 128K | 752.73 | 37.98 | 174.1s | 177.5s |
| FastLLM FP8 DFlash5 | 128K | 755.54 | 45.78 | 173.5s | 176.3s |
| FastLLM FP8 DFlash7 | 128K | 754.39 | **47.70** | 173.7s | 176.4s |
| Definitive 0.2 NVFP4 MTP3 | 200K | 623.05 | **50.01** | 328.7s | 331.2s |
| FastLLM NVFP4 MTP3 | 200K | **667.76** | 27.62 | **306.7s** | **311.3s** |
| FastLLM FP8 DFlash3 | 200K | 629.16 | 32.42 | 325.5s | 329.4s |
| FastLLM FP8 DFlash5 | 200K | 629.01 | 30.74 | 325.6s | 329.7s |
| FastLLM FP8 DFlash7 | 200K | 627.60 | 35.18 | 326.3s | 329.9s |

长输入端到端：FastLLM NVFP4 总耗时相对 Definitive 0.2 少约 46.6% / 4.6% / 6.0%。  
长输出：Definitive 0.2 在 200K 仍约 50 tok/s，FastLLM 掉到约 28。  
DFlash 三档用的是 **FP8 主模型**，不能写成 NVFP4+DFlash。

### 3.5 短答总耗时（出 128 token，50K 档）

| 输入 | FastLLM 中文 | Definitive 0.2 中文 | FastLLM 英文 | Definitive 0.2 英文 |
|---:|---:|---:|---:|---:|
| 0.5K | **2.4** | 2.7 | 6.0（5K） | **2.2** |
| 50K | **25.0** | 26.8 | **48.7** | 48.8 |
| 100K | **58.7** | 63.0 | **113.4** | 122.9 |
| 150K | **99.0** | 108.5 | **195.7** | 214.1 |
| 180K | **125.9** | 137.3 | **255.6** | 277.8 |

50K 起 FastLLM 总耗时更短约 7–9%。中文大约比英文快一倍（词表，不是某一套引擎独有）。Definitive 0.2 的优势在长 decode，不在短答总时间。

### 3.6 仓库宣传 100 tok/s

本机同口径 Definitive 0.1 大约 **69.6**，不是 100。参考机单核更强。DFlash 英文短续写 + draft 5/7 可以到 147 / 201，中文思考仍是 50–64。日常中文对话不会变成 180。见 [test_results/对照-仓库100与官方尺子.md](test_results/对照-仓库100与官方尺子.md)。

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

Definitive 0.2 **没有**做同尺 849 题质量赛。质量结论只对 FastLLM 的 FP8 / NVFP4 负责。

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
| [README.md](README.md) | English homepage |
| [README.zh-CN.md](README.zh-CN.md) | **我们的说明（本页，简体中文）** |
| [docs/ours/](docs/ours/) | 本机踩坑原文（FastLLM / Definitive 0.2 / Definitive 0.1） |
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

不改推理源码，不转权重，不打兼容补丁。修的是能稳定对外服务的 Docker 跑法：FastLLM 只用官方 wheel；Definitive 0.2 烤 `libxcb1`、双缓存、`SERVICE_SCOPE=lan`、launcher 冒烟后 wait PID。详见 [docs/ours/](docs/ours/)。
