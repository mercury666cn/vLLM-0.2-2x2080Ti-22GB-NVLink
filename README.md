# vLLM-2080Ti-Definitive 0.2 for 2x RTX 2080 Ti 22GB NVLink

Language: English | [Simplified Chinese](README.zh-CN.md)

> **Read this first.** Measured results on one dual RTX 2080 Ti 22GB + NVLink box.  
> Upstream text is in [docs/upstream/](docs/upstream/). Our notes are in [docs/ours/](docs/ours/). Raw benches are in [test_results/](test_results/).

**Eval model: several weights of Qwen3.8-27B, not another size.** Every FP8 / NVFP4 / DFlash number on this page is that 27B:

| Weights | Directory | Used for |
|---|---|---|
| FP8 | `Qwen3.8-27B-FP8` | Baseline, three DFlash rows, 9/12 quality A |
| NVFP4 | `Qwen3.8-27B-NVFP4` | Daily FastLLM, Definitive 0.2 240K text profile, 9/14 quality B |

Not Qwen3-8B, not Qwen3.8-Flash-Next, not any other parameter count.

**Names:** **Definitive 0.2 / Definitive 0.1** are release lines of [vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive), **not** official vLLM version numbers. This machine pins 0.2 at tag `v0.2.1-pre3` (engine is upstream vLLM 0.27.1) and 0.1 at `0.1.17` (do not deploy). `start-v02` and `native/vllm-2080ti:v0.2.1-pre3-sm75` point at that 0.2 line.

Sister repo: [FastLLM for 2x RTX 2080 Ti 22GB NVLink](https://github.com/mercury666cn/FastLLM-2x2080Ti-22GB-NVLink). Upstream source: [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) `v0.2.1-pre3`.

**Prefer Docker.** Do not compile FastLLM master on CUDA 12.1 / SM75. Only one stack can own both GPUs.

## 1. Conclusions (read this first)

Same box, same Docker, **both the FP8 and NVFP4 weights of Qwen3.8-27B were measured**. Daily serving moved to the FastLLM sister repo on 2026-09-14 (NVFP4 + FP16 KV + 240K + thinking off, **with vision**). This repo is still the **vLLM-2080Ti-Definitive 0.2 deploy note**.

**If you deploy Definitive 0.2: NVFP4 + FP8 KV + MTP3 + 240K text profile (`text-only`).** Long decode is still steadier than FastLLM, but this profile has no vision.

| Route | Result on this box |
|---|---|
| Official FP8 + 128K + FP16-KV | Empty start fails. KV wants ~4.64 GiB; only ~3.5–3.7 GiB left |
| Definitive 0.2 NVFP4 + FP8 KV + 240K **text** | `text-only`, no vision; 245600-token input ran; 200K Chinese decode still ~50 tok/s |
| FastLLM NVFP4 + FP16 KV + 240K (daily, sister repo) | Full weights, **with vision**; thinking-off 849-item 64.15%, NIAH 15/15; 235K OOMs |

Do not mix the Definitive 0.2 text profile with FastLLM daily. Do not shrink FP8 128K and call it “official 128K passed”.

Do not compare quality to the 9/12 48.14% / 49.15% numbers (thinking on, scorer reads `content` only). After 9/14 thinking-off, both FastLLM stacks tied and NIAH was 15/15. See [test_results/FP8FP8-vs-NVFP4FP16-20260914.md](test_results/FP8FP8-vs-NVFP4FP16-20260914.md).

On the fixed English ruler, Definitive 0.2 NVFP4 decode is slower than FP8 (lower MTP accept rate). The reason to pick NVFP4 here is **240K text context** and steadier long decode than FastLLM, not winning every short ruler.

**Do not deploy Definitive 0.1.** NVFP4 fails to load (`lm_head.weight_scale`); official 128K FP16-KV empty-starts fail; the advertised 100 tok/s is a stronger-single-core reference box. Same ruler here is about **69.6**. See [docs/ours/runtime-v0.1.md](docs/ours/runtime-v0.1.md).

## 2. Reference machine

| Item | Value |
|---|---|
| Board | RD450X |
| Memory | DDR4 2133 ECC, 16G × 8, 4 channels (~125 GiB) |
| CPU | 2× Xeon E5-2686 v4, 72 threads |
| GPU | 2× RTX 2080 Ti **22GB mod**, SM75, NVLink, **PCIe x4** |
| Concurrency | 1 (`max_batch=1` / `MAX_NUM_SEQS=1`) |
| Eval model | **Qwen3.8-27B** (FP8 and NVFP4 weights) |
| Example | LAN IP:8000, model root `$MODEL_ROOT` |

22G mod + x4 is not official 11G / x16. Capacity and bandwidth will differ on other boxes.

| Stack | Runtime | CUDA / Python | Local image |
|---|---|---|---|
| FastLLM | Official `ftllm[all]` 0.1.8.2 wheel | 12.1 / 3.10 | `native/fastllm:official-wheel` |
| Definitive 0.2 | fork `v0.2.1-pre3`, vLLM 0.27.1 | 13.0 / 3.12 | `native/vllm-2080ti:v0.2.1-pre3-sm75` |
| Definitive 0.1 (do not deploy) | fork 0.1.17, vLLM 0.1.15 | 12.8 / 3.11 | `native/vllm-2080ti:v0.1.17` |

## 3. Speed cross-bench (full tables)

Methods live in the result files. Older speed numbers are **thinking off**. Do not mix them with thinking-on runs.

### 3.1 Fixed English ruler (4096 in / 128 out, temperature=0, MTP3, median of 3)

| Runtime / model | Mode | Prefill tok/s | Decode tok/s | TTFT |
|---|---:|---:|---:|---:|
| Definitive 0.1 FP8 | MTP3 | **1680.57** | 48.41 | 2.437s |
| Definitive 0.1 NVFP4 | MTP3 | unsupported | unsupported | weight load failed |
| Definitive 0.2 FP8 | MTP3 | 1344.61 | 52.45 | 3.046s |
| Definitive 0.2 NVFP4 | MTP3 | 1207.49 | 34.56 | 3.392s |
| FastLLM FP8 | MTP3 | 1356.45 | 66.31 | 3.020s |
| FastLLM NVFP4 | MTP3 | **1535.18** | **74.91** | **2.668s** |
| FastLLM FP8 | DFlash7 | 1393.14 | 63.31 | 2.940s |

- Definitive 0.2 FP8 vs 0.1 FP8: prefill **-20.0%**, decode **+8.3%**
- FastLLM NVFP4 vs FastLLM FP8: prefill **+13.2%**, decode **+13.0%**
- FastLLM DFlash7 vs FastLLM FP8 MTP3: decode **-4.5%** (DFlash is not faster here)

### 3.2 Project English ruler (random bag, temperature=0.7, 4096)

| Runtime / model | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Definitive 0.1 FP8 MTP3 | 1692.79 | 61.81 |
| Definitive 0.2 FP8 MTP3 | 1359.79 | 76.19 |
| Definitive 0.2 NVFP4 MTP3 | 1212.01 | 40.94 |
| FastLLM FP8 MTP3 | 1337.49 | 74.37 |
| FastLLM NVFP4 MTP3 | 1537.38 | 78.79 |
| FastLLM FP8 DFlash7 | 1381.83 | 63.62 |

Definitive 0.2 MTP accept rate swings hard with the prompt: FP8 ~53.3%–98.9%, NVFP4 ~42.3%–84.4%. Do not crown a winner from one prompt.

### 3.3 Long-context live request (smoke, 32 tokens out)

| Stack | In / out | Result | TTFT | Prefill |
|---|---:|---|---:|---:|
| Definitive 0.1 FP8 official 128K | start | capacity short | - | - |
| Definitive 0.2 FP8 official 128K | start | capacity short | - | - |
| Definitive 0.2 NVFP4 MTP3 official 240K text | 245600 / 32 | **pass** | 378.316s | 649.19 |
| FastLLM FP8 MTP3 128K | 130900 / 32 | **pass** | 167.969s | 779.31 |
| FastLLM NVFP4 MTP3 128K | 130900 / 32 | **pass** | 157.263s | 832.36 |
| FastLLM FP8 DFlash7 128K | 130900 / 32 | **pass** | 171.383s | 763.79 |

### 3.4 Chinese long context (exact 64K / 128K / 200K, 128 out, thinking off)

| Runtime | Context | Prefill | Decode | TTFT | Total |
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

Long-input e2e: FastLLM NVFP4 total time is about 46.6% / 4.6% / 6.0% shorter than Definitive 0.2.  
Long output: Definitive 0.2 stays ~50 tok/s at 200K; FastLLM drops to ~28.  
The three DFlash rows are **FP8 main weights**, not NVFP4+DFlash.

### 3.5 Short-answer wall time (128 tokens out, 50K band)

| Input | FastLLM ZH | Definitive 0.2 ZH | FastLLM EN | Definitive 0.2 EN |
|---:|---:|---:|---:|---:|
| 0.5K | **2.4** | 2.7 | 6.0 (5K) | **2.2** |
| 50K | **25.0** | 26.8 | **48.7** | 48.8 |
| 100K | **58.7** | 63.0 | **113.4** | 122.9 |
| 150K | **99.0** | 108.5 | **195.7** | 214.1 |
| 180K | **125.9** | 137.3 | **255.6** | 277.8 |

From 50K up, FastLLM is about 7–9% quicker. Chinese is about 2× faster than English (tokenizer, not one engine). Definitive 0.2 wins on long decode, not short-answer wall time.

### 3.6 Advertised 100 tok/s

Same-ruler Definitive 0.1 here is about **69.6**, not 100. The reference box has a stronger single core. DFlash English short continue + draft 5/7 can hit 147 / 201; Chinese thinking stays 50–64. Daily Chinese chat will not become 180. See [test_results/对照-仓库100与官方尺子.md](test_results/对照-仓库100与官方尺子.md).

## 4. Quality cross-bench (FastLLM FP8 vs NVFP4, 849 items)

Same image, MTP3, FP8 KV, 210K, temperature=0, serial concurrency. No weight conversion, no compat patches.

### 4.1 Deduped (production gate passed)

| Category | FP8 | NVFP4 | Delta |
|---|---:|---:|---:|
| Chinese knowledge | 33.50% | **42.00%** | +8.50pp |
| Math | 69.50% | **70.50%** | +1.00pp |
| HumanEval | 65.24% | 65.24% | 0 |
| IFEval | 74.89% | 74.22% | -0.67pp |
| Long context | 45.72% | 42.94% | -2.79pp |
| Multi-turn JSON | 0.00% | 0.00% | 0 |
| **Equal-weight overall** | **48.14%** | **49.15%** | **+1.01pp** |

Ratio 102.1%. No API errors on 849 items. Long-context gap is mostly NIAH (FP8 12/15, NVFP4 8/15). Multi-turn is 0 on both: the model writes Chinese key `预算`, the item wants `budget`.

### 4.2 First pass (not deduped, gate failed; kept)

Overall 100.44%, but long context −6.78pp. NIAH first pass 7/15, rerun 4/15. “It fits” is not “retrieval is equally stable”. Raw files are in `test_results/`.

Full eval: CPU ~6.2% / 72 cores, GPU ~94%. **Do not swap the CPU for single-concurrency serving.**

Definitive 0.2 has **no** same-ruler 849 quality duel. Quality claims cover FastLLM FP8 / NVFP4 only.

### 4.3 2026-09-14 thinking-off duel (FastLLM daily kept B)

Same 849 items, thinking off. A: FP8 weights + FP8 KV. B: NVFP4 weights + FP16 KV. Overall 64.18% / 64.15%, NIAH 15/15. Do not compare to the 9/12 thinking-on 48%/49%. Report: [test_results/FP8FP8-vs-NVFP4FP16-20260914.md](test_results/FP8FP8-vs-NVFP4FP16-20260914.md).

## 5. Suggested deploy (Docker)

Image: `native/vllm-2080ti:v0.2.1-pre3-sm75`. Weights: `$MODEL_ROOT/Qwen3.8-27B-NVFP4`.

```text
--gpus all --network host --ipc host
mount: $MODEL_ROOT(ro) + $VLLM_CACHE + $TRITON_CACHE + $RUN_LOGS
ln -sfn /opt/venv /workspace/.venv
./launcher.sh --non-interactive \
  --set PROFILE=qwen27b/w4a16/normal/fp8kv-240K-mtp3-text-only.env \
  --set MODEL_DIR=$MODEL_ROOT/Qwen3.8-27B-NVFP4 \
  --set PORT=8000 \
  --set MAX_MODEL_LEN=210000 \
  --set SERVED_NAME=qwen38-v02-nvfp4-mtp3-210k \
  --set MAX_NUM_SEQS=1 \
  --set SERVICE_SCOPE=lan
then wait on the PID; do not exit after exec
```

Required: bake in `libxcb1`; mount both caches; `SERVICE_SCOPE=lan`; wait the PID after launcher smoke.  
Reference script: [scripts/start-v02.sh](scripts/start-v02.sh).

Do not send CUDA / weight blobs through a high-ratio proxy. Chatbox: API Host `http://<IP>:8000/v1`, **leave API Path empty**.

## 6. Layout

| Path | What |
|---|---|
| [README.md](README.md) | **This page (English)** |
| [README.zh-CN.md](README.zh-CN.md) | Simplified Chinese |
| [docs/ours/](docs/ours/) | Local notes (FastLLM / Definitive 0.2 / Definitive 0.1) |
| [docs/upstream/](docs/upstream/) | Unmodified upstream READMEs |
| [test_results/](test_results/) | Speed JSON/CSV, quality jsonl, failure logs, cross-bench md |
| [quality_eval/](quality_eval/) | Suite manifest and baseline (`suite.jsonl` is too large for git) |
| [scripts/](scripts/) | Mutual-exclusion start scripts |

Cross-bench markdown:

- [test_results/原生Docker三方案对照-20260912.md](test_results/原生Docker三方案对照-20260912.md)
- [test_results/NVFP4生产验收-20260912.md](test_results/NVFP4生产验收-20260912.md)
- [test_results/NVFP4整体能力验收-20260912.md](test_results/NVFP4整体能力验收-20260912.md)
- [test_results/NVFP4中文长上下文对照-20260912.md](test_results/NVFP4中文长上下文对照-20260912.md)
- [test_results/中英文总耗时对照-50k-20260912.md](test_results/中英文总耗时对照-50k-20260912.md)
- [test_results/对照-仓库100与官方尺子.md](test_results/对照-仓库100与官方尺子.md)
- [test_results/FP8FP8-vs-NVFP4FP16-20260914.md](test_results/FP8FP8-vs-NVFP4FP16-20260914.md)
- [test_results/并发1-2-4-1k30k-20260914.md](test_results/并发1-2-4-1k30k-20260914.md)

## 7. What we changed / did not change

No inference source edits, no weight conversion, no compat patches. We only fixed a Docker recipe that stays up: FastLLM uses the official wheel; Definitive 0.2 bakes `libxcb1`, dual caches, `SERVICE_SCOPE=lan`, and waits the PID after launcher smoke. See [docs/ours/](docs/ours/).
