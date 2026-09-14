# vLLM-2080Ti-Definitive 0.1 运行时踩坑

这是 [vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) 的 0.1 线，不是官方 vLLM 的 0.1。本机钉 `0.1.17`，**不要部署**。

对照环境：2× RTX 2080 Ti 22GB（SM75，NVLink），TP=2，并发=1，纯文本。  
镜像：`native/vllm-2080ti:v0.1.17`（约 30GB，CUDA 12.8，fork 0.1.17 / 运行时 vLLM 0.1.15）。  
同层别名：`native/vllm-2080ti:v0.1.17-cu128-fixed`。

路径一律用占位符，方便转发：

- `$MODEL_ROOT`：只读模型根目录
- FP8 权重：`$MODEL_ROOT/Qwen3.8-27B-FP8`
- NVFP4 权重：`$MODEL_ROOT/Qwen3.8-27B-NVFP4`（这套加载不了）

日常切换脚本**不会**拉 Definitive 0.1。镜像留着只为对照。

## 官方什么样：能跑 / 不能跑

官方这条线是 **Definitive 0.1 + CUDA 12.8 + FP8**。在这套双 22GB 上实测：

| 官方意图 | 结果 | 原因 |
|---|---|---|
| FP8 + MTP3，短上下文（4K 级） | 能跑 | 原生加载 FP8，短英文预填充是三套里最快的一档（约 1680 tok/s） |
| FP8 + MTP3，官方 128K + FP16-KV | 空启动失败 | 启动估算要 4.64 GiB KV，实际只剩约 3.72 GiB，上限大约 102400 tokens。容量不够，不是崩溃 |
| NVFP4 / `compressed-tensors` | 加载失败 | 已进入 Qwen3.5 权重加载，随后报 `There is no module or parameter named 'lm_head.weight_scale'` |
| 和 Definitive 0.2 / FastLLM 同卡双开 | 不能 | 两张卡只能跑一套 |

同一只读 NVFP4 目录：Definitive 0.2 会选 `MarlinNvFp4LinearKernel`，FastLLM 也能直接加载。所以不是权重坏了，也不是 Docker 问题，是 **Definitive 0.1 的 NVFP4 加载实现不完整**。

## 我们修了什么

只修构建，不改推理源码，不转权重，不打兼容补丁。

1. **作废错误 CUDA 的 torch 层，整镜重建。**  
   第一版用 `uv 0.5.31`，它还不认 `UV_TORCH_BACKEND`，结果装成 `torch 2.11.0+cu130`。镜像看起来是 CUDA 12.8 路线，实际 torch 对不上。  
   **修法：** 从干净 tag 用 `uv 0.12.10` 重建。现用层实测 `torch 2.11.0+cu128`，`torch.version.cuda=12.8`。旧层不要再用。  
   NVFP4 失败栈、128K 容量结论，都是用重建后的镜像复测的。

2. **没有给 Definitive 0.1 补 `lm_head.weight_scale`。**  
   那是官方 Definitive 0.1 缺的能力。补一行加载逻辑就算兼容补丁，本轮明确不做。要 NVFP4 就换 Definitive 0.2 或 FastLLM。

3. **没有把 128K profile 改短再宣称通过。**  
   官方 128K + FP16-KV 过不了，就记“容量不足”。缩短 `max_model_len` 冒充 128K 不算官方路线通过。

## 启动注意

- 只用重建后的 `v0.1.17` / `v0.1.17-cu128-fixed`。看到 cu130 的 torch 就是旧层。
- 不要给 CUDA / 模型下载走高倍率代理。小工具包可以走本机代理，权重大包不要。
- 要测 Definitive 0.1，先停掉 FastLLM 和 Definitive 0.2。

## 实测还会踩的坑

- Docker 不是低速根因。Definitive 0.1、Definitive 0.2、FastLLM 都在同一套 Docker / GPU / 只读模型挂载下测过，只有 Definitive 0.1 认不出 NVFP4。速度差来自运行时版本、CUDA 和内核，不是“装进容器就变慢”。
- 短英文 FP8 预填充快，不代表它能上 NVFP4 生产，也不代表 128K 官方 profile 能起。
- 项目测速页默认是随机英文词袋，和官方固定 chat 尺子不是同一条。Definitive 0.1 只适合当 FP8 短英文历史对照。

## 什么时候用它

确认“NVFP4 失败是 Definitive 0.1 路径问题”，或和 Definitive 0.2 FP8 做短英文对照。生产、中文长上下文、NVFP4 210K 都不要用这套。
