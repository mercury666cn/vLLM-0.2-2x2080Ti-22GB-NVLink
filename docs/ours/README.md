# 运行时踩坑（可转发）

对照机器：2× RTX 2080 Ti 22GB（SM75，NVLink）。建议 Docker。两套互斥，一次只跑一套。  
FP8 和 NVFP4 都测过，**建议 NVFP4**，纯文本上下文可以开到 240K。

文中 **Definitive 0.2 / 0.1** = [vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) 的发布线，不是官方 vLLM 版本号。

公开部署说明（`mercury666cn`）：

- [FastLLM for 2x RTX 2080 Ti 22GB NVLink](https://github.com/mercury666cn/FastLLM-2x2080Ti-22GB-NVLink)
- [vLLM-2080Ti-Definitive 0.2 for 2x RTX 2080 Ti 22GB NVLink](https://github.com/mercury666cn/vLLM-2080Ti-Definitive-0.2-2x2080Ti-22GB-NVLink)

本仓细目：

- [FastLLM](runtime-FastLLM.md)：中文长输入首选；官方 wheel；NVFP4 + MTP3 + 210K
- [Definitive 0.2](runtime-v0.2.md)：长输出更稳；官方 pre3 + FP8 KV；启动坑最多。Definitive 0.1 不好用（NVFP4 加载失败、主频追不上官方 100），见 Definitive 0.2 仓说明，不再单独维护
- [Definitive 0.1](runtime-v0.1.md)：历史对照，不要部署

文里的 `$MODEL_ROOT` / `$VLLM_CACHE` / `$TRITON_CACHE` 是占位符，换成你自己的目录即可。
