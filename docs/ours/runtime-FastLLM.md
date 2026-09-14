# FastLLM 运行时踩坑

对照环境：2× RTX 2080 Ti 22GB（SM75，NVLink），TP=2，并发=1。  
镜像：`native/fastllm:official-wheel`（约 16.5GB，官方 `ftllm[all]` 0.1.8.2，CUDA 12.1，Python 3.10）。  
切换：`start-fastllm`。日常模型名：`qwen38-fastllm-nvfp4-mtp3-cn-210k`（**NVFP4 + FP16 KV + 240K + 关思考，带图像**）。端口 `:8000`。不要写成「纯文本档」——那是 Definitive 0.2 的 `text-only` profile。

路径一律用占位符：

- `$MODEL_ROOT`：只读模型根目录
- NVFP4：`$MODEL_ROOT/Qwen3.8-27B-NVFP4`
- FP8：`$MODEL_ROOT/Qwen3.8-27B-FP8`

这套没有 Definitive 0.2 那种 cubin 双缓存。启动快，热重启大约几十秒。

## 官方什么样：能跑 / 不能跑

官方能用的是 **发布 wheel**，不是这台机器上现场编 master。

| 官方意图 | 结果 | 原因 |
|---|---|---|
| 官方 wheel 加载 NVFP4 / FP8，MTP3，FP16 KV，240K | 能跑 | 只读目录原生加载；FP16 KV 实打约 230K 稳，235K 会 OOM |
| 官方 wheel + DFlash7（FP8 主模型 + 官方 draft，block size 8） | 能跑，但本机不更快 | 固定英文 4K 尺子上，DFlash7 decode 比 FP8 MTP3 慢约 4.5% |
| 官方 master 源码在 CUDA 12.1 / SM75 现场编译 | 编不过 | 必须用官方 wheel 镜像。本机修过的源码提交不能拿来当“官方对照” |
| 官方 DFlash 当 NVFP4 对照 | 不能这么比 | 实测 DFlash 用的是 FP8 主模型，比的是服务速度，不能把差值全算到量化头上 |
| 和 Definitive 0.2 同卡双开 | 不能 | 两张卡只能跑一套 |

长中文实请求（精确 64K / 128K / 200K 输入，出 128 token，关思考）：NVFP4 + MTP3 三档预填充和总耗时都优于当时的 Definitive 0.2。200K decode 会掉到约 27.6 tok/s。

## 我们修了什么

不改 FastLLM 推理源码，不用本机修复提交，不转权重。

1. **放弃源码编译，固化官方 wheel 镜像。**  
   官方 master 在 CUDA 12.1 / SM75 编不过。有人会想用本机修过的提交硬编。  
   **修法：** 只保留 `native/fastllm:official-wheel`。对照口径是官方 wheel，不是私有补丁。

2. **把生产参数冻死，避免各测各的。**  
   公平对照固定为：

   ```text
   --device cuda --tp 0,1 --max_batch 1 --gpu_mem_ratio 0.95
   --tokens 240000 --max_context_length 240000
   --kv_cache_dtype float16 --prefix_cache false --mtp 3
   --enable_thinking false
   --host 127.0.0.1 --port 8001
   ```

   对外 `:8000` 走兼容层。改其中任何一项，就不要和旧的 4K / 128K / 200K / 质量门禁数字并排。

3. **200K 冒烟按错 token 比会直接打爆上下文。**  
   按 0.62 tok/字去拼 200K，实际会到二十多万 token，接口报 `prompt too long`。  
   **修法：** 先用短请求看接口返回的真实 `prompt_tokens` / 字数，再按实测比拼长文。本机一次成功冒烟大约 199855 prompt tokens。

4. **质量评分“先写算数”会误判长上下文。**  
   两路评测同时写同一份结果时，同一 `case_id` 出现重复行。如果按先写的算，长上下文会假失败。  
   **修法：** 评分改为同一题 **后写覆盖先写**。门禁数字以去重后的为准。

5. **IFEval 缺 NLTK 数据时不要卡死。**  
   官方 IFEval 评分要 `punkt_tab`，本机下载常被拦。  
   **修法：** 评分脚本在 NLTK 数据缺失时回退到 `PunktSentenceTokenizer` / `TreebankWordTokenizer`，题还能判。

6. **C-Eval 科目名按数据集实情改。**  
   题库里没有 `college_mathematics`，硬拉会失败。  
   **修法：** 改用实际存在的 `advanced_mathematics`。

7. **评测容器会把生产“看起来像崩了”。**  
   残留的质量引擎会再拉一套 FP8 占卡，甚至 `docker rm -f` 掉生产容器。内核旧 Xid 不是这次的崩。  
   **修法：** 切换脚本启动前清掉评测容器和旧候选容器。评测控制脚本不再当日常入口。

8. **互斥切换，避免两套抢卡。**  
   `start-fastllm` / `start-v02` 同一时刻只留一套，都绑 `:8000`。测速页只改模型名。

没有做的事也写清楚：没有给 FastLLM 打 SM75 兼容补丁，没有把 DFlash 设成默认生产（本机 MTP3 更快），没有和 Definitive 0.2 做同尺质量赛。

## 启动注意

- 单请求服务，并发先留 1。`max_batch=1` 下多人是排队，不是批处理。
- `docker restart` 大约 46 秒能回来，适合热重启。
- 用 `start-fastllm`，不要和 Definitive 0.2 手搓双开。
- **生产默认关思考。** Agent / 日常用 `qwen38-fastllm-nvfp4-mtp3-cn-210k`。Chatbox 要看推理，选 `qwen3.8-27b-think-low` / `medium` / `xhigh`（兼容层只改参数，不重载模型）。思考档最大输出拉到 8192+；`xhigh` 容易把额度写光、不出正文。旧的 4K / 128K / 200K 数字是关思考测的，不要和开思考混排。
- 不要给 CUDA / 权重走高倍率代理。

## 实测还会踩的坑

- **长输出会掉。** 200K 中文 decode 约 27–28 tok/s；测速页 180K 英文也掉到约 28。这是长上下文输出短板，不是没起来。
- **长输入端到端它更快。** 相对 Definitive 0.2，64K / 128K / 200K 总耗时大约少 47% / 5% / 6%。64K 等第一口，差距最明显。
- **DFlash 不是银弹。** 仓库推荐 draft=7。本机固定英文尺子上 MTP3 NVFP4 仍然更快。DFlash 三档用的是 FP8 主模型，不要写成“NVFP4+DFlash”。
- **质量：9/12 开思考 48%/49% 不要和 9/14 关思考 64% 横比。** 打分只读 `content`。9/14 当场对打（关思考）：FP8 权+FP8 KV 64.18% vs NVFP4 权+FP16 KV 64.15%，NIAH 两边 15/15。9/12 NIAH 8/15 不能证明 NVFP4 权变笨。详见 `test_results/FP8FP8-vs-NVFP4FP16-20260914.md`。  
  多轮 JSON 两边都是 0：模型爱写中文键 `预算`，题面要 `budget`。这是格式，不是量化把模型变笨。
- **中文知识表面分不要过度解读。** NVFP4 中文知识 +8.5pp，更像发挥波动，不能证明“量化变聪明”。可靠结论是没有系统性变笨。
- **单并发别换 CPU。** 全量评测时整机 CPU 大约 6% / 72 核，GPU 大约 94%。干活的是 GPU 和内核。
- **测速页默认英文词袋不是中文生产口径。** 要测中文，用页面的「中文测试」。

## 什么时候用它

大段中文贴进去、等第一口出来，优先这套。后面还要持续吐得快，切到 Definitive 0.2 对照。质量结论目前只对 FastLLM 自己的 FP8 / NVFP4 负责。
