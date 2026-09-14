# v0.2 运行时踩坑

对照环境：2× RTX 2080 Ti 22GB（SM75，NVLink），TP=2，并发=1，纯文本。  
镜像：`native/vllm-2080ti:v0.2.1-pre3-sm75`（约 42.7GB，CUDA 13.0，fork `v0.2.1-pre3`，运行时 vLLM 0.27.1）。  
切换：`start-v02`。日常模型名：`qwen38-v02-nvfp4-mtp3-210k`。端口 `:8000`，必须听 `0.0.0.0`。

路径一律用占位符：

- `$MODEL_ROOT`：只读模型根目录
- `$VLLM_CACHE`：宿主机上的 vLLM 编译缓存，挂到容器 `/root/.cache/vllm`
- `$TRITON_CACHE`：宿主机上的 Triton cubin 缓存，挂到容器 `/workspace/triton-cache`
- `$RUN_LOGS`：启动日志目录，挂到容器 `/workspace/run-logs`

两处缓存缺一不可。索引里写的是容器内绝对路径 `/workspace/triton-cache/*.cubin`，只挂 `$VLLM_CACHE`，重建容器就会报 `cubin not found`。

## 官方什么样：能跑 / 不能跑

官方这条线是 **v0.2.1-pre3 + CUDA 13 + PyTorch 2.13 + SM75**。权重走 `compressed-tensors`，KV 用 FP8 才能在双 22GB 上到超长上下文。

| 官方意图 | 结果 | 原因 |
|---|---|---|
| NVFP4 + `profiles/qwen27b/w4a16/normal/fp8kv-240K-mtp3-text-only.env` | 能跑 | 原生选 `MarlinNvFp4LinearKernel`。实测 245600 输入 / 32 输出，TTFT 约 378s，预填充约 649 tok/s，约 20 GiB/卡。不是空启动 |
| 同上，把 `MAX_MODEL_LEN` 收到 210000 | 能跑 | 日常脚本这样和 FastLLM 对齐 |
| FP8 + 官方 128K + FP16-KV | 空启动失败 | 要 4.64 GiB KV，实际约 3.48 GiB，上限大约 95200 tokens |
| 默认 `SERVICE_SCOPE=local` 给局域网测速页用 | 连不上 | 只绑 `127.0.0.1`。机器本机 curl 通，别的电脑会以为服务崩了 |
| `launcher.sh --non-interactive` 当 Docker PID 1 直接 `exec` | 容器会自己停 | 官方启动器是服务管理器：拉起 vLLM、冒烟、打印 `START OK` 然后退出 |
| 官方镜像缺 `libxcb.so.1` 直接起引擎 | EngineCore 起不来 | `opencv-python-headless` 导入失败。要先有 `libxcb1`，或烤进镜像 |
| 和 FastLLM 同卡双开 | 不能 | 两张卡只能跑一套 |

官方 launcher 还要求容器内 `/workspace/.venv` 指向 `/opt/venv`，否则找不到运行时。

## 我们修了什么

不转权重，不改 vLLM 推理源码。修的是**能稳定对外服务**的运行方式。

1. **烤进 `libxcb1`，以后不再现场 `apt`。**  
   第一次启动卡在缺 `libxcb.so.1`。后来每次容器里 `apt-get update` 还会把 universe 拉一遍，又慢又像“又在装环境”。  
   **修法：** 只从 jammy main 装 `libxcb1` 及其依赖（大约 500KB），`docker commit` 回同一标签。现用镜像里已经有库。

2. **双缓存持久化，修好重启必挂。**  
   只挂 vLLM 缓存时，AOT 索引指向容器里未持久化的 Triton cubin，重启约 120 秒启动保护会主动结束。  
   **修法：** `$VLLM_CACHE` 和 `$TRITON_CACHE` 一起挂。两处齐了之后原生重编译过一次，再实测容器重启，大约 80 秒恢复。

3. **冒烟成功后挂住进程。**  
   官方 `--non-interactive` 打印 `START OK` 就退出，Docker 认为主进程结束，容器 `exit 0`。从外面看就是“刚就绪就崩了”。  
   **修法：** launcher 返回后读 PID 文件，`while kill -0 "$pid"` 挂住；容器 `restart=unless-stopped`。

4. **改成局域网监听。**  
   默认 `Scope: local` / `Bind: 127.0.0.1:8000`。  
   **修法：** `--set SERVICE_SCOPE=lan`，听 `0.0.0.0:8000`。切换脚本写死这条。

5. **补上 venv 软链。**  
   官方镜像 Python 在 `/opt/venv`，launcher 认 `/workspace/.venv`。  
   **修法：** 启动前 `ln -sfn /opt/venv /workspace/.venv`。

6. **没有给官方 128K FP16-KV 开小灶。**  
   容量不够就记不够。要超长上下文走官方 NVFP4 + FP8-KV 的 240K profile，而不是把 FP8/FP16-KV 改短冒充。

7. **没有在启动脚本里再走一遍 apt。**  
   交叉测已经能跑，缺的小库应留在镜像里。现场装包会让人误以为环境没固化。

## 启动注意

切换脚本已经按上面写死。手搓时不要漏：

```text
--gpus all --network host --ipc host
挂载: $MODEL_ROOT(只读) + $VLLM_CACHE + $TRITON_CACHE + $RUN_LOGS
ln -sfn /opt/venv /workspace/.venv
./launcher.sh --non-interactive
  --set PROFILE=qwen27b/w4a16/normal/fp8kv-240K-mtp3-text-only.env
  --set MODEL_DIR=$MODEL_ROOT/Qwen3.8-27B-NVFP4
  --set PORT=8000
  --set MAX_MODEL_LEN=210000
  --set SERVED_NAME=qwen38-v02-nvfp4-mtp3-210k
  --set MAX_NUM_SEQS=1
  --set SERVICE_SCOPE=lan
然后 wait 住 PID，不要 exec 完就结束
```

- 不要给 CUDA / 权重走高倍率代理。
- 用 `start-v02`，它会先停 FastLLM 和评测残留容器。

## 实测还会踩的坑

- 启动后段可能连续打 `No available shared memory broadcast block found in 60 seconds`。随后 Health check / `/v1/models` 通了，当噪声。
- MTP 接受率随提示词暴涨暴跌。日志里 FP8 大约 53%–99%，NVFP4 大约 42%–84%。同一套 NVFP4，固定英文尺子 decode 可以只有 34，测速页随机英文又能到 50+。**不要拿一条提示词定终身。**
- 不能说“NVFP4 比 FP8 慢 10%”当常数。v0.2 上这条固定尺子：预填充慢约 10%，decode 慢约 34%；换 FastLLM，NVFP4 反而比 FP8 快约 13%。慢的是推测解码接受率，不是权重格式本身。
- 长中文（精确 64K/128K/200K 输入，出 128 token）：64K 预填充大约 564 tok/s，当时 FastLLM 大约 1077，总耗时差近一倍。200K decode 仍能到约 50 tok/s，FastLLM 会掉到约 28。长输入等第一口，v0.2 不占优；长输出更稳。
- 测速页默认英文词袋 + 英文写作后缀，不能代表中文生产。要看中文，用页面的「中文测试」。
- 评测容器、旧候选容器会抢 GPU，看起来像生产崩了。用切换脚本，不要手搓漏停。

## 什么时候用它

看长对话 / 长输出是否掉速，或和 FastLLM 做同端口切换对照。大段中文贴进去要比预填充，先看 FastLLM 那份文档。
