# sparse_atten_spec

基于 vLLM 的稀疏注意力自推测解码移植，包含 StreamingLLM 和 Vegas。两者使用同一个模型生成草稿，以完整 KV 验证；稀疏化只作用于草稿 attention，不删除完整 KV。

- `streamingllm`：草稿使用 sink 和 recent window。
- `vegas`：验证阶段收集注意力重要性，下一轮草稿使用每层选出的历史 KV 和 recent tokens。

这是固定版本的源码覆盖层和补丁，不是完整 vLLM，也不能在本仓库直接执行 `pip install -e .`。它是移植实现，不是作者原仓库的原样镜像。

## 文件范围

- `overlay/vllm/v1/spec_decode/sparse_attn/`：共享 proposer、两种算法、归约与 top-k 内核。
- `overlay/tools/`：CPU 接入检查、GPU 内核检查、采样检查及模型验证入口。
- `patches/vllm-integration.patch`：配置、模型 runner、attention backend 和 FA Python 接口接入。
- `flash-attention/hopper/`：修改过的六个 FA3 源文件，便于阅读。
- `patches/flash-attention.patch`：这些 FA3 文件相对指定基础版本的补丁。
- `upstream.json`：基础代码与参考算法的固定提交。

不包含模型权重、数据集、日志、虚拟环境、编译产物和访问凭据。

## 准备源码

在 Linux CUDA 环境运行。先克隆本仓库，再准备匹配版本的 vLLM。以下命令创建新的源码目录，不覆盖现有 SSH 项目。

```sh
git clone https://github.com/Ethan-el/sparse_atten_spec.git
git clone https://github.com/vllm-project/vllm.git vllm-sparse
git -C vllm-sparse checkout --detach 25ac790b81f18daa149197e2b46e37428e75bcf6
git -C vllm-sparse apply --check ../sparse_atten_spec/patches/vllm-integration.patch
git -C vllm-sparse apply ../sparse_atten_spec/patches/vllm-integration.patch
cp -R sparse_atten_spec/overlay/vllm/. vllm-sparse/vllm/
cp -R sparse_atten_spec/overlay/tools/. vllm-sparse/tools/
```

随后按该版本 vLLM 的环境说明使用 `uv` 配置 `.venv`。已有匹配的 vLLM 二进制时可优先使用预编译 editable 安装方式，避免重编整个工程。预编译产物必须与源码、Torch 和 CUDA 匹配；这些源码准备命令不包含环境安装，也不保证任意 PyPI wheel 可复用。

## StreamingLLM 与 Vegas 启动

在已安装覆盖层的 `vllm-sparse` 环境中运行以下 Python 示例。切换 `algorithm` 即可使用另一种算法；模型可改为本地路径。

```python
from vllm import LLM, SamplingParams

algorithm = "streamingllm"  # Vegas 需要先完成下方 FA3 构建
llm = LLM(
    model="Qwen/Qwen3-8B",
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=1,
    enforce_eager=True,
    attention_config={"backend": "FLASH_ATTN"},
    speculative_config={
        "method": "sparse_attn",
        "sparse_attn_algorithm": algorithm,
        "num_speculative_tokens": 6,
        "sparse_attn_ratio": 0.07,
        "sparse_attn_min_tokens": 256,
    },
)
outputs = llm.generate(
    ["Explain how sparse attention helps speculative decoding."],
    SamplingParams(temperature=0, max_tokens=128),
)
print(outputs[0].outputs[0].text)
```

## Vegas 的 FA3 依赖

当前 Vegas 移植要求 SM90 Hopper、BF16 及带 Vegas 扩展接口的 FA3。普通 FA3 wheel 不提供这里的 scores 和 sparse slots 参数；不能仅修改算法配置就直接运行。RTX 4090 可验证独立选择内核，不能验证该 FA3 路径。

```sh
git clone https://github.com/vllm-project/flash-attention.git flash-attention-vegas-build
git -C flash-attention-vegas-build checkout --detach 9cd61de38763d712bb6ce56e2a02cc2bf718c89f
git -C flash-attention-vegas-build submodule update --init --recursive
git -C flash-attention-vegas-build apply --check ../sparse_atten_spec/patches/flash-attention.patch
git -C flash-attention-vegas-build apply ../sparse_atten_spec/patches/flash-attention.patch
```

必须获取该版本的 CUTLASS 子模块。使用 vLLM 的 `VLLM_FLASH_ATTN_SRC_DIR` 指向这个完整目录，按照固定版本中的 `docs/contributing/incremental_build.md` 构建和安装 `_vllm_fa3_C`，替换前备份原共享库。不要用外部 FA 仓库的 Python wrapper 覆盖 vLLM 内置 wrapper。

这里只提供源码准备流程，未提供已验证的 Vegas wheel。Vegas FA3 尚未编译或进行 Hopper 端到端验证，不能视为已验证可运行版本。

## 验证入口

在已安装的 `vllm-sparse` 目录中按需要运行，GPU 命令会占用显卡。

```sh
.venv/bin/python tools/check_streamingllm_cpu.py
.venv/bin/python tools/check_vegas_kernels_gpu.py
.venv/bin/python tools/check_streamingllm_sampling_gpu.py
.venv/bin/python tools/validate_streamingllm_gpu.py --model /path/to/model --algorithm streamingllm
```

已有 RTX 4090 检查覆盖 StreamingLLM 模型回归、Vegas 独立归约/top-k、状态、slot 映射及图回放逻辑。Vegas 的真实 FA3 attention、完整模型验证和性能对比仍需在兼容环境中完成。默认 BF16 路径存在数值差异，不能把所有逐 token 比较都视为已通过。

## 来源与许可证

主许可证为 Apache-2.0；vLLM 和 Vegas 衍生代码保留原版权声明。FlashAttention 文件保留 BSD-3-Clause 许可，详见 `licenses/FLASH_ATTENTION_LICENSE`。来源和移植修改见 `NOTICE` 与 `upstream.json`。本项目包含 AI 辅助移植，不代表任何上游项目背书。
