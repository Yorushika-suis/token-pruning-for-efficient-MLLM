# CDPruner 复现：环境、图像 VQA 与 VideoMME

本文对应：

- 代码：[`ThirdParty/CDPruner`](../../ThirdParty/CDPruner)
- 论文：<https://arxiv.org/abs/2506.10967>
- 本文审计的代码 commit：`9541616c40fcd5625de1cdb8ea6c33c129eb7864`

CDPruner 是 training-free 的视觉 token 剪枝。它用视觉编码器的文本塔把当前问题编成条件，在「问题相关度 × token 相似度」的核矩阵上做 Fast MAP DPP，保留固定个数的视觉 token。实现在 [`llava/model/llava_arch.py`](../../ThirdParty/CDPruner/llava/model/llava_arch.py) 的 `encode_images`（约 140–186 行）。CLIP 文本塔在 [`clip_encoder.py`](../../ThirdParty/CDPruner/llava/model/multimodal_encoder/clip_encoder.py) 的 `load_text_tower`。

论文主表是 **LLaVA-1.5-7B 的图像 benchmark**（576 个视觉 token 里留 128 / 64 / 32）。论文没有 VideoMME。下面第 3 节是把同一套 DPP 接到本仓库已有的 LLaVA-Video × VideoMME 评测上，分数不能写成论文复现。

可执行入口：

- 图像：[`scripts/cdpruner/run_llava15.sh`](../../scripts/cdpruner/run_llava15.sh)
- 视频：[`scripts/cdpruner/run_videomme.sh`](../../scripts/cdpruner/run_videomme.sh)

下文 `$ROOT` 是仓库根。不改 `ThirdParty/` 里的文件。

## 1. 环境

README 要求 Python 3.10，`pip install -e .` 会装 `torch==2.1.2`、`torchvision==0.16.2`、`transformers==4.37.2`。本机 GPU 是 B200（compute capability 10.0）。`torch==2.1.2+cu121` 没有 `sm_100` 内核，模型一上 GPU 就会报 `no kernel image is available`。与 TransPrune 相同，评测环境改用 `torch==2.14.0+cu130`、`torchvision==0.29.0+cu130`，`transformers` 保持 `4.37.2`。这是硬件要求，不是论文设置。`pip check` 对 `torch==2.1.2` 的告警可以忽略。FlashAttention 和 `deepspeed` 不是这条 training-free 评测所需要的。

```bash
export ROOT=/data0/lyx/token-pruning-for-efficient-MLLM
export CD="$ROOT/ThirdParty/CDPruner"
source /opt/miniconda3/etc/profile.d/conda.sh
conda create -n cdpruner python=3.10 -y
conda activate cdpruner
python -m pip install --upgrade pip
cd "$CD"
python -m pip install -e .
python -m pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
```

安装后做环境门禁：

```bash
cd "$CD"
python - <<'PY'
import torch, transformers, torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
python -m compileall -q llava
git rev-parse HEAD   # 9541616c40fcd5625de1cdb8ea6c33c129eb7864
```

权重和离线缓存与仓库其他评测一致：

```bash
export HF_HOME="$ROOT/weights"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
```

LLaVA-1.5 必须用旧版 `llava_llama` checkpoint（目录里有 `mm_projector.bin`）。加载器只看路径最后一级是否包含 `llava`。本机可用 snapshot 是 `haotian-from-hf`，不要把 Transformers 格式的 `6f798d...` 直接传给 `--model-path`。

```bash
SNAP="$ROOT/weights/hub/models--liuhaotian--llava-v1.5-7b/snapshots/haotian-from-hf"
test -s "$SNAP/config.json"
test -s "$SNAP/mm_projector.bin"
ln -sfn "$SNAP" "$ROOT/weights/llava-v1.5-7b"
```

视觉塔和 CDPruner 的文本塔都是 `openai/clip-vit-large-patch14-336`，本机缓存在 `weights/hub/models--openai--clip-vit-large-patch14-336`。`builder.py` 在加载视觉塔之后会调用 `load_text_tower`。

VideoMME 不使用这个 `cdpruner` 环境。它走已经装好 LLaVA-Video 和 `lmms-eval` 的 `fastvid` 环境，见第 3 节。

## 2. 在 VQA 上复现

论文表 1 在 LLaVA-1.5-7B 上比较保留 token 数。这些数用来核对协议，不是本机已经跑出的结果。

| 设置 | VQAv2 | GQA | POPE | MME | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| 全部 576 token | 78.5 | 61.9 | 85.9 | 1506.5 | 未剪枝上界 |
| 保留 128（↓77.8%） | 76.6 | 59.9 | 87.7 | 1431.4 | 论文主对比行 |
| 保留 64（↓88.9%） | 75.4 | 58.6 | 87.5 | 1415.1 | README 示例之一 |
| 保留 32（↓94.4%） | 73.6 | 57.0 | — | — | 表中 CDPruner 行被截断处仍有其余列 |

脚本默认 `TOKEN=128`。换预算：`TOKEN=64 bash scripts/cdpruner/run_llava15.sh gqa`。

### 2.1 本机有数据、脚本会打分的三项

评测根目录是 `ThirdParty/LLaVA/playground/data/eval`，不是 CDPruner 仓库里的 `playground/`（那边是空的）。

| Benchmark | 问题文件 | 图像 | 评分 |
| --- | --- | --- | --- |
| GQA | `gqa/llava_gqa_testdev_balanced.jsonl`（12,578） | `gqa/data/images` | [`scripts/repro/score_visionzip.py`](../../scripts/repro/score_visionzip.py) 的 Accuracy |
| POPE | `pope/llava_pope_test.jsonl` | `pope/val2014` | 同上，对 `pope/coco/coco_pope_{random,popular,adversarial}.json` |
| MME | `MME/llava_mme.jsonl` | `MME/MME_Benchmark_release_version` | 同上，Perception / Cognition |

官方 `scripts/v1_5/eval/gqa.sh` 最后调用 GQA 的 `eval/eval.py`。该脚本在出 Accuracy 之前就要 scene graph 和 choices，当前目录没有这些文件，会在出分前退出。对齐论文 GQA 列只需要小写后的 exact match，所以脚本把答案写成 `answers/gqa/merge.jsonl`，再交给 `score_visionzip.py`。POPE、MME 同样走这个评分器，不调用 CDPruner 自带 shell 里的绝对路径。

推理参数与发布脚本一致：`temperature=0`、`conv-mode=vicuna_v1`、`--visual_token_num`。`model_vqa_loader.py` 会去掉 “Answer the question using a single word or phrase.” 再把剩余问题文本传给 `generate(..., texts=question)`，DPP 的条件就是这句问题。

一进程一张卡。`load_pretrained_model` 使用 `device_map`，多卡可见时 7B 会被切开。脚本按 `CUDA_VISIBLE_DEVICES` 切 chunk，每张卡一个进程。

```bash
# 默认 8 卡可见时会按卡数切分；单卡：
CUDA_VISIBLE_DEVICES=0 bash scripts/cdpruner/run_llava15.sh gqa
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/cdpruner/run_llava15.sh pope
TOKEN=64 bash scripts/cdpruner/run_llava15.sh mme
bash scripts/cdpruner/run_llava15.sh all
```

答案和分数：

```text
runs/cdpruner-v1.5/<task>/vtn_<N>/
├── command.txt          # commit、TOKEN、GPU
├── answers/<task>/merge.jsonl
└── metrics/<task>.json
```

### 2.2 论文表里其余数据集

CDPruner 的 `scripts/v1_5/eval/` 还有 VQAv2、TextVQA、MMBench、MMBench-CN、ScienceQA、VizWiz、MM-Vet，以及 `scripts/v1_6/eval/` 的 LLaVA-NeXT。本机没有这些数据，`run_llava15.sh` 不包含它们。文件名以官方 [`EVAL.md`](../../ThirdParty/CDPruner/EVAL.md) 为准；VQAv2 / MMBench / MM-Vet 的论文分数来自评测服务器，本地只能得到提交文件。LLaVA-1.5-13B 和 LLaVA-NeXT 的权重本机也没有，不要用 7B 的分数代替那些行。

## 3. 迁移到 VideoMME

宿主与 FastVID 全量评测相同：`fastvid` 环境、LLaVA-Video-7B-Qwen2、`lmms-eval --model llava_vid --tasks videomme`、64 帧、`mm_spatial_pool_mode=average`。权重和 VideoMME parquet 已在 `weights/`。入口是 `scripts/cdpruner/run_videomme.sh`。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/cdpruner/run_videomme.sh
# 保留比例，默认 0.10，与 FastVID 的 retention 对照
CDPRUNER_RETENTION_RATIO=0.25 bash scripts/cdpruner/run_videomme.sh
# 绝对条数优先于比例（只计空间 token，换行 token 另计）
CDPRUNER_VISUAL_TOKEN_NUM=2048 bash scripts/cdpruner/run_videomme.sh
```

日志在 `runs/cdpruner/full_videomme/full_launch.log`。跑完的标志与 FastVID 脚本相同：日志里有 `videomme_perception_score`，且没有 `Traceback`。

### 3.1 剪枝插在哪里

LLaVA-Video 的 checkpoint 把 `mm_newline_position` 设为 `grid`：空间池化之后每个 13×13 块末尾有一个换行 token，FastVID 用 `(index + 1) % 14 == 0` 认出它们。CDPruner 不改这个布局。钩子在换行 token 已经插入之后，只在空间 token 上做 DPP，换行 token 全部保留。

相似度用 mm projector 之后的特征（与图像实现一致）。图像实现的相关度来自 CLIP 文本塔；LLaVA-Video 的视觉编码器是 SigLIP，没有 CLIP 文本空间。钩子改为加载同一份 `google/siglip-so400m-patch14-384`，用它的文本塔编码当前问题，并用 `visual_projection` 把池化后的 SigLIP patch 特征投到同一空间。问题文本来自 `llava_vid.generate_until` 里的 `contexts`（VideoMME 的题目和选项），不是前面拼上的时间说明。

`mm_newline_position` 为 `one_token` 或 `no_token` 时，钩子改为在 `get_2dPool` 的展平序列上剪，因为后面不再按 14 的周期插入换行。本仓库的 VideoMME 命令不会改 checkpoint 的 `grid`。

FastVID 在 [`modeling_qwen2.py`](../../ThirdParty/FastVID/fastvid_llavavideo/LLaVA-NeXT/llava/model/language_model/modeling_qwen2.py) 的 `if seq_length > 1` 里还会再剪一次，并且把序列长度写成 `64 * 13 * 13`。CDPruner 先剪过之后这个假设不成立。`CDPRUNER=1` 时钩子给这次前向加上 `cdpruner_skip_fastvid`，跳过该块，避免剪两次。这是运行时包一层，不改 ThirdParty 文件。

### 3.2 和论文图像设置的差别

| 项目 | 图像复现 | VideoMME |
| --- | --- | --- |
| 模型 | LLaVA-1.5-7B | LLaVA-Video-7B-Qwen2 |
| 环境 | `cdpruner` | `fastvid` |
| 条件编码器 | CLIP 文本塔 | SigLIP 文本塔 |
| 预算 | 绝对个数，默认 128 / 576 | 空间 token 的比例，默认 0.10；或 `CDPRUNER_VISUAL_TOKEN_NUM` |
| 分数含义 | 对齐论文 GQA / POPE / MME | 本仓库扩展，无论文对照值 |

钩子代码在 [`scripts/cdpruner/dpp.py`](../../scripts/cdpruner/dpp.py) 和 [`scripts/cdpruner/hook_videomme.py`](../../scripts/cdpruner/hook_videomme.py)。`run_videomme.sh` 把 `scripts/cdpruner` 放在 `PYTHONPATH` 最前，由该目录的 `sitecustomize.py` 先加载 `scripts/repro` 的 VideoMME 本地 parquet 补丁，再安装 CDPruner 钩子。
