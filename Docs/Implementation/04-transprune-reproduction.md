# TransPrune 复现：数据、环境与运行命令

本文对应：

- 代码：`[ThirdParty/TransPrune](../../ThirdParty/TransPrune)`
- 论文：`[ThirdParty/papers/TransPrune.pdf](../../ThirdParty/papers/TransPrune.pdf)`
- 本文审计的代码 commit：`c9e8284f5808f7692c40590ffc99537e9220a01d`

先给出边界：当前 checkout 可以直接复现 **LLaVA-v1.5-7B 的标准图像 benchmark 评测**。论文中的 LLaVA-NeXT、Qwen2.5-VL、Video-LLaVA、VisionZip/CDPruner 组合实验没有在这个仓库中提供相应的完整适配和命令，不能把下面的 LLaVA-v1.5 结果冒充为这些实验的严格复现。

可执行入口是 [`scripts/transprune/run_llava15.sh`](../../scripts/transprune/run_llava15.sh)。它会建立下面要求的模型目录名、按「一进程一张卡」切分，并在 GQA 上调用本地 Accuracy 评分。下面的命令与该脚本一致。

## 1. 论文实验与复现目标

论文的标准图像实验使用以下数据集：MME、MMBench、SEED-Bench、ScienceQA、VQAv2、POPE、GQA 和 TextVQA。论文还报告 TGIF/MSVD 上的 Video-LLaVA，以及 LLaVA-NeXT-7B 和 Qwen2.5-VL-7B 的结果。

TransPrune 的核心设置为：


| 参数                              | 论文/仓库设置                                            |
| --------------------------------- | -------------------------------------------------------- |
| 模型                              | LLaVA-v1.5-7B                                            |
| accumulation layers               | `[7,8,9,10,11,12]`                                       |
| pruning layers                    | `[7,9,12]`                                               |
| `image_token_ratio_list`          | `[0.875,0.625,0.125]`                                    |
| 实际比例（代码会在前面插入`1.0`） | `1.0 -> 0.875 -> 0.625 -> 0.125`                         |
| `alpha`                           | 代码中 TTV 与 IGA 直接相加，对应论文的`alpha=0.5`        |
| decoding                          | `temperature=0`、`conv-mode=vicuna_v1`、`use_cache=True` |
| GPU                               | 论文：A100 40 GB；代码要求 CUDA 和 FP16                  |

LLaVA-v1.5 的 576 个视觉 token 在三个剪枝层后约为 `504 -> 360 -> 72` 个。论文表 1 将两种预算称为 `TransPrune-High` 和 `TransPrune-Low`，但论文 PDF 和当前公开脚本没有给出这两个名称与比例列表的逐项映射；当前脚本明确提供的就是上面的比例列表。因此先以该列表复现 released-script 结果，再报告实际 token 数、TFLOPs 和指标，不要仅凭名称声称已经对齐 High/Low。

### 1.1 论文表 1 的 LLaVA-v1.5-7B 对照值

这些数值用于检查协议是否一致，不是本机已经跑出的结果：


| 方法            | TFLOPs | MMEP | VQAv2 | SEED-I | TextVQA | ScienceQA | POPE | GQA  | MMBench |
| --------------- | ------ | ---- | ----- | ------ | ------- | --------- | ---- | ---- | ------- |
| Base（100%）    | 3.82   | 1506 | 78.5  | 66.2   | 58.2    | 69.5      | 85.9 | 61.9 | 64.6    |
| TransPrune-High | 1.56   | 1540 | 77.9  | 65.7   | 57.8    | 69.5      | 85.0 | 61.4 | 66.0    |
| TransPrune-Low  | 1.19   | 1491 | 76.6  | 64.2   | 56.5    | 68.7      | 85.1 | 60.0 | 65.6    |

准确率先对齐，再测效率。评测脚本不会自动计算论文表中的 TFLOPs；TFLOPs、latency 和显存需要在同一硬件、同一输入长度和同一 attention backend 下单独测量。

## 2. 目录和权重

本文命令统一使用以下变量。`ROOT` 必须是仓库根，而不是 `ThirdParty`：

```bash
export ROOT=/data0/lyx/token-pruning-for-efficient-MLLM
export TP="$ROOT/ThirdParty/TransPrune"
export DATA="$ROOT/weights"
export EVAL_ROOT="$ROOT/ThirdParty/LLaVA/playground/data/eval"
export RUN="$ROOT/runs/transprune-v1.5"
mkdir -p "$RUN"
```

### 2.1 LLaVA-v1.5 权重

TransPrune 的 `builder.py` 要求旧版 LLaVA checkpoint 格式：`config.json` 的 `model_type` 应为 `llava_llama`，并且模型目录中有 `mm_projector.bin`。本机可用目录是：

权重文件在 Hugging Face snapshot 里，但传给评测脚本的路径最后一段必须包含 `llava`。`get_model_name_from_path` 只取最后一级目录名，`builder.py` 再用 `'llava' in model_name` 决定是否走 LLaVA / TransPrune 加载。`haotian-from-hf` 不满足这个条件，直接把它当作 `--model-path` 会按普通因果语言模型去加载 `llava_llama` 权重。

```bash
SNAP="$DATA/hub/models--liuhaotian--llava-v1.5-7b/snapshots/haotian-from-hf"
test -s "$SNAP/config.json"
test -s "$SNAP/mm_projector.bin"
ln -sfn "$SNAP" "$DATA/llava-v1.5-7b"
export MODEL="$DATA/llava-v1.5-7b"
```

不要直接使用本地 Hugging Face `refs/main` 当前指向的 `6f798d...` snapshot：它的架构是 Transformers 的 `LlavaForConditionalGeneration`，没有 `mm_projector.bin`，与 `LlavaLlamaForCausalLM_TransPrune` 不兼容。若本机没有 `haotian-from-hf`，应重新获取与 TransPrune/LLaVA-v1.5 旧代码兼容的 `llava-v1.5-7b` 权重，而不是只下载一个 Transformers snapshot。

视觉塔由 config 指向 `openai/clip-vit-large-patch14-336`。本机缓存目录为 `weights/hub/models--openai--clip-vit-large-patch14-336`；运行前固定 Hugging Face cache，避免程序联网或误用另一份模型：

```bash
export HF_HOME="$DATA"
export HUGGINGFACE_HUB_CACHE="$DATA/hub"
export TRANSFORMERS_CACHE="$DATA/hub"
```

### 2.2 数据清单

TransPrune 仓库只提供评测代码和 shell 模板，不包含以下 benchmark 的完整数据。准备数据时保留官方文件名，或者同步修改命令中的路径。下表路径是相对评测根目录的示意；本机 GQA 评测根目录是 `$EVAL_ROOT`，其他未下载的数据可放在 `$DATA/eval` 或另一个自定义根目录。


| Benchmark  | 代码入口                          | 需要的文件/目录（相对评测根目录）                                                                                 | 当前仓库状态                                     |
| ---------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| MME        | `llava/eval/model_vqa_loader.py`  | `MME/llava_mme.jsonl`、`MME/MME_Benchmark_release_version/`；官方 `convert_answer_to_mme.py` 和 `eval_tool/`        | 未随 TransPrune 提供                             |
| MMBench    | `llava/eval/model_vqa_mmbench.py` | `mmbench/mmbench_dev_20230712.tsv`                                                                                  | 未随 TransPrune 提供；图片以内嵌 base64 存在 TSV |
| SEED-Bench | `llava/eval/model_vqa_loader.py`  | `seed_bench/llava-seed-bench.jsonl`、`seed_bench/`、`SEED-Bench.json`                                               | 未随 TransPrune 提供                             |
| ScienceQA  | `llava/eval/model_vqa_science.py` | `scienceqa/llava_test_CQM-A.json`、`scienceqa/images/test/`、`scienceqa/pid_splits.json`、`scienceqa/problems.json` | 未随 TransPrune 提供                             |
| VQAv2      | `llava/eval/model_vqa_loader.py`  | `vqav2/llava_vqav2_mscoco_test-dev2015.jsonl`、`vqav2/test2015/`、`llava_vqav2_mscoco_test2015.jsonl`               | 未随 TransPrune 提供                             |
| POPE       | `llava/eval/model_vqa_loader.py`  | `pope/llava_pope_test.jsonl`、`pope/val2014/`、`pope/coco/coco_pope_*.json`                                         | 未随 TransPrune 提供                             |
| GQA        | `llava/eval/model_vqa_loader.py`  | `gqa/llava_gqa_testdev_balanced.jsonl`、`gqa/data/images/`、`gqa/data/eval/eval.py`                                 | 标准 testdev-balanced 已在工作区的 `ThirdParty/LLaVA/playground/data/eval/gqa/`；不在 TransPrune 仓库内 |
| TextVQA    | `llava/eval/model_vqa_loader.py`  | `textvqa/llava_textvqa_val_v051_ocr.jsonl`、`textvqa/train_images/`、`textvqa/TextVQA_0.5.1_val.json`               | 未随 TransPrune 提供                             |

当前工作区已经有一份可直接用于标准 GQA 推理的测试资产：

```text
$ROOT/ThirdParty/LLaVA/playground/data/eval/gqa/
├── llava_gqa_testdev_balanced.jsonl       # 12,578 条问题
└── data/
    ├── images/                            # 398 张图片
    ├── testdev_balanced_questions.json     # GQA gold questions
    └── eval/eval.py                        # GQA evaluator（完整指标还需 sceneGraphs/choices）
```

已核对：JSONL 有 12,578 行、398 张不重复图片且文件都在，`question_id`/`image` 字段正是 `model_vqa_loader.py` 所需格式。它不在 TransPrune git 仓库内，是工作区中其他 LLaVA/评测准备步骤留下的共享数据。

论文表 1 的 GQA 列是 balanced Accuracy。用 `scripts/repro/score_visionzip.py` 对 `testdev_balanced_questions.json` 做小写后的 exact match 即可，不依赖 scene graph。官方 `eval/eval.py --tier testdev_balanced` 在算 Accuracy 之前就会无条件读取：

- `testdev_balanced_sceneGraphs.json`
- `testdev_balanced_choices.json`
- `testdev_balanced_all_questions.json`（默认问题文件名；当前目录里的文件叫 `testdev_balanced_questions.json`）

因此即使用户补了 eval 包，不改文件名、不补 scene graph 和 choices，这条官方命令也会在出分前退出。完整 consistency/validity 才需要那些文件；对齐表 1 的 Accuracy 不要走这条命令。

`weights/datasets/gqa/processed/mt_gqa/` 是另一个用于 MetaCompress 多轮训练的 GQA-only 重建产物，不是 TransPrune 标准 testdev 评测数据，也不要把它直接传给下面的 GQA 命令。

因此，“未随 TransPrune 提供”不等于“官方没有使用说明”：TransPrune 的 `scripts/transprune/v1_5_test/gqa.sh` 已经说明了输入文件名、模型调用参数、答案合并、`convert_gqa_for_eval.py` 和 `eval/eval.py` 的顺序；只是脚本使用作者机器上的 `/home/usr/liao/...` 绝对路径，README 没有附带数据下载器。数据来源应使用 GQA 官方发布版本，并保存下载日期、文件 SHA-256 和数据版本。

## 3. 环境

README 和 `pyproject.toml` 给出的基础环境是 Python 3.10，主要版本固定如下：

```bash
conda create -n transprune python=3.10 -y
conda activate transprune
python -m pip install --upgrade pip
cd "$TP"
python -m pip install -e .
```

`pyproject.toml` 会安装：`torch==2.1.2`、`torchvision==0.16.2`、`transformers==4.37.2`、`tokenizers==0.15.1`、`accelerate==0.21.0`、`einops==0.6.1`、`timm==0.6.13` 等。本机 GPU 是 B200（compute capability 10.0）。`torch==2.1.2+cu121` 没有 `sm_100` 内核，模型一在 GPU 上初始化权重就会报 `CUDA error: no kernel image is available for execution on the device`。当前 `transprune` 环境改为与本机 `llava` 环境相同的 `torch==2.14.0+cu130`、`torchvision==0.29.0+cu130`，`transformers` 仍是 `4.37.2`。这是硬件要求，不是论文设置。`pip check` 会提示与 `llava` 包声明的 `torch==2.1.2` 冲突，可以忽略。训练相关的 `deepspeed==0.12.6`、`ninja`、`wandb` 是可选依赖，本文的 training-free 评测不需要安装它们：

```bash
# 只有要使用仓库中的训练脚本时才安装
python -m pip install 'deepspeed==0.12.6' ninja wandb
```

安装后先做环境门禁：

```bash
cd "$TP"
python - <<'PY'
import torch, transformers, torchvision
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('transformers', transformers.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
PY
python -m compileall -q llava
git rev-parse HEAD
```

论文使用 A100 40 GB 并启用 FlashAttention。当前 `llava/model/builder.py` 在 `method=transprune` 分支显式设置 `use_flash_attention_2=False`，因此“安装了 flash-attn”并不会自动得到论文的效率配置。准确率复现先保持 released code 的默认行为；若要对齐 TFLOPs/latency，必须在相同 CUDA、FlashAttention 版本和 A100 上显式改 loader，并把这个改动记录到实验 manifest。

发布代码在剪枝后重建 4D attention mask 时仍传入剪枝前的 `seq_length`。`transformers==4.37.2` 的 `AttentionMaskConverter.to_4d` 会把这条 2D mask 扩成与 query 长度不一致的 4D mask，第一张图就会在 layer 7 报 shape mismatch。本仓库已在 `llava/model/modeling_llama_transprune.py` 的 `transprune` 前向里改成使用剪枝后的 `hidden_states.shape[1]`，prefill 的 `past_key_values_length` 置 0。这是让发布脚本能跑完的本地补丁，不改变保留比例或打分公式。

## 4. 统一评测参数

在每个命令前执行：

```bash
cd "$TP"
# 一个进程只能看见一张卡。load_pretrained_model 默认 device_map="auto"，
# 若这里列出多张卡，7B 会被切开；剪枝代码会直接调用下一层的 q_proj/k_proj，
# hidden state 和权重不在同一张卡时会在第一次剪枝失败。
export CUDA_VISIBLE_DEVICES=0
export CKPT=llava-v1.5-7b
export METHOD=transprune   # Base 门禁用 llava
COMMON_ARGS=(
  --model-path "$MODEL"
  --temperature 0
  --method "$METHOD"
  --accmulating_layer_list '[7,8,9,10,11,12]'
  --pruning_layer_list '[7,9,12]'
  --image_token_ratio_list '[0.875,0.625,0.125]'
  --conv-mode vicuna_v1
)
```

代码中的参数名拼写是 `accmulating_layer_list`（少一个 `u`），不要改成论文中的自然语言拼写。所有 `model_vqa_*` 脚本都通过 `eval()` 解析这三个 list 参数，必须保留外层引号。`--method llava` 会加载未剪枝的 `LlavaLlamaForCausalLM`；省略 `--method` 时 argparse 得到 `None`，同样不会进入 TransPrune 分支，但脚本里应显式写成 `llava`，避免和「忘了传参」混在一起。

单卡命令使用 `--num-chunks 1 --chunk-idx 0`，并且 `CUDA_VISIBLE_DEVICES` 只含一张卡。多卡时每张卡一个进程、一个 `chunk-idx`，进程内再把 `CUDA_VISIBLE_DEVICES` 设成该卡。分片结束后按 `IDX` 从 0 到 `CHUNKS-1` 拼接；不要让多个进程写同一个 answers 文件。

## 5. 运行命令

以下手写命令把原始答案放在 `$RUN/<benchmark>/`。`scripts/transprune/run_llava15.sh` 的对应目录是 `$RUN/<benchmark>/$METHOD/`（默认 `METHOD=transprune`），这样 Base（`METHOD=llava`）和 TransPrune 不会互相覆盖。评测结果和中间 JSON 不要覆盖原始答案。

### 5.1 GQA

```bash
export GQA="$EVAL_ROOT/gqa"
export OUT="$RUN/gqa"
mkdir -p "$OUT"

python -m llava.eval.model_vqa_loader \
  "${COMMON_ARGS[@]}" \
  --question-file "$GQA/llava_gqa_testdev_balanced.jsonl" \
  --image-folder "$GQA/data/images" \
  --answers-file "$OUT/$CKPT.jsonl" \
  --num-chunks 1 --chunk-idx 0

mkdir -p "$OUT/answers/gqa"
cp "$OUT/$CKPT.jsonl" "$OUT/answers/gqa/merge.jsonl"
python "$ROOT/scripts/repro/score_visionzip.py" \
  --run-dir "$OUT" --eval-root "$EVAL_ROOT" --tasks gqa
```

`score_visionzip.py` 复用工作区已有的 GQA Accuracy 评分，与 VisionZip 的 token 选择无关。它读取 `$OUT/answers/gqa/merge.jsonl`，内部再用 `ThirdParty/LLaVA/scripts/convert_gqa_for_eval.py` 转成预测文件，因此对 TransPrune 和 `--method llava` 都适用。结果写入 `$OUT/metrics/gqa.json` 和 `$OUT/metrics/gqa.txt`。不要再调用 `eval/eval.py --tier testdev_balanced`：当前数据缺少它启动时就要的 scene graph、choices，以及默认文件名 `testdev_balanced_all_questions.json`。

### 5.2 MMBench

MMBench 的图片已经嵌入 TSV，不能额外传 `--image-folder`：

```bash
export OUT="$RUN/mmbench"
mkdir -p "$OUT/upload"

python -m llava.eval.model_vqa_mmbench \
  "${COMMON_ARGS[@]}" \
  --question-file "$DATA/eval/mmbench/mmbench_dev_20230712.tsv" \
  --answers-file "$OUT/$CKPT.jsonl" \
  --single-pred-prompt \
  --num-chunks 1 --chunk-idx 0

python scripts/convert_mmbench_for_submission.py \
  --annotation-file "$DATA/eval/mmbench/mmbench_dev_20230712.tsv" \
  --result-dir "$OUT" \
  --upload-dir "$OUT/upload" \
  --experiment "$CKPT"
```

`upload/$CKPT.xlsx` 是官方提交文件。MMBench dev 的论文分数来自评测服务器，本机没有对应打分脚本，不能把生成答案数或 xlsx 行数当成准确率。

### 5.3 MME

MME 的官方转换和评分脚本不在 TransPrune 仓库中，按 MME 发布包放置 `convert_answer_to_mme.py` 与 `eval_tool/`：

```bash
export MME="$DATA/eval/MME"
export OUT="$RUN/mme"
mkdir -p "$OUT" "$MME/answers"

python -m llava.eval.model_vqa_loader \
  "${COMMON_ARGS[@]}" \
  --question-file "$MME/llava_mme.jsonl" \
  --image-folder "$MME/MME_Benchmark_release_version" \
  --answers-file "$MME/answers/$CKPT.jsonl" \
  --num-chunks 1 --chunk-idx 0

# 下面两条命令的参数格式来自 scripts/transprune/v1_5_test/mme.sh。
cd "$MME"
python convert_answer_to_mme.py --experiment "$CKPT"
cd eval_tool
python calculation.py --results_dir "$MME/answers/$CKPT"
cd "$TP"
cp "$MME/answers/$CKPT.jsonl" "$OUT/$CKPT.jsonl"
```

论文表中的 `MMEP` 是 MME 的 perception 分数；同时保存 MME 的原始子任务结果，避免只保留一个聚合数。

### 5.4 POPE

```bash
export OUT="$RUN/pope"
mkdir -p "$OUT"

python -m llava.eval.model_vqa_loader \
  "${COMMON_ARGS[@]}" \
  --question-file "$DATA/eval/pope/llava_pope_test.jsonl" \
  --image-folder "$DATA/eval/pope/val2014" \
  --answers-file "$OUT/$CKPT.jsonl" \
  --num-chunks 1 --chunk-idx 0

python llava/eval/eval_pope.py \
  --annotation-dir "$DATA/eval/pope/coco" \
  --question-file "$DATA/eval/pope/llava_pope_test.jsonl" \
  --result-file "$OUT/$CKPT.jsonl"
```

POPE evaluator 会分别打印 `random`、`popular`、`adversarial` 类别的 Accuracy/Precision/Recall/F1/Yes ratio；完整保留三类输出。

### 5.5 TextVQA

```bash
export OUT="$RUN/textvqa"
mkdir -p "$OUT"

python -m llava.eval.model_vqa_loader \
  "${COMMON_ARGS[@]}" \
  --question-file "$DATA/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl" \
  --image-folder "$DATA/eval/textvqa/train_images" \
  --answers-file "$OUT/$CKPT.jsonl" \
  --num-chunks 1 --chunk-idx 0

python -m llava.eval.eval_textvqa \
  --annotation-file "$DATA/eval/textvqa/TextVQA_0.5.1_val.json" \
  --result-file "$OUT/$CKPT.jsonl"
```

### 5.6 ScienceQA

```bash
export SQA="$DATA/eval/scienceqa"
export OUT="$RUN/scienceqa"
mkdir -p "$OUT"

python -m llava.eval.model_vqa_science \
  "${COMMON_ARGS[@]}" \
  --question-file "$SQA/llava_test_CQM-A.json" \
  --image-folder "$SQA/images/test" \
  --answers-file "$OUT/$CKPT.jsonl" \
  --single-pred-prompt \
  --num-chunks 1 --chunk-idx 0

python llava/eval/eval_science_qa.py \
  --base-dir "$SQA" \
  --result-file "$OUT/$CKPT.jsonl" \
  --output-file "$OUT/${CKPT}_output.json" \
  --output-result "$OUT/${CKPT}_result.json"
```

`eval_science_qa.py` 默认读取 `pid_splits.json` 的 `test` split，并同时报告总 Accuracy 和 image-only Accuracy。

### 5.7 VQAv2

```bash
export VQA="$DATA/eval/vqav2"
export OUT="$RUN/vqav2"
export VQA_SPLIT=llava_vqav2_mscoco_test-dev2015
mkdir -p "$OUT" "$VQA/answers/$VQA_SPLIT/$CKPT"

python -m llava.eval.model_vqa_loader \
  "${COMMON_ARGS[@]}" \
  --question-file "$VQA/$VQA_SPLIT.jsonl" \
  --image-folder "$VQA/test2015" \
  --answers-file "$VQA/answers/$VQA_SPLIT/$CKPT/1_0.jsonl" \
  --num-chunks 1 --chunk-idx 0

# converter 固定读取 answers/$SPLIT/$CKPT/merge.jsonl
# 和同目录上一级的 llava_vqav2_mscoco_test2015.jsonl（不是 test-dev 问题文件）。
# 多卡时按 IDX 顺序拼成这个 merge.jsonl，再执行 converter。
cat "$VQA/answers/$VQA_SPLIT/$CKPT/1_0.jsonl" > \
  "$VQA/answers/$VQA_SPLIT/$CKPT/merge.jsonl"
python scripts/convert_vqav2_for_submission.py \
  --split "$VQA_SPLIT" --ckpt "$CKPT" --dir "$DATA/eval/vqav2"
cp "$VQA/answers/$VQA_SPLIT/$CKPT/merge.jsonl" "$OUT/$CKPT.jsonl"
```

`answers_upload/$VQA_SPLIT/$CKPT.json` 是 VQA 评测服务器的提交文件。test-dev 准确率要等服务器返回，本机没有标注时不能从这份 JSON 算出论文表里的 VQAv2。

### 5.8 SEED-Bench

```bash
export SEED="$DATA/eval/seed_bench"
export OUT="$RUN/seed"
mkdir -p "$OUT"

python -m llava.eval.model_vqa_loader \
  "${COMMON_ARGS[@]}" \
  --question-file "$SEED/llava-seed-bench.jsonl" \
  --image-folder "$SEED" \
  --answers-file "$OUT/$CKPT.jsonl" \
  --num-chunks 1 --chunk-idx 0

python scripts/convert_seed_for_submission.py \
  --annotation-file "$SEED/SEED-Bench.json" \
  --result-file "$OUT/$CKPT.jsonl" \
  --result-upload-file "$OUT/${CKPT}_upload.jsonl"
```

`${CKPT}_upload.jsonl` 是 SEED 官方提交文件，本机 convert 脚本不计算 SEED-I 准确率。

注意 `scripts/transprune/v1_6_test/seed.sh` 中写的是 `--method llava`，那条命令不会启用 TransPrune；复现 v1.5 论文主表应使用上面的 `--method transprune`。

## 6. 多卡运行与结果合并

仓库脚本的多卡模式是“每个 GPU 一个进程、每个进程一个 chunk”，例如：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
    "${COMMON_ARGS[@]}" \
    --question-file "$EVAL_ROOT/gqa/llava_gqa_testdev_balanced.jsonl" \
    --image-folder "$EVAL_ROOT/gqa/data/images" \
    --answers-file "$RUN/gqa/${CHUNKS}_${IDX}.jsonl" \
    --num-chunks "$CHUNKS" --chunk-idx "$IDX" &
done
wait
: > "$RUN/gqa/merge.jsonl"
for IDX in $(seq 0 $((CHUNKS-1))); do
  cat "$RUN/gqa/${CHUNKS}_${IDX}.jsonl" >> "$RUN/gqa/merge.jsonl"
done
```

按 `IDX` 循环拼接。`${CHUNKS}_*.jsonl` 的 glob 是字典序，`CHUNKS>=10` 时 `10_10` 会排在 `10_2` 前面。这些 evaluator 按 `question_id` 对齐，顺序不改变分数，但合并文件仍应按分片编号写，并在评测前检查答案数、question id 去重数和输入 question 数相同。每个子进程的 `CUDA_VISIBLE_DEVICES` 只能是一张卡。

## 7. 复现验收与记录

每个 benchmark 至少保存：

```text
$RUN/<benchmark>/
  <model>.jsonl                 # 原始答案，不覆盖
  command.txt                   # 完整命令和环境变量
  env.pip-freeze.txt            # python -m pip freeze
  nvidia-smi.txt
  result.json or evaluator.log
```

建议按以下顺序验收：

1. 先用 `METHOD=llava bash scripts/transprune/run_llava15.sh gqa` 跑未剪枝 LLaVA，检查 Base 分数和 question/answer 对齐；这一步失败时不要调剪枝参数。
2. 再用默认的 `bash scripts/transprune/run_llava15.sh gqa`（`[7..12] / [7,9,12] / [0.875,0.625,0.125]`）跑完整 v1.5 评测，并报告实际 token 数和全部指标。
3. 再测另一组 token budget，才能讨论论文的 High/Low；不要只凭脚本目录名给结果命名。
4. 最后才在 A100 上测 latency、显存和 TFLOPs。当前 loader 默认关闭 FlashAttention，这些数字不能直接与论文表 1 的效率列比较。

记录版本时至少包含：TransPrune commit、模型目录及 `config.json`/权重 SHA-256、vision tower revision、数据文件 SHA-256、Python/PyTorch/Transformers 版本、GPU 型号、CUDA/FlashAttention 状态、完整命令和 evaluator 输出。

## 8. 当前 checkout 的不可直接复现项

- **LLaVA-NeXT-7B、Qwen2.5-VL-7B**：论文有表 2/3，但当前 TransPrune 代码只提供 LLaVA/Llama 路径，没有对应的 Qwen2.5-VL 或完整 LLaVA-NeXT 适配。
- **TGIF、MSVD / Video-LLaVA**：论文表 6 的 Video-LLaVA 评测脚本和视频数据不在本 checkout，不能用标准 Video-MME 代替。
- **VisionZip/CDPruner 组合**：论文表 4/5 依赖外部 projector pruning 实现，TransPrune 仓库没有这些依赖和权重。
- **效率数字**：论文使用 A100 40 GB 和 FlashAttention；当前 `method=transprune` loader 强制 `use_flash_attention_2=False`，所以只执行准确率命令不会得到论文的 TFLOPs/latency。
- **注意力 mask**：发布代码在剪枝后用剪枝前的序列长度重建 4D mask，在 `transformers==4.37.2` 上会报 shape mismatch。本仓库已改为剪枝后的长度，见第 3 节。报告里要写明这个补丁。

这些限制不影响验证 TransPrune 的核心剪枝逻辑和 v1.5 标准 benchmark 评测，但必须在报告中标为“部分复现”或“扩展实验”，不能写成论文全部结果已严格重现。
