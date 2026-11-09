# VLA-ACL 落地：图像、视频与环境

方法合同见 [`06-vla-acl-methods.md`](06-vla-acl-methods.md)，图像迁移见 [`07-vla-acl-migration.md`](07-vla-acl-migration.md)。本文只记录已经写进 `scripts/vla_acl/` 的实现、两条环境，以及怎么跑。不改 `ThirdParty/`。

本机是 RTX 4090，驱动 535.129.03。用 `torch==2.1.2+cu121`。不要装面向 B200 的 `cu130` 轮子，也不要跑 `scripts/repro/cuda_env.sh`（它按 CUDA 13 的 `nvcc` 查找）。系统里的 `nvcc` 是 11.6，与这两套环境无关。

根分区 `/`（含 `/opt/conda` 和 `/root/.cache/pip`）已经写满。环境和 pip 缓存都放在 `/root/Test`。不要 `conda create --clone base`：base 是 Python 1.13.1，没有这份 torch，而且人在满的根分区上。不要在没装好 cu121 torch 时直接 `pip install accelerate`，pip 会去拉最新的 CUDA 13 torch，下载阶段就会把根分区写满。

下文 `$ROOT` 是仓库根。图像和视频各用一个解释器，不要混在同一个 env 里：两边都有名为 `llava` 的包，但是两份代码。任何 pip 之前先：

```bash
export PIP_CACHE_DIR=/root/Test/pip-cache
mkdir -p "$PIP_CACHE_DIR"
```

## 1. 实现了什么

Scorer 是一个 5 层 Transformer，给视觉 patch 打分。底座冻结。训练用 soft top-K 和黑图门控，损失是答案交叉熵加上全 token 教师的 KL。推理才真正删 token，并保留原网格位置。

| | 图像 | 视频 |
| --- | --- | --- |
| 脚本 | `train_gqa.py`、`hook.py`、`run_train.sh`、`run_llava15.sh` | `split_videomme.py`、`train_video.py`、`hook_video.py`、`run_train_video.sh`、`run_videomme.sh` |
| 模型 | LLaVA-1.5-7B | LLaVA-Video-7B-Qwen2 |
| 特征 | CLIP patch，576×1024，projector 之前 | SigLIP 池化成每帧 13×13，projector 之前 |
| 训练数据 | MetaCompress 的 2 万会话子集（6 万条问题），去掉 testdev 题号 | Video-MME 划分里的 180 个训练视频（540 题） |
| 评测 | GQA / POPE / MME | Video-MME held-out，720 个视频、2160 题，64 帧 |
| 预算 | K=128 与 K=57，各训一个 scorer | `RETENTION=0.10` 每帧 17 个；`0.25` 每帧 42 个。各训一个 scorer |

图像 K=128 的 GQA 是 58.71%，K=57 是 55.84%。这两份权重不能加载到视频上。

视频训练和图像有三处不同，评测协议不变：

- SigLIP 按 4 帧一块前向，避免 64 帧一次进编码器。
- 训练只用 8 帧均匀抽样。教师看这 8 帧的全部空间 token。学生在进语言模型前按每帧 K 做 hard top-K，换行 token 全部保留；反向用 straight-through 把梯度送回 soft top-K。
- 评测仍是 64 帧。不要把 8 帧训练写成和 FastVID 相同的训练设置。

比较只使用 held-out 视频。已有 FastVID / VidCom2 日志在这 2160 题上的分数：

| 方法 | 保留率 | overall | short / medium / long |
| --- | ---: | ---: | --- |
| FastVID | 0.10 | 60.97 | 71.08 / 60.26 / 51.05 |
| VidCom2 | 0.10 | 57.45 | 63.45 / 58.97 / 49.65 |
| FastVID | 0.25 | 63.70 | 73.90 / 63.25 / 53.45 |
| VidCom2 | 0.25 | 62.22 | 72.82 / 63.53 / 49.79 |

这四行来自 `runs/vla-acl-video/heldout_baselines.json`，不是全量 2700 题。全量分数仍以 [`Docs/Reports/2026-09-19.md`](../Reports/2026-09-19.md) 为准，不放进这张表。

## 2. 安装图像环境

图像不新装 torch。`/root/Test/envs/metacompress` 已经是 `torch==2.1.2+cu121`，并且能 import `ThirdParty/LLaVA`。直接用它，或把它克隆到 `/root/Test`：

```bash
# 若上次中断的 venv 还占着这个路径，先删掉再克隆
rm -rf /root/Test/envs/vla-acl-image
conda create --prefix /root/Test/envs/vla-acl-image --clone /root/Test/envs/metacompress -y
```

```bash
/root/Test/envs/vla-acl-image/bin/python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
import llava
print(llava.__file__)
PY
```

`torch.__version__` 应为 `2.1.2+cu121`。`llava.__file__` 应落在 `ThirdParty/LLaVA`。

## 3. 安装视频环境

视频同样从 `metacompress` 克隆，这样不用再下载 torch。克隆之后把 `llava` 换成 FastVID 的 LLaVA-NeXT，并补上 `lmms-eval`、`decord`、`pyarrow`。不要用图像那个解释器，也不要升级 torch。

```bash
rm -rf /root/Test/envs/vla-acl-video
conda create --prefix /root/Test/envs/vla-acl-video --clone /root/Test/envs/metacompress -y
conda activate vla-acl-video
export ROOT=/root/Test/Lyx/token-pruning-for-efficient-MLLM
export PIP_CACHE_DIR=/root/Test/pip-cache
export TMPDIR=/root/Test/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python -m pip install -e "$ROOT/ThirdParty/FastVID/fastvid_llavavideo/LLaVA-NeXT" --no-deps
python -m pip install -e "$ROOT/ThirdParty/FastVID/fastvid_llavavideo/lmms-eval" --no-deps
python -m pip install 'transformers==4.44.2' 'huggingface-hub==0.23.5' \
  'datasets==2.20.0' 'numpy==1.26.4' 'pillow==11.2.1' \
  'accelerate==0.29.3' decord pyarrow loguru opencv-python-headless
```

`--no-deps` 避免 `pip install -e` 把已经克隆好的 cu121 torch 换成 PyPI 上的 CUDA 13 包。不要单独 `pip install datasets`：datasets 5.x 会把 `huggingface-hub` 升到 2.x，而这份 transformers 只接受 `huggingface-hub<1`。FastVID 的 `pyproject.toml` 写的是 transformers 的某个 git 提交；本机访问 GitHub 会 TLS 中断，且默认克隆目录在已满的 `/tmp`。`4.44.2` 是同一时期、且带有 `SlidingWindowCache` 的 PyPI 发行版，走镜像安装。`4.40.2` 太旧，导入 `llava_qwen` 会失败。`OpenCLIP not installed` 可以忽略，LLaVA-Video 用的是 SigLIP。`TMPDIR` 放到 `/root/Test`，避免临时文件再写满根分区。`lmms-eval` 声明的 `openai`、`wandb`、`yt-dlp` 等未安装项是其它 benchmark 的依赖，Video-MME 这条链路不用装。要对齐的是 `numpy==1.26.4`、`accelerate==0.29.3`、`datasets==2.20.0`、`pillow==11.2.1`。`av==14.4.0` 在镜像上只有源码包，本机没有 ffmpeg 开发库，不要从源码编译；读视频用 `decord`，保留已经装上的 `av` 轮子即可。

```bash
python - <<'PY'
import torch, decord, pyarrow
import llava, lmms_eval
print(torch.__version__, torch.cuda.get_device_name(0))
print(llava.__file__)
print(lmms_eval.__file__)
PY
```

`llava.__file__` 应落在 `ThirdParty/FastVID/fastvid_llavavideo/LLaVA-NeXT`。缺哪个 import 再补哪个包，不要重跑不带 `--no-deps` 的 `pip install -e .`。

## 4. 命令

在 `$ROOT` 下执行。训练和评测不要占同一张卡。卡 3、卡 5 是 49GB、大约一半空闲时可以用。

图像，K=128：

```bash
export PY_IMG=/root/Test/envs/vla-acl-image/bin/python   # 或 metacompress 的 python

CUDA_VISIBLE_DEVICES=3 PYTHON=$PY_IMG bash scripts/vla_acl/run_train.sh
CUDA_VISIBLE_DEVICES=3,5 PYTHON=$PY_IMG bash scripts/vla_acl/run_llava15.sh gqa
```

`pope`、`mme`、`all` 替换最后一个参数。权重在 `runs/vla-acl-v1.5/train/k128/patch_scorer.pt`。冒烟：`LIMIT=1`。

视频。训练默认 8 帧。评测是 64 帧的 held-out。两个比例各训一个 scorer，不能混用。

```bash
export PY_VID=/root/Test/envs/vla-acl-video/bin/python

PYTHON=$PY_VID python scripts/vla_acl/split_videomme.py

CUDA_VISIBLE_DEVICES=3 RETENTION=0.10 PYTHON=$PY_VID bash scripts/vla_acl/run_train_video.sh
CUDA_VISIBLE_DEVICES=3 RETENTION=0.10 PYTHON=$PY_VID bash scripts/vla_acl/run_videomme.sh heldout

CUDA_VISIBLE_DEVICES=5 RETENTION=0.25 PYTHON=$PY_VID bash scripts/vla_acl/run_train_video.sh
CUDA_VISIBLE_DEVICES=5 RETENTION=0.25 PYTHON=$PY_VID bash scripts/vla_acl/run_videomme.sh heldout
```

划分在 `runs/vla-acl-video/split.json`（180 / 720）。scorer 在 `runs/vla-acl-video/train/r010/` 和 `r025/`。held-out 基线重算：

```bash
PYTHON=$PY_VID python scripts/vla_acl/score_heldout_videomme.py \
  --split runs/vla-acl-video/split.json \
  --output runs/vla-acl-video/heldout_baselines.json \
  --logs runs/fastvid/full_videomme/logs/snapshots__86f52acccbac267e034a120b97b04d45c3490a4c/20260917_024551_samples_videomme.jsonl \
        runs/fastvid/full_videomme_r025/logs/snapshots__86f52acccbac267e034a120b97b04d45c3490a4c/20260919_230012_samples_videomme.jsonl \
        runs/vidcom2/full_videomme_r010/logs/snapshots__86f52acccbac267e034a120b97b04d45c3490a4c/20260919_214028_samples_videomme.jsonl \
        runs/vidcom2/full_videomme/logs/snapshots__86f52acccbac267e034a120b97b04d45c3490a4c/20260918_152149_samples_videomme.jsonl
```

`LIMIT=1` 只跑一条训练样本，用来确认前向，不是分数。`full` 会评全部 2700 题，其中包含训练视频。
