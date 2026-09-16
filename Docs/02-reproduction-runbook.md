# FastVID / VidCom2 / VisionZip 复现步骤

从环境到权重到官方评测的操作手册。产物契约、probe 打在哪一层绑定，见 [`01-artifact-collection.md`](01-artifact-collection.md)。

下文 `$ROOT` 指本仓库根路径。`ThirdParty/` 不进 git，clone 时按第 1 节钉 commit。

本阶段 **不改** `ThirdParty/` 文件（VisionZip 底座 LLaVA 的 SDPA 默认除外，见第 8.1 节）。官方分数用仓库里的 smoke 脚本先跑通；`REPRO_PROBE=1` 已由 [`scripts/repro/`](../scripts/repro) 实现（`PYTHONPATH=$ROOT/scripts/repro` 加载 `sitecustomize.py`）。

## 0. 第一波范围

| 方法 | 模型 | Task | 声明保留率 | 对齐目标 |
|---|---|---|---|---|
| FastVID | LLaVA-Video-7B | Video-MME | 0.10 | 官方 log overall **60.7%**（short/med/long 71.7 / 59.1 / 51.2） |
| FastVID | LLaVA-OneVision-7B | Video-MME | 0.10 | **57.3%**（68.0 / 55.8 / 48.2） |
| FastVID | Qwen2.5-VL-7B | Video-MME | 0.27（注释称实际 ~24.1%） | **62.9%**（73.9 / 62.0 / 52.8） |
| VidCom2 | LLaVA-OneVision-7B | Video-MME | `R_RATIO=0.25` | 论文 Table 2 约 **58.6** |
| VidCom2 | LLaVA-Video-7B | Video-MME | `R_RATIO=0.25` | 论文 LLaVA-Video @ R=25% 相对 upper bound 约 93.6% |
| VisionZip | LLaVA-1.5-7B | 图像（先 GQA / MME / POPE） | retain 64 = `dominant=54, contextual=10` | GQA 55.1 / MME 1690 / POPE 77.0 / Avg **94.0%** of vanilla |

第二波（本文不写全命令）：VidCom2 的 `mlvu_dev,longvideobench_val_v,mvbench`；VisionZip 其余图像表与 LLaVA-NeXT；三个方法的跨方法视频对比。

不做：

- FastVID README 的 FLOPs / 7.1× prefilling（官方 `eval.sh` 不产出这些数）。
- VidCom2 的 Qwen（必须 `git checkout qwen`）。
- VisionZip 论文视频表（走 Video-LLaVA，本 checkout 没有）；Qwen2.5-VL 文件的 video 路径不跑 VisionZip。

三个方法依赖互相打架。必须 **三个独立 venv**（FastVID 的三个子目录可以共用名为 `fastvid` 的 conda env，但不要和 VidCom2 / VisionZip 混）。

## 1. 钉死的源码

```bash
cd $ROOT/ThirdParty
# 若目录已在，核对 HEAD，不要随手 git pull
git -C FastVID   rev-parse HEAD   # a40a109cb8df20e8c2c17c6850f8269717369990
git -C VidCom2   rev-parse HEAD   # 944a17d14f4848ad080a65eab2b0711a760fa154
git -C VisionZip rev-parse HEAD   # 8f86b55c6f000eb033e6912538af2dd7dcb30502
git -C LLaVA     rev-parse HEAD   # c121f0432da27facab705978f83c4ada465e46fd
```

| 方法 | 远程 | Commit |
|---|---|---|
| FastVID | https://github.com/LunarShen/FastVID.git | `a40a109cb8df20e8c2c17c6850f8269717369990` |
| VidCom2 | https://github.com/xuyang-liu16/VidCom2.git | `944a17d14f4848ad080a65eab2b0711a760fa154`（分支 `main`） |
| VisionZip | https://github.com/dvlab-research/VisionZip.git | `8f86b55c6f000eb033e6912538af2dd7dcb30502` |
| LLaVA（VisionZip 底座，不是 LLaVA-NeXT） | https://github.com/haotian-liu/LLaVA.git | `c121f0432da27facab705978f83c4ada465e46fd` |

缺目录时：

```bash
git clone https://github.com/LunarShen/FastVID.git FastVID
git -C FastVID checkout a40a109cb8df20e8c2c17c6850f8269717369990
# VidCom2 / VisionZip / LLaVA 同理
```

把 SHA 写入 `$ROOT/runs/<method>/<run_id>/provenance.json`。

## 2. 硬件、磁盘、网络

- GPU：官方 FastVID log 用 4 进程；VidCom2 README 用 8 进程。单卡把 `--num_processes` 改成 1，分数应仍可对齐，吞吐不行。
- CUDA：FastVID 官方 wheel 是 **cu12 + torch 2.1** 的 flash-attn。VidCom2 钉 `torch==2.1.2`。本机（B200）toolkit 是 **`/usr/local/cuda-13.0`**。`torch==2.1.2+cu121` 最高 `sm_90`，在 B200 上 `torch.zeros(1).cuda()` 会 `no kernel image`。**FastVID / VidCom2 / VisionZip 三套 env 都要换成 cu130 Torch**（配方同第 8.1 节），smoke 用 `attn_implementation=sdpa`。换完必须 **卸掉** 旧的 `flash-attn` wheel，否则 transformers import 时加载 `flash_attn_2_cuda*.so` 会 `undefined symbol`。官方 FA2 log 的分数在本机不保证逐点对齐。
- 磁盘：每个 7B 模型约 15–20 GB；Video-MME 全量视频远大于 zhr 的 300-short（~3 GB）。lmms-eval 会按 task 自己拉数据集。
- 网络：`export HF_ENDPOINT=https://hf-mirror.com`（能直连 huggingface.co 则可不设）。

权重建议统一放到 `$ROOT/weights`，评测时：

```bash
export HF_HOME="$ROOT/weights"
# 仅当 snapshots/<sha>/config.json 真实存在后再开。只写了 refs/main、目录是空的，等于没下到。
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

本集群请用 `python $ROOT/utils/download_weights.py`（ModelScope）。Hub 上钉死的 commit 在 ModelScope 上经常列出 **空文件列表** 却仍把 `refs/main` 写成该 SHA；下载器必须把空列表当成失败并回落到 `master`（[`utils/hf_cli.py`](../utils/hf_cli.py)）。stage0 以 `snapshots/*/config.json` 为准，SHA 对不上只警告。

`liuhaotian/llava-v1.5-7b` 在 ModelScope 上没有原名。`AI-ModelScope/llava-v1.5-7b` 目前 **404**；下载器会回落到 `llava-hf/llava-1.5-7b-hf`（Transformers 转换）。该格式在 haotian-liu/LLaVA 里会把全部 LLM 权重当成 missing。下完后跑 `python $ROOT/utils/convert_llava_hf_to_haotian.py`，把 snapshot 转成 `model_type=llava_llama` + `mm_projector.bin`。VisionZip 还要另下 CLIP（`openai/clip-vit-large-patch14-336` → `AI-ModelScope/clip-vit-large-patch14-336`），否则 `vision_tower.load_model` 在 `HF_HUB_OFFLINE=1` 下会卡住或失败。**FastVID / VidCom2 不需要这个 LLaVA-1.5 仓。**

## 3. 权重下载（共用手法，钉 revision）

官方 `eval.sh` 写的是浮动 `pretrained=lmms-lab/...`。要对齐，按 zhr 的方式下指定 revision，并补 `refs/main`（transformers 走 `main` 时才会命中钉死的快照）。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$ROOT/weights"
python - <<'EOF'
from huggingface_hub import snapshot_download

# 与 zhr / LLaVA-Video 评测共用；revision 来自 ThirdParty/zhr/scripts/adts_harness.py
snapshot_download("lmms-lab/LLaVA-Video-7B-Qwen2",
                  revision="013210b3aff822f1558b166d39c1046dd109520f")
snapshot_download("google/siglip-so400m-patch14-384",
                  revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3")

# 官方脚本未钉 revision 的模型：下载当下 main，立刻记下 snapshot 目录名里的 commit
snapshot_download("lmms-lab/llava-onevision-qwen2-7b-ov")
snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
snapshot_download("liuhaotian/llava-v1.5-7b")
EOF

# 仅当走 pinned revision 时需要手工补 refs/main（hf_hub 按 sha 下载不会写它）
cd "$HF_HOME/hub"
mkdir -p models--lmms-lab--LLaVA-Video-7B-Qwen2/refs \
         models--google--siglip-so400m-patch14-384/refs
echo -n 013210b3aff822f1558b166d39c1046dd109520f \
  > models--lmms-lab--LLaVA-Video-7B-Qwen2/refs/main
echo -n 9fdffc58afc957d1a03a25b10dba0329ab15c2a3 \
  > models--google--siglip-so400m-patch14-384/refs/main
```

LLaVA-OV / Qwen2.5-VL / LLaVA-1.5 若未钉 revision：把 `hub/models--.../refs/main` 的内容和 `snapshots/` 目录名记进 provenance，下次复现用同一 sha。

Vision tower：LLaVA-Video / OneVision 都会再拉 `google/siglip-so400m-patch14-384`。缺 refs 时会出现 zhr README 里的 `cached main ref ... is not pinned`。

## 4. Video-MME 数据

lmms-eval 的 `videomme` task 默认 `dataset_path: lmms-lab/Video-MME`。第一次跑会拉 parquet 和视频。为了以后 mask 能按 `videoID` 对齐，建议额外钉一份元数据：

```bash
python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    "lmms-lab/Video-MME",
    "videomme/test-00000-of-00001.parquet",
    repo_type="dataset",
    revision="ead1408f75b618502df9a1d8e0950166bf0a2a0b",
)
EOF
```

该 parquet revision 与 [`ThirdParty/zhr/data/short_subset.json`](../ThirdParty/zhr/data/short_subset.json) 一致。zhr 的 300 short ID 只用于以后做跨方法 mask 普查，**不是** lmms-eval 全量 Video-MME（2700 题 / 900 视频）。第一波评测跑官方全量；probe 落地后可用这 300 个 ID 做子集采集。

**parquet ≠ 视频。** `utils/download_dataset.py` 只拉 metadata。lmms-eval 的 `videomme_doc_to_visual` 要 `$HF_HOME/videomme/data/{videoID}.mp4`。官方 `task.py` 会 `snapshot_download` 全量 `videos_chunked_*.zip`（~101 GB），本机既到不了 huggingface.co，也不该整包下。

Smoke（`--limit 4`）只抽前几个 `videoID`：

```bash
export HF_HOME="$ROOT/weights"
python $ROOT/utils/extract_videomme_smoke.py
# 默认: fFjv93ACGo8 N1cdUjctpG8 HIjX8OPuf-w HwnB8aCn8yE
# 用 zhr 的 zip 索引 + ModelScope HTTP Range，写到 $HF_HOME/videomme/data/
```

离线时 `datasets.load_dataset("lmms-lab/Video-MME")` 仍会打 Hub。smoke 把 `$ROOT/scripts/repro` 放进 `PYTHONPATH`，`sitecustomize.py` 会把该名字转到本地 parquet。

`HF_HUB_OFFLINE=1` 必须在 **mp4 已就位** 且 **模型 snapshot 里有 config.json** 之后再开。

若要自备视频字节校验，可复用 zhr 的 `data/videomme_short_manifest.json` + `scripts/extract_videomme.py`，路径改到 `$ROOT`，不要用脚本里的 `/mnt/tidalfs-...`。

## 5. Stage0 检查清单（每次评测前）

在对应 venv 里确认：

1. `git rev-parse HEAD` 等于第 1 节的 SHA。
2. `python -c 'import llava, inspect; print(inspect.getfile(llava))'`（或 FastVID-Qwen 的 `lmms_eval`）指向 **当前子目录**，不是另一个方法的 site-packages。
3. 模型 snapshot 存在；LLaVA-Video 的 `refs/main` 等于 pinned revision。
4. FastVID / VidCom2 的 LLaVA-Video：`mm_spatial_pool_mode=average`（VidCom2 README 已写进 `--model_args`；FastVID-Video 的 `eval.sh` 同样）。checkpoint 默认 bilinear → 14×14，坐标系错了还不报错。
5. 超参与第 6–8 节表格逐字相同（尤其 FastVID-Qwen 的 `retention_ratio=0.27` 和 `DySeg_tau=0.84`）。
6. `pip freeze` 存到 `runs/<method>/<run_id>/env.pip-freeze.txt`。
7. `REPRO_PROBE=1` 时必须带 `PYTHONPATH=$ROOT/scripts/repro`。未设置则不要把该目录放进 path（避免误加载 sitecustomize 以外的顶层模块名）。smoke 脚本已处理。

## 6. FastVID

工作目录必须进 **对应子文件夹**（README 原话）。三个子目录各带一份 LLaVA-NeXT / lmms-eval fork，不能串。

### 6.1 环境

**不要直接跑官方 `scripts/create_env.sh`。** 在本机上它会踩这几处：

| 官方脚本 | 实际问题 |
|---|---|
| `pip install --upgrade pip` | conda 环境里会把 `/envs/fastvid/bin/pip` 换残，随后 `pip: No such file or directory` |
| `cd ../lmms-eval` | 相对路径是相对 **子目录**（`fastvid_llavavideo/lmms-eval`），不是 `$ROOT/ThirdParty/FastVID/lmms-eval` |
| `pip install -e .`（lmms-eval） | `pyproject.toml` 钉死 `av==14.4.0`，该版本 **wheel 已从 PyPI 下架**，pip 会源码编译，要求 Cython + FFmpeg 7 的 `libav*`。本机没有这些库就会在 `avformat` 上失败 |
| `source activate` + 裸 `pip` | 应用 `python -m pip`，并确认 `which python` 在 `envs/fastvid` 下 |

用仓库脚本（LLaVA-Video 默认）：

```bash
# 已有坏掉的 fastvid env 时先删：conda deactivate && conda env remove -n fastvid -y
bash $ROOT/utils/create_fastvid_env.sh llavavideo
conda activate fastvid
```

另外两个栈：`llavaonevision`、`qwen25vl`（Qwen 子目录没有 LLaVA-NeXT）。三套要并存时：

```bash
FASTVID_ENV_NAME=fastvid_ov bash $ROOT/utils/create_fastvid_env.sh llavaonevision
```

脚本相对官方 `create_env.sh` 的改动：

- 不升级 pip；一律 `python -m pip`。
- 先装 **`av==14.2.0` 的 binary wheel**（同 14.x、仍有 PyPI wheel），再 `pip install -e lmms-eval --no-deps`，避免 pip 回头去编 14.4.0。
- LLaVA-NeXT 用 `pip install -e . --no-deps`，**不要** `.[train]`（会拉另一份 transformers git pin，冲掉 4.51.3）。
- 再钉：`accelerate==1.1.0`，`datasets==2.20.0`，`transformers==4.51.3`，`huggingface_hub==0.34.4`，`pyarrow==20.0.0`，`sentencepiece==0.1.99`。
- flash-attn 用官方 wheel `2.7.2.post1+cu12torch2.1cxx11abiFALSE-cp310-...`。lmms-eval 会拉 **torch==2.1.2 / torchvision==0.16.2**；wheel 必须对上这条，对不上就不要装。

若只想补当前已经建好的 env（`av` 编译失败那一步）：

```bash
conda activate fastvid
python -m pip install --only-binary=:all: 'av==14.2.0'
cd $ROOT/ThirdParty/FastVID/fastvid_llavavideo/lmms-eval
python -m pip install -e . --no-deps
# 其余依赖按 utils/create_fastvid_env.sh 里从 pyproject 抽出（跳过 av==14.4.0）再装
```

手工从源码编 `av==14.4.0` 需要 FFmpeg **7** 的开发库，Ubuntu 自带的 FFmpeg 4 不够；不要走这条路。

### 6.2 超参（官方 `eval.sh`）

公共：`DySeg_c=8`，`STPrune_d=0.4`，`DTM_p=4`，`DTM_beta=0.6`，`batch_size=1`，`attn_implementation=flash_attention_2`。

| 模型 | 帧 | retention | 其余 |
|---|---|---|---|
| LLaVA-Video | 64，`force_sample=True` | 0.10 | `mm_spatial_pool_mode=average`，`conv_template=qwen_1_5` |
| LLaVA-OV | 32 | 0.10 | `model_name=llava_qwen`，`conv_template=qwen_1_5` |
| Qwen2.5-VL | 动态分辨率 | 0.27 | `DySeg_tau=0.84`，`DySeg_ignore=0.95`（仅此栈） |

OV/Video 的 `DySeg_tau=0.9`。

### 6.3 Smoke

不要改 `ThirdParty` 里的 `eval.sh`。用仓库脚本（单卡、`--limit 4`、`sdpa`）：

```bash
export HF_HOME="$ROOT/weights"
bash $ROOT/scripts/repro/smoke_fastvid_llava_vid.sh
# 产物: $ROOT/runs/fastvid/smoke/logs/
```

应出现 `videomme_perception_score`（n=4 分数无意义）。`pretrained` 指向本地 `snapshots/<sha>`，避免离线 Hub 查找。

带 probe：

```bash
export REPRO_PROBE=1
export REPRO_PROBE_DIR="$ROOT/runs/fastvid/smoke_probe"
bash $ROOT/scripts/repro/smoke_fastvid_llava_vid.sh "$REPRO_PROBE_DIR"
# masks/*.npz + sidecars/journal.jsonl
```

### 6.4 全量

```bash
export HF_HOME="$ROOT/weights"
export HF_HUB_OFFLINE=1
cd $ROOT/ThirdParty/FastVID/fastvid_llavavideo
bash scripts/eval.sh     # 4 卡；单卡改 --num_processes=1
# 日志：scripts/llava_vid.log 与 ./logs/
```

OneVision / Qwen 同理，分别进 `fastvid_llavaonevision`、`fastvid_qwen25vl`。

对齐：overall 与 short/medium/long 对照第 0 节。跨 GPU 型号不保证逐位相同；差几个百分点先查抽帧、`mm_spatial_pool_mode`、flash-attn 是否真的启用。

### 6.5 Probe 插法

实现在 [`scripts/repro/probe_fastvid.py`](../scripts/repro/probe_fastvid.py)。算法写在 `Qwen2Model.forward` 体内，没有独立 selector 函数。包装 **仍调用 original**；用仅针对该 `source_file` 的 `sys.settrace` 在 `keep_indexs` 已是排序后的 1-D tensor、decoder 循环之前 clone。`sitecustomize` 在 import `modeling_qwen2` 后打补丁，并 wrap `LlavaVid.generate_until` 写入 `videoID` + `question_id`。进程退出时 `atexit` 还原绑定。

```bash
export REPRO_PROBE=1
export REPRO_PROBE_METHOD=fastvid
export REPRO_PROBE_DIR="$ROOT/runs/fastvid/<run_id>"
export PYTHONPATH="$ROOT/scripts/repro${PYTHONPATH:+:$PYTHONPATH}"
bash scripts/eval.sh   # 或仓库 smoke 脚本
```

density 的 `torch.rand * 1e-6` 使 context 索引非确定性。COMPLETE 门禁只卡 salient top-k 与 `segment_sizes`；context 索引照写。

函数身份：记录实际 `forward` 的 `source_file_sha256`，确认来自当前子目录的 `modeling_qwen2.py`，而不是 pip 的 transformers。

## 7. VidCom2

只在 **`main`** 复现 LLaVA。Qwen 数字需要 `origin/qwen`，第一波不做。

### 7.1 环境

```bash
cd $ROOT/ThirdParty/VidCom2
conda create -n VidCom2 python=3.10 -y
conda activate VidCom2
pip install --upgrade pip
pip install -e ".[train]"
cd lmms-eval && pip install -e . && cd ..

# token_compressor 不在 setuptools packages.find 里
export PYTHONPATH="$ROOT/ThirdParty/VidCom2${PYTHONPATH:+:$PYTHONPATH}"
```

`pyproject.toml` 钉 `torch==2.1.2`、`torchvision==0.16.2`、`datasets==2.16.1`，transformers 来自 GitHub 固定 commit `1c39974a4c4036fd641bc1191cc32799f85715a4`。这和 FastVID 的 `transformers==4.51.3` 冲突，所以不能共用 venv。

强烈建议 Flash Attention 2（效率表依赖它；精度表也应开，与 README 命令一致）。

### 7.2 Smoke

```bash
export HF_HOME="$ROOT/weights"
bash $ROOT/scripts/repro/smoke_vidcom2_llava_vid.sh
```

必须看到 `[VidCom2] Successfully integrated VidCom2 with LLaVA-Video-7B.` 没有这句话等于 `PYTHONPATH` 或 `COMPRESSOR` 没生效。`REPRO_PROBE=1` 时同样用该脚本，产物在 `$ROOT/runs/vidcom2/smoke_probe/`。

### 7.3 全量（第一波：只 Video-MME）

LLaVA-Video：

```bash
export PYTHONPATH="$ROOT/ThirdParty/VidCom2"
export HF_HOME="$ROOT/weights"
export HF_HUB_OFFLINE=1
COMPRESSOR=vidcom2 R_RATIO=0.25 accelerate launch --num_processes=8 \
  -m lmms_eval \
  --model llava_vid \
  --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average,attn_implementation=flash_attention_2 \
  --tasks videomme \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_vid \
  --output_path $ROOT/runs/vidcom2/videomme_llava_vid/logs/
```

LLaVA-OV（论文 Table 2 主模型，32×196）：

```bash
COMPRESSOR=vidcom2 R_RATIO=0.25 accelerate launch --num_processes=8 \
  -m lmms_eval \
  --model llava_onevision \
  --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,attn_implementation=flash_attention_2 \
  --tasks videomme \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_onevision \
  --output_path $ROOT/runs/vidcom2/videomme_llava_ov/logs/
```

对照：先跑一遍 **不设** `COMPRESSOR` 的 upper bound，再跑 `vidcom2`。OV @ R=25% 论文 Video-MME overall 约 58.6，且接近 upper bound。

第二波把 `--tasks` 扩成 `videomme,mlvu_dev,longvideobench_val_v,mvbench`（README 原命令）。

### 7.4 Probe 插法

```bash
export REPRO_PROBE=1
export REPRO_PROBE_DIR="$ROOT/runs/vidcom2/<run_id>"
# 其余同 7.3，必须仍带 COMPRESSOR=vidcom2
```

包装 `token_compressor.vidcom2.vidcom2.vidcom2_compression`（以及如需 features 的 `select_outlier_indices` / `compute_scales`）。接入点已经是 lmms-eval 里的 `types.MethodType(cus_prepare_inputs_labels_for_multimodal, ...)`——probe 打在压缩函数上，不要再替换一次 `prepare_inputs_labels_for_multimodal`。

LLaVA-Video 的 `global_idx` 含 newline；布尔 mask 记在 13×13 网格上，newline 单独计数。

## 8. VisionZip

仓库内 **没有** `eval.sh`。论文表 = 外部 LLaVA 环境 + 加载后两行 wrap。

### 8.1 环境（LLaVA-1.5 + VisionZip）

这是 **haotian-liu/LLaVA**，和 FastVID / VidCom2 的 LLaVA-NeXT **不是同一棵树**。conda env 名用 `llava`，不要和 `fastvid` 混。

**不要按官方 README 原样做完。** 官方步骤在本机上会踩这些：

| 官方步骤 | 实际问题 |
|---|---|
| `pip install --upgrade pip` | 和 FastVID 一样，conda 里可能把 `envs/llava/bin/pip` 换残。用 `python -m pip`。 |
| `pip install -e .` | `pyproject.toml` 钉 `torch==2.1.2` / `torchvision==0.16.2`。2.1.2+cu121 没有 B200 的 `sm_100`，`torch.zeros(1).cuda()` 会警告不兼容。 |
| `pip install flash-attn --no-build-isolation` | 连环失败：setuptools 83 没有 `pkg_resources`（Torch 2.1 的 `cpp_extension` 还在 import）；`CUDA_HOME=/usr/local/cuda-13.1` 目录不存在；toolkit 13.0 vs torch cu121 触发 CUDA mismatch；PyPI / git 源码为 `sm_100` 编 2.8.x 在 CUDA 13 上仍会 ninja 失败，也没有 `torch 2.14+cu130` 的预编译 wheel。 |
| `pip install -e ".[train]"` | 第一波只评测，不需要 deepspeed。 |

本机 toolkit 核对应是：

```bash
# 必须是 /usr/local/cuda-13.0/bin/nvcc，release 13.0
# 不要用 PATH 里的 /usr/bin/nvcc（Ubuntu 自带的 11.5）
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset CUDA_PATH
which nvcc && nvcc --version
```

安装（评测够用，不装 flash-attn）：

```bash
conda create -n llava python=3.10 -y
conda activate llava

cd $ROOT/ThirdParty/LLaVA
# 若目录已在，核对 HEAD；缺则 clone 后 checkout 第 1 节 SHA
python -m pip install -e .

# 2.1.2 装完立刻换成与 nvcc 13.0 / B200 对齐的包
python -m pip install --upgrade --force-reinstall \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130

# --force-reinstall 会把 numpy/pillow/markupsafe 拉高，和 gradio / sklearn 1.2.2 打架
python -m pip install "numpy==1.26.4" "pillow==10.4.0" "markupsafe==2.1.5"

# 之后不要再裸跑 pip install -e .（会把 torch==2.1.2 拉回来）。补包用 --no-deps。
```

验收：

```bash
python - <<'PY'
import inspect, torch, llava
from llava.model.builder import load_pretrained_model
print(torch.__version__, torch.version.cuda)          # 例如 2.14.0+cu130 / 13.0
print(torch.cuda.get_device_capability(0))            # (10, 0)
print(inspect.getfile(llava))                         # .../ThirdParty/LLaVA/llava/__init__.py
x = torch.zeros(1, device="cuda")
print("cuda ok", x)
# 加载后 model.config._attn_implementation 应为 sdpa
PY
```

注意力默认走 **SDPA**，不要编 flash-attn。官方 loader 不传 `attn_implementation` 时是 eager；本机在 `llava/model/builder.py` 加了默认 `sdpa`（`use_flash_attn=True` 仍强制 FA2）。CLIP tower **不要**改成 SDPA：VisionZip 要从 CLIP attention map 里选 token。`train_mem.py` 同样改成 `attn_implementation="sdpa"`。

这是第 8 节允许的 **唯一** ThirdParty 改动。干净 checkout 后补这两处：

```python
# llava/model/builder.py，use_flash_attn 分支改为：
if use_flash_attn:
    kwargs['attn_implementation'] = 'flash_attention_2'
else:
    kwargs.setdefault('attn_implementation', 'sdpa')
```

```python
# llava/train/train_mem.py
train(attn_implementation="sdpa")
```

再装 VisionZip：

```bash
conda activate llava
cd $ROOT/ThirdParty/VisionZip
python -m pip install -e . --no-deps
# editable 的 find_packages(where='visionzip') 可能装不出可 import 的包；评测脚本已把 ThirdParty/VisionZip 放进 PYTHONPATH
python -m pip install protobuf
```

评测用 LLaVA 自带 `llava.eval.model_vqa_loader` 或 lmms-eval。不要把这个 env 的 `llava` 包和 FastVID 子目录的 `LLaVA-NeXT` 混在 `PYTHONPATH` 里。

### 8.2 加载时插入（官方用法，也是 probe 的挂载点）

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from visionzip import visionzip

model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, get_model_name_from_path(model_path)
)
# retain 64 visual tokens（论文 LLaVA-1.5 最激进一档）
model = visionzip(model, dominant=54, contextual=10)
```

其它档：retain 128 → `dominant=108, contextual=20`；retain 192 → `dominant=162, contextual=30`。第一波只跑 retain 64。

`visionzip()` 会替换 CLIP attention/layer/tower 与 `prepare_inputs_labels_for_multimodal`。之后按 LLaVA 或 lmms-eval 原样评 GQA / MME / POPE。

### 8.3 Smoke 与全量

- Smoke：

```bash
export HF_HOME="$ROOT/weights"
conda activate llava
python $ROOT/scripts/repro/stage0.py --method visionzip --run-dir $ROOT/runs/visionzip/smoke
python $ROOT/scripts/repro/smoke_visionzip.py
# 期望 VISIONZIP_N_OUT 在 60–70（retain 64），并打印一段文本
# REPRO_PROBE=1 python .../smoke_visionzip.py
```

确认视觉 token 约为 64，而不是 576。`llava` env 缺 `protobuf` 时 LlamaTokenizer 会在加载时报错。
- 全量：沿用 LLaVA 的 GQA / MME / POPE 脚本，或 lmms-eval 同名 task。预期见表 0。Avg 94.0% 是相对 vanilla LLaVA-1.5-7B，需要一份未 wrap 的对照 run。

### 8.4 Probe 插法

`visionzip()` 已经是 monkeypatch。probe 应 wrap `CLIPVisionTower_VisionZip.forward`（在 `visionzip()` 调用 **之后** 再包一层），采集 `all_indices`、contextual `target_indices`、`n_in`/`n_out`。

不要在 `visionzip()` 之前包 `CLIPVisionTower.forward`：会被替换掉。

记录 API 参数 `dominant=54` 与内部 `_info["dominant"]=53`（CLS 另计）。manifest 写有效视觉 token = 54 + 10 = 64。

Qwen2.5-VL 路径（非第一波）：把 lmms-eval / LLaMA-Factory 的

```python
from transformers import Qwen2_5_VLForConditionalGeneration
```

换成 `from qwen2_5vl_visionzip import Qwen2_5_VLForConditionalGeneration`，并按要对齐的表修改约 L1916 的 0.65/0.05（70%）或 0.45/0.05（50%）。`batch_size=1`。不要用这条路径复现视频数字。

## 9. 把官方分数收进 v4-lite

每次 COMPLETE run 至少归档：

```
$ROOT/runs/<method>/<run_id>/
  provenance.json          # commit、model revision、freeze sha256、超参
  env.pip-freeze.txt
  scores/results.json      # 从 lmms-eval 聚合结果摘出：task、metric、value；去掉 output_path 和时间
  sidecars/official.log    # 原始 tee 日志（execution，不进 science 哈希）
```

`scores/results.json` 示例：

```json
{
  "task": "videomme",
  "metric": "videomme_perception_score",
  "overall": 60.6667,
  "short": 71.7,
  "medium": 59.1,
  "long": 51.2
}
```

probe 落地后，同一 `run_id` 下再填 `masks/` 与 `manifest.json`。分数 run 可以先 COMPLETE（`artifacts` 只有 scores）；mask 作为后续 upstream 接到同一 provenance 上。

## 10. 已知坑

| 症状 | 原因与处理 |
|---|---|
| FastVID 两次 mask 对不上 | density `torch.rand * 1e-6`。只对 salient / DySeg 做硬门。 |
| VidCom2 日志没有 Successfully integrated | `COMPRESSOR` 未设或 `PYTHONPATH` 找不到 `token_compressor`。 |
| LLaVA-Video 网格 14×14 | 未覆盖 `mm_spatial_pool_mode=average`。 |
| VisionZip retain 数是 65 不是 64 | 忘了 API `dominant` 含 CLS 偏移；应用 54+10。 |
| VisionZip 仍是 576 token | `visionzip()` 没调用，或包到了错误的 `llava` 安装。 |
| `import llava` 指到另一个方法 | 三个 venv / 三个 FastVID 子目录串了。stage0 查 `inspect.getfile`。 |
| transformers / torch 被换掉 | `pip install -e` 拉了 pyproject 全量依赖。对照 freeze。 |
| Qwen VidCom2 无压缩 | 你在 `main` 上跑 Qwen。第一波不要跑；需要时切 `qwen` branch。 |
| VisionZip Qwen 视频无剪枝 | 该文件 video 路径不走 VisionZip。 |
| `cached main ref is not pinned` | 按第 3 节补 `refs/main`。 |
| 官方 log 路径是 `/home/shenleqi3/...` | 作者机器路径。用 `HF_HOME` + snapshot，不要抄绝对路径。 |
| `pip: No such file or directory`（fastvid env） | 官方 `create_env.sh` 里 `pip install --upgrade pip` 把 conda 的 pip 换残了。删 env，改用 `utils/create_fastvid_env.sh`，一律 `python -m pip`。 |
| `Failed to build 'av'` / 找不到 `avformat` | lmms-eval 钉 `av==14.4.0`，PyPI 已下架该版本 wheel，pip 源码编译还要 FFmpeg 7。改装 `av==14.2.0` wheel，并对 lmms-eval 用 `--no-deps`。 |
| `cd .../ThirdParty/FastVID/lmms-eval` 不存在 | lmms-eval 在 `fastvid_llavavideo/lmms-eval`（以及 OV / Qwen 各自一份），不在 FastVID 根目录。 |
| `No module named 'pkg_resources'`（编 flash-attn） | setuptools ≥82 去掉了它。VisionZip 环境不要编 flash-attn；若仍要编，先 `pip install "setuptools<81"`。 |
| `No such file or directory: '.../cuda-13.1/bin/nvcc'` | 本机只有 `/usr/local/cuda-13.0`。`export CUDA_HOME=/usr/local/cuda-13.0`。 |
| CUDA 13.0 mismatches PyTorch 12.1 | LLaVA 官方钉的是 cu121。B200 上换成 cu130 Torch，见 8.1。 |
| flash-attn ninja / GitHub wheel 404 | 没有 `torch 2.14+cu130+cp310` 的官方 wheel；源码 `sm_100` + CUDA 13 也编不过。VisionZip 走 SDPA。 |
| `llava requires torch==2.1.2` pip 警告 | 故意的。忽略；不要再无 `--no-deps` 地 `pip install -e .`。 |
| `refs/main` 有 SHA 但 `snapshots/` 空或没有 `config.json` | ModelScope 对 Hub SHA 返回空文件列表仍写了 refs。卸掉空 pin，让 `hf_cli` 回落到 master 再下。 |
| FastVID/VidCom2 `no kernel image` / sm_100 | 仍是 torch 2.1.2+cu121。换 cu130，见第 2 节。 |
| `flash_attn_2_cuda ... undefined symbol` | cu130 之后旧 FA2 wheel 还在。`pip uninstall flash-attn`。 |
| Offline `Couldn't reach the Hub for dataset Video-MME` | parquet 在本地但 `load_dataset` 仍走 Hub。smoke 的 `PYTHONPATH=$ROOT/scripts/repro` 会重定向。 |
| VisionZip LLM 权重全部 newly initialized | 下到了 `llava-hf` 转换格式。需要原始 LLaVA dump。 |
| VisionZip 卡在 load / generate | 缺 CLIP snapshot。下 `openai/clip-vit-large-patch14-336`。 |
| `LlamaTokenizer ... protobuf` | `python -m pip install protobuf`。 |
| numpy 2 / pillow 12 把 sklearn 或 gradio 打崩 | cu130 的 `--force-reinstall` 连带升级。钉回 `numpy==1.26.4` `pillow==10.4.0` `markupsafe==2.1.5`。 |

## 11. 推荐顺序

1. `python utils/download_weights.py`（确认 snapshot 里有 `config.json`）+ `python utils/extract_videomme_smoke.py`。
2. `bash scripts/repro/smoke_fastvid_llava_vid.sh`（官方输出，probe 关）。
3. 同脚本 `REPRO_PROBE=1` 再跑，检查 `masks/` 与 journal。
4. `bash scripts/repro/smoke_vidcom2_llava_vid.sh`，同样再开 probe。
5. `python scripts/repro/smoke_visionzip.py`，同样再开 probe。
6. 全量 Video-MME / 图像表（仍用 `sdpa`）。
7. 300 short Video-MME ID 的跨方法 mask 普查。
