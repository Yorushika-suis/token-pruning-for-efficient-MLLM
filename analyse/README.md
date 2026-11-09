# analyse/：keep map 与视觉层可视化

读现有 probe NPZ，把保留的视觉 token 铺回空间网格，叠在原图/视频帧上。层输出不在 NPZ 里，按需 hook **视觉编码器**（Video-MME：SigLIP；VisionZip：CLIP）。

## 目录

| 目录 | 内容 |
|---|---|
| `lib/` | 共享库：路径、NPZ、视频帧、拼图、抽题、题目文本、层 hook、几何变换与再推理 |
| `pipeline/` | 第一次抽题（`run_keep_and_layers.py --keys-only`）以及 keep+layers 一键 |
| `keep/` | keep 拼图、抽样画 keep、位置频率、混合图 |
| `layers/` | SigLIP/CLIP 层能量图 |
| `geom/` | 旋转/翻转后重新推理，以及把各变换排成一排 |
| `out/` | 产物（gitignore） |

题目原文：`python analyse/lib/questions.py --dataset videomme --key VIDEOID__QID`

## FastVID 换 keep ratio 做全量复现

默认脚本是 `fastvid_retention_ratio=0.10`，产物写在 `runs/fastvid/full_videomme` 和 `runs/fastvid/full_videomme_probe`。换比例时 **另开目录**，不要覆盖 0.10 那次。

按 0.04 / 0.06 / 0.08（FastVID）和 VidCom2 的 `R_RATIO=0.10 / 0.06` 顺序全量复现：

```bash
bash scripts/repro/sweep_retention_videomme.sh
# 只要 FastVID 或只要 VidCom2：
bash scripts/repro/sweep_retention_videomme.sh --fastvid-only
bash scripts/repro/sweep_retention_videomme.sh --vidcom2-only
```

VidCom2 的 keep 比例环境变量是 `R_RATIO`（默认 0.25），不是 `FASTVID_RETENTION_RATIO`。单次手动示例：

```bash
conda activate fastvid
export HF_HOME="$PWD/weights"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# 例：0.25，分数日志与 mask 都进 r025
REPRO_PROBE=1 \
REPRO_PROBE_DIR="$PWD/runs/fastvid/full_videomme_probe_r025" \
FASTVID_RETENTION_RATIO=0.25 \
bash scripts/repro/full_fastvid_llava_vid.sh "$PWD/runs/fastvid/full_videomme_r025"
```

- `FASTVID_RETENTION_RATIO`：论文默认 0.10；对齐 VidCom2 时用 0.25。
- 第一个参数：lmms-eval 输出目录（`logs/`、`pretrained_path.txt`）。
- `REPRO_PROBE_DIR`：2700 个 NPZ 的 mask 根目录（其下 `masks/`）。
- 中断后续跑：`REPRO_PROBE=1` 默认跳过已有 mask NPZ 的题（`REPRO_RESUME=0` 则全量重跑）。已有 NPZ 但没有 `predictions/` 缓存答案时，这些题的 Video-MME 选项会空着，不影响 mask。
- 画图时把 `--mask-dirs fastvid=` 指到新的 `.../masks`。

VidCom2 对应的是 `R_RATIO` + `scripts/repro/full_vidcom2_llava_vid.sh`。

## 能比什么

| 数据集 | 方法 | 网格 |
|---|---|---|
| `videomme` | `fastvid`, `vidcom2`（可拼图对比） | 每帧 13×13，评测 64 帧再均匀抽 8 帧 |
| `gqa` / `mme` / `pope` | `visionzip` | CLIP 24×24（dominant；contextual 未写入 NPZ） |

跨数据集硬拼会直接报错。Video-MME 的 key 必须带题目，形如 `1NcGHbFWBFA__573-1`（FastVID 三题 mask 不同）。

## 命令

在仓库根目录，用带 **numpy + 视频解码** 的环境跑 keep 图（`fastvid` 有 decord/cv2）。VisionZip 图像用 `llava`（Pillow）。层图需要 **torch + transformers + GPU 更好**。

```bash
# 1. 单题 keep；对比 FastVID vs VidCom2
python analyse/keep/show_keep.py \
  --methods fastvid,vidcom2 --dataset videomme --key 1NcGHbFWBFA__573-1 \
  --out analyse/out/keep.png

python analyse/keep/show_keep.py \
  --methods visionzip --dataset mme --key code_reasoning/0020.png \
  --out analyse/out/vz.png

# 2. FastVID vs VidCom2：一次抽题，keep + layers 共用同一批 --key
python analyse/pipeline/run_keep_and_layers.py \
  --methods fastvid,vidcom2 --dataset videomme --n 8 --seed 0 \
  --out-dir analyse/out/videomme_pair
# 只写名单（keys.txt / keys.json / questions.txt，含题目原文）
python analyse/pipeline/run_keep_and_layers.py --n 8 --seed 0 --keys-only \
  --out-dir analyse/out/videomme_pair
# 之后分两次跑，读同一份名单
python analyse/keep/sample_keep.py --methods fastvid,vidcom2 --dataset videomme \
  --keys-file analyse/out/videomme_pair/keys.txt --out-dir analyse/out/sample_keep
python analyse/layers/show_layers.py --methods fastvid,vidcom2 --dataset videomme \
  --keys-file analyse/out/videomme_pair/keys.txt --out-dir analyse/out/layers/videomme

# 2b. 仅 keep、自行抽样
python analyse/keep/sample_keep.py \
  --methods fastvid,vidcom2 --dataset videomme --n 8 --seed 0 \
  --out-dir analyse/out/sample_keep

# 3. keep + 视觉编码器第一层 / 中间均匀 3 层 / 最后一层
python analyse/layers/show_layers.py \
  --methods fastvid,vidcom2 --dataset videomme --key 1NcGHbFWBFA__573-1 \
  --out-dir analyse/out/layers

python analyse/layers/show_layers.py \
  --methods visionzip --dataset gqa --n 4 --seed 0 \
  --out-dir analyse/out/layers

# 4. 层能量 heatmap + keep 变暗（不叠原图；未选中默认 dim=0.55）
python analyse/keep/show_keep_attn.py \
  --methods fastvid,vidcom2 --dataset videomme \
  --keys-file analyse/out/videomme_pair/keys.txt \
  --cache-dir analyse/out/videomme_pair/layers \
  --out-dir analyse/out/videomme_pair/keep_attn

# 5. 13×13 位置选择频率（默认两方法全部共享 key、全部帧）
python analyse/keep/show_keep_freq.py \
  --methods fastvid,vidcom2 --dataset videomme \
  --out-dir analyse/out/videomme_pair/keep_freq

# 6. 旋转/翻转 64 帧后重新跑 keep（分环境；同一 --out-dir）
python analyse/geom/show_geom.py --methods fastvid --dataset videomme \
  --keys-file analyse/out/videomme_pair/keys.txt \
  --out-dir analyse/out/videomme_pair/geom
# VidCom2 env:
python analyse/geom/show_geom.py --methods vidcom2 --dataset videomme \
  --keys-file analyse/out/videomme_pair/keys.txt \
  --out-dir analyse/out/videomme_pair/geom

# 6b. 各变换排成一排（列 = geom）
python analyse/geom/show_geom_row.py \
  --geom-dir analyse/out/videomme_pair/geom \
  --keys-file analyse/out/videomme_pair/keys.txt
```

产物目录（`run_keep_and_layers` / `sample_keep` / `show_layers`）：

```
analyse/out/videomme_pair/
  keys.txt
  questions.txt            # 题目原文 + 选项 + 答案
  jsons/keys.json          # 含 questions
  jsons/questions.json
  keep/
    keys.txt
    pngs/*.png
    jsons/*.json          # 含 keys.json
  layers/
    keys.txt
    pngs/{key}_{method}_layers.png
    jsons/{key}_layers.json
    jsons/keys.json
    cache/                # 层 dump，可复用
  keep_attn/
    keys.txt
    pngs/{key}_{method}_keep_attn.png
    jsons/
  keep_freq/
    pngs/keep_freq.png
    jsons/keep_freq.json
    freq_{method}.npy
  geom/
    keep/pngs/{key}_{geom}.png
    layers/pngs/{key}_{geom}_{method}_layers.png
    keep_attn/pngs/{key}_{geom}_{method}_keep_attn.png
    row/pngs/{key}_{method}_keep.png   # 列 = 变换
    row/pngs/{key}_keep.png            # 两方法拼在同一排
```

```bash
# 7. 把 rot90/180/270/hflip 排成一排（只拼图，不再推理）
python analyse/geom/show_geom_row.py \
  --methods fastvid,vidcom2 \
  --keys-file analyse/out/videomme_pair/keys.txt \
  --geom-dir analyse/out/videomme_pair/geom \
  --out-dir analyse/out/videomme_pair/geom/row
# 只要 keep；加上 orig 列时再给原始 NPZ
python analyse/geom/show_geom_row.py --kinds keep \
  --orig-mask-dirs "fastvid=$PWD/runs/fastvid/full_videomme_probe/masks,vidcom2=$PWD/runs/vidcom2/full_videomme_probe_r010/masks" \
  --geoms orig,rot90,rot180,rot270,hflip \
  --geom-dir analyse/out/videomme_pair/geom
```

已有 png/json 会 skip；加 `--force` 才重画。

层 dump 缓存在 `layers/cache/<method>/<key>/layer_{i}.npy`（`--out-dir` 之下）。

## 产物

- `*.png`：行 = 帧（或单图），列 = 方法或层号。保留 patch 较亮，丢掉的变暗。
- 同名 `*.json`：retention、`spatial_kept`、帧下标、FastVID 还原说明。
- `show_layers`：`analyse/out/layers/<key>/keep.png` 与 `<method>_layers.png`。

## 实现要点

- FastVID 的 `keep_indices` 是整段 LLM 序列；`lib/masks.py` 去掉文本前缀和每行 newline 后映到 13×13。NPZ 里的 `n_out` 曾用「全部原始 newline 数」去减，和格子上的 patch 数不必相等，标题同时写 `spatial_kept`。
- 64 帧协议与评测一致：`np.linspace(0, n_mp4-1, 64)`，再对这 64 帧 `linspace` 出 8 帧；对比时各方法同一组下标。
- 不改 `ThirdParty/`。层 hook 只 `register_forward_hook`，前向结束后 remove。
- `analyse/out/` 已 gitignore。
