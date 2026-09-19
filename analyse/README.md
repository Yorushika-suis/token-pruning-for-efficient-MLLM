# analyse/：keep map 与视觉层可视化

读现有 probe NPZ，把保留的视觉 token 铺回空间网格，叠在原图/视频帧上。层输出不在 NPZ 里，按需 hook **视觉编码器**（Video-MME：SigLIP；VisionZip：CLIP）。

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
python analyse/show_keep.py \
  --methods fastvid,vidcom2 --dataset videomme --key 1NcGHbFWBFA__573-1 \
  --out analyse/out/keep.png

python analyse/show_keep.py \
  --methods visionzip --dataset mme --key code_reasoning/0020.png \
  --out analyse/out/vz.png

# 2. FastVID vs VidCom2：一次抽题，keep + layers 共用同一批 --key
python analyse/run_keep_and_layers.py \
  --methods fastvid,vidcom2 --dataset videomme --n 8 --seed 0 \
  --out-dir analyse/out/videomme_pair
# 只写名单（keys.txt / keys.json，含每条 --key）
python analyse/run_keep_and_layers.py --n 8 --seed 0 --keys-only \
  --out-dir analyse/out/videomme_pair
# 之后分两次跑，读同一份名单
python analyse/sample_keep.py --methods fastvid,vidcom2 --dataset videomme \
  --keys-file analyse/out/videomme_pair/keys.txt --out-dir analyse/out/sample_keep
python analyse/show_layers.py --methods fastvid,vidcom2 --dataset videomme \
  --keys-file analyse/out/videomme_pair/keys.txt --out-dir analyse/out/layers/videomme

# 2b. 仅 keep、自行抽样
python analyse/sample_keep.py \
  --methods fastvid,vidcom2 --dataset videomme --n 8 --seed 0 \
  --out-dir analyse/out/sample_keep

# 3. keep + 视觉编码器第一层 / 中间均匀 3 层 / 最后一层
python analyse/show_layers.py \
  --methods fastvid,vidcom2 --dataset videomme --key 1NcGHbFWBFA__573-1 \
  --out-dir analyse/out/layers

python analyse/show_layers.py \
  --methods visionzip --dataset gqa --n 4 --seed 0 \
  --out-dir analyse/out/layers

# 4. 层能量 heatmap + keep 变暗（不叠原图；未选中默认 dim=0.55）
python analyse/show_keep_attn.py \
  --methods fastvid,vidcom2 --dataset videomme \
  --keys-file analyse/out/videomme_pair/keys.txt \
  --cache-dir analyse/out/videomme_pair/layers \
  --out-dir analyse/out/videomme_pair/keep_attn

# 5. 13×13 位置选择频率（默认两方法全部共享 key、全部帧）
python analyse/show_keep_freq.py \
  --methods fastvid,vidcom2 --dataset videomme \
  --out-dir analyse/out/videomme_pair/keep_freq

# 6. 对 8 帧做 rot90/180/270 + 水平翻转，复用 keep / layers / keep_attn
python analyse/show_geom.py \
  --methods fastvid,vidcom2 --dataset videomme \
  --key 1evyOuQz-jM__845-3 \
  --out-dir analyse/out/videomme_pair/geom
# 或 --keys-file analyse/out/videomme_pair/keys.txt  /  --n 1 --seed 0
```

产物目录（`run_keep_and_layers` / `sample_keep` / `show_layers`）：

```
analyse/out/videomme_pair/
  keys.txt
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
```

已有 png/json 会 skip；加 `--force` 才重画。

层 dump 缓存在 `layers/cache/<method>/<key>/layer_{i}.npy`（`--out-dir` 之下）。

## 产物

- `*.png`：行 = 帧（或单图），列 = 方法或层号。保留 patch 较亮，丢掉的变暗。
- 同名 `*.json`：retention、`spatial_kept`、帧下标、FastVID 还原说明。
- `show_layers`：`analyse/out/layers/<key>/keep.png` 与 `<method>_layers.png`。

## 实现要点

- FastVID 的 `keep_indices` 是整段 LLM 序列；`masks.py` 去掉文本前缀和每行 newline 后映到 13×13。NPZ 里的 `n_out` 曾用「全部原始 newline 数」去减，和格子上的 patch 数不必相等，标题同时写 `spatial_kept`。
- 64 帧协议与评测一致：`np.linspace(0, n_mp4-1, 64)`，再对这 64 帧 `linspace` 出 8 帧；对比时各方法同一组下标。
- 不改 `ThirdParty/`。层 hook 只 `register_forward_hook`，前向结束后 remove。
- `analyse/out/` 已 gitignore。
