# FlashVID ADTS 空间偏置研究 · 评测环境与样本对齐

这个仓库用来对齐"相同模型 + 相同样本 + 相同抽帧"的评测环境。baseline 是 [FlashVID](https://github.com/Fanziyang-v/FlashVID)（training-free 的视频 token 压缩框架，挂在 LLaVA-Video-7B-Qwen2 上），我们研究它 ADTS token 选择环节的空间位置偏置。照本文档走完，你应当能在自己机器上得到与本项目一致的：模型权重 revision、300 个视频样本（字节级可校验）、每视频 64 帧的确定性抽帧、以及 stage1 的 token 选择 mask。

下文 `$ROOT` 一律指你克隆本仓库的本地路径。

## 这个项目里的"评测"指什么

不是 LLM 精度 benchmark。语言模型的权重被钉在 CPU 上并装了 tripwire（`scripts/adts_harness.py` 的 `install_llm_tripwire`，任何 LLM 前向都会直接抛错），所有评测都在视觉侧：

| 脚本 | 作用 |
|---|---|
| `scripts/stage0_assertions.py` | 环境与资产门禁断言（A0–A7） |
| `scripts/stage1_collect.py` | GPU：采集选择 mask。300 视频 × 3 几何臂（native/hflip/shift）+ 30 合成视频 × 3 臂 = 990 项，每项 4 档保留率（0.10/0.15/0.20/0.25） |
| `scripts/stage1_analyze.py` | CPU：置换检验等统计分析（固定 seed 20260828） |
| `scripts/stage1_figures.py` | 出图 fig1–fig10 |
| `scripts/story_figures.py` | 科普版中文图（需中文字体） |

`stage2_*.py` 已作废、`scripts/v4/` 实施中，见第 7 节红线。

## 硬件与系统

- 一张 NVIDIA GPU 就够：视觉侧只跑 SigLIP 视觉塔 + 投影层，7B LLM 不上 GPU（内部用 8×H20 是吞吐配置，不是门槛）。
- CUDA 12.8（对应 torch 2.10.0+cu128），其他版本未验证。
- Python 3.11（本项目用 3.11.13）。
- 磁盘：权重 ~19G + 视频 ~3G。
- 网络需可达：hf-mirror.com（或 huggingface.co）、codeload.github.com。

## 仓库结构

```
scripts/   评测、环境、数据脚本 + v4/ 新管线模块
tests/     4 个核心测试 + tests/v4/
data/      short_subset.json（300 个样本 ID）
           videomme_short_manifest.json（每个视频的字节数与 CRC32）
           videomme_zip_index.json（20 个 zip 卷的中央目录索引）
env/       实际运行环境的 pip freeze（参考对照用，不是安装清单）
```

`weights/`、`data/videos_short/`、`code/` 不在仓库里（体积原因），按第 1–3 节自备；`runs/`、`results/` 是输出目录，脚本会自动创建。

## 1. 环境搭建

### 1.1 自己的机器（公开网络）

```bash
python3.11 -m venv ~/flashvid_venv
source ~/flashvid_venv/bin/activate

# torch 走 cu128 索引（py3.11 有现成 wheel）
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# 其余依赖（版本对照本项目实际运行环境）
pip install transformers==4.57.3 decord==0.6.0 flash-attn==2.8.3 accelerate==1.13.0 \
    einops==0.8.2 timm==1.0.25 opencv-python-headless==4.13.0.92 av==16.1.0 \
    numpy==1.26.4 matplotlib==3.10.8 pyarrow==23.0.1 huggingface_hub==0.36.2 \
    pytest==9.0.2 requests safetensors sentencepiece protobuf shortuuid einops-exts
```

说明：

- flash-attn 没有匹配的预编译 wheel 时会触发源码编译，比较久，见它官方 README。
- decord 0.6.0 在 PyPI 有 manylinux 通用 wheel，直接装；装完 `pip check` 报 platform 警告属于元数据噪音，可忽略。
- `env/requirements.lock` 和 `env/create-env_20260905.pip-freeze.txt` 是本项目实际环境的全量 pip freeze，供对照，不要拿来 `pip install -r`（大部分是底座镜像自带的包）。

### 1.2 内部 GPU 容器

```bash
bash scripts/create_env.sh
```

脚本用 base Python 建 venv（`--system-site-packages` 继承底座的 torch/flash_attn），只装 transformers==4.57.3 等少量包，写 .pth，然后跑验收门（模块 import、精确版本、源码位置、GPU 拓扑）。默认按 8 卡校验，单卡机器先 `FLASHVID_EXPECTED_GPU_COUNT=1`；venv 位置用 `FLASHVID_VENV_PATH` 改。

### 1.3 FlashVID 源码（两条路径都需要）

```bash
# 先按第 5 节的路径替换表把 scripts/fetch_code.sh:5 的 PROJECT_ROOT 改成你的 $ROOT
bash scripts/fetch_code.sh    # 拉取 pinned commit 到 code/sources/FlashVID-<sha>
ln -s "$ROOT/code/sources/FlashVID-983cce6e30d7a8012442bfc7557d3afa61b3572d" "$ROOT/code/FlashVID"
```

然后**必须**让 Python 能 import 到它（等价于 create_env.sh 里的 .pth）：

```bash
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
echo "$ROOT/code/FlashVID" > "$SITE/flashvid_project_source.pth"
# 或者每个会话 export PYTHONPATH="$ROOT/code/FlashVID"
```

不要 `pip install -e code/FlashVID`：它的 pyproject 会从 PyPI 拉 lmms-eval、可能重装 torch，破坏版本对齐。lmms_eval 不需要装（stage1 管线不用它）。

## 2. 模型准备

两个仓库，revision 都精确 pin 在 `scripts/adts_harness.py:65-68`：

| 模型 | repo | pinned revision |
|---|---|---|
| 主模型 | `lmms-lab/LLaVA-Video-7B-Qwen2` | `013210b3aff822f1558b166d39c1046dd109520f` |
| 视觉塔 | `google/siglip-so400m-patch14-384` | `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` |

`scripts/dl_weights.sh` 可以用（hf-mirror、断点续传、失败重试 40 次），但它下的是 main、不钉 revision，跨时间不保证对齐。推荐钉着 revision 下，并且**手工补 refs/main**：

```bash
export HF_ENDPOINT=https://hf-mirror.com    # 能直连 huggingface.co 的话可不设
export HF_HOME="$ROOT/weights"
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download('lmms-lab/LLaVA-Video-7B-Qwen2',
                  revision='013210b3aff822f1558b166d39c1046dd109520f')
snapshot_download('google/siglip-so400m-patch14-384',
                  revision='9fdffc58afc957d1a03a25b10dba0329ab15c2a3')
EOF

# hf_hub 按 sha 下载不会写 refs/main，而 harness 硬性校验 refs/main == pin，所以要手工补：
cd "$ROOT/weights/hub"
mkdir -p models--lmms-lab--LLaVA-Video-7B-Qwen2/refs models--google--siglip-so400m-patch14-384/refs
echo -n 013210b3aff822f1558b166d39c1046dd109520f > models--lmms-lab--LLaVA-Video-7B-Qwen2/refs/main
echo -n 9fdffc58afc957d1a03a25b10dba0329ab15c2a3 > models--google--siglip-so400m-patch14-384/refs/main
```

权重就位后建议 `export HF_HUB_OFFLINE=1`：harness 全程离线解析本地快照，避免 transformers 联网时对 HF 的 HEAD 请求卡顿。

如果权重放在别的位置，可用环境变量 `FLASHVID_MODEL_SNAPSHOT` / `FLASHVID_VISION_TOWER_SNAPSHOT` 直接指向快照目录（跳过 refs 校验）。注意**只对 stage1 管线有效**，`scripts/v4/` 的 runtime 会明确拒绝这两个变量。

权重就位后先跑 `python scripts/stage0_assertions.py`，它会断言模型、源码、数据的基本一致性。

## 3. 样本与数据

样本 = Video-MME test 划分里 `duration=="short"` 的**全部** 300 个 videoID（排序去重后写入 `data/short_subset.json`）。这是全量普查，不是随机抽样，所以"对齐"的含义是**字节相同**，不是"重新选一遍"。

stage1 主流程只读 `short_subset.json` 里的 `short_video_ids`。parquet（QA 元数据，421K）是这份清单的来源，主要用于再校验：用 `scripts/inspect_parquet.py` 从 parquet 重新生成清单，应当得到完全相同的 300 个 ID。

```bash
# (a) 下载 Video-MME 的 test parquet
export HF_ENDPOINT=https://hf-mirror.com
python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download('lmms-lab/Video-MME', 'videomme/test-00000-of-00001.parquet',
                repo_type='dataset',
                revision='ead1408f75b618502df9a1d8e0950166bf0a2a0b')
EOF

# (b) 把 data/short_subset.json 的 "parquet" 字段改成你机器上 parquet 的实际路径

# (c) 抽视频：先按第 5 节路径替换表改 scripts/extract_videomme.py:30 的 W
python scripts/extract_videomme.py index     # 建/缓存 20 个 zip 卷的中央目录索引
python scripts/extract_videomme.py extract   # HTTP Range 只拉 300 个成员（~3G），断点续传
```

抽取完成后做字节级校验：`data/videomme_short_manifest.json` 记录了每个视频的字节数和 CRC32。跑 stage1 时 `scripts/data_manifest.py` 的 `require_video_files()` 也会强制 `data/videos_short/` 与清单**精确一致**——缺一个、多一个都直接报错。

## 4. 抽帧：为什么仓库里没有帧文件

帧不落盘。`scripts/adts_harness.py:436-443` 的 `load_video_frames()` 用 decord 打开视频，`np.linspace(0, len(vr)-1, 64, dtype=int)` 均匀取 64 帧，直接进 SigLIP 384×384（BICUBIC、各向异性 resize）预处理，没有随机性。

所以**视频文件字节一致（上面的 CRC 校验）+ decord 版本一致（0.6.0）= 抽到的帧是同一批**，不需要同步任何帧文件。

## 5. 跑评测

先做一次性的路径替换。脚本里硬编码了原机器的绝对路径 `/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial`，把它全部替换成你的 `$ROOT`。一条 sed 搞定（在仓库根执行）：

```bash
OLD=/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial
NEW=/path/to/your/token-pruning-for-efficient-MLLM
grep -rl "$OLD" scripts tests | xargs sed -i "s|$OLD|$NEW|g"
```

涉及的文件（行号供核对）：

| 文件 | 位置 |
|---|---|
| `scripts/stage1_collect.py` | 47、69 |
| `scripts/stage1_analyze.py` | 27 |
| `scripts/stage1_figures.py` | 23 |
| `scripts/story_figures.py` | 24 |
| `scripts/stage0_assertions.py` | 19、34 |
| `scripts/extract_videomme.py` | 30 |
| `scripts/inspect_parquet.py` | 12 |
| `scripts/create_env.sh` | 5（另有 6–7 行的 `/root/miniconda3`、`/root/flashvid_venv`，走路径 B 才需要改） |
| `scripts/fetch_code.sh` | 5 |
| `scripts/dl_weights.sh` | 5（11 行的 `/root/miniconda3/bin/python` 同上） |
| `scripts/stage2_*.py` | 已作废，可不管 |
| `tests/v4/build_d2_fixtures.py` | 16（v4 工具，不跑可不管） |

`scripts/adts_harness.py`、`scripts/data_manifest.py`、`scripts/v4/*` 用相对定位，不用改。`stage1_figures.py:20`、`story_figures.py:68` 里的 `sys.path.insert(0, "/root/FlashVID")` 是历史残留死路径，在你机器上无害（不存在的 sys.path 条目会被跳过），可忽略或删掉；llava 的 import 靠 1.3 节的 .pth 解决。

然后按顺序跑：

```bash
python scripts/stage0_assertions.py                                          # 门禁断言
python scripts/stage1_collect.py --shard 0 --nshards 1 --device cuda:0       # 建议 --limit 5 先试跑
python scripts/stage1_analyze.py                                             # 输出 results/stage1_analysis.json
```

stage1_collect 用 append-only journal 断点续跑，中断后重跑同一命令会自动跳过已完成项；要加速就 `--shard i --nshards N` 分片多进程。30 个合成对照视频由脚本自己生成（seed 1000+i），不需要额外数据。

## 6. 可视化

```bash
python scripts/stage1_figures.py --frame 32 --n-examples 5    # results/figures/fig1..fig10
python scripts/story_figures.py                               # results/figures/story/
```

两步有顺序依赖：story_figures 读 stage1_figures 写出的 `results/figures/example_videos.json`。story_figures 需要中文字体：`sudo apt install fonts-wqy-zenhei`，装完如有旧缓存 `rm -rf ~/.cache/matplotlib`。stage1_figures 没有中文，不需要字体。

## 7. 结果对齐与红线

**对齐方法**：比 `runs/stage1/{video_id}__{arm}.npz` 里的 `keep_*` 数组（选择 mask）。同硬件 + 同代码 + 同权重 revision + 同视频文件下，选择 mask 和分析 p 值理论上逐位一致。**跨 GPU 型号/驱动/CUDA 版本不保证逐位**——fp16 kernel 差异可能翻转边缘 argmax。个别帧对不上时，先查 batch/GPU 差异，再怀疑代码。

**红线（勿踩）**：

- `scripts/stage2_*.py` 保留只作历史参考，其产物与分析已作废，不要用于新结论。
- `scripts/v4/` 是实施中的新管线，**在本仓库内不可运行**：依赖未上传的 `docs/implementation/v4_20260905/gate_status.json`、`d2_i6_selection_fixture.json`、`docs/reviews/v3_v4_20260905/v4_candidate_a_r4.md` 以及 runs 旧产物。放进来是供代码审阅的。
- `scripts/llm_gateway.py`、`scripts/lit_verify_driver.py`（内部文献核验工具）和 `tests/` 下 4 个审计目录不在仓库里；`tests/v4/build_*.py` 引用了未上传的路径，pytest 不会收集它们，保留只为完整性。
- `scripts/gpu_throughput_benchmark.py` 是内部吞吐工具，非复现必需；跑它需要 `--environment-manifest` 传入环境清单。

**跑测试**：在仓库根执行 `pytest tests/`，应收集到 138 项。不要单独跑 `pytest tests/v4/`——v4 包的导入依赖同批次其他测试先把 `scripts/` 插进 sys.path。

## 8. 常见坑

| 症状 | 原因与解法 |
|---|---|
| 坐标系断言失败（14×14 而不是 13×13） | checkpoint 默认 `mm_spatial_pool_mode=bilinear`；必须按 harness 覆写成 `average`/`frame`（169 格坐标系）。别绕过断言。 |
| `cached main ref ... is not pinned` | refs/main 缺失或与 pin 不一致，按第 2 节手工补 |
| `missing exact cached snapshot for ...` | 快照目录不存在或位置不对；或用 SNAPSHOT 环境变量显式指定（仅 stage1 有效） |
| figures 脚本 `ImportError: llava` | 1.3 节的 .pth 没做 |
| stage1_collect 报 video assets ID mismatch | `data/videos_short/` 与清单不一致，重新对照 CRC |
| v4 报缺 gate_status.json | 预期行为，见第 7 节 |
| story 图中文显示成方框 | 装 fonts-wqy-zenhei，清 matplotlib 缓存 |

## 9. 与原始实验目录的差异

本仓库文件来自原始实验目录的副本，除以下几处外与原文件一致：

1. **脱敏**（改的是上传副本，原文件未动）：
   - `env/requirements.lock`、`env/create-env_20260905.pip-freeze.txt`：删除 1 行内部源包（mcore_adapter），3 行内部 `file://` 路径替换为 `file:///REDACTED_INTERNAL/PATH` 占位；
   - `scripts/create_env.sh:10` 的 qscode 容器名、`scripts/gpu_throughput_benchmark.py:32` 的 pod 名替换为 REDACTED 占位。
2. **重命名**：env 清单原文件名带 pod 信息（`create-env_qs-249918-...`），改为 `create-env_20260905.*`；另一份 20260903 的 freeze 与 20260905 字节级相同（pip-check 仅行序不同），故只保留一对。
3. **未包含**：`weights/`、`data/videos_short/`、`code/`、`runs/`、`results/`、`docs/`、`report/`（大文件或研究文档），以及 `scripts/llm_gateway.py`、`scripts/lit_verify_driver.py`、`tests/test_llm_gateway.py`、`tests/test_lit_verify_driver.py`、`tests/` 四个审计目录。

对不上或发现缺东西，直接联系仓库 owner。
