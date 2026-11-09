# MetaCompress 复现准备：数据、权重与环境

本文记录基于论文 [`ThirdParty/papers/MetaCompass.pdf`](../../ThirdParty/papers/MetaCompass.pdf)、当前仓库资产和 [`ThirdParty/MetaCompress`](../../ThirdParty/MetaCompress) 实现所做的复现审计。

文中统一使用：

```bash
ROOT=/data0/lyx/token-pruning-for-efficient-MLLM
```

审计时间为 **2026-09-22**。后续若论文作者发布代码、处理后的多轮数据或预训练权重，应以作者版本为准，并重新核对本文结论。

---

## 迁移记录（2026-09-28）

仓库已从旧机器迁到本机。第 5 节仍是 2026-09-24 在 B200 / CUDA 13 上的审计，不要把那里的环境当成当前可运行环境。

| 项目 | 本机 |
|---|---|
| `ROOT` | `/root/Test/Lyx/token-pruning-for-efficient-MLLM` |
| 解释器 | `/root/Test/envs/metacompress/bin/python` |
| GPU / 驱动 | 8× RTX 4090，驱动 535.129.03，CUDA 12.2 |
| PyTorch | `2.1.2+cu121`，`torchvision==0.16.2+cu121`，与 LLaVA `pyproject.toml` 锁定版本一致 |
| 安装方式 | `pip install -e ThirdParty/LLaVA --no-deps` 与 `pip install -e ThirdParty/MetaCompress --no-deps` |

不要在本机安装旧文档中的 PyTorch `2.14.0+cu130`。那个轮子面向 B200，当前驱动无法使用。根分区 overlay 已满，conda 包缓存、pip 缓存和虚拟环境都放在 `/root/Test`，不要写入 `/opt/conda`。

`scripts/metacompress/run_train_eval.sh` 的默认 `PYTHON` 已指向上述解释器；解释器不存在时脚本会直接退出。`ROOT` 仍由脚本位置推导，数据路径不用改。MT-GQA 训练会话、16,488 张训练图、4,061 个测试会话，以及 `haotian-from-hf` 和 CLIP 权重在迁移后仍然齐全。正式训练前先选一块空闲 24 GB 卡，不要使用默认的 `GPU=0`，除非当时它是空闲的。

---

## 0. 结论

| 项目 | 当前状态 | 能否用于论文复现 |
|---|---|---|
| `weights/videomme` | 900 个视频，和本地 2700 条标准 Video-MME 元数据完整对应 | 可用于标准 Video-MME 扩展实验，**不能替代论文的 MT-Video-MME** |
| `weights/hub/datasets--lmms-lab--Video-MME` | 标准 Video-MME parquet 完整 | 同上 |
| `weights/hub/models--liuhaotian--llava-v1.5-7b` | 活跃 snapshot 完整 | 可作为论文模型列表中的一个起点 |
| 其他本地 VLM 权重 | Qwen2.5-VL、LLaVA-Video、LLaVA-OneVision 等 | 可做扩展实验，但不是论文主表所用模型 |
| MT-VQA-v2 | 未找到作者处理后的多轮版本 | 对齐 MT-VQA-v2 分数时需要；**对齐 MT-GQA 分数不需要** |
| MT-GQA | 未找到作者处理后的多轮文件；GQA `train-balanced` 问题、20K GQA-only 会话和 16,488 张所需图像已按公开原始数据重建 | 可启动数据接入和训练适配工作，并用论文 Base / Sample 分数校验协议；仍不是作者官方 20K 清单 |
| ConvBench | 官方 Hugging Face 数据已放在 `weights/datasets/convbench`：`ConvBench.xlsx` 有 578 条对话，图片目录 2,070 个文件，缺 1 张 | 可开始 Base / Sample 链路；评分裁判和 MT-VQA-v2 训练权重仍缺，不能直接对表 1 |
| MT-Video-MME | 未找到 | 论文视频实验所需，标准 Video-MME 不能直接替代 |
| MetaCompress 预训练 checkpoint | 未找到 | 当前实现只能随机初始化或自行训练，不能直接复现论文精度 |
| `metacompress` Python 包 | 专用 conda 环境 `/opt/miniconda3/envs/metacompress` 已创建；LLaVA 已是该环境的 editable install，压缩包本身尚未 `pip install` | 按第 5 节做 `--no-deps` editable install 后即可持久使用 |

因此，当前仓库可以验证 MetaCompress 的核心算法和接口，也具备 LLaVA-1.5-7B、MT-GQA 候选测试集以及 GQA-only 训练图像的实验基础。MT-GQA 的测试会话数和训练子集已经可以按当前重建规则生成，但作者没有公布问题顺序、对话历史、官方训练子集和 MetaCompress checkpoint，因而仍不能仅凭样本数宣称严格复现。本文后续把**对齐论文中 LLaVA-1.5-7B 的 MT-GQA 分数**作为当前主目标，并通过未压缩 Base、确定性 Sample、最后 MetaCompress 的三级门禁逐步排除协议错误。

---

## 1. 论文复现目标

论文中的复现对象不只是一个 token 压缩层，还包括以下完整实验协议：

1. 在视觉编码器和语言模型之间插入 MetaCompress。
2. 冻结原始 LVLM，仅训练轻量压缩模块。
3. 使用目标 benchmark 对应的多轮训练数据训练；对齐 MT-GQA 时使用 GQA `train-balanced`，不要求 MT-VQA-v2。
4. 在 MT-VQA-v2、MT-GQA 和 ConvBench 上评测多轮对话能力。
5. 以约 90% 的视觉 token 压缩比例比较精度、吞吐量和显存占用。
6. 视频实验使用论文构造的 MT-Video-MME，而不是标准单轮 Video-MME。

论文报告的主要模型包括：

- LLaVA-1.5-7B / 13B
- LLaVA-NeXT-7B / 13B
- InternLM-XComposer-2.5-7B

论文给出的典型训练设置包括约 20K 训练样本、2 个 epoch、总 batch size 36、SGD、学习率 `1e-3`、梯度裁剪 `1e-2`、`alpha_entropy=1` 和 `alpha_collapse=1`。这些超参数可以照用，但它们不能弥补训练样本选择和多轮转换规则的缺失。

论文表 7 消除了“必须使用 MT-VQA-v2 才能验证 MT-GQA”的歧义：LLaVA-1.5-7B 在 **MT-GQA 训练、MT-GQA 评测**时为 `58.43`，而从 MT-VQA-v2 迁移到 MT-GQA 时为 `57.45`。因此，当前目标应只依赖 GQA：使用 `train-balanced` 构造训练会话，使用 `testdev-balanced` 构造 4061 个测试会话。MT-VQA-v2 只在复现跨数据集迁移或论文其他列时才需要。

### 1.1 必须区分的三个复现等级

| 等级 | 目标 | 当前可行性 |
|---|---|---|
| 算法复现 | 验证论文公式、token 选择、相似度剪枝、蒸馏损失和训练冻结策略 | 可行 |
| 模型复现 | 在某个本地 LVLM 上训练出可用压缩器，并验证压缩前后输出 | 数据接入和模型适配完成后可行 |
| 数值对齐 | 在候选协议下使 Base、Sample 和 Ours 依次接近论文 MT-GQA 分数 | 可执行，是本文当前目标 |
| 严格复现 | 证明数据顺序、训练样本、实现和 checkpoint 均与作者完全相同 | 当前不可证明，缺少作者产物 |

本文后续所有“可复现”结论均按这个边界解释。

### 1.2 MT-GQA 的目标分数和验收顺序

优先复现本地已有权重的 **LLaVA-1.5-7B**。论文表 1 在 90% reduction rate 下报告：

| 方法 | Acc1 | Acc2 | Acc3 | Avg |
|---|---:|---:|---:|---:|
| Base | 61.76 | 64.07 | 65.35 | 63.73 |
| Random，3 个随机种子均值 | 54.60 | 57.07 | 59.31 | 56.99 |
| Sample | 55.06 | 57.89 | 59.74 | 57.56 |
| MetaCompress / Ours | **55.95** | **58.71** | **60.64** | **58.43** |

论文表 4 在 70% reduction rate 下的补充目标是 `58.62 / 60.96 / 63.64 / 61.07`。第一阶段只对齐主表的 90% 设置，70% 设置用于排除偶然匹配。

验收必须按以下顺序进行：

1. **Base 门禁**：不启用任何压缩，先对齐 `61.76 / 64.07 / 65.35`。失败说明会话构建、prompt、历史策略、生成配置、模型 revision 或评分器尚未对齐，此时不得训练 MetaCompress。
2. **Sample 门禁**：用确定性等距采样对齐 `55.06 / 57.89 / 59.74`。Base 已通过而 Sample 失败，通常说明压缩位置、保留 token 数或特殊 token 处理错误。
3. **MetaCompress 门禁**：前两项通过后，再训练并对齐 `55.95 / 58.71 / 60.64`。只有此时分数差异才主要指向压缩模块或训练协议。
4. **70% 交叉检查**：同一 checkpoint 或按论文设置训练的对应 checkpoint 应复现表 4 的趋势，防止只针对 90% 设置调参。

建议预先登记工程容差：Base 和 Sample 的 `Avg` 误差不超过 `0.3` 个百分点、任一轮不超过 `0.5`；MetaCompress 的三个种子均值采用相同容差。实际分数必须完整报告，不能只报告“通过”。这个容差只用于工程验收，不等于证明与官方实现逐样本一致。

---

## 2. `weights/` 资产审计

### 2.1 标准 Video-MME 数据完整

本地元数据：

```text
weights/hub/datasets--lmms-lab--Video-MME/
└── snapshots/206061640ec3ed96edcb957c06cea3691b24220d/
    └── videomme/test-00000-of-00001.parquet
```

审计结果：

- parquet 大小：405,167 bytes
- SHA-256：`7fffab8ed38ecc2f9f0eca4c44d8a11636f0ee96116ede83580c8b9ae0faf986`
- 题目数：2700
- 唯一视频数：900
- `short` / `medium` / `long` 各 900 题
- 本地视频目录：`weights/videomme/data`
- MP4 数量：900
- 元数据引用和视频文件一一对应，缺失 0、额外 0

这说明本地**标准 Video-MME** 已经齐全，无需重复下载。

但论文视频实验使用的是作者将 Video-MME 改造成 500 个多轮对话后的 **MT-Video-MME**。本地 2700 条单轮选择题没有论文所用的会话分组、轮次顺序和 prompt 模板，因此不能直接复现论文对应表格。

### 2.2 模型权重状态

本地 Hugging Face cache 中以下模型的活跃 snapshot 均有配置文件，索引引用的权重 shard 未发现缺失：

| 模型 | 本地 revision | 与论文的关系 |
|---|---|---|
| `liuhaotian/llava-v1.5-7b` | `6f798d78731e34b2fa643748a3c55cacc7afc5e2` | 现代 Transformers 格式；论文模型之一，但不用于本文旧版 LLaVA builder 命令 |
| `liuhaotian/llava-v1.5-7b` | `haotian-from-hf` | 本地旧版 `llava_llama` 格式；本文脚本的兼容输入 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `78d6bca5309f4e513181315c9ad3e68db4243a73` | 非论文主模型，可做扩展实验 |
| `lmms-lab/LLaVA-Video-7B-Qwen2` | `86f52acccbac267e034a120b97b04d45c3490a4c` | 非论文主模型，可做视频扩展 |
| `lmms-lab/llava-onevision-qwen2-7b-ov` | `39a0447bdada76be2e21ed484f4aef23337022e5` | 非论文主模型，可做扩展实验 |
| `google/siglip-so400m-patch14-384` | `ff792b42c7709e0444bffb50bd2faf66216b9dc8` | 视觉编码器资产 |
| `openai/clip-vit-large-patch14-336` | `96971839baa62a3493f8f1465dbc0a59d1b135ca` | LLaVA-1.5 视觉编码器资产 |

本地未发现论文其余主模型的完整对应权重：

- LLaVA-1.5-13B
- LLaVA-NeXT-7B / 13B
- InternLM-XComposer-2.5-7B

建议先以本地 LLaVA-1.5-7B 完成端到端链路，再决定是否下载其他大模型。否则数据协议尚未对齐时下载数十 GB 权重不会推进数值复现。

### 2.3 缺失的数据和权重

`weights/` 下未找到以下内容：

- MT-VQA-v2
- 作者处理后的 MT-GQA 会话文件；当前只有 GQA-only 重建会话，原始 `testdev_balanced` 已存在于 `ThirdParty/LLaVA`
- MT-Video-MME
- 论文使用的约 20K 样本清单或随机种子
- 论文训练好的 MetaCompress checkpoint

当前 GQA-only 训练重建已经可用：`train_balanced_questions.json`、20,000 个三轮会话、60,000 个问题和 16,488 张训练图像均已准备完成。它不能替代作者未公开的 MT-GQA/MT-VQA-v2 联合 20K 数据，也不能消除官方样本清单、转换脚本和 checkpoint 缺失带来的严格复现限制。

---

## 3. MT-GQA 本地数据与仍未公开的协议

### 3.1 已确认可重建的部分

本地已经包含 LLaVA 标准 GQA 测试资产：

```text
ThirdParty/LLaVA/playground/data/eval/gqa/
├── llava_gqa_testdev_balanced.jsonl
└── data/
    ├── testdev_balanced_questions.json
    ├── images/*.jpg
    └── eval/eval.py
```

审计结果：

- `testdev_balanced_questions.json` 有 12,578 个问题，SHA-256 为 `14039069c0b3c797c7aa9bcd5f4c2aa4b5976e02c0b6773e1a584d942a03a318`；
- 问题属于 398 张图像，398 张对应图片均存在；
- 对每张图的问题数取 `floor(count / 3)` 后求和，**恰好为 4061**；
- 4061 个三轮会话使用 12,183 个问题，每张图末尾不足三问的余数共 395 个问题；
- `llava_gqa_testdev_balanced.jsonl` 有 12,578 行，SHA-256 为 `7905fa531896163975f9242cd2ad9c69f612bda4e4ce29fcf81fe3078ea711ca`。

这与论文“MT-GQA 从 GQA testdev-balanced 构造，共 4061 个三轮对话”的描述精确吻合，足以确定主体规则是**同图问题每三个组成一个会话，并丢弃每张图不足三个的尾部**。但问题的组内顺序和对话执行方式仍需通过 Base 分数校验。

### 3.2 仍未公开的部分

论文中的 MT-VQA-v2 和 MT-GQA 是由原始 VQAv2 / GQA 转换而来的多轮数据。即使下载了原始数据，仍至少需要确定：

- 同图问题采用源文件顺序、question ID 顺序还是随机顺序；
- 多轮历史使用模型生成答案、标准答案，还是每轮独立评测；
- 图像只在首轮输入还是每轮重复输入；
- 是否过滤重复、短答案或特定题型；
- 训练、验证和测试划分如何继承；
- 20K 训练子集的确切样本 ID 和抽样随机种子；
- 系统 prompt、用户 prompt 和答案格式；
- ConvBench 和 MT-Video-MME 的转换及评分脚本。

这些细节会直接影响训练分布和最终分数。因此：

> 下载 VQAv2、GQA 和标准 Video-MME 只能准备原始材料；在作者发布处理脚本或转换后数据之前，不能声称严格复现了论文结果。

对于本文的 MT-GQA 目标，上述结论应更精确地理解为：测试集的**成员数量和同图三问分组规则**已有强证据，训练侧也已准备一个可审计的 GQA-only 20K 子集及其全部图像；但问题顺序、历史答案来源、图片是否只在首轮输入、prompt 和生成设置尚未公开，20K 子集也不是作者官方清单。

截至审计日期，论文官方仓库 [`MArSha1147/MetaCompress`](https://github.com/MArSha1147/MetaCompress) 的 `main` HEAD 为 `70a620eaec3d03368bfff9b85c2e278871c595ea`，README 仍表示实现将后续发布。当前 [`ThirdParty/MetaCompress`](../../ThirdParty/MetaCompress) 是根据论文补写的本地实现，不是作者官方代码。

---

## 4. 数据应该如何下载

### 4.1 优先等待或获取作者发布产物

精确复现应优先获取以下作者产物：

1. MT-VQA-v2 和 MT-GQA 的处理后数据或确定性构建脚本；
2. 20K 训练子集的样本 ID；
3. ConvBench 和 MT-Video-MME 的数据及评分脚本；
4. 官方 MetaCompress checkpoint；
5. 模型适配代码和完整训练配置。

可以只读检查远端状态：

```bash
git -C "$ROOT/ThirdParty/MetaCompress" ls-remote origin HEAD
```

不要直接在当前目录执行 `git pull` 覆盖本地补写实现。作者代码发布后，建议先克隆到独立目录进行差异核对：

```bash
git clone https://github.com/MArSha1147/MetaCompress.git \
  "$ROOT/ThirdParty/MetaCompress-official"
```

### 4.2 下载 VQAv2 原始数据

本节不是当前 MT-GQA 分数对齐的前置步骤。只有继续复现 MT-VQA-v2、ConvBench 迁移结果或论文的联合训练设置时才需要下载。

官方入口：<https://visualqa.org/download.html>

至少需要：

- VQAv2 train questions
- VQAv2 validation questions
- VQAv2 train annotations
- VQAv2 validation annotations
- COCO `train2014` images
- COCO `val2014` images

建议目录：

```bash
mkdir -p "$ROOT/weights/datasets/vqav2/raw"
cd "$ROOT/weights/datasets/vqav2/raw"

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget https://s.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
```

下载 URL 若有变化，以 VQAv2 官方下载页为准。解压后应保留原始 JSON 和图像目录，不要直接改写原文件；多轮转换结果应写入单独的 `processed/`，并记录转换脚本版本和随机种子。

### 4.3 下载 GQA 原始数据

官方入口：<https://cs.stanford.edu/people/dorarad/gqa/download.html>

当前仓库已经具备 `testdev_balanced` 的 12,578 个问题、398 张所需图像和官方 evaluator，**不需要重新下载测试部分**。训练侧的 GQA-only 第一版重建也已经完成：`train_balanced_questions.json`、20,000 个三轮会话、60,000 个问题和全部 16,488 张所需训练图像均已落盘。仍需补充的不是原始 GQA 文件，而是作者未公开的官方多轮转换结果、20K 样本清单和 checkpoint。

建议目录：

```bash
mkdir -p "$ROOT/weights/datasets/gqa/raw"
```

GQA 的大文件链接由官方页面和云盘维护，建议从页面选择 `images`、`questions` 和 evaluation 资源，而不是在文档中固定可能失效的云盘 ID。下载后必须先确认 split 名称、问题数、图像覆盖率和文件哈希，再构建训练会话；不得从本地 `testdev_balanced` 抽取训练样本。

当前已完成 Questions 包的最小解压：

```text
weights/datasets/gqa/raw/questions1.2.zip
weights/datasets/gqa/raw/questions/train_balanced_questions.json
```

`train_balanced_questions.json` 的 SHA-256 为
`c600b8301c67ee3ac1b4b9af8a35f776589f045f245fd5b15c50c97fa244d6b`，包含 943,000 个问题。构造脚本为 [`scripts/metacompress/build_mt_gqa.py`](../../scripts/metacompress/build_mt_gqa.py)，执行：

```bash
python3 scripts/metacompress/build_mt_gqa.py \
  --questions weights/datasets/gqa/raw/questions/train_balanced_questions.json \
  --output-dir weights/datasets/gqa/processed/mt_gqa \
  --num-dialogs 20000 \
  --seed 42
```

当前生成的是 **GQA-only 非官方重建训练子集**，不是作者未公开的联合 MT-VQA-v2/MT-GQA 20K 清单。输出目录包含 20,000 个三轮会话、60,000 个问题和 16,488 个图像 ID：

```text
weights/datasets/gqa/processed/mt_gqa/
├── mt_gqa_train_balanced_20000.jsonl
├── mt_gqa_train_balanced_20000.json
├── image_ids.txt
├── question_ids.txt
├── missing_image_ids.txt
├── images/*.jpg                 # 16,488 个训练图像
└── manifest.json
```

训练图像已从官方 `images.zip` 选择性下载完成。该服务器支持 HTTP Range，且 ZIP 中每张图片都是独立的 deflate 成员，因此没有下载 20.3GB 全压缩包，而是读取 14,506,043 字节的中央目录后只抓取 `image_ids.txt` 中的 16,488 张图片。选择性下载脚本为 [`scripts/metacompress/fetch_gqa_images.py`](../../scripts/metacompress/fetch_gqa_images.py)，中央目录索引保存在 `weights/datasets/gqa/raw/images.zip.central-directory.bin`。执行方式（已有文件会自动跳过）为：

```bash
/opt/miniconda3/envs/metacompress/bin/python \
  scripts/metacompress/fetch_gqa_images.py \
  --archive-url https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip \
  --central-directory weights/datasets/gqa/raw/images.zip.central-directory.bin \
  --image-ids weights/datasets/gqa/processed/mt_gqa/image_ids.txt \
  --output weights/datasets/gqa/processed/mt_gqa/images \
  --manifest weights/datasets/gqa/processed/mt_gqa/manifest.json \
  --workers 8
```

实际完成结果已经写入 [`manifest.json`](../../weights/datasets/gqa/processed/mt_gqa/manifest.json)：

```text
selected_dialog_count       = 20,000
selected_question_count     = 60,000
selected_image_count        = 16,488
available_image_count       = 16,488
missing_image_count         = 0
training_ready              = true
missing_image_ids.txt       = empty
extracted JPEG bytes        = 2,333,009,319
```

图片输出目录为 `weights/datasets/gqa/processed/mt_gqa/images/`。这里下载的是公开 GQA 原始图像的子集，不是作者未公开的 MT-GQA 处理后数据；Scene Graph 包对 MetaCompress 的训练和 MT-GQA accuracy 不是必需的。可用以下命令复核文件名、数量和 JPEG 头部：

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("weights/datasets/gqa/processed/mt_gqa")
ids = set((root / "image_ids.txt").read_text().split())
images = {p.stem: p for p in (root / "images").glob("*.jpg")}
assert ids == set(images)
assert len(ids) == 16_488
assert all(path.open("rb").read(3) == b"\xff\xd8\xff" for path in images.values())
assert (root / "missing_image_ids.txt").read_text() == ""
print(f"validated {len(images)} GQA training images")
PY
```

### 4.4 ConvBench

ConvBench 不是 MetaCompress 作者构造的数据。它是 Liu 等人发布的多轮视觉对话评测，论文引用为 arXiv:2403.20194，共 577 段对话。官方仓库 [shirlyliu64/ConvBench](https://github.com/shirlyliu64/ConvBench) 的 Dataset 链接是 <https://huggingface.co/datasets/liushuo12345/ConvBench>。2026-09-29 已下载并放到：

```text
weights/datasets/convbench/
├── ConvBench.xlsx
└── visit_bench_images/    # 2,070 个文件，约 718 MB
```

`ConvBench.xlsx` 的工作表 `multi_turn_benchmark` 有 578 个 ID（1 到 578），列包括三轮 instruction、instruction category、reference answer，以及 `image_id`。论文写的是 577 段，本地多 1 条，使用前要保留这 578 条并在结果里写明条数。ID 131 引用的 `5_tested_skill_exercise_b85365f832794357.png` 不在图片目录中，其余 577 张图都在。Hugging Face 页面上的 240 行预览只是图片文件夹的自动视图，不是这 578 段对话。

MT-Video-MME 仍然没有。它是作者把 Video-MME 改成多轮后的数据，标准 Video-MME 不能替代。

### 4.5 标准 Video-MME

当前机器不需要重复下载。若在新机器上只需重新下载 parquet 元数据，可运行：

```bash
/opt/miniconda3/envs/metacompress/bin/python "$ROOT/utils/download_dataset.py"
```

该脚本只下载 parquet，不下载约 100 GB 的视频文件。完整数据集入口：<https://huggingface.co/datasets/lmms-lab/Video-MME>。本仓库的视频准备和核验流程另见 [`02-reproduction-runbook.md`](02-reproduction-runbook.md)。

---

## 5. 环境状态与安装方式

### 5.1 当前环境

复现统一使用专用环境，不要再用 `llava` 环境跑 MetaCompress：

```text
/opt/miniconda3/envs/metacompress
```

对应解释器：

```text
/opt/miniconda3/envs/metacompress/bin/python
```

该环境已在 **2026-09-24** 创建。已核对的核心版本：

| 包 | 当前版本 |
|---|---|
| Python | 3.10.21 |
| PyTorch | 2.14.0+cu130 |
| torchvision | 0.29.0+cu130 |
| transformers | 4.37.2 |
| accelerate | 0.21.0 |
| llava | 1.2.2.post1，editable 指向 `ThirdParty/LLaVA` |
| metacompress | **尚未 pip 安装**；安装前只能靠 `PYTHONPATH` 或脚本里的 `sys.path` |

`scripts/metacompress/*.py` 仍会把 `ThirdParty/LLaVA` 和 `ThirdParty/MetaCompress` 插到 `sys.path`，因此不安装也能跑。正式训练和评测应做下面的持久化安装，避免依赖调用方是否 export 了 `PYTHONPATH`。

不要在这个环境里执行：

```bash
pip install -e "$ROOT/ThirdParty/LLaVA[train]"
```

LLaVA 已经是 editable install。`[train]` extra 会按旧版依赖声明重装 PyTorch / torchvision，可能破坏当前为 B200 / CUDA 13 准备的 `2.14.0+cu130`。

### 5.2 持久化安装 MetaCompress

这是推荐方式。只安装本地压缩包，**不要让 pip 解析它的 `torch>=2.0` 依赖**：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate metacompress
cd /data0/lyx/token-pruning-for-efficient-MLLM

python -m pip install --no-deps -e ThirdParty/MetaCompress

python -c "import metacompress; print(metacompress.__file__)"
pip show metacompress
```

预期输出：

```text
.../ThirdParty/MetaCompress/metacompress/__init__.py
Name: metacompress
Version: 0.1.0
Editable project location: /data0/lyx/token-pruning-for-efficient-MLLM/ThirdParty/MetaCompress
```

`--no-deps` 是必须的：`pyproject.toml` 只声明 `torch>=2.0`，pip 一旦解析依赖，可能改写已经装好的 CUDA 13 轮子。压缩模块本身只使用现有 PyTorch。

安装后新开的 shell 里 `import metacompress` 即可，不必再设 `PYTHONPATH`。源码改动会立刻生效，因为这是 editable install。

若以后作者发布官方仓库，先克隆到 `ThirdParty/MetaCompress-official`，再决定是否把 editable 路径改过去；不要直接 `git pull` 覆盖当前补写实现。

### 5.3 不安装、只用 PYTHONPATH

仅用于临时检查，不是正式实验路径：

```bash
export PYTHONPATH="$ROOT/ThirdParty/MetaCompress:$ROOT/ThirdParty/LLaVA:${PYTHONPATH:-}"
/opt/miniconda3/envs/metacompress/bin/python -c \
  "from metacompress import MetaCompress; print(MetaCompress)"
```

### 5.4 可选依赖

当前训练和 MT-GQA 评测脚本不需要 Hugging Face `datasets` 或 DeepSpeed。`accelerate` 已在环境中。若以后改用 HF Datasets 组织数据：

```bash
conda activate metacompress
python -m pip install "datasets==2.20.0"
```

DeepSpeed 不是算法必需。若要装，单独选择与 PyTorch 2.14 / CUDA 13 兼容的版本，不要套用旧 LLaVA 锁定文件。

### 5.5 安装后验证

```bash
cd "$ROOT"
conda activate metacompress

python -c "from metacompress import MetaCompress, MetaCompressLoss; print(MetaCompress, MetaCompressLoss)"
python -m unittest discover -s ThirdParty/MetaCompress/tests -v
python -m compileall -q \
  scripts/metacompress ThirdParty/MetaCompress/metacompress ThirdParty/MetaCompress/tests
```

验证通过后，后续命令一律使用：

```bash
PYTHON=/opt/miniconda3/envs/metacompress/bin/python
# 或
conda activate metacompress
```

---

## 6. 当前实现如何使用

完整 API 和示例见 [`ThirdParty/MetaCompress/README.md`](../../ThirdParty/MetaCompress/README.md)。核心组件包括：

- `MetaCompress`：根据池化后的 query 和原视觉 token 的 key 生成可学习压缩矩阵；
- `compress_visual_tokens`：在保留模型特定前缀或后缀 token 的情况下压缩视觉特征；
- `MetaCompressLoss`：组合预测分布蒸馏、行熵和 collapse 正则项；
- `distillation_forward` / `optimization_step`：执行冻结 LVLM 的教师-学生前向及压缩器更新；
- `FixedCompressionMatrix` / `fit_fixed_compression_matrix`：实现论文附录中的逐样本固定矩阵分析。

### 6.1 基础压缩调用

```python
import torch
from metacompress import MetaCompress

compressor = MetaCompress(
    input_dim=4096,
    compression_dim=256,
    reduction_rate=0.90,
)

visual_tokens = torch.randn(1, 576, 4096)
compressed = compressor(visual_tokens, target_tokens=58)
print(compressed.shape)  # [1, 58, 4096]
```

`58 / 576` 大约保留 10% 视觉 token，对应约 90% 压缩。这里仅演示 API，不代表论文一定使用 58 个 token：当前实现按 `int(576 * 0.1)` 自动计算时得到 57，而四舍五入得到 58。论文未说明取整规则，正式对齐必须分别记录 `m=57` 和 `m=58` 的 Sample 分数，用第 7 节的门禁确定协议。LLaVA-1.5 的 projector 输出是 576 个 patch token，不应把文本或不存在的 CLS token 计入 `n`。

### 6.2 加载训练好的压缩器

```python
from metacompress import MetaCompress

compressor = MetaCompress.from_pretrained(
    "checkpoints/metacompress",
    map_location="cpu",
)
compressor.eval()
```

训练结束后可用 `compressor.save_pretrained("checkpoints/metacompress")` 保存 `config.json` 和 `pytorch_model.bin`。推理时，压缩器的 `input_dim` 必须与目标 LVLM 经过 vision projector 后的 embedding 维度相同。

当前仓库没有论文 checkpoint。随机初始化的压缩器只能用于形状和接口测试，不能期待获得论文报告的精度。

### 6.3 训练调用

```python
import torch
from metacompress import (
    MetaCompress,
    MetaCompressLoss,
    build_paper_optimizer,
    distillation_forward,
    freeze_module,
    optimization_step,
)

compressor = MetaCompress(
    input_dim=4096,
    compression_dim=256,
    reduction_rate=0.90,
)
freeze_module(lvlm)
compressor.train()
optimizer = build_paper_optimizer(compressor, learning_rate=1e-3)
criterion = MetaCompressLoss(entropy_weight=1.0, collapse_weight=1.0)

result = distillation_forward(
    compressor,
    visual_tokens,
    model_forward,
    criterion=criterion,
    token_mask=response_mask,
)
grad_norm = optimization_step(
    result,
    compressor,
    optimizer,
    max_grad_norm=1e-2,
)

print(result.losses.loss, grad_norm)
```

这里的 `model_forward` 必须由具体 LVLM 适配器提供，并满足当前库定义的接口。它需要：

1. 接收替换后的视觉特征，并在闭包中捕获文本、attention mask 和标签；
2. 对完整视觉 token 和压缩视觉 token 两个分支运行同一个冻结 LVLM；
3. 分别抽取 teacher 和 student 的 assistant response 位置，返回时间维和词表维严格对齐的 logits；
4. 保留文本 prompt 和标签掩码的一致性；
5. 对 LLaVA 中视觉 patch 的换行、分隔符或特殊 token 做显式处理。

当前仓库已经补上 LLaVA-1.5 的最小端到端入口：[`build_mt_gqa_eval.py`](../../scripts/metacompress/build_mt_gqa_eval.py) 构造候选测试会话，[`run_mt_gqa.py`](../../scripts/metacompress/run_mt_gqa.py) 运行 Base/Sample/MetaCompress 多轮生成，[`score_mt_gqa.py`](../../scripts/metacompress/score_mt_gqa.py) 逐轮评分，[`train_llava_metacompress.py`](../../scripts/metacompress/train_llava_metacompress.py) 在冻结 LLaVA 的前提下训练压缩器。它们使用本节的重建协议，不能消除作者未公开的官方会话、20K 清单和 checkpoint 所带来的严格复现限制。

---

## 7. MT-GQA 分数对齐步骤

本节是后续实现和实验的执行规范。对应的 converter、multi-turn evaluator、LLaVA 推理入口和训练适配器已经落在 `scripts/metacompress/`；下面命令可以直接执行。命令默认使用当前仓库的**非官方 GQA-only 重建协议**，所得结果只能称为“在重建协议下的复现尝试”，不能直接宣称官方逐样本复现。

### 7.1 固定实验身份

在产生任何分数前先固定以下内容：

| 项目 | 第一版设置 |
|---|---|
| 基础模型 | `liuhaotian/llava-v1.5-7b` |
| 本地兼容路径 | `weights/hub/models--liuhaotian--llava-v1.5-7b/snapshots/haotian-from-hf` |
| 视觉编码器 | `openai/clip-vit-large-patch14-336`，使用模型自带 image processor |
| 对话模板 | `vicuna_v1` |
| 问题后缀 | `Answer the question using a single word or phrase.` |
| 解码 | greedy，`do_sample=false`，`temperature=0`，`num_beams=1`，`max_new_tokens=128` |
| 数值精度 | 首轮使用 FP16，与论文 A6000 环境相符；另外记录实际 GPU 和 CUDA/PyTorch 版本 |
| 目标 | 表 1，90% reduction rate，MT-GQA 三轮和平均 accuracy |

每个实验目录必须保存完整配置，不能依赖代码默认值。运行期间禁止改模型权重、vision projector、tokenizer 或 conversation template。

`haotian-from-hf` 是本地旧版 LLaVA loader 能识别的 `llava_llama` snapshot。另一个本地 `6f798d...` snapshot 是现代 Transformers `LlavaForConditionalGeneration` 格式，不能直接作为下面旧版 `ThirdParty/LLaVA` builder 的输入；不要在同一组结果中混用两个格式。

下面各命令共享这些 shell 变量；在新终端重新执行本段即可：

```bash
cd "$ROOT"
PYTHON=/opt/miniconda3/envs/metacompress/bin/python
MODEL="$ROOT/weights/hub/models--liuhaotian--llava-v1.5-7b/snapshots/haotian-from-hf"
DIALOGS="$ROOT/weights/datasets/gqa/processed/mt_gqa_testdev/mt_gqa_testdev_balanced_4061.json"
IMAGE_ROOT="$ROOT/ThirdParty/LLaVA/playground/data/eval/gqa/data/images"
RUN_DIR="$ROOT/outputs/metacompress/mt_gqa"
mkdir -p "$RUN_DIR"
```

### 7.2 构造候选 MT-GQA testdev

从 `data/testdev_balanced_questions.json` 构建会话，第一版只采用下面这一种预注册规则：

1. 使用保持 JSON 插入顺序的 parser 读取问题字典。
2. 遍历源文件，把 question ID 按出现顺序追加到对应 `imageId` 的有序列表；**不对 image ID 或 question ID 排序，也不 shuffle**。
3. 对每张图的列表从头开始切成连续的三问块，丢弃该图最后的 0 至 2 个余数问题。
4. 每个 turn 保存 `question_id`、`question` 和短答案字段 `answer`；`fullAnswer` 不用于 accuracy gold。
5. 每个会话保存 `dialog_id`、`image_id`、图片相对路径和三个 turns。另存 manifest，记录源文件 SHA-256、converter commit、构建时间和输出 SHA-256。

converter 必须用断言验证：

```text
image groups       = 398
dialogs            = 4061
turns per dialog   = 3
used question ids  = 12183，且全部唯一
dropped questions  = 395
missing images     = 0
```

任何断言失败都应终止构建。测试集 converter 不接收随机种子，因为第一版协议没有随机操作。

实际构造命令：

```bash
cd "$ROOT"
TEST_Q="$ROOT/ThirdParty/LLaVA/playground/data/eval/gqa/data/testdev_balanced_questions.json"
TEST_IMAGES="$ROOT/ThirdParty/LLaVA/playground/data/eval/gqa/data/images"
TEST_OUT="$ROOT/weights/datasets/gqa/processed/mt_gqa_testdev"

/opt/miniconda3/envs/metacompress/bin/python \
  scripts/metacompress/build_mt_gqa_eval.py \
  --questions "$TEST_Q" \
  --image-root "$TEST_IMAGES" \
  --output-dir "$TEST_OUT"
```

输出的 `mt_gqa_testdev_balanced_4061.json` 是推理和评分脚本的输入；其 manifest 应显示 `398 / 4061 / 12183 / 395 / 0` 五个统计量。

### 7.3 固定多轮执行和评分语义

第一版采用真正的连续三轮对话，而不是把三问作为三个独立样本：

1. 第一轮输入一次 image token、问题 1 和固定的单词/短语回答提示。
2. greedy 生成回答 1，并把**模型实际生成的文本**作为 assistant history。
3. 在同一 conversation 中追加问题 2，生成回答 2；然后以相同方式执行问题 3。
4. 图像只在第一轮出现一次。允许通过 KV cache 复用上下文，但 cached 和重新前向完整 conversation 必须在小样本上产生相同 token。
5. 对 MetaCompress 和 Sample，视觉序列在第一轮前只压缩一次，并在三轮中复用同一份结果。

评分沿用本地 LLaVA GQA 转换逻辑：对已经 `strip()` 的模型输出执行 `rstrip(".").lower()`，然后与 GQA 的短答案 `answer` 做严格字符串相等比较。每轮分母都必须是 4061：

```text
Acc_t = 100 * correct_turn_t / 4061
Avg   = (Acc_1 + Acc_2 + Acc_3) / 3
```

保存每个 question ID 的原始输出、归一化输出、gold、是否正确、turn index 和 dialog ID。不要直接调用要求全部 12,578 个预测的原版 `eval.py`；MT-GQA 只包含其中 12,183 问，应实现等价的子集逐轮 exact-match evaluator。

本地实现对应关系如下：`run_mt_gqa.py` 在同一个 `Conversation` 中连续执行三轮，并把上一轮模型输出追加为 assistant history；`score_mt_gqa.py` 读取 JSONL，按上述 `strip -> rstrip(".") -> lower` 规则评分。先用 `--limit 8` 做 smoke run；正式评分时去掉 `--limit`，让评分器看到全部 4061 个会话。

### 7.4 先对齐 Base，确定数据协议

关闭所有 token reduction，运行 4061 个会话。目标为：

```text
Acc1=61.76, Acc2=64.07, Acc3=65.35, Avg=63.73
```

若未通过第 1.2 节的门禁，只允许按以下顺序检查有限的协议变量，每次只改变一个，并保存完整结果：

1. 确认 checkpoint revision、image processor、`vicuna_v1` 模板、greedy 解码和答案归一化完全固定。
2. 比较 generated history、ground-truth 短 `answer` history 和自然句 `fullAnswer` history。generated history 是默认值，后两者只用于识别作者是否使用 teacher-forced history。
3. 比较图像仅首轮输入与每轮重复输入。
4. 比较源 JSON 顺序与同图 question ID 数值排序。不要尝试无边界随机搜索问题顺序。
5. 最后才比较“同一会话但每轮独立推理”的解释。

诊断原则：Acc1 已接近而 Acc2/Acc3 明显偏离，优先检查 history 和 KV cache；Avg 接近而各轮分布错位，优先检查组内问题顺序；三轮整体平移，优先检查模型 revision、prompt、image preprocessing 和答案归一化。确定协议后把选择固化到 manifest，后续 Sample 和 Ours 不得再改变。

可直接运行的命令（先 smoke，再完整评测）：

```bash
cd "$ROOT"
MODEL="$ROOT/weights/hub/models--liuhaotian--llava-v1.5-7b/snapshots/haotian-from-hf"
DIALOGS="$ROOT/weights/datasets/gqa/processed/mt_gqa_testdev/mt_gqa_testdev_balanced_4061.json"
IMAGE_ROOT="$ROOT/ThirdParty/LLaVA/playground/data/eval/gqa/data/images"
RUN_DIR="$ROOT/outputs/metacompress/mt_gqa"
PYTHON=/opt/miniconda3/envs/metacompress/bin/python

# 8-dialog smoke test; --expected-dialogs makes partial scoring explicit.
$PYTHON scripts/metacompress/run_mt_gqa.py \
  --model-path "$MODEL" --dialogs "$DIALOGS" --image-root "$IMAGE_ROOT" \
  --method base --output "$RUN_DIR/base_smoke_predictions.jsonl" --limit 8
$PYTHON scripts/metacompress/score_mt_gqa.py \
  --dialogs "$DIALOGS" --predictions "$RUN_DIR/base_smoke_predictions.jsonl" \
  --output "$RUN_DIR/base_smoke_metrics.json" --limit 8 --expected-dialogs 8

# Full Base gate: omit --limit and keep this prediction file immutable.
$PYTHON scripts/metacompress/run_mt_gqa.py \
  --model-path "$MODEL" --dialogs "$DIALOGS" --image-root "$IMAGE_ROOT" \
  --method base --output "$RUN_DIR/base_predictions.jsonl"
$PYTHON scripts/metacompress/score_mt_gqa.py \
  --dialogs "$DIALOGS" --predictions "$RUN_DIR/base_predictions.jsonl" \
  --output "$RUN_DIR/base_metrics.json"
```

脚本会在 full score 阶段强制每轮均为 4061 个样本；smoke 分数只用于检查模型加载、图片预处理、prompt 和输出格式，不能与论文的 4061-dialog 分数比较。

### 7.5 用 Sample 验证压缩插入点

LLaVA-1.5-7B 的视觉 tower 和 `mm_projector` 产生 `[B, 576, 4096]` patch embeddings。Sample 和 MetaCompress 都应作用在 `mm_projector` 之后、视觉 embeddings 与文本 embeddings 拼接之前，即 `llava/model/llava_arch.py` 的 `prepare_inputs_labels_for_multimodal` 路径内。LLaVA-1.5 这 576 个位置不包含要额外保留的 CLS 或 newline token。

论文没有说明 90% reduction 的取整规则。必须注册并比较：

- `m=57`：`floor(576 * 0.1)`，当前 `MetaCompress._target_tokens` 的行为；
- `m=58`：`round(576 * 0.1)`，README 示例的行为。

Sample 使用覆盖首尾的确定性等距索引；索引生成公式、57/58 的选择和最终索引数组必须写入结果文件。目标为 `55.06 / 57.89 / 59.74 / 57.56`。只有 Base 已通过但两个 Sample 版本都失败时，才检查压缩位置和等距索引公式；不得用训练 MetaCompress 来绕过这个门禁。

Sample 的正式命令如下；两次运行共用同一份 Base 对话和解码配置：

```bash
# m=57 (floor(576 * 0.1))
$PYTHON scripts/metacompress/run_mt_gqa.py \
  --model-path "$MODEL" --dialogs "$DIALOGS" --image-root "$IMAGE_ROOT" \
  --method sample --target-tokens 57 \
  --output "$RUN_DIR/sample_m57_predictions.jsonl"
$PYTHON scripts/metacompress/score_mt_gqa.py \
  --dialogs "$DIALOGS" --predictions "$RUN_DIR/sample_m57_predictions.jsonl" \
  --output "$RUN_DIR/sample_m57_metrics.json"

# m=58 (rounding candidate)
$PYTHON scripts/metacompress/run_mt_gqa.py \
  --model-path "$MODEL" --dialogs "$DIALOGS" --image-root "$IMAGE_ROOT" \
  --method sample --target-tokens 58 \
  --output "$RUN_DIR/sample_m58_predictions.jsonl"
$PYTHON scripts/metacompress/score_mt_gqa.py \
  --dialogs "$DIALOGS" --predictions "$RUN_DIR/sample_m58_predictions.jsonl" \
  --output "$RUN_DIR/sample_m58_metrics.json"
```

### 7.6 准备 MT-GQA train-balanced

第 4.3 节列出的 GQA train-balanced 问题和训练图片已经准备完成。当前可直接使用的文件为：

```text
weights/datasets/gqa/raw/questions/train_balanced_questions.json
weights/datasets/gqa/processed/mt_gqa/mt_gqa_train_balanced_20000.jsonl
weights/datasets/gqa/processed/mt_gqa/images/*.jpg
```

训练数据的构造和使用仍需遵守以下规则：

1. 对训练问题采用与最终 testdev 协议相同的同图排序和连续三问分组规则。
2. 断言训练 question ID 不与 testdev 重叠，并记录训练/testdev image ID 的交集。
3. 把一个三轮会话定义为一个 training item；保留短答案作为 teacher-forced assistant 文本，并只在 assistant response token 位置计算蒸馏损失。
4. 从完整训练会话中确定性抽取 20,000 项作为第一版。固定种子并保存**完整 dialog/question ID 清单**，不能只保存 seed。
5. 论文没有说明 20K 是精确数量、每个数据集各自数量还是联合数量，也没有公布抽样 seed。对 MT-GQA-only 主目标，20,000 个 GQA 三轮会话是可审计的第一版假设，不得写成论文已确认事实。

训练集只用于优化压缩器。testdev 的问题、答案和 Base/Sample 错误样本不得进入训练数据或用于选择训练样本。

已生成的 GQA-only 第一版训练 manifest 使用 `seed=42`，位置为 [`weights/datasets/gqa/processed/mt_gqa/manifest.json`](../../weights/datasets/gqa/processed/mt_gqa/manifest.json)。当前 `training_ready=true`，`available_image_count=16488` 且 `missing_image_count=0`，因此数据准备门禁已经通过，可以进入 LLaVA 数据加载器和 MetaCompress 训练适配器阶段。该 20K 选择仍是可审计的工程假设，不是论文官方样本清单；启动训练前仍应保存 manifest 和完整的 dialog/question ID 清单。

### 7.7 实现并验证 LLaVA 训练适配器

适配器必须满足以下不变量：

1. 冻结 vision tower、`mm_projector`、LLM 和所有 embedding；优化器只能看到 MetaCompress 参数。
2. 每张图先计算一次 projector 后的 576 个 visual tokens。teacher 使用完整 tokens，student 使用同一输入经 MetaCompress 得到的 `m` 个 tokens。
3. teacher 前向放在 `no_grad` 下；student 的梯度只能经过压缩后的 visual tokens 回到 MetaCompress。
4. 完整和压缩输入的总序列长度不同，不能直接比较全部 logits。两路前向必须分别抽取同一组三轮 assistant response token 对应的 logits，再验证 shape、token ID 和 mask 完全对齐后计算 KL。
5. padding、文本 prompt、标准答案和 response mask 在两路中一致；image token 位置和视觉 token 位置全部从 loss mask 排除。
6. 在 8 至 32 个会话上做过拟合测试，并逐步验证 loss 有限且下降、压缩矩阵每行和为 1、没有 NaN/Inf、LVLM 梯度全部为 `None`、保存再加载 checkpoint 后输出一致。

完成这些单元和小样本检查后才能启动完整训练。

本地训练适配器的 8-dialog 形状/梯度 smoke 命令为：

```bash
TRAIN_DIALOGS="$ROOT/weights/datasets/gqa/processed/mt_gqa/mt_gqa_train_balanced_20000.json"
TRAIN_IMAGES="$ROOT/weights/datasets/gqa/processed/mt_gqa/images"

$PYTHON scripts/metacompress/train_llava_metacompress.py \
  --model-path "$MODEL" \
  --dialogs "$TRAIN_DIALOGS" \
  --image-root "$TRAIN_IMAGES" \
  --output "$RUN_DIR/train_smoke" \
  --testdev-question-ids "$ROOT/weights/datasets/gqa/processed/mt_gqa_testdev/question_ids.txt" \
  --testdev-image-ids "$ROOT/weights/datasets/gqa/processed/mt_gqa_testdev/image_ids.txt" \
  --target-tokens 57 --pool-mode max \
  --epochs 1 --batch-size 2 --limit 8 --seed 42
```

该命令需要可用 CUDA 和完整的旧版 LLaVA-1.5-7B 权重；它会生成 `run_config.json`、`train_metrics.jsonl` 和 `checkpoint-final/`。若出现 response mask、NaN 或冻结参数错误，应先停止，不要运行完整训练。

### 7.8 按论文设置训练 MetaCompress

论文明确给出的配置必须原样使用：

| 配置 | 值 |
|---|---|
| optimizer | SGD |
| learning rate | `1e-3` |
| epochs | 2 |
| global batch size | 36 |
| max gradient norm | `1e-2` |
| `alpha_entropy` | 1 |
| `alpha_collapse` | 1 |
| pooling kernel | 3 |
| initialization | `Wq = Wk`，高斯标准差 `1/sqrt(d_c)`；`omega=1` |

未公开或论文内部不一致的配置必须显式记录，不能悄悄采用本地默认值：

- 主文公式描述 average pooling，附录 8.2 指定 fractional max pooling；正式对齐优先运行附录的 `max` 设置，再将 `avg` 作为受控消融。
- 论文没有给出 `d_c`；本地默认 `256` 只是待验证假设。
- 论文称 absolute positional embedding，但没有说明 learned、sinusoidal、长度和初始化；本地默认 sinusoidal 不是官方事实。
- 论文没有说明 57/58 的取整、SGD momentum、weight decay、KL temperature、数据 shuffle seed 和 mixed precision 细节。未说明的 optimizer 参数第一版设为 PyTorch SGD 默认值，并逐项记录。

训练损失为 `KL(teacher || student) + L_entropy + L_collapse`。第一版在三个固定随机种子上训练，逐 step 记录总 loss、三个分量、裁剪前 gradient norm、学习率、投影熵和最大列质量，并保存每个 epoch 的 checkpoint。若需要枚举未公开变量，使用预先登记的小型矩阵，不根据单个 testdev 结果无限调参。

当前脚本按论文主设置执行单进程梯度累积；`--batch-size 36` 是一个 optimizer step 的累积样本数。每个 epoch 使用 `seed + epoch - 1` 生成确定性 dialog shuffle，`train_metrics.jsonl` 还记录 source index、投影行和误差、最大列质量和裁剪前 gradient norm。正式训练命令（每个 seed 使用独立输出目录）为：

```bash
for SEED in 0 1 2; do
  $PYTHON scripts/metacompress/train_llava_metacompress.py \
    --model-path "$MODEL" \
    --dialogs "$TRAIN_DIALOGS" \
    --image-root "$TRAIN_IMAGES" \
    --output "$RUN_DIR/train_seed_${SEED}" \
    --testdev-question-ids "$ROOT/weights/datasets/gqa/processed/mt_gqa_testdev/question_ids.txt" \
    --testdev-image-ids "$ROOT/weights/datasets/gqa/processed/mt_gqa_testdev/image_ids.txt" \
    --target-tokens 57 --compression-dim 256 \
    --pool-mode max --position-embedding sinusoidal \
    --epochs 2 --batch-size 36 \
    --learning-rate 1e-3 --max-grad-norm 1e-2 --seed "$SEED"
done
```

`--pool-mode max`、sinusoidal position、`d_c=256` 和 `m=57` 是第一版待验证假设；若 Base/Sample 已通过而 Ours 仍偏离，再按预注册矩阵改为 `avg`、`m=58` 或其他位置编码，并保留完整输出目录。

### 7.9 最终评测和结果判定

对每个训练 seed 使用第 7.4 节已经冻结的 MT-GQA 协议评测，生成：

```text
base_metrics.json
sample_m57_metrics.json
sample_m58_metrics.json
metacompress_seed_<seed>_metrics.json
predictions/<method>.jsonl
predictions/<method>.config.json
manifests/testdev.json
manifests/train_subset_<seed>.json
configs/<run>.json
```

最终报告至少同时列出 Base、选定的 Sample、MetaCompress 每个 seed、三种子均值/标准差和论文目标，并报告相对 Base 的下降。判定顺序如下：

| 现象 | 结论和下一步 |
|---|---|
| Base 不对齐 | 数据或推理协议错误；停止训练侧调参 |
| Base 对齐，Sample 不对齐 | 压缩位置、token 数或采样索引错误 |
| Base/Sample 对齐，小样本训练失败 | adapter、logit 对齐、loss 或冻结策略错误 |
| 前三项通过，Ours 低于 58.43 | 优先检查 pooling、position embedding、`d_c`、训练 item 定义和 20K 子集 |
| Ours 对齐但 Base/Sample 未对齐 | 不能作为代码正确性证据，属于训练对测试协议的偶然补偿 |
| Base、Sample、Ours 和 70% 检查均通过 | 可称为“在重建协议下数值对齐论文”；作者数据/代码发布前仍不能称逐样本严格复现 |

这条路径的关键不是先得到 `58.43`，而是让 `63.73 -> 57.56 -> 58.43` 三个独立锚点依次成立。这样最终分数才对 MetaCompress 实现正确性有诊断价值。

训练后运行 MetaCompress 评测：

```bash
for SEED in 0 1 2; do
  $PYTHON scripts/metacompress/run_mt_gqa.py \
    --model-path "$MODEL" --dialogs "$DIALOGS" --image-root "$IMAGE_ROOT" \
    --method metacompress --target-tokens 57 \
    --compressor "$RUN_DIR/train_seed_${SEED}/checkpoint-final" \
    --output "$RUN_DIR/metacompress_seed_${SEED}_predictions.jsonl"
  $PYTHON scripts/metacompress/score_mt_gqa.py \
    --dialogs "$DIALOGS" \
    --predictions "$RUN_DIR/metacompress_seed_${SEED}_predictions.jsonl" \
    --output "$RUN_DIR/metacompress_seed_${SEED}_metrics.json"
done
```

`score_mt_gqa.py` 的 full run 会拒绝缺失或重复的 dialog/turn 预测，并在 metrics JSON 中写出 `acc1`、`acc2`、`acc3`、`avg` 及逐题 `.scored.jsonl`。把这些 JSON 和 `run_config.json` 一起归档后再计算三 seed 均值/标准差。

---

## 8. 复现记录清单

每次正式训练至少记录：

- 基础模型名称、revision 和本地 snapshot；
- 视觉编码器名称和 revision；
- MetaCompress git commit 或源码 diff；
- 数据集名称、版本、样本数和 SHA-256；
- 多轮构建脚本版本、训练/测试 manifest、问题顺序、history 模式、随机种子及会话统计；
- Base、Sample、MetaCompress 的逐轮原始预测和 exact-match 明细；
- `input_dim`、`compression_dim`、精确输入/输出 token 数、取整规则、`reduction_rate`、pooling 模式、`pool_kernel_size` 和位置编码模式；
- 优化器、学习率、batch size、epoch、梯度裁剪和 `alpha`；
- GPU 型号、PyTorch / CUDA 版本和分布式策略；
- checkpoint、日志、评测输出和失败样本；
- 实验属于论文复现还是非官方扩展。

只有上述信息可追溯，后续作者代码发布时才能可靠比较本地实现与官方实现的差异。

---

## 9. ConvBench 复现

ConvBench 用来检查压缩插入是否正确，不用来代替 MT-GQA 的数值对齐。论文第 6.1 节写明：压缩器在 MT-VQA-v2 上训练，再零样本评到 ConvBench，不在 ConvBench 上训练。三轮分别对应感知、推理和创造，表 1 的列名是 S1、S2、S3。

LLaVA-1.5-7B、90% 压缩下，论文表 1 的目标是：

| 方法 | S1 | S2 | S3 | Avg |
|---|---:|---:|---:|---:|
| Base | 4.33 | 5.72 | 5.55 | 5.20 |
| Sample | 3.64 | 4.85 | 3.81 | 4.10 |
| MetaCompress | 4.33 | 3.99 | 4.16 | 4.16 |

官方 ConvBench 用 GPT-3.5-turbo 做裁判。MetaCompress 脚注改成 Llama-3.1-8B-Instruct，但没有公布裁判 prompt，本机也跑不了该模型。当前评分改用 [Mint](https://mintcn.macaron.xin) 的 `glm-5.2`，prompt 仍是官方 pairwise 模板。因此分数用于检查 Base 和 Sample 的相对趋势，不能直接写成表 1 的复现。

验收顺序和 MT-GQA 相同：

1. **Base**：不压缩，使用「图像 token 留在后续轮次」的协议。第一问放入图像，后两问不再插入新图像，但第一问的图像 token 仍留在上下文里。历史使用模型自己生成的回答，不用参考答案替换。每段对话只编码一次图像。Base 对不上 4.33 / 5.72 / 5.55 / 5.20 时，先查 prompt、历史和裁判，不评压缩器。
2. **Sample**：LLaVA-1.5 仍是 576 个视觉 token。先跑 `m=57`，即 `floor(576 × 0.1)`。目标是 3.64 / 4.85 / 3.81 / 4.10。Base 已对齐而 Sample 偏离，才检查压缩插入点和保留 token 数。
3. **MetaCompress**：论文的 4.16 对应 MT-VQA-v2 上训出的权重。当前 GQA-only checkpoint 只能确认 ConvBench 前向能跑通，不能用来对 4.16。MT-VQA-v2 训练数据尚未准备。

当前可执行路径如下。`scripts/metacompress/build_convbench.py` 把 `ConvBench.xlsx` 写成 MT-GQA 格式的 `outputs/metacompress/convbench/convbench_dialogs.json`，并跳过缺图的 ID 131，实际生成 577 条。生成仍用 `run_mt_gqa.py` 的 `persist` 协议，`--question-suffix` 传空字符串，避免把 GQA 的短答案提示加到开放式问题上。

评分脚本是 `scripts/metacompress/score_convbench.py`。它使用官方 `ThirdParty/ConvBenchEval/prompts.py` 的标准 pairwise prompt 和 `pairwise.npy` 的 A/B 位置，不用 perfect perception / reasoning 消融。裁判是 [Mint](https://mintcn.macaron.xin) 上的 `glm-5.2`，接口为 OpenAI 兼容的 `https://mintcn.macaron.xin/v1/chat/completions`。密钥写在仓库根目录 `.env` 的 `CONV_BENCH_API_KEY`。该文件已在 `.gitignore` 中，评分脚本和 `drafts/cmd.sh` 都会读取它。S1、S2、S3 是模型回答胜过人工参考的比例乘以 100。论文分母是 577；缺图的 ID 131 不在这 577 条里。`scripts/metacompress/convbench_judge_blacklist.json` 里的 6 条对话（260、355、408、418、474、520）裁判没有给出可用结论，不调用裁判，也不计入分子和分母，因此当前分母是 571。Avg 是这三项的平均。论文表 1 的裁判是 Llama-3.1-8B-Instruct，因此这组分数不能写成表 1 的复现。

命令在 `drafts/cmd.sh` 末尾，只使用 GPU 7，顺序是构建 JSON、Base 生成并评分、Sample `m=57` 生成并评分。不在 ConvBench 上训练，也不评现有 GQA checkpoint。
