# VLA-ACL 迁移：接到 LLaVA-1.5 的 GQA / POPE / MME

本文对应方法合同 [`06-vla-acl-methods.md`](06-vla-acl-methods.md)。不复现 OpenVLA-OFT / LIBERO。

迁移方式与仓库里已有两条线对齐：

- 评测宿主跟 CDPruner、TransPrune、VisionZip 的图像线：LLaVA-1.5-7B，GQA / POPE / MME，同一 token 预算。见 [`05-cdpruner-reproduction.md`](05-cdpruner-reproduction.md) 第 2 节。
- 训练形态跟 MetaCompress：冻结 LLaVA，只训一个插在视觉特征和 LLM 之间的小模块；教师是全视觉 token 的同一模型。见 [`scripts/metacompress/train_llava_metacompress.py`](../../scripts/metacompress/train_llava_metacompress.py)。

不改 `ThirdParty/`。训练和推理都放在 `scripts/vla_acl/`，评测复用 [`scripts/cdpruner/run_llava15.sh`](../../scripts/cdpruner/run_llava15.sh) 的切卡、答案路径和 [`scripts/repro/score_visionzip.py`](../../scripts/repro/score_visionzip.py)。

下文 `$ROOT` 是仓库根。

## 1. 迁移后还保留什么

| 原方法 | 迁到 LLaVA-1.5 |
| --- | --- |
| 可学习 Patch Scorer，只训它 | 保留。输入换成 CLIP patch（projector 之前） |
| 每视角独立 top-K | 单图，退化为全局 top-K。LLaVA-1.5 只有一张图、576 个 patch |
| Soft top-K（LapSum）+ α 从软到尖 | 保留，训练用 |
| 黑图门控 `s·X + √(1−s²)·b` | 保留，训练用。`b` 用同一 CLIP 预处理的全黑图 |
| 推理 hard gather | 保留。剪完再进 projector 和 LLM |
| 位置保持 | 保留原 patch 的 position，不要把留下的 token 重编号成 `0..K-1` |
| 损失：L1(动作, GT) + L1(动作, teacher) | 改成文本。学生答案贴近 GT 短答案，也贴近全 token 教师的答案分布 |
| OpenVLA / RLDS / LIBERO scorer 权重 | 不用 |

分数只和同模型、同预算下的 Base / CDPruner / TransPrune / VisionZip 比，不和 VLA-ACL 论文的 LIBERO success rate 比。

## 2. 本地资产

这些路径本机已经有，迁移不另下机器人数据。

| 用途 | 路径 |
| --- | --- |
| 底座 | `weights/hub/models--liuhaotian--llava-v1.5-7b/snapshots/haotian-from-hf`（`llava_llama` + `mm_projector.bin`）。传给加载器的路径最后一级须含 `llava`，沿用 CDPruner 的 `weights/llava-v1.5-7b` 链接 |
| 视觉塔 | `weights/hub/models--openai--clip-vit-large-patch14-336` |
| 训练问题 | `weights/datasets/gqa/raw/questions`（GQA questions1.2）里的 `train_balanced_questions.json` |
| 训练图 | MetaCompress 已经用过的 GQA 图。图像根与 [`scripts/metacompress/`](../../scripts/metacompress) 现用目录一致，不另建一份 |
| GQA 评测 | `ThirdParty/LLaVA/playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl` 与 `gqa/data/images` |
| POPE | `.../eval/pope/llava_pope_test.jsonl`、`pope/val2014`、`pope/coco/coco_pope_*.json` |
| MME | `.../eval/MME/llava_mme.jsonl` 与 `MME_Benchmark_release_version` |

训练集用 GQA `train_balanced`，并从中去掉出现在 testdev 的 question id，避免评测题进训练。这和 MetaCompress 训练时排除 testdev id 的做法相同。

POPE、MME 只评测，不训练。VideoMME 不在这一步里：视频线要先处理 64 帧和换行 token，和 CDPruner 第 3 节是另一项工作。

## 3. 插在哪一层

LLaVA-1.5 的视觉序列是 CLIP ViT 的 576 个 patch（336 / 14，不含 CLS），再经 `mm_projector` 进 Vicuna。

```
pixel → CLIP patch X (B, 576, 1024)
            ├─ 教师：project(X) → LLM → 答案 logits（no grad）
            └─ scorer(X) → soft/hard 选择
                 训练：X̃ = gate(X, s, b) → project → LLM
                 推理：X_keep = X[:, topk] → project → LLM（position 用原下标）
```

Scorer 吃 **projector 之前** 的 1024 维 patch，和 VLA-ACL 吃 vision backbone、不吃 LLM hidden 一致。结构仍是 5 层、hidden 1024、16 头、Gelu、pre-norm 的 Transformer，最后一层 Linear 出 `(B, 576)` logits。`vision_dim == 1024` 时 `input_proj` 是 Identity。

推理预算与 CDPruner 表对齐：`TOKEN ∈ {128, 64, 32}`，默认 128。训练时的 `K` 必须和这次评测的 `TOKEN` 相同；换预算就换一个 scorer，不要用 K=128 的权重去评 32。

## 4. 损失

动作 L1 换成答案上的两项，权重仍默认各 0.5，并按权重和归一化：

```text
L = (μ / W) · CE(student, y_gt) + (λ / W) · KL(teacher_logits ‖ student_logits)
W = μ + λ
```

- `y_gt`：GQA 的短 `answer`。prompt 与评测一致，带 “Answer the question using a single word or phrase.”
- teacher：同一冻结 LLaVA、全部 576 个 token、`temperature=0` 的前向。KL 只加在 assistant 答案 token 上，mask 对齐方式跟 MetaCompress 的蒸馏相同：两条序列长度不同时，各自取出答案位置再算，不要直接对整段 logits。
- 底座、CLIP、projector 全部冻结。优化器仍是 AdamW、`lr=1e-4`、cosine 到 0。步数先按 GQA train_balanced 过 1 个 epoch 定，不照搬 VLA-ACL 的 5000 step（那是 LIBERO 轨迹，不是这批问答）。

Soft top-K 与黑图门控按方法合同原样搬：α 从 2.0 余弦退火到 0.1；`√(1−s²)` 对 detach 后的 `s` 计算。训练日志里每隔若干 step 用 hard top-K 再跑一次不反传的前向，看和推理的差距。

## 5. 推理与打分

实现放在 `scripts/vla_acl/`，运行时挂到 LLaVA 的 `encode_images`（或等价的 vision→projector 入口），不改 `ThirdParty/LLaVA`。挂法与 CDPruner 的 `sitecustomize.py` 相同：`PYTHONPATH` 最前，环境变量打开剪枝并传入 checkpoint 和 `TOKEN`。

评测命令与现有图像线相同的三件事：

- `temperature=0`，`conv-mode=vicuna_v1`
- 一进程一张卡；`device_map` 不要把 7B 切到多卡
- 答案写到 `runs/vla-acl-v1.5/<task>/vtn_<N>/answers/`，再用 `score_visionzip.py` 出 GQA Accuracy、POPE、MME Perception/Cognition

对照行是同一脚本跑出的未剪枝 LLaVA，以及同 `TOKEN` 的 CDPruner / VisionZip。TransPrune 的预算是层内比例，不是固定 128/64/32，和它对表时要写明实际留下的 token 数。

## 6. 和原论文、和现有钩子的差别

| 项目 | VLA-ACL 原文 | 本迁移 |
| --- | --- | --- |
| 模型 | OpenVLA-OFT-7B | LLaVA-1.5-7B |
| 监督 | 连续动作 L1 | 短答案 CE + 教师 KL |
| 选择范围 | 每相机 K | 单图 576 里留 K |
| 数据 | LIBERO RLDS | GQA train_balanced |
| 评测 | LIBERO success | GQA / POPE / MME |
| 发布 scorer | `owendudu/VLA-ACL` | 不能加载，特征维和任务都不同 |

CDPruner 的 VideoMME 钩子是 training-free，问题文本只做 DPP 条件。这里 scorer 必须先在 GQA 上训完，推理时不再读文本塔；问题只通过冻结 LLM 的 prompt 起作用。

## 7. 完成标准

1. 冻结 LLaVA 上，K=128 的 scorer 能训完一个 epoch，hard top-K 的 GQA 训练 loss 随 α 变尖而接近 soft 路径。
2. `TOKEN=128` 的 GQA / POPE / MME 跑完，目录里有 `metrics/*.json`，且未剪枝对照用同一评分器。
3. 64 和 32 各有自己的 scorer，或明确写成「未按该 K 重训、只作参考」。
4. 文档和结果里不出现 LIBERO 分数，也不把这三列写成 VLA-ACL 论文复现。

图像和视频的落地命令、环境安装见 [`08-vla-acl-run.md`](08-vla-acl-run.md)。
