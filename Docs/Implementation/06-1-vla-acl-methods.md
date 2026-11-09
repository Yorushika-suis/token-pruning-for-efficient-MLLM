# VLA-ACL 方法归纳：Action-Consistent Visual Token Pruning

本文对应：

- 代码：[`ThirdParty/VLA-ACL`](../../ThirdParty/VLA-ACL)（主体在 [`src/openvla-oft`](../../ThirdParty/VLA-ACL/src/openvla-oft)）
- 论文：<https://arxiv.org/abs/2610.08133>
- 发布权重：<https://huggingface.co/owendudu/VLA-ACL>
- 本文审计的代码 commit：`40b7a7a17f2f5a15acab0564ba1e304bcf6d150d`

目标不是复现跑通命令，而是把 **VLA-ACL 相对 OpenVLA-OFT 的新增方法** 抽成可重实现的合同：输入/输出、公式、训练–推理落差、关键超参与代码锚点。后续新实现应以本文为准，不必整仓库 fork OpenVLA-OFT；底座可换成任意连续动作 VLA，但下列组件语义应对齐。

下文路径均相对 `ThirdParty/VLA-ACL/src/openvla-oft/`，除非写明仓库根。

---

## 0. 一句话与边界

**VLA-ACL**：在冻结的 fine-tuned OpenVLA-OFT 上，训练一个轻量 **Patch Scorer**；训练时用可微 soft top-K + 黑图 embedding 门控，使「剪枝后学生动作」同时贴近 **全上下文教师动作** 与 **GT 动作**；推理时对每个相机视角 hard top-K，只把选中的视觉 patch 送进 LLM，并用 **位置保持的 position ids** 保留原序列几何。

| 属于 VLA-ACL 方法（应重实现） | 属于底座脚手架（可替换） |
| --- | --- |
| `PatchScorer` 结构与输入特征位置 | OpenVLA-OFT / Prismatic / LIBERO / RLDS |
| LapSum soft top-K、α 余弦退火 | FiLM / LoRA / diffusion action head（剪枝未支持 diffusion） |
| 黑图 gating：`X̃ = s·X + √(1−s²)·b` | 具体 vision backbone（DINO+SigLIP fused） |
| Action consistency loss（μ·L1_GT + λ·L1_teacher） | 数据管线、wandb、LIBERO eval harness |
| 每视角 top-K、hard gather、position-preserving ids | 发布权重文件名约定 |

剪枝只支持 **连续 L1 regression action head**；diffusion 路径会直接 `NotImplementedError`。

---

## 1. 问题设定与整体流水线

设定（与代码默认一致）：

- 策略：`OpenVLA-OFT`，多图输入默认 `num_images_in_input=2`（第三人称 + wrist），可选 proprio token。
- 视觉特征：backbone 输出 patch embeddings `X ∈ R^{B×N×D_v}`（投影到 LLM 之前）。OpenVLA fused backbone 下，单视角约 **256** patch（224 / 14），两视角 `N≈512`；`D_v = embed_dim(DINO)+embed_dim(SigLIP)`。
- 预算：每视角保留 `K` 个 patch（发布常用 `K∈{32,64}`），总视觉 token ≈ `num_views · K`。
- 冻结：VLA、projector、proprio projector、action head 全部 `eval` + `requires_grad_(False)`；**只训 scorer**。

```
训练（soft，序列长度不变）
  pixels → frozen vision → X
                         ├→ teacher: project(X) → LLM → â_teacher
                         └→ scorer(X) → logits → soft_top_k/view → s
                            X̃ = s·X + √(1−s²)·b
                            project(X̃) → LLM → â_student
                            L = μ̂·‖â_student−a_gt‖₁ + λ̂·‖â_student−â_teacher‖₁

推理（hard，序列缩短）
  pixels → frozen vision → X → scorer → per-view top-K indices
  gather projected patches → [BOS, kept, proprio?, lang+action]
  position_ids = 未剪枝序列中的原始位置 → LLM → action head
```

---

## 2. Patch Scorer

文件：[`prismatic/models/patch_scorer.py`](../../ThirdParty/VLA-ACL/src/openvla-oft/prismatic/models/patch_scorer.py)

| 项 | 默认 |
| --- | --- |
| 输入 | `vision_tokens`，形状 `(B, N, vision_dim)`，**未**经 LLM projector |
| `input_proj` | `vision_dim == hidden_dim` 时为 `Identity`，否则 `Linear` |
| 主体 | `nn.TransformerEncoder`，`norm_first=True`，`activation=gelu`，`dropout=0` |
| 层数 / 隐维 / 头数 | `5` / `1024` / `16`（与 HF 发布 scorer 一致） |
| 输出 | `logit_head: Linear(hidden, 1)` → `(B, N)` logits |

```python
# 前向语义
logits = logit_head(encoder(input_proj(vision_tokens))).squeeze(-1)  # (B, N)
```

**每视角 top-K**（推理与 hard metric 共用）`select_top_k_per_view`：

1. 将 `(B, N)` reshape 为 `(B, num_views, patches_per_view)`，要求 `N % num_views == 0`。
2. 各视角独立 `topk(k)`。
3. 加上视角 offset，flatten 后 **按全局下标排序**，得到 `(B, num_views·k)`。

新实现必须保持「**按相机独立选 K，再拼回全局下标**」，不要在全序列上做一次 top-K（会偏向某一视角）。

---

## 3. Soft Top-K（LapSum）

文件：[`prismatic/training/scorer_train_utils.py`](../../ThirdParty/VLA-ACL/src/openvla-oft/prismatic/training/scorer_train_utils.py) 中 `SoftTopK` / `soft_top_k`

引用 Struski et al. 2025 的 LapSum soft top-K：

1. 对每行 scores，二分求阈值 `t`，使  
   `∑_i LaplaceCDF((score_i − t) / α) = K`。
2. 前向：`s_i = LaplaceCDF((score_i − t) / α) ∈ (0,1)`，且 `∑_i s_i ≈ K`。
3. 反向：用 LapSum 闭式 VJP（对 Laplace PDF 做中心化），不对 `k`/`α` 求梯度。

训练时 **按视角** reshape 后调用：

```text
(B, N) → (B·V, N/V) → soft_top_k(..., k=K, alpha=α) → (B, N)
```

**α 余弦退火**（`train_scorer.get_selection_scores`）：

| 符号（配置名） | 默认 | 含义 |
| --- | ---: | --- |
| `soft_top_k_max_tau` | `2.0` | 训练初 α（更软） |
| `soft_top_k_tau` | `0.1` | 训练末 α（更尖） |
| `soft_top_k_end_step` | `max_steps` | 退火终点 step |

```text
progress = min(step, end) / end
α = τ_end + 0.5 · (τ_start − τ_end) · (1 + cos(π · progress))
```

α 越小选择越接近 hard top-K。新实现应保留「先软后尖」的日程，否则训练早期梯度噪声大、后期与推理落差大。

---

## 4. 黑图 Embedding 与 Soft Gating

同文件：`get_black_pixel_values`、`gate_vision_tokens`

**黑图特征 `b`（Eq. 4.4）**

- 用与训练相同的 `image_processor.apply_transform`，对全黑 RGB 图做归一化。
- 无 FiLM：整次 run 算一次 `b`（batch 维可广播）。
- 有 FiLM：`b` 依赖语言，每个 batch 重算。

**门控（Eq. 4.5）**

```text
X̃ = s ⊙ X + √(1 − s²) ⊙ b
```

实现细节：

- `s` 升维到 `(B, N, 1)`。
- `√(1−s²)` 对 **`s.detach()`** 计算，避免 `s→1` 时导数爆炸；对 `s` 的梯度只走 `s·X` 项。
- 门控在 **vision 空间** 做完，再 `project`；不是先投影再门控。

直觉：`s≈1` 保留原 patch；`s≈0` 换成「无信息」黑图特征，序列长度仍为 `N`，便于端到端反传，且不改变 LLM 注意力形状。

---

## 5. Action Consistency Loss

同文件：`compute_scorer_training_loss`；调用侧权重 `gt_weight=μ`、`teacher_weight=λ`，默认均为 `0.5`。

设 `W = μ + λ`（须 `>0`）：

```text
L = (μ/W) · L1(â_student, a_gt) + (λ/W) · L1(â_student, â_teacher)
```

| 项 | 来源 | 作用 |
| --- | --- | --- |
| GT L1 | RLDS batch `actions` | 任务正确性 |
| Teacher L1 | 同 batch 全上下文、`torch.no_grad()` 前向 | 与未剪枝策略行为一致（蒸馏） |

`â_teacher` 仅在 `teacher_weight > 0` 时计算。日志还会记 position / orientation / gripper 分量 L1，以及 teacher–GT gap（不进损失）。

**学生前向**（训练）与底座一致：拼 multimodal embeddings →（可选 gradient checkpoint）LLM → 按 action mask 取 hidden → `L1RegressionActionHead.predict_action`，输出 `(B, NUM_ACTIONS_CHUNK, ACTION_DIM)`。

---

## 6. 推理：Hard Top-K 与 Position-Preserving IDs

入口：

- 模型：[`OpenVLAForActionPrediction.predict_action`](../../ThirdParty/VLA-ACL/src/openvla-oft/prismatic/extern/hf/modeling_prismatic.py)
- 包装：[`get_vla_action` / `get_patch_scorer`](../../ThirdParty/VLA-ACL/src/openvla-oft/experiments/robot/openvla_utils.py)
- Eval：[`run_libero_eval.py`](../../ThirdParty/VLA-ACL/src/openvla-oft/experiments/robot/libero/run_libero_eval.py) 的 `PatchScorerConfig`

步骤：

1. `patch_features = vision_backbone(...)`，`projected = projector(patch_features)`。
2. `keep = select_top_k_per_view(scorer(patch_features), K, num_views)`；当前推理取 batch 第 0 行 `[0]`（eval batch size=1）。
3. `projected = projected[:, keep]`，再可选 append proprio。
4. 构造 **position-preserving** ids（`build_preserving_multimodal_position_ids`）：

未剪枝逻辑位置（1-based 于 BOS 之后）：

```text
[BOS=0 | vision 0..N-1 at positions 1..N | proprio & language/action 接在 N+1 ...]
```

剪枝后序列为 `[BOS | kept_vision | proprio? | tail]`，但：

- kept vision 的 position = **原全局 patch 下标 + 1**；
- proprio 与 tail 的 position = **未剪枝时** 的 `1+N .. 1+N+proprio+tail-1`（不因视觉变短而左移）。

这样 RoPE / 绝对位置仍对应「该 patch 在完整网格中的位置」，避免 gather 后被重新编号成 `1..K`。

训练日志里的 **hard top-K metric**（`compute_hard_top_k_metrics`）走同一套 gather + preserving position ids，用于监控 train/infer gap；**不反传**。

可视化：`overlay_pruned_patches` 按 keep mask 把丢掉的 patch 半透明压灰，写入 LIBERO rollout 视频。

---

## 7. 训练循环合同

脚本：[`vla-scripts/train_scorer.py`](../../ThirdParty/VLA-ACL/src/openvla-oft/vla-scripts/train_scorer.py)

| 项 | 默认 / 约定 |
| --- | --- |
| 优化器 | AdamW，只优化 scorer；`lr=1e-4`，cosine → 0，`max_steps=5000` |
| batch | `8`，`grad_accumulation_steps=1` |
| dtype | VLA / scorer / heads：`bfloat16` |
| 数据 | 与 `vla_path` 同 suite 的 RLDS（如 `libero_10_no_noops`），并用 checkpoint 内 `dataset_statistics.json` |
| 增强 | `image_aug=True` |
| 存盘 | `patch_scorer_head--{step}_checkpoint.pt`（仅 `state_dict`） |
| 断言 | `use_l1_regression=True` |

单 step 顺序（与代码一致）：

1. `X = vision(pixels)`；`projected = project(+proprio)`。
2. `â_teacher = predict(projected)`（no grad，若 λ>0）。
3. `logits = scorer(X)` → `s = soft_top_k_per_view(logits, α(step))`。
4. `X̃ = gate(X, s, b)` → `â_student = predict(project(X̃))`。
5. `L` backward → optimizer / scheduler。

Run id 编码了 suite、`K`、μ、λ、batch、lr，便于对照发布命名 `scorer-libero-{suite}-k{K}.pt`。

---

## 8. 评测合同（LIBERO）

配置字段 `patch_scorer.*`：

| 字段 | 含义 |
| --- | --- |
| `enabled` | 是否启用剪枝 |
| `checkpoint` | scorer `state_dict` 路径 |
| `prune_topk` | 每视角 K，**必须与训练 `soft_top_k` 一致** |
| `hidden_dim` / `num_layers` / `num_heads` | 必须与训练架构一致（默认 1024/5/16） |

约束：

- 使用 **本地** OpenVLA-OFT checkpoint，使 `predict_action` 带 `patch_scorer` 参数（HF Hub 远程 modeling 可能不同步）。
- Scorer 与 VLA 的 LIBERO suite 对齐：`spatial` / `object` / `goal` / `10`。
- 示例：`K=64` 时 `--patch_scorer.prune_topk 64`。

官方训练 / 评测入口见仓库根 [`README.md`](../../ThirdParty/VLA-ACL/README.md)；环境见 [`SETUP.md`](../../ThirdParty/VLA-ACL/src/openvla-oft/SETUP.md)、[`LIBERO.md`](../../ThirdParty/VLA-ACL/src/openvla-oft/LIBERO.md)。

---

## 9. 代码地图（重实现时优先读这些）

| 角色 | 路径 |
| --- | --- |
| Scorer + per-view top-K | `prismatic/models/patch_scorer.py` |
| Soft top-K、gating、loss、学生/教师前向 | `prismatic/training/scorer_train_utils.py` |
| 训练入口与退火 / hard metric | `vla-scripts/train_scorer.py` |
| 推理剪枝 + preserving position ids | `prismatic/extern/hf/modeling_prismatic.py`（`predict_action`、`build_preserving_multimodal_position_ids`） |
| Eval 加载与调用 | `experiments/robot/openvla_utils.py`、`experiments/robot/libero/run_libero_eval.py` |
| Eval YAML 示例 | `experiments/patch_scorer_eval_config.yaml` |

OpenVLA-OFT 其余目录（RLDS、finetune、ALOHA 等）对理解 VLA-ACL **算法** 非必需。

---

## 10. 新实现检查清单

实现新版本时，建议按下列语义做单元级对照（不必 bit-exact）：

1. **特征阶段**：scorer 吃 **projector 之前** 的 patch；gating 也在该空间。
2. **选择粒度**：soft / hard 都是 **每视角 K**，不是全局 K。
3. **可微路径**：训练用 LapSum soft top-K + 黑图门控保长度；推理用 gather 缩短长度。
4. **损失**：归一化后的 μ/λ 加权 L1（GT + teacher）；底座冻结。
5. **位置**：hard 路径使用 **原网格位置** 的 position ids，tail 不因视觉缩短而重编号。
6. **退火**：α 从软到尖；最终 α 与 K 和发布 checkpoint 一致。
7. **动作头**：连续 L1；勿假设 diffusion 已支持剪枝。
8. **suite 绑定**：scorer 与 fine-tuned VLA、数据统计同一 LIBERO suite。

与本仓库其他视觉 token 剪枝（VisionZip / FastVID / CDPruner / TransPrune / MetaCompress）的本质差别：VLA-ACL 的监督信号是 **动作一致性**，不是 VQA 答案、注意力或 DPP 多样性；且是 **learned scorer + frozen policy**，不是纯 training-free。

---

## 11. 公式 ↔ 代码索引

| 论文概念（代码注释中的编号） | 实现 |
| --- | --- |
| Soft top-K / LapSum（Eq. 2.7） | `soft_top_k` / `SoftTopK` |
| Per-view hard indices（Eq. 4.2） | `select_top_k_per_view` |
| 黑图 embedding `b`（Eq. 4.4） | `get_black_pixel_values` + `get_vision_patch_embeddings` |
| Soft prune `X̃`（Eq. 4.5–4.6） | `gate_vision_tokens` + 学生 `predict_actions` |
| Teacher actions（Eq. 4.7） | 全上下文 `predict_actions`（no grad） |
| Consistency loss（Eq. 4.8） | `compute_scorer_training_loss` |
| Inference top-K + pos ids | `predict_action` + `build_preserving_multimodal_position_ids` |
