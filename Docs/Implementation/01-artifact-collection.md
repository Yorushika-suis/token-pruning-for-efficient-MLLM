# 复现产物收集：从 zhr 抽 v4-lite 契约

本文说明如何把 [`ThirdParty/zhr`](../ThirdParty/zhr) 里已经验证过的「复现产物可核对」做法，应用到 FastVID、VidCom2、VisionZip 的论文复现上。配套操作步骤见 [`02-reproduction-runbook.md`](02-reproduction-runbook.md)。

本阶段只定合同。probe 实现放在仓库根的 [`scripts/repro/`](../scripts/repro/)，**不改** `ThirdParty/` 源码。`PYTHONPATH=$ROOT/scripts/repro` 时 `sitecustomize.py` 在 `REPRO_PROBE=1` 下安装包装。`ThirdParty/` 被 `.gitignore` 忽略，文档里的路径和 commit 以本机 checkout 为准。

下文 `$ROOT` 指本仓库根路径。

## 1. zhr 在做什么、我们借什么

zhr 的目标不是复现 FlashVID 论文分数，而是研究 ADTS token 选择的空间位置偏置。它有两层产物：

| 层 | 位置 | 用途 |
|---|---|---|
| Stage0–2 | [`ThirdParty/zhr/scripts/adts_harness.py`](../ThirdParty/zhr/scripts/adts_harness.py)、`stage1_collect.py` | monkeypatch 官方选择函数，写 mask NPZ + jsonl journal |
| V4 | [`ThirdParty/zhr/scripts/v4/`](../ThirdParty/zhr/scripts/v4) | 失败即关的科学产物：content-hash NPZ、D3 provenance、runtime closure |

论文复现需要 LLM 前向，不能装 zhr 的 LLM tripwire，也不需要 FlashVID 专属的 13×13 几何臂。要迁移的是 **观察官方代码、把科学内容和执行噪声分开、把输入钉死**。

### 1.1 要借鉴

**观察、不替换。** probe 只 wrap 官方函数，退出时还原绑定。不要写一套「等效」selector 去替官方实现。zhr 的教训写在 `adts_harness.py` 开头：

- `flashvid.utils.ALL_TOKEN_SELECTION_METHOD[...]` 是运行时查 dict，改 dict 条目才生效。
- `flashvid.llava_arch` 已经 `from .utils import flashvid_compression`，必须打在 `llava_arch` 上，打在 `utils` 上静默失效。

V4 把同一原则做成 [`capture.py`](../ThirdParty/zhr/scripts/v4/capture.py) 的 `VisionCapture` / `CompressionCapture`：`register_forward_hook` + 模块函数替换，`__exit__` 还原。

**具名 capture contract。** 每个采集点有名字、dtype、期望 shape。zhr 的轴是 `pixels / sig729 / attn729 / projected729 / pooled169 / attn169`（[`numerics.py`](../ThirdParty/zhr/scripts/v4/numerics.py)）。三个方法几何不同，必须各自登记一套，不能硬套 64 帧 × 169 token。

**科学内容 vs 执行元数据。** [`artifacts.py`](../ThirdParty/zhr/scripts/v4/artifacts.py) 的 `FORBIDDEN_SCIENCE_KEYS` 禁止把这些字段写进 science 哈希：

`run_id`, `started_at`, `finished_at`, `generated_at`, `mtime`, `duration`, `elapsed_seconds`, `physical_path`, `pid`, `worker_pid`, `gpu_uuid`, `heartbeat`, `wall_time`, `log_path`

同一科学内容跨机器应得到同一 `content_sha256`。时间戳、GPU UUID、日志路径只进 `execution`。

**钉死输入。** 源码 commit SHA、模型 revision、数据字节校验、`pip freeze`。对应 zhr 的 `fetch_code.sh`、`create_env.sh`、`data_manifest.py`、以及「`snapshot_download` 之后手工补 `refs/main`」。

**原子写出 + canonical JSON。** science JSON 用 `sort_keys=True, separators=(',', ':'), allow_nan=False`。NPZ 禁止 pickle。失败证据另存 forensic NPZ（允许 NaN/Inf），且不得进入科学依赖。

**函数身份。** 记录被 hook 函数的 `module`、`qualname`、`source_file`、`source_file_sha256`、`first_line`（`capture.identity`）。防止「看起来在跑官方代码、实际绑到了另一份实现」。

**可续跑。** `REPRO_PROBE=1` 时（默认 `REPRO_RESUME=1`）`LlavaVid.generate_until` 若 `masks/` 里已有该题 NPZ 则跳过推理；新完成的题会把答案写到 `predictions/`。`REPRO_RESUME=0` 则全量重跑。目录遍历不能当覆盖范围的权威来源（JuiceFS / 多卡写盘会漏或重复），但续跑以已落盘的 NPZ 为准。

### 1.2 明确不照搬

- FlashVID 专属几何：`FRAMES=64, GRID=13, NV=169, BUDGETS R010–R025`（[`spec.py`](../ThirdParty/zhr/scripts/v4/spec.py)）。
- D2/D3/D4 评审门、全量 `ldconfig` runtime closure、独立重写 selector 再 bit-exact 对照。
- LLM tripwire。论文分数需要语言模型。
- 几何扰动臂（hflip / shift）。那是空间偏置实验，不是论文表复现。

落地形态称为 **v4-lite**：同一套 probe 契约 + 轻量 manifest，挂在官方 `lmms-eval` / `visionzip()` 外面。

```
env pin + data/weight pin
        ↓
  stage0 assertions
        ↓
  official eval  ←── runtime probes (REPRO_PROBE=1)
        ↓                    ↓
   lmms-eval scores      mask NPZ (+ optional features)
        ↓                    ↓
           science manifest
```

## 2. v4-lite 契约

### 2.1 产物种类

| `kind` | 是否进 science 哈希 | 内容 |
|---|---|---|
| `provenance` | 是 | 源码 commit、被 hook 函数身份、模型 revision、数据清单哈希、关键超参、`pip freeze` 的 sha256 |
| `masks` | 是 | 每条样本的 keep 索引 / 布尔 mask / 实际保留比例 / 分段信息 |
| `features` | 可选 | 选择前的分数或注意力（体积大，默认关） |
| `scores` | 是 | lmms-eval 聚合指标 + 逐样本答案（从官方 log / `results.json` 摘出，去掉时间戳路径） |
| `execution` | 否 | `run_id`、墙钟、GPU 名、pid、日志路径 |

`features` 默认关闭。需要对照「为什么选了这些 token」时再开，并在 manifest 里登记 `upstream` 关系。

### 2.2 目录

```
$ROOT/runs/<method>/<run_id>/
  plan.json                 # 方法、模型、超参、task、probe 开关（无时间戳）
  provenance.json           # 钉死输入
  manifest.json             # science + execution
  masks/<sample_id>.npz
  scores/results.json       # 从官方输出摘出的科学部分
  sidecars/                 # journal.jsonl、unit_timings、失败 forensic
  env.pip-freeze.txt
```

`<method>` 取 `fastvid` / `vidcom2` / `visionzip`。`<run_id>` 只出现在路径和 `execution` 里，不进入 science 哈希。

建议 `run_id` 形如 `20260912T143000Z_<8-hex>`，但格式本身不是科学内容。

### 2.3 Canonical JSON 与哈希

科学对象先经 `canonical()`：

```python
json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
```

`content_sha256 = sha256(canonical(science))`。`science` 里不得出现第 1.1 节的禁止键。集合类字段（coverage、artifacts、upstreams）按逻辑键排序后再哈希，避免写入顺序进入身份。

### 2.4 NPZ 合同

每个 `masks/<sample_id>.npz`：

- 禁止 `allow_pickle=True`。
- 保留成员 `meta_json`：canonical JSON 的 UTF-8 字节，存成一维 `uint8`。
- 其余成员名只允许 `[A-Za-z0-9_-]`。
- 数组必须 finite（forensic 失败文件除外）、C-contiguous、小端。
- `meta.coverage` 非空，且为已排序的逻辑键列表。

推荐通用成员（方法可以多写，不能少写带 `*` 的）：

| 成员 | 含义 |
|---|---|
| `keep_indices` * | 最终保留的全局 token 索引，`int32`，一维，已排序 |
| `n_in` * | 压缩前视觉 token 数（不含文本） |
| `n_out` * | 压缩后视觉 token 数 |
| `retention` * | `n_out / n_in`，`float32` 标量 |
| `frame_keep_k` | 每帧保留个数，`int32`，长度 = 帧数 |
| `segment_lengths` | FastVID DySeg 段长；其他方法可缺省 |
| `meta_json` * | 见上 |

`sample_id` 对 Video-MME 用 **`[videoID, question_id]`**（一视频三题，只按 `videoID` 写 NPZ 会覆盖）。`meta_json` 仍记录 `videoID`。对图像任务用官方 dataset 的稳定 id（如 `question_id`），禁止用行号。文件名把 key 用 `__` 拼进 `masks/`。

### 2.5 Manifest 骨架

```json
{
  "science": {
    "schema_version": "repro-artifact/v4-lite-1",
    "method": "fastvid",
    "model": "lmms-lab/LLaVA-Video-7B-Qwen2",
    "task": "videomme",
    "hyperparams": {},
    "source_commit": "40-char-sha",
    "model_revision": "40-char-sha-or-tag",
    "dataset_sha256": "64-hex",
    "provenance_sha256": "64-hex",
    "expected_coverage": [["video_id"]],
    "artifacts": [{"key": ["video_id"], "file_type": "npz-v4-lite", "bytes": 1, "sha256": "64-hex"}],
    "upstreams": [],
    "scores": {}
  },
  "execution": {
    "run_id": "...",
    "status": "RUNNING",
    "started_at": "...",
    "finished_at": null,
    "artifact_paths": [],
    "sidecars": []
  },
  "content_hash_spec": "v4-json-1",
  "content_sha256": null
}
```

`status=COMPLETE` 时才填写 `content_sha256`。覆盖范围以 `expected_coverage` 为准，不以目录 listing 为准。缺一个、多一个都不得标 COMPLETE。

### 2.6 Journal

`sidecars/journal.jsonl` 每行：

```json
{"key": ["-ApL8d6tX5U", "001-1"], "config_hash": "64-hex", "ok": true}
```

`config_hash` 覆盖一切会改变 mask 或分数的字段（方法、retention、帧数、`mm_spatial_pool_mode`、attn backend、probe 版本）。改超参后旧 journal 行全部作废。

### 2.7 失败产物

官方函数抛错或 capture 形状不对时：

1. 把当时的索引 / 分数写入 `sidecars/failures/<key>/observation.npz`（forensic，允许非有限值）。
2. 写 `difference.json`（异常类型、traceback、成员名）。
3. journal 记 `ok: false`。
4. 该文件 **不得** 列入 `science.artifacts`。

### 2.8 Probe 开关

环境变量：

| 变量 | 含义 |
|---|---|
| `REPRO_PROBE=1` | 打开观察包装 |
| `REPRO_PROBE_DIR` | 产物根目录，默认 `$ROOT/runs/<method>/<run_id>` |
| `REPRO_PROBE_FEATURES=1` | 额外写分数 / 注意力（默认关） |

probe 必须在官方压缩 **返回值不变** 的前提下工作。包装函数先调 original，再 clone 需要的张量；禁止改索引、禁止改 `hidden_states`。退出时还原所有绑定。若还原失败，stage0 应直接报错，而不是带着脏 monkeypatch 去评测。

## 3. 三个方法的 capture 合同

原则：runtime wrap（`types.MethodType` / 模块函数替换 / `register_forward_hook`），打在 **实际被调用的那一层绑定** 上。打错 module 会得到空 mask 且不报错——这是 zhr 已经踩过的坑。

### 3.1 FastVID

压缩发生在 LLM prefilling，不是视觉塔。

| 栈 | 官方函数 | 路径（相对 `ThirdParty/FastVID/`） |
|---|---|---|
| LLaVA-Video | `Qwen2Model.forward` | `fastvid_llavavideo/LLaVA-NeXT/llava/model/language_model/modeling_qwen2.py` 约 L893，`if seq_length > 1: ### FastVID` |
| LLaVA-OneVision | `Qwen2Model.forward` | `fastvid_llavaonevision/.../modeling_qwen2.py` 约 L901 |
| Qwen2.5-VL | `Qwen2_5_VLForConditionalGeneration.forward` | `fastvid_qwen25vl/lmms-eval/lmms_eval/models/qwenvlutils/modeling_qwen2_5_vl.py` 约 L1889 |

侧输入（可选 features）：`LlavaQwenForCausalLM.generate` → `set_my_kwargs`；`llava_arch.encode_images` / `SigLipVisionAbstract` 提供 `frame_global_features` 与 `frame_attn_weights`。

**算法阶段（采集点按这个切）：** DySeg（帧全局余弦 + 低相似度切段）→ 按 `retention_ratio` 与 `STPrune_d` 分预算 → salient = attention top-k → context = density 峰 → DTM 把未保留 token 并进 context → 用 `keep_indexs` 重建 `hidden_states` 并重映射 RoPE。

**必采**

| 字段 | 说明 |
|---|---|
| `segment_sizes` | DySeg 段长，和为帧数 |
| `keep_indexs` | 最终绝对索引（含文本前缀/后缀；视觉部分可再拆） |
| `n_in`, `n_out`, `retention` | 视觉 token 计 |
| `frm_top_indexes` / `cur_salient_indexs` | 每帧 salient |
| `frm_context_indexes` | 每帧 density context |

**确定性。** density 分支有 `torch.rand(...) * 1e-6` 破并列。跨 run **不保证** bit-exact。分数对齐看 Video-MME；mask 硬门只要求 salient top-k 稳定。density 索引照记，但不作为 COMPLETE 的逐位条件。

LLaVA-Video 在剪之前丢掉 newline（`(index+1) % 14 != 0`），剪完再挂回去。`n_in` 用「纯视觉 64×13×13」计，newline 另记，避免和 VidCom2 的 182/帧对不上。

Hook：wrap 上述 `forward`（内部仍调 original）。选择逻辑是 inline 的，不能只包一个具名 selector。实现用仅限该 `source_file` 的 `sys.settrace`，在 `keep_indexs` 已是排序 1-D tensor 且 `hidden_states` 长度已经对齐之后 clone，然后关掉 tracer，避免扫 decoder。不要改算法。

### 3.2 VidCom2

压缩是纯函数，优先 wrap 函数本身，而不是再包一层 `prepare_inputs_labels_for_multimodal`。

| 符号 | 路径（相对 `ThirdParty/VidCom2/`） |
|---|---|
| `vidcom2_compression` / `select_outlier_indices` / `compute_scales` | `token_compressor/vidcom2/vidcom2.py` |
| 接入：`COMPRESSOR=vidcom2` 时替换 `prepare_inputs_labels_for_multimodal` | `lmms-eval/lmms_eval/models/llava_vid.py` 约 L214；`llava_onevision.py` 约 L157 |
| 替换实现 | `token_compressor/vidcom2/models/llava.py` 的 `cus_prepare_inputs_labels_for_multimodal` |

`MODEL_SPECS`：`llava_ov` 每帧 196、linear mapper；`llava_vid` 每帧 169、grid=13（再补每行 newline → 论文写 182 token/帧）。

LLaVA-Video 路径（`mm_newline_position == "grid"`）先 `reshape` 成 flat，再 `add_token_per_grid`，然后：

```python
vidcom2_compression(flattened_image_feature, model="llava_vid",
                    base_scale=R_RATIO, img_feat=image_feature)
```

LLaVA-OV 路径 flatten `(T,M,C) → (T*M,C)`，默认 `model="llava_ov"`。

**必采**

| 字段 | 说明 |
|---|---|
| `vid_score`, `frame_score` | `[T, M]`，仅 `REPRO_PROBE_FEATURES=1` |
| `scales` | 每帧动态保留率 |
| `ks` + 每帧 local indices | `select_outlier_indices` |
| `global_idx` | `_map_linear_offset` 或 `_map_grid_vid` |
| 布尔 keep mask | 原始 `[T, M]` 网格上；跨方法对比的首选产物 |

**确定性。** 无 FastVID 那种 `torch.rand`。同一输入、同一 dtype 下 top-k 应稳定。并列时 `torch.topk(..., sorted=False)` 再 `.sort()`，文档要求记录 `ks` 与排好序的 local indices。

**论文 vs 代码。** 论文 Algorithm 1 写 cosine uniqueness；发布代码是 L2-normalized 多尺度 Gaussian（α = 2^{-3..1}）+ 最低方差 50% 通道筛选。复现对齐 **代码**，不是伪代码。把这个差异写进 `provenance.notes`。

**不要走的路径。** `token_compressor/vidcom2/models/qwen2_vl.py` 把 `vidcom2_compression` 的 **feature 返回值当成 indices**。probe 与第一波复现只走 LLaVA；Qwen 需 `qwen` branch 上的 Qwen2.5/3 adapter。

LLaVA-Video 必须 `mm_spatial_pool_mode=average`。checkpoint 默认 bilinear 会变成 14×14，坐标系全错且往往不报错。

### 3.3 VisionZip

加载后整体 monkeypatch，真正选 token 在视觉塔。

| 符号 | 路径（相对 `ThirdParty/VisionZip/`） |
|---|---|
| `visionzip()` | `visionzip/main.py`：替换 `CLIPAttention.forward`、`CLIPEncoderLayer.forward`、`CLIPVisionTower.forward`、`prepare_inputs_labels_for_multimodal` |
| 选择 / 合并 | `visionzip/clip_encoder.py` 的 `CLIPVisionTower_VisionZip.forward` |
| 配置 | `visionzip/utils.py` 的 `apply_info`：`_info["dominant"]=dominant-1`，`_info["contextual"]=contextual`；layer schedule `r = [0]*22 + [1] + [0]`（倒数第二层） |

算法（CLIP 倒数第二层）：

1. `attn_weights = attentions[-2]`；`metric = layers[-2].metric`（key 均值，来自被替换的 `CLIPAttention_forward`）。
2. Dominant：CLS→patch attention 对 head 求和，`topk(dominant_num)`，并永远保留 CLS（index 0）。
3. Contextual：在未入选 patch 上按步长取 target，用 key 余弦把其余 token 并进 target。

**必采**

| 字段 | 说明 |
|---|---|
| `cls_attention_sum` | 仅 features |
| `topk_indices` / `all_indices` | dominant（含 CLS） |
| `target_indices` | contextual 在「非 dominant」子集上的下标 |
| `assign_one_hot` | 仅 features |
| `n_in`, `n_out` | CLIP 576 patch+CLS → `dominant+contextual`（retain 64 时 API 为 54+10） |

**API 偏移。** `visionzip(model, dominant=D, contextual=C)` 内部存 `dominant=D-1`，因为 CLS 额外保留。retain 64 = `dominant=54, contextual=10`。manifest 同时记 API 参数和有效 token 数。

**Qwen2.5-VL**（非第一波）：`Qwen2_5_VL/qwen2_5vl_visionzip.py` 约 L1916。代码默认 `dominant_num = int(0.65 * N)`、`contextual_num = max(int(0.05 * N), 1)`，对应 README「70%」。要对齐「50%」表需改成 `0.45+0.05`。`batch_size=1`。该文件的 **video 路径不跑 VisionZip**。

论文视频表走的是 Video-LLaVA（每帧 256→17）。本 checkout 的 `visionzip/` 没有 Video-LLaVA fork。与 FastVID/VidCom2 做视频对比需要后加包装，不算本契约的必采项。

## 4. 跨方法可比较的最小公共集

三个方法几何不同，不要把 FastVID 的绝对 `keep_indexs` 直接减 VidCom2 的 `global_idx`。可比较的层：

1. **每帧布尔 mask**（在该方法自己的空间网格上）：FastVID 13×13（Video）或 OV 网格；VidCom2 `[T,M]`；VisionZip CLIP 24×24 patch（去掉 CLS 后）。
2. **实际保留比例** `n_out/n_in`（声明 retention 经常不等于实现值，尤其 FastVID-Qwen 的 0.27 → ~24.1%）。
3. **分数** 只在相同模型、相同 task、相同抽帧协议下比。

把网格定义写进每条 mask 的 `meta_json`：`grid_h`, `grid_w`, `frames`, `newline_policy`。

## 5. 与代码的边界

| 契约（本文） | 实现（[`scripts/repro/`](../scripts/repro)） |
|---|---|
| 观察、不替换 | `probe_fastvid.py` / `probe_vidcom2.py` / `visionzip_probe.py` |
| manifest / NPZ 字段 | `artifacts.write_npz`、`session.ProbeSession` |
| journal | `sidecars/journal.jsonl`；Video-MME key 含 `question_id` |
| 挂到评测 | `sitecustomize.py` + `PYTHONPATH`；`LlavaVid.generate_until` 写 thread-local 样本 id |

`REPRO_PROBE=1` 现已可用。关闭 probe 时官方分数仍应能跑通（仓库 smoke 脚本默认关）。
