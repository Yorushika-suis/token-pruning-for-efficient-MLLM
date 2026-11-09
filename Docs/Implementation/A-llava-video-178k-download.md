# 按 FlexSelect 索引下载 LLaVA-Video-178K 的一部分

FlexSelect 在 LLaVA-Video-178K 上训练 token selector，在 Video-MME 上测试。训练清单不是一份冻结的视频 ID，而是 [`output_rnd05.yaml`](../../weights/datasets/llava-video-178k/sample/output_rnd05.yaml)：8 个子集、24 个 json，每个 json `sampling_strategy: random:5%`。yaml 里被注释掉的 perceptiontest、activitynetqa、nextqa、llava_hound 不下载。

抽样代码在 [`ThirdParty/flexselect/train/LLaVA-Video/llava/train/train.py`](../../ThirdParty/flexselect/train/LLaVA-Video/llava/train/train.py)。`random:5%` 先按 `math.ceil(5 * len / 100)` 算出条数，再 `random.shuffle`，取前 N 条（约 1041–1043 行）。`random.seed(1234)` 写在 `__main__`（1760 行），`train()` 里先 `get_model`（1529 行），数据集要到 1744 行才建。模型初始化若再碰随机数，题号就和「先设种子、立刻 shuffle」不完全一致。下面的命令走后一种，用来决定下哪些视频。

这批视频是给之后的 VLA-ACL 训练准备的。当前 `scripts/vla_acl/train_video.py` 只读 Video-MME 和 `decode/output`，还不能直接吃这些 json。

下文 `$ROOT` 是仓库根。直连 `huggingface.co` 会超时，用镜像。

## 1. 已经在本地的索引

| 内容 | 路径 |
| --- | --- |
| 抽样配方 | `$ROOT/weights/datasets/llava-video-178k/sample/output_rnd05.yaml` |
| 24 个 json | `$ROOT/weights/datasets/llava-video-178k/raw/<子集>/*.json` |

json 大约 2 GiB。来源是 `yunzhuyunzhu/token_selector_train_data` 的 yaml，以及 `lmms-lab/LLaVA-Video-178K` 里这 8 个子集的 json。视频还没有下。

每条记录的 `video` 是子集内部的相对路径，例如 `liwei_youtube_videos/videos/youtube_video_2024/ytb_YVQwAEKZpaU.mp4`。`data_source` 是子集目录名。有的路径没有 `.mp4` 后缀。

## 2. 视频怎么存放

Hugging Face 上没有单条 mp4。8 个子集的视频打在 **230** 个 `*_videos_*.tar.gz` 里，每个大约 5 GiB，合计 **1117 GiB**。压缩包内部的路径就是 json 里的 `video` 字段。

因此只能整包下载。做法是：下一个包，解出抽中的视频，立刻删掉压缩包。磁盘上最终只留 mp4；网络上仍要拉完 1117 GiB。

## 3. 抽中多少、占多少空间

对本地 24 个 json 执行 `random.seed(1234)` 后 shuffle、向上取整 5%。得到 **66789** 条问答、**56288** 个不重复视频。全集有 **178680** 个不重复视频。同一条视频会同时出现在 caption、选择题和开放题里，所以 5% 的题目碰到了大约 31.5% 的视频。

解压体积按「该子集压缩包总大小 × 抽中视频占该子集视频数的比例」估计。同一时长档里各条视频大小接近。合计大约 **377 GiB**。过程中最多再多占一个约 5 GiB 的压缩包。

| 子集 | 全集视频 | 抽中 | 压缩包 | 解压后 |
| --- | ---: | ---: | ---: | ---: |
| `0_30_s_academic_v0_1` | 12150 | 2962 | 8 个，35.3 GiB | 8.6 GiB |
| `0_30_s_youtube_v0_1` | 79346 | 23282 | 19 个，92.8 GiB | 27.2 GiB |
| `30_60_s_academic_v0_1` | 10514 | 3637 | 10 个，44.9 GiB | 15.5 GiB |
| `30_60_s_youtube_v0_1` | 19995 | 6988 | 13 个，62.6 GiB | 21.9 GiB |
| `1_2_m_academic_v0_1` | 4627 | 1506 | 14 个，66.8 GiB | 21.7 GiB |
| `1_2_m_youtube_v0_1` | 24239 | 8264 | 50 个，247.6 GiB | 84.4 GiB |
| `2_3_m_academic_v0_1` | 3124 | 1143 | 18 个，85.0 GiB | 31.1 GiB |
| `2_3_m_youtube_v0_1` | 24685 | 8506 | 98 个，481.6 GiB | 166.0 GiB |
| 合计 | 178680 | 56288 | 230 个，1116.6 GiB | 376.5 GiB |

`/root/Test`（`/dev/sdb1`）写这份文档时只剩大约 120 GB，放不下。默认输出在仓库的 `weights/` 下，开跑前把 `VID` 改到有约 380 GiB 空闲的目录。本容器里能看见 `sdc`、`sdd` 两块 18.2T 盘，但没有 `/dev/sdc` 节点，临时建节点后读盘返回 `Operation not permitted`，不能在这里格式化或挂载。

## 4. 下载命令

输出在 `$VID/<子集>/` 下，相对路径与 json 的 `video` 字段一致。中断后续跑：已经解出的 mp4 还在；若某个 `.tar.gz` 只下了一半，先删掉那个残包，否则脚本会把它当成完整包。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export ROOT=/root/Test/Lyx/token-pruning-for-efficient-MLLM
RAW=$ROOT/weights/datasets/llava-video-178k/raw
VID=$ROOT/weights/datasets/llava-video-178k/videos
mkdir -p "$VID"

python3 - <<'PY'
import json, math, random, tarfile, urllib.request
from pathlib import Path

raw = Path("/root/Test/Lyx/token-pruning-for-efficient-MLLM/weights/datasets/llava-video-178k/raw")
videos = Path("/root/Test/Lyx/token-pruning-for-efficient-MLLM/weights/datasets/llava-video-178k/videos")
yaml = Path("/root/Test/Lyx/token-pruning-for-efficient-MLLM/weights/datasets/llava-video-178k/sample/output_rnd05.yaml")
mirror = "https://hf-mirror.com/datasets/lmms-lab/LLaVA-Video-178K/resolve/main/"

jsons = []
for line in yaml.read_text().splitlines():
    line = line.strip()
    if line.startswith("- json_path:"):
        jsons.append(raw / line.split("LLaVA-Video-178K/")[-1])

random.seed(1234)
need = set()
for path in jsons:
    data = json.load(path.open())
    k = math.ceil(5 * len(data) / 100)
    random.shuffle(data)
    for item in data[:k]:
        rel = str(item["video"])
        need.add((item["data_source"], rel))
        if not rel.endswith(".mp4"):
            need.add((item["data_source"], rel + ".mp4"))
print(f"needed names {len(need)}", flush=True)

subsets = sorted({src for src, _ in need})
for subset in subsets:
    rows = json.load(urllib.request.urlopen(
        "https://hf-mirror.com/api/datasets/lmms-lab/LLaVA-Video-178K/tree/main/" + subset, timeout=60
    ))
    tars = sorted(r["path"].split("/")[-1] for r in rows if str(r.get("path", "")).endswith(".tar.gz"))
    wanted = {rel for src, rel in need if src == subset}
    for name in tars:
        dest = raw / subset / name
        if not dest.exists():
            url = mirror + f"{subset}/{name}"
            print(f"download {subset}/{name}", flush=True)
            urllib.request.urlretrieve(url, dest)
        out = videos / subset
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.lstrip("./") in wanted]
            if members:
                print(f"extract {len(members)} from {name}", flush=True)
                tar.extractall(out, members)
        dest.unlink()
print("done", videos)
PY
```

改存放位置时，同时改 shell 里的 `VID` 和 Python 里的 `videos`。`raw` 只用来暂存当前这一个压缩包。
