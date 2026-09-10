"""Inspect the Video-MME test parquet and pin down the `short` subset.

The plan needs exactly the short-duration videos (expected 300 unique videoIDs / 900 QA).
Everything downstream — the 2:1 fit/eval split, the ZIP64 extraction manifest — keys off
this, so report the real schema rather than trusting the dataset card.
"""
import collections
import json

import pyarrow.parquet as pq

W = "/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial"
PARQUET = (
    f"{W}/weights/hub/datasets--lmms-lab--Video-MME/snapshots/"
    "ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet"
)

tbl = pq.read_table(PARQUET)
print("rows:", tbl.num_rows, " columns:", tbl.num_columns, flush=True)
print("schema:", flush=True)
for name, typ in zip(tbl.schema.names, tbl.schema.types):
    print(f"   {name:22s} {typ}", flush=True)

d = tbl.to_pydict()

print("\n=== first row ===", flush=True)
for k in tbl.schema.names:
    v = d[k][0]
    s = str(v)
    print(f"   {k:22s} = {s[:110]}", flush=True)

# Which column carries short/medium/long?
dur_col = None
for cand in ("duration", "duration_category", "task_type", "category"):
    if cand in d:
        vals = collections.Counter(d[cand])
        print(f"\n=== value counts for {cand} ===", flush=True)
        for k, v in vals.most_common(12):
            print(f"   {str(k)[:40]:42s} {v}", flush=True)
        if set(vals) & {"short", "medium", "long"}:
            dur_col = cand

print("\nduration column detected:", dur_col, flush=True)
assert dur_col, "could not find the short/medium/long column"

vid_col = "videoID" if "videoID" in d else ("video_id" if "video_id" in d else None)
assert vid_col, f"no video id column among {tbl.schema.names}"
print("video id column:", vid_col, flush=True)

short_rows = [i for i, x in enumerate(d[dur_col]) if x == "short"]
short_ids = sorted({d[vid_col][i] for i in short_rows})
print(f"\nshort QA rows : {len(short_rows)}", flush=True)
print(f"short videos  : {len(short_ids)} unique", flush=True)
print("sample ids    :", short_ids[:5], flush=True)

by_dur = collections.Counter()
for i, x in enumerate(d[dur_col]):
    by_dur[x] += 1
print("QA rows per duration:", dict(by_dur), flush=True)
uniq_all = {x: set() for x in by_dur}
for i, x in enumerate(d[dur_col]):
    uniq_all[x].add(d[vid_col][i])
print("unique videos per duration:", {k: len(v) for k, v in uniq_all.items()}, flush=True)

out = f"{W}/data/short_subset.json"
with open(out, "w") as fh:
    json.dump(
        {
            "duration_column": dur_col,
            "video_id_column": vid_col,
            "short_video_ids": short_ids,
            "n_short_qa_rows": len(short_rows),
            "parquet": PARQUET,
        },
        fh,
        indent=1,
    )
print("\nwrote", out, flush=True)
