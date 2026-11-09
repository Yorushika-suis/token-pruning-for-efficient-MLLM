#!/usr/bin/env python3
"""Convert the official ConvBench workbook into the MT-GQA dialog JSON."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CELL_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def column_letters(cell_ref: str) -> str:
    return "".join(char for char in cell_ref if char.isalpha())


def load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("m:si", NS):
        strings.append("".join(text.text or "" for text in item.iter(f"{CELL_NS}t")))
    return strings


def cell_text(cell: ET.Element, strings: list[str]) -> str:
    value = cell.find("m:v", NS)
    if value is None or value.text is None:
        return ""
    if cell.attrib.get("t") == "s":
        return strings[int(value.text)]
    return value.text


def load_rows(workbook: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(workbook) as archive:
        strings = load_shared_strings(archive)
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = sheet.findall("m:sheetData/m:row", NS)
    header = [cell_text(cell, strings) for cell in rows[0].findall("m:c", NS)]
    records: list[dict[str, str]] = []
    for row in rows[1:]:
        values = {column_letters(cell.attrib.get("r", "")): cell_text(cell, strings) for cell in row.findall("m:c", NS)}
        if not values.get("A", "").strip():
            continue
        records.append({name: values.get(letter, "") for letter, name in zip("ABCDEFGHIJKLMN", header)})
    return records


def build_dialogs(records: list[dict[str, str]], image_root: Path) -> tuple[list[dict], list[dict]]:
    dialogs: list[dict] = []
    skipped: list[dict] = []
    for record in records:
        convbench_id = int(record["ID"])
        image_name = record["image_id"]
        if not (image_root / image_name).is_file():
            skipped.append({"convbench_id": convbench_id, "image_id": image_name})
            continue
        turns = []
        for turn_index, question_key, answer_key in (
            (1, "The_first_turn_instruction", "first_turn_answer"),
            (2, "The_second_turn_instruction", "second_turn_answer"),
            (3, "The_third_turn_instruction", "third_turn_answer"),
        ):
            turns.append(
                {
                    "question_id": f"{convbench_id}-{turn_index}",
                    "question": record[question_key],
                    "answer": record[answer_key],
                    "full_answer": record[answer_key],
                }
            )
        dialogs.append(
            {
                "dialog_id": f"convbench_{convbench_id:04d}",
                "convbench_id": convbench_id,
                "source_split": "convbench",
                "image_id": image_name,
                "image": image_name,
                "image_description": record["instruction-conditioned-caption"],
                "third_turn_demands": record["third_turn_demands"],
                "turns": turns,
            }
        )
    return dialogs, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_rows(args.workbook)
    dialogs, skipped = build_dialogs(records, args.image_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dialog_path = args.output_dir / "convbench_dialogs.json"
    manifest_path = args.output_dir / "manifest.json"
    dialog_path.write_text(json.dumps(dialogs, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "dataset": "ConvBench official workbook",
        "workbook": str(args.workbook),
        "image_root": str(args.image_root),
        "source_row_count": len(records),
        "dialog_count": len(dialogs),
        "skipped": skipped,
        "paper_dialog_count": 577,
        "outputs": {"json": dialog_path.name},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
