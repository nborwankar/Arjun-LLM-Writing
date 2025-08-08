#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE: convert_persona_whatsapp_to_openai.py

INPUT FILES:
- Training (persona schema, WhatsApp Outgoing only):
  /Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/persona_training_dataset.whatsapp_outgoing_only.jsonl
  - Each line: JSON object with fields like instruction, input, output, source, chunk_id, timestamp

- Validation (persona schema, WhatsApp Outgoing only):
  /Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/persona_validation_dataset.whatsapp_outgoing_only.jsonl
  - Same schema as training

OUTPUT FILES:
- OpenAI Chat Format (Training):
  /Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/arjun_voice_training_whatsapp.jsonl
  - Each line: {"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

- OpenAI Chat Format (Validation):
  /Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/arjun_voice_validation_whatsapp.jsonl
  - Same schema as training

Version history:
- v1.0 (2025-08-07): Initial conversion utility with progress meters and validation.

Notes (10th grader friendly):
- We take each WhatsApp example (your message and the reply), and reshape it into the format OpenAI needs:
  - The "user" message is the question/instruction, optionally followed by extra input text.
  - The "assistant" message is the answer/output.
- We also skip broken lines and keep a small log.
"""

import json
from pathlib import Path
from typing import Iterable

# Absolute paths
TRAIN_IN = Path("/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/persona_training_dataset.whatsapp_outgoing_only.jsonl")
VAL_IN = Path("/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/persona_validation_dataset.whatsapp_outgoing_only.jsonl")
TRAIN_OUT = Path("/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/arjun_voice_training_whatsapp.jsonl")
VAL_OUT = Path("/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/arjun_voice_validation_whatsapp.jsonl")

METER_EVERY = 1000  # progress meter frequency


def to_openai_messages(obj: dict) -> dict | None:
    """Convert one persona-format object to OpenAI chat format.

    Returns None if required fields are missing or empty.
    """
    instr = (obj.get("instruction") or "").strip()
    out = (obj.get("output") or "").strip()
    inp = (obj.get("input") or "").strip()

    if not instr or not out:
        return None

    if inp:
        user_text = instr + "\n\n" + inp
    else:
        user_text = instr

    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": out},
        ]
    }


def convert_file(src: Path, dst: Path) -> tuple[int, int]:
    """Convert a persona JSONL file to OpenAI chat JSONL.

    Returns (written_lines, skipped_lines).
    """
    written = 0
    skipped = 0

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                skipped += 1
                continue

            rec = to_openai_messages(obj)
            if rec is None:
                skipped += 1
                continue

            json.dump(rec, fout, ensure_ascii=False)
            fout.write("\n")
            written += 1

            if written % METER_EVERY == 0:
                print(f"... converted {written} lines from {src.name}")

    print(f"Done {src.name}: wrote={written}, skipped={skipped}, out={dst.name}")
    return written, skipped


def main():
    if not TRAIN_IN.exists():
        print(f"❌ Missing: {TRAIN_IN}")
        return 1
    if not VAL_IN.exists():
        print(f"❌ Missing: {VAL_IN}")
        return 1

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)

    w1, s1 = convert_file(TRAIN_IN, TRAIN_OUT)
    w2, s2 = convert_file(VAL_IN,   VAL_OUT)

    print("Conversion complete.")
    print({
        "train_written": w1, "train_skipped": s1,
        "val_written": w2, "val_skipped": s2,
        "train_out": str(TRAIN_OUT), "val_out": str(VAL_OUT),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
