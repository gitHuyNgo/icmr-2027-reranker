#!/usr/bin/env python3
"""Regenerate three infrastructure-only synthetic images; never research data."""
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1] / "data" / "smoke"


def main() -> None:
    images = ROOT / "images"
    images.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (color, shape) in enumerate((('red', 'square'), ('blue', 'circle'), ('green', 'triangle')), 1):
        identifier = f"smoke_{index:03d}"
        image = Image.new("RGB", (224, 224), "white")
        draw = ImageDraw.Draw(image)
        if shape == "square":
            draw.rectangle((48, 48, 176, 176), fill=color)
        elif shape == "circle":
            draw.ellipse((48, 48, 176, 176), fill=color)
        else:
            draw.polygon(((112, 40), (40, 184), (184, 184)), fill=color)
        image.save(images / f"{identifier}.png")
        records.append({"id": identifier, "image": f"images/{identifier}.png",
                        "question": "What color is the shape?", "answer": color,
                        "infrastructure_only": True})
    with (ROOT / "samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print("Generated 3 infrastructure-only smoke samples. Never use their metrics in the paper.")


if __name__ == "__main__":
    main()
