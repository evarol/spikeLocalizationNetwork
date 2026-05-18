"""Stitch three aggregate-projections PNGs vertically into a single
comparison figure, with a header banner per row.

Default: MP raw  ·  MP+DREDge  ·  CNN-SLN all-spike postd. ep20
Goal: showcase the headline improvement in one figure for the README.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HEADER_H = 60   # px banner per row
BG = (0, 0, 0)
TEXT_FG = (240, 240, 240)


def load_font(size: int = 28):
    # Prefer a clean sans on macOS; fall back to default
    for candidate in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mp_raw",    default="figures/aggregate_projections_raw.png", type=Path)
    p.add_argument("--mp_dredge", default="figures/aggregate_projections_mp_dredge.png", type=Path)
    p.add_argument("--method",    default="figures/aggregate_projections_postdredge_cnn_all_ep20.png", type=Path,
                   help="The learned method to feature.")
    p.add_argument("--mp_raw_label",    default="MP raw  (no motion correction)  ·  ρ̄ = 0.267")
    p.add_argument("--mp_dredge_label", default="MP + DREDge canonical  (standard pipeline)  ·  ρ̄ = 0.568")
    p.add_argument("--method_label",    default="CNN-SLN all-spike post-DREDge ep20  (this work)  ·  ρ̄ = 0.663")
    p.add_argument("--out", default="figures/aggregate_3method_comparison.png", type=Path)
    p.add_argument("--banner_h", type=int, default=HEADER_H)
    p.add_argument("--font_size", type=int, default=28)
    args = p.parse_args()

    panels = []
    for path, label in [
        (args.mp_raw,    args.mp_raw_label),
        (args.mp_dredge, args.mp_dredge_label),
        (args.method,    args.method_label),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"input missing: {path}")
        panels.append((Image.open(path).convert("RGB"), label))

    # All three panels should share width (they were rendered with the same style.fig_w)
    widths = [p[0].width for p in panels]
    W = max(widths)
    # Resize any panels narrower than the max (preserving aspect ratio)
    resized = []
    for img, lbl in panels:
        if img.width != W:
            new_h = int(img.height * (W / img.width))
            img = img.resize((W, new_h), Image.LANCZOS)
        resized.append((img, lbl))

    total_h = sum(img.height + args.banner_h for img, _ in resized)
    out = Image.new("RGB", (W, total_h), BG)
    draw = ImageDraw.Draw(out)
    font = load_font(args.font_size)

    y = 0
    for img, lbl in resized:
        # Banner with the row label
        draw.rectangle([(0, y), (W, y + args.banner_h)], fill=BG)
        # Center the text vertically in the banner
        bbox = draw.textbbox((0, 0), lbl, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_y = y + (args.banner_h - text_h) // 2
        draw.text(((W - text_w) // 2, text_y), lbl, fill=TEXT_FG, font=font)
        y += args.banner_h
        out.paste(img, (0, y))
        y += img.height

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out, "PNG", optimize=True)
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)  size = {W}x{total_h}")


if __name__ == "__main__":
    main()
