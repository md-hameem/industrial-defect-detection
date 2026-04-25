from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
import math
import re


def safe_filename(name: str) -> str:
    """Convert folder names like 'SKIP CAE' into safe file names."""
    name = name.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]", "", name)


def load_font(size=18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()


def make_collage(
    image_files,
    output_file,
    title=None,
    cols=4,
    thumb_w=420,
    thumb_h=300,
    margin=25,
    bg_color="white",
):
    image_files = list(image_files)

    if not image_files:
        print(f"Skipped empty folder: {output_file}")
        return

    label_h = 50
    title_h = 70 if title else 0

    rows = math.ceil(len(image_files) / cols)

    cell_w = thumb_w + margin
    cell_h = thumb_h + label_h + margin

    collage_w = cols * cell_w + margin
    collage_h = rows * cell_h + margin + title_h

    canvas = Image.new("RGB", (collage_w, collage_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(28)
    label_font = load_font(16)

    if title:
        draw.text((margin, 20), title, fill="black", font=title_font)

    for idx, file in enumerate(image_files):
        row = idx // cols
        col = idx % cols

        x = margin + col * cell_w
        y = margin + title_h + row * cell_h

        img = Image.open(file).convert("RGB")
        img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)

        img_x = x + (thumb_w - img.width) // 2
        img_y = y

        canvas.paste(img, (img_x, img_y))

        label = file.stem
        label = label.replace("_", " ")

        # Shorten very long labels
        if len(label) > 42:
            label = label[:39] + "..."

        draw.text((x, y + thumb_h + 8), label, fill="black", font=label_font)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_file, quality=95)

    print(f"Saved: {output_file}")
    print(f"  Images included: {len(image_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create one collage PNG for each figure subfolder."
    )

    parser.add_argument(
        "--root",
        default="outputs/figures",
        help="Root figures directory. Default: outputs/figures",
    )

    parser.add_argument(
        "--out",
        default="outputs/figures/collages",
        help="Output directory for collages. Default: outputs/figures/collages",
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns in each collage. Default: 4",
    )

    parser.add_argument(
        "--thumb-width",
        type=int,
        default=420,
        help="Thumbnail width. Default: 420",
    )

    parser.add_argument(
        "--thumb-height",
        type=int,
        default=300,
        help="Thumbnail height. Default: 300",
    )

    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.out)

    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    subfolders = [
        folder for folder in sorted(root.iterdir())
        if folder.is_dir() and folder.name.lower() != "collages"
    ]

    if not subfolders:
        print(f"No subfolders found in: {root}")
        return

    print(f"Found {len(subfolders)} folders.")

    for folder in subfolders:
        image_files = sorted(folder.glob("*.png"))

        if not image_files:
            print(f"Skipping {folder.name}: no PNG files found.")
            continue

        output_file = output_dir / f"{safe_filename(folder.name)}_collage.png"

        make_collage(
            image_files=image_files,
            output_file=output_file,
            title=f"{folder.name} Figures",
            cols=args.cols,
            thumb_w=args.thumb_width,
            thumb_h=args.thumb_height,
        )

    print("\nDone. All collages saved in:")
    print(output_dir)


if __name__ == "__main__":
    main()