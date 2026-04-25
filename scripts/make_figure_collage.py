from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
import math

def make_collage(input_dir, output_file, pattern="*.png", cols=4, thumb_w=420, thumb_h=300):
    input_dir = Path(input_dir)
    output_file = Path(output_file)

    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir} matching {pattern}")

    label_h = 45
    margin = 25
    cell_w = thumb_w + margin
    cell_h = thumb_h + label_h + margin

    rows = math.ceil(len(files) / cols)

    collage_w = cols * cell_w + margin
    collage_h = rows * cell_h + margin

    canvas = Image.new("RGB", (collage_w, collage_h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for idx, file in enumerate(files):
        row = idx // cols
        col = idx % cols

        x = margin + col * cell_w
        y = margin + row * cell_h

        img = Image.open(file).convert("RGB")
        img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)

        img_x = x + (thumb_w - img.width) // 2
        img_y = y

        canvas.paste(img, (img_x, img_y))

        label = file.stem.replace("cae_", "").replace("_", " ")
        draw.text((x, y + thumb_h + 8), label, fill="black", font=font)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_file, quality=95)
    print(f"Saved collage: {output_file}")
    print(f"Included {len(files)} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=420)
    parser.add_argument("--thumb-height", type=int, default=300)

    args = parser.parse_args()

    make_collage(
        args.input_dir,
        args.out,
        pattern=args.pattern,
        cols=args.cols,
        thumb_w=args.thumb_width,
        thumb_h=args.thumb_height,
    )