from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
import math
import re
import textwrap


FOLDER_DESCRIPTIONS = {
    "CAE": "Convolutional Autoencoder results across MVTec AD categories, including training behavior and reconstruction quality.",
    "DAE": "Denoising Autoencoder training results across MVTec AD categories.",
    "VAE": "Variational Autoencoder training results across MVTec AD categories.",
    "SKIP CAE": "Skip-connection CAE results, including loss curves, ROC curves, anomaly scores, and reconstruction examples.",
    "PATCHCORE": "PatchCore anomaly detection ROC curves across MVTec AD categories.",
    "CNN": "CNN classification performance visualizations, including training curves and confusion matrix.",
    "CROSS DATASET": "Cross-dataset generalization results and performance visualizations.",
    "NEU": "NEU surface defect dataset category and distribution visualizations.",
}


SECTION_RULES = [
    ("Dataset Overview", ["dataset", "categories", "distribution", "samples"]),
    ("Training Curves", ["training", "train", "loss_curve"]),
    ("Loss Curves", ["loss"]),
    ("Reconstructions", ["reconstruction", "reconstructions"]),
    ("ROC Curves", ["roc"]),
    ("Anomaly Scores", ["scores"]),
    ("Performance Comparison", ["comparison", "bar", "heatmap", "radar", "metrics", "performance"]),
    ("Generalization", ["generalization", "cross_dataset"]),
    ("Confusion Matrix", ["confusion"]),
    ("Other Figures", []),
]


def safe_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]", "", name)


def load_font(size: int, bold: bool = False):
    candidates = []

    if bold:
        candidates = [
            "arialbd.ttf",
            "Arial Bold.ttf",
            "DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "arial.ttf",
            "Arial.ttf",
            "DejaVuSans.ttf",
        ]

    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except:
            pass

    return ImageFont.load_default()


def detect_section(file: Path) -> str:
    stem = file.stem.lower()

    for section, keywords in SECTION_RULES:
        if not keywords:
            continue
        if any(keyword in stem for keyword in keywords):
            return section

    return "Other Figures"


def section_sort_key(file: Path):
    section = detect_section(file)
    section_order = [name for name, _ in SECTION_RULES]
    return section_order.index(section), file.stem.lower()


def pretty_caption(file: Path, folder_name: str) -> str:
    stem = file.stem

    # Remove repeated model prefix from captions
    prefixes = [
        folder_name.lower().replace(" ", "_"),
        "cae",
        "dae",
        "vae",
        "skip_cae",
        "patchcore",
        "cnn",
        "cross_dataset",
        "neu",
    ]

    clean = stem.lower()

    for prefix in prefixes:
        if clean.startswith(prefix + "_"):
            clean = clean[len(prefix) + 1:]

    clean = clean.replace("_", " ")

    replacements = {
        "roc": "ROC curve",
        "auc": "AUC",
        "mvtec": "MVTec",
        "neu": "NEU",
        "cae": "CAE",
        "dae": "DAE",
        "vae": "VAE",
        "cnn": "CNN",
    }

    words = []
    for word in clean.split():
        words.append(replacements.get(word, word.capitalize()))

    return " ".join(words)


def draw_wrapped_text(draw, text, xy, font, fill, max_width_chars, line_spacing=6):
    x, y = xy
    lines = textwrap.wrap(text, width=max_width_chars)

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_spacing

    return y


def draw_panel_label(draw, label, box, font):
    x, y, w, h = box
    pad = 10

    label_bbox = draw.textbbox((0, 0), label, font=font)
    label_w = label_bbox[2] - label_bbox[0] + 2 * pad
    label_h = label_bbox[3] - label_bbox[1] + 2 * pad

    draw.rounded_rectangle(
        [x + 12, y + 12, x + 12 + label_w, y + 12 + label_h],
        radius=10,
        fill=(30, 30, 30),
    )
    draw.text((x + 12 + pad, y + 12 + pad - 2), label, font=font, fill="white")


def make_professional_collage(
    folder: Path,
    output_file: Path,
    cols: int = 4,
    thumb_w: int = 720,
    thumb_h: int = 500,
    dpi: int = 300,
):
    image_files = sorted(folder.glob("*.png"), key=section_sort_key)

    if not image_files:
        print(f"Skipping {folder.name}: no PNG files found.")
        return

    folder_name = folder.name
    description = FOLDER_DESCRIPTIONS.get(
        folder_name,
        f"Summary figure plate for {folder_name} results.",
    )

    margin = 70
    gutter = 38
    header_h = 250
    section_h = 70
    caption_h = 95
    footer_h = 80

    cell_w = thumb_w
    cell_h = thumb_h + caption_h

    # Group by section
    grouped = {}
    for file in image_files:
        grouped.setdefault(detect_section(file), []).append(file)

    ordered_sections = [
        section for section, _ in SECTION_RULES
        if section in grouped
    ]

    # Estimate canvas height
    total_h = header_h + margin + footer_h
    for section in ordered_sections:
        n = len(grouped[section])
        rows = math.ceil(n / cols)
        total_h += section_h + rows * cell_h + max(0, rows - 1) * gutter + margin

    canvas_w = margin * 2 + cols * cell_w + (cols - 1) * gutter
    canvas_h = total_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(54, bold=True)
    subtitle_font = load_font(27)
    section_font = load_font(34, bold=True)
    caption_font = load_font(22)
    panel_font = load_font(24, bold=True)
    footer_font = load_font(20)

    # Header
    title = f"{folder_name} Figure Summary"
    draw.text((margin, 48), title, font=title_font, fill=(20, 20, 20))

    draw_wrapped_text(
        draw,
        description,
        (margin, 125),
        subtitle_font,
        fill=(70, 70, 70),
        max_width_chars=95,
        line_spacing=8,
    )

    draw.text(
        (margin, 205),
        f"Total figures included: {len(image_files)}",
        font=footer_font,
        fill=(90, 90, 90),
    )

    # Header underline
    draw.line(
        [(margin, header_h - 20), (canvas_w - margin, header_h - 20)],
        fill=(190, 190, 190),
        width=3,
    )

    y = header_h
    panel_index = 0

    for section in ordered_sections:
        files = grouped[section]

        # Section title
        draw.rounded_rectangle(
            [margin, y, canvas_w - margin, y + section_h - 10],
            radius=18,
            fill=(242, 242, 242),
        )
        draw.text(
            (margin + 24, y + 15),
            f"{section} ({len(files)})",
            font=section_font,
            fill=(25, 25, 25),
        )

        y += section_h

        for idx, file in enumerate(files):
            row = idx // cols
            col = idx % cols

            x = margin + col * (cell_w + gutter)
            yy = y + row * (cell_h + gutter)

            # Panel background
            draw.rounded_rectangle(
                [x - 12, yy - 12, x + cell_w + 12, yy + cell_h - 8],
                radius=20,
                fill=(248, 248, 248),
                outline=(220, 220, 220),
                width=2,
            )

            img = Image.open(file).convert("RGB")
            img.thumbnail((thumb_w - 30, thumb_h - 30), Image.LANCZOS)

            img_x = x + (cell_w - img.width) // 2
            img_y = yy + (thumb_h - img.height) // 2

            canvas.paste(img, (img_x, img_y))

            panel_label = f"({chr(97 + panel_index % 26)})"
            if panel_index >= 26:
                panel_label = f"({chr(97 + (panel_index // 26) - 1)}{chr(97 + panel_index % 26)})"

            draw_panel_label(
                draw,
                panel_label,
                (x, yy, cell_w, thumb_h),
                panel_font,
            )

            caption = pretty_caption(file, folder_name)
            caption = f"{panel_label} {caption}"

            draw_wrapped_text(
                draw,
                caption,
                (x + 10, yy + thumb_h + 15),
                caption_font,
                fill=(40, 40, 40),
                max_width_chars=42,
                line_spacing=5,
            )

            panel_index += 1

        rows = math.ceil(len(files) / cols)
        y += rows * cell_h + max(0, rows - 1) * gutter + margin

    # Footer
    footer_text = "Generated automatically from outputs/figures subfolders. Captions are derived from file names."
    draw.line(
        [(margin, canvas_h - footer_h), (canvas_w - margin, canvas_h - footer_h)],
        fill=(210, 210, 210),
        width=2,
    )
    draw.text(
        (margin, canvas_h - footer_h + 25),
        footer_text,
        font=footer_font,
        fill=(100, 100, 100),
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_file, dpi=(dpi, dpi), quality=95)

    print(f"Saved: {output_file}")
    print(f"  Included: {len(image_files)} images")
    print(f"  Size: {canvas_w} x {canvas_h}px")
    print(f"  DPI: {dpi}")


def main():
    parser = argparse.ArgumentParser(
        description="Create high-resolution professional figure collages for each subfolder."
    )

    parser.add_argument(
        "--root",
        default="outputs/figures",
        help="Root figures directory. Default: outputs/figures",
    )

    parser.add_argument(
        "--out",
        default="outputs/figures/professional_collages",
        help="Output directory. Default: outputs/figures/professional_collages",
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns. Default: 4",
    )

    parser.add_argument(
        "--thumb-width",
        type=int,
        default=720,
        help="Thumbnail width in pixels. Default: 720",
    )

    parser.add_argument(
        "--thumb-height",
        type=int,
        default=500,
        help="Thumbnail height in pixels. Default: 500",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG DPI metadata. Default: 300",
    )

    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.out)

    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    subfolders = [
        folder for folder in sorted(root.iterdir())
        if folder.is_dir()
        and folder.name.lower() not in ["collages", "professional_collages"]
    ]

    if not subfolders:
        print(f"No figure subfolders found in {root}")
        return

    print(f"Found {len(subfolders)} figure folders.")

    for folder in subfolders:
        output_file = output_dir / f"{safe_filename(folder.name)}_professional_collage.png"

        make_professional_collage(
            folder=folder,
            output_file=output_file,
            cols=args.cols,
            thumb_w=args.thumb_width,
            thumb_h=args.thumb_height,
            dpi=args.dpi,
        )

    print("\nDone.")
    print(f"Professional collages saved in: {output_dir}")


if __name__ == "__main__":
    main()