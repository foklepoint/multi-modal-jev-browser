"""Compose a side-by-side demo GIF from recorded runs.

    uv run --with pillow python examples/make_demo_gif.py demo.gif --run tmp/demo/jev "Jev + LLM fallback" --run tmp/demo/llm "LLM only"

Each run directory comes from `mmjb run ... --record DIR` and holds steps.jsonl (one line per step: n, t, frame, label, cost)
and result.json (outcome, seconds, steps, cost). Time is real: every run starts at 0:00 and the GIF plays them sped up, with a clock.
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

PANEL_WIDTH = 440
PANEL_HEIGHT = 306
TOP_BAR = 34
BOTTOM_BAR = 62
GAP = 10
BACKGROUND = (16, 20, 24)
GREEN = (84, 214, 136)
AMBER = (255, 190, 90)
GREY = (150, 160, 175)
WHITE = (240, 244, 250)
FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size):
    for path in FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def read_run(directory):
    steps = []
    with open(os.path.join(directory, "steps.jsonl")) as handle:
        for line in handle:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    with open(os.path.join(directory, "result.json")) as handle:
        result = json.load(handle)
    for step in steps:
        image = Image.open(os.path.join(directory, step["frame"])).convert("RGB")
        step["image"] = image.resize((PANEL_WIDTH, PANEL_HEIGHT))
    return steps, result


def clock(seconds):
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def current_step(steps, now):
    chosen = None
    for step in steps:
        if step["t"] <= now:
            chosen = step
    return chosen


def draw_panel(canvas, left, title, steps, result, now):
    draw = ImageDraw.Draw(canvas)
    small = load_font(13)
    medium = load_font(16)
    big = load_font(19)
    draw.text((left + 8, 8), title, font=medium, fill=WHITE)
    step = current_step(steps, now)
    finished = now >= result["seconds"] - 0.05
    if step is None:
        canvas.paste(steps[0]["image"], (left, TOP_BAR))
        shade = Image.new("RGBA", (PANEL_WIDTH, PANEL_HEIGHT), (16, 20, 24, 170))
        canvas.paste(shade, (left, TOP_BAR), shade)
        label, cost, number = "starting", 0.0, 0
    else:
        canvas.paste(step["image"], (left, TOP_BAR))
        label, cost, number = step["label"], step["cost"], step["n"]
    y = TOP_BAR + PANEL_HEIGHT
    draw.rectangle([left, y, left + PANEL_WIDTH, y + BOTTOM_BAR], fill=(26, 32, 40))
    draw.text((left + 8, y + 6), f"step {number}   {label}"[:52], font=small, fill=WHITE)
    if finished:
        colour = GREEN if result["outcome"] == "verified" else AMBER
        draw.text((left + 8, y + 28), f"{result['outcome']}   {clock(result['seconds'])}   ${result['cost']:.4f}", font=big, fill=colour)
        draw.text((left + 8, y + 49), f"{result['steps']} steps, {result['seconds']:.1f} s", font=small, fill=GREY)
    else:
        draw.text((left + 8, y + 28), f"{clock(now)}   ${cost:.4f}", font=big, fill=GREY)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--run", nargs=2, action="append", metavar=("DIR", "TITLE"), required=True)
    parser.add_argument("--seconds", type=float, default=14.0, help="GIF length for the longest run")
    parser.add_argument("--fps", type=float, default=3.0)
    args = parser.parse_args()
    runs = [(title, *read_run(directory)) for directory, title in args.run]
    longest = max(result["seconds"] for _title, _steps, result in runs)
    speed = longest / args.seconds
    width = len(runs) * PANEL_WIDTH + (len(runs) - 1) * GAP
    height = TOP_BAR + PANEL_HEIGHT + BOTTOM_BAR + 30
    frames = []
    durations = []
    total_frames = int(args.seconds * args.fps)
    for index in range(total_frames + 1):
        now = longest * index / total_frames
        canvas = Image.new("RGB", (width, height), BACKGROUND)
        for position, (title, steps, result) in enumerate(runs):
            draw_panel(canvas, position * (PANEL_WIDTH + GAP), title, steps, result, now)
        draw = ImageDraw.Draw(canvas)
        draw.text((8, height - 24), f"real time, played {speed:.1f}x faster   clock {clock(now)}", font=load_font(13), fill=GREY)
        frames.append(canvas.quantize(colors=96, method=Image.Quantize.MEDIANCUT))
        durations.append(int(1000 / args.fps))
    durations[-1] = 4000
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True, disposal=2)
    size = os.path.getsize(args.output)
    print(f"wrote {args.output}: {len(frames)} frames, {size / (1024 * 1024):.1f} MB, {speed:.1f}x speed", file=sys.stderr)


if __name__ == "__main__":
    main()
