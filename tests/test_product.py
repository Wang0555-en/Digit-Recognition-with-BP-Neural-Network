"""Check the shipped model, long-string segmentation, and leading-zero preservation."""
import atexit
import csv
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "bin/x64/Release/BP.exe"
IMAGE = Path(tempfile.gettempdir()) / "bp_product_long_digits.pgm"
atexit.register(lambda: IMAGE.unlink(missing_ok=True))


def save_pgm(path, pixels, width, height):
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode() + bytes(pixels))


def recognize(path):
    run = subprocess.run(
        [str(EXE), "--recognize", str(path)], cwd=ROOT,
        text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    match = re.search(r"RESULT=(.*)\nCOUNT=(\d+)", run.stdout.replace("\r", ""))
    assert match, run.stdout
    return match.group(1), int(match.group(2))


def main():
    assert EXE.exists(), "Build Release first."
    candidates = {str(i): [] for i in range(10)}
    with (ROOT / "MNIST/mnist_test.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            label = row[0]
            if len(candidates[label]) < 12:
                candidates[label].append(bytes(map(int, row[1:])))
            if all(len(values) == 12 for values in candidates.values()):
                break

    # Pick examples the shipped model recognizes individually. This keeps the
    # test focused on product wiring and segmentation, rather than model accuracy.
    chosen = {}
    for label, images in candidates.items():
        for pixels in images:
            canvas = bytearray(36 * 36)
            for y in range(28):
                canvas[(y + 4) * 36 + 4:(y + 4) * 36 + 32] = pixels[y * 28:y * 28 + 28]
            save_pgm(IMAGE, canvas, 36, 36)
            if recognize(IMAGE) == (label, 1):
                chosen[label] = pixels
                break
        assert label in chosen, f"No recognized fixture for {label}"

    expected = "0123456789012"
    gap, height = 8, 44
    width = len(expected) * 28 + (len(expected) + 1) * gap
    canvas = bytearray(width * height)
    for index, label in enumerate(expected):
        pixels = chosen[label]
        for y in range(28):
            start = (y + 8) * width + gap + index * (28 + gap)
            canvas[start:start + 28] = pixels[y * 28:y * 28 + 28]
    save_pgm(IMAGE, canvas, width, height)
    actual = recognize(IMAGE)
    assert actual == (expected, len(expected)), (actual, expected)
    print("PASS: shipped model, 13-digit ordering, and leading zero.")


if __name__ == "__main__":
    main()
