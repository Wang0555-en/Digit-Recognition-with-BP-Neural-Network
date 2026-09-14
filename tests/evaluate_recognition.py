"""Fixed, unselected MNIST images through the actual executable and segmentation.

Reports coverage separately from accepted accuracy; rejected digits are not correct.
"""
import csv
import json
import subprocess
import tempfile
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=300)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stats = {k: dict(total=0, correct=0, wrong=0, rejected=0) for k in ('original', 'dim')}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'digit.pgm'
        with (ROOT / 'MNIST/mnist_test.csv').open() as f:
            for index, row in enumerate(csv.reader(f)):
                if index >= args.count:
                    break
                for kind, factor in [('original', 1.0), ('dim', 0.35)]:
                    canvas = bytearray(36 * 36)
                    pixels = bytes(round(int(v) * factor) for v in row[1:])
                    for y in range(28):
                        canvas[(y+4)*36+4:(y+4)*36+32] = pixels[y*28:y*28+28]
                    path.write_bytes(b'P5\n36 36\n255\n' + canvas)
                    run = subprocess.run([str(ROOT / 'bin/x64/Release/BP.exe'), '--recognize', str(path)],
                                         cwd=ROOT, capture_output=True, text=True, errors='replace', timeout=30)
                    assert run.returncode in (0, 3), run.stdout + run.stderr
                    result = next(s[7:] for s in run.stdout.splitlines() if s.startswith('RESULT='))
                    stat = stats[kind]
                    stat['total'] += 1
                    stat['rejected' if '?' in result or not result else 'correct' if result == row[0] else 'wrong'] += 1
    for s in stats.values():
        accepted = s['correct'] + s['wrong']
        s['coverage'] = accepted / s['total']
        s['accepted_accuracy'] = s['correct'] / accepted if accepted else None
        s['correct_over_total'] = s['correct'] / s['total']
    args.output.write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
