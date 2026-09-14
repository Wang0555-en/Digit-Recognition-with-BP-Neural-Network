"""Launch full-data BP training with backups; no third-party Python packages needed."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.3)
    parser.add_argument('--resume', action='store_true', help='Continue from model.bin; restart learning-rate schedule')
    parser.add_argument('--check-data', action='store_true', help='Count source rows without training or changing models')
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.lr) or not 0 < args.lr <= 1:
        parser.error('epochs >= 1 and 0 < lr <= 1 required')
    counts = {}
    for name in ('mnist_train.csv', 'mnist_test.csv'):
        path = ROOT / name
        if not path.is_file():
            path = ROOT / 'MNIST' / name
        with path.open(encoding='utf-8') as source:
            count = sum(1 for line in source if line.strip())
        if count == 0:
            parser.error(f'Empty dataset: {path}')
        counts[name] = count
        print(f'{path}: {count} samples', flush=True)
    if args.check_data:
        return 0
    exe = ROOT / 'bin/x64/Release/BP.exe'
    if not exe.is_file():
        parser.error('Build Release first: python tools/build.py')
    if args.resume and not (ROOT / 'model.bin').is_file():
        parser.error('--resume requires model.bin')
    # 原训练直接覆盖模型和CSV。启动前备份，备份失败则不进入训练。
    backup = ROOT / 'training_backups' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup.mkdir(parents=True)
    for name in ('model.bin', 'model.checkpoint.bin', 'training_dataset.csv', 'training_history.csv'):
        path = ROOT / name
        if path.exists():
            shutil.copy2(path, backup / name)
    (backup / 'run.json').write_text(json.dumps({**vars(args), 'counts': counts}, indent=2), encoding='utf-8')
    print(f'Backup: {backup}', flush=True)
    print('Close BP Studio before training; reopen it after completion to load the new model.', flush=True)
    command = [str(exe), '--train-full', str(args.epochs), str(args.lr)]
    if args.resume:
        command.append('--resume')
    code = subprocess.call(command, cwd=ROOT)
    print('Training complete. New model: model.bin' if code == 0 else
          'Training failed/interrupted. Keep the backup; inspect the last model.checkpoint.bin.', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
