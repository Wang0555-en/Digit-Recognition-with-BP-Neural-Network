"""Exercise full-data command, resume, checkpoints and invalid input in isolation."""
import csv
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / 'bin/x64/Release/BP.exe'


def main():
    with tempfile.TemporaryDirectory(prefix='bp-full-training-') as name:
        folder = Path(name)
        rows = [[label] + [(i+label) % 256 for i in range(784)] for label in (0, 2, 5, 9)]
        for filename in ('mnist_train.csv', 'mnist_test.csv'):
            with (folder/filename).open('w', newline='') as file:
                csv.writer(file).writerows(rows)

        def run(*args, code=0):
            proc = subprocess.run([str(EXE), *args], cwd=folder, capture_output=True,
                                  text=True, encoding='utf-8', errors='replace', timeout=120)
            assert proc.returncode == code, proc.stdout + proc.stderr
            return proc.stdout + proc.stderr

        run('--train-full', '1', '0.05', '--resume', code=2)
        for args in [('0', '.3'), ('1', 'nan'), ('1', '0'), ('2bad', '.3')]:
            run('--train-full', *args, code=4)
        output = run('--train-full', '2', '0.05')
        assert 'Samples 4/4' in output and '从头训练' in output
        exported = list(csv.reader((folder/'training_dataset.csv').open()))
        assert sorted(exported) == sorted([[str(v) for v in row] for row in rows])
        history = list(csv.reader((folder/'training_history.csv').open()))
        assert len(history) == 3
        model = folder/'model.bin'
        checkpoint = folder/'model.checkpoint.bin'
        assert model.read_bytes() == checkpoint.read_bytes()
        assert not (folder/'model.bin.tmp').exists()
        before = model.read_bytes()
        assert '继续训练' in run('--train-full', '1', '0.01', '--resume')
        assert model.read_bytes() != before
        assert 'ACCURACY=' in run('--evaluate', '-1')
        preserved = model.read_bytes()
        (folder/'mnist_train.csv').write_text('10,' + ','.join(['0']*784) + '\n')
        assert '数据无效' in run('--train-full', '1', '.1', code=3)
        assert model.read_bytes() == preserved
    print('PASS: full command, all rows, progress, checkpoint, fresh/resume, invalid data, preserved model.')


if __name__ == '__main__':
    main()
