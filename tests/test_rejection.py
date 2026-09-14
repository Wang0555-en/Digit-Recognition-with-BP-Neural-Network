"""Exercise real CLI with controlled valid models, without modifying model.bin."""
import math
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / 'bin/x64/Release/BP.exe'

def constant_model(scores):
    data = bytearray(struct.pack('<i', 2))
    for n, m in [(784, 128), (128, 10)]:
        data += struct.pack('<ii', n, m) + bytes(n*m*8)
        data += struct.pack('<ii', 1, m)
        values = [math.log(p/(1-p)) for p in scores] if m == 10 else [0]*m
        data += struct.pack('<'+'d'*m, *values)
    return data

def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        pixels = bytearray(40*40)
        for y in range(10,30):
            pixels[y*40+16:y*40+24] = bytes([255])*8
        (root/'digit.pgm').write_bytes(b'P5\n40 40\n255\n'+pixels)
        cases = [([.1]*10, '?', 1), ([.8,.7]+[.05]*8, '?', 1),
                 ([.9]+[.05]*9, '0', 0)]
        for scores, expected, rejected in cases:
            (root/'model.bin').write_bytes(constant_model(scores))
            run = subprocess.run([str(EXE), '--recognize', str(root/'digit.pgm')], cwd=root,
                                 capture_output=True, text=True, errors='replace', timeout=30)
            assert run.returncode == 0, run.stdout+run.stderr
            assert f'RESULT={expected}\nCOUNT=1\nUNCERTAIN={rejected}' in run.stdout, run.stdout
        # Even a high model score must not confirm an obviously wide touching region.
        pixels = bytearray(40*40)
        for y in range(16,24):
            pixels[y*40+5:y*40+35] = bytes([255])*30
        (root/'digit.pgm').write_bytes(b'P5\n40 40\n255\n'+pixels)
        run = subprocess.run([str(EXE), '--recognize', str(root/'digit.pgm')], cwd=root,
                             capture_output=True, text=True, errors='replace', timeout=30)
        assert run.returncode == 0 and 'RESULT=?\nCOUNT=1\nUNCERTAIN=1' in run.stdout, run.stdout
    print('PASS: low scores, ambiguous scores, accepted zero, touching rejection.')

if __name__ == '__main__':
    main()
