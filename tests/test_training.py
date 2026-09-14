"""用少量人工构造的样本检查完整训练流程，不用于评价识别准确率。"""
import csv
import argparse
import math
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "BP.exe"


def run(folder, expected_code=0):
    result = subprocess.run(
        [str(EXE), "--console"], input="1\n", text=True, encoding="utf-8",
        errors="replace", cwd=folder, capture_output=True, timeout=120,
    )
    assert result.returncode == expected_code, result.stdout + result.stderr
    return result.stdout + result.stderr


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.reader(file))


def main():
    assert EXE.exists(), "请先编译生成 BP.exe，并将 OpenCV 的 bin 目录加入 PATH。"
    # 覆盖 0~255 全部灰度值，以及数字标签的边界值。
    rows = [[label] + [(i + label) % 256 for i in range(784)] for label in (0, 3, 9)]
    expected = [[str(value) for value in row] for row in rows]
    with tempfile.TemporaryDirectory(prefix="bp-training-test-") as name:
        folder = Path(name)
        for filename in ("mnist_train.csv", "mnist_test.csv"):
            with (folder / filename).open("w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerows(rows)

        for attempt in range(2):
            run(folder)
            assert read_csv(folder / "training_dataset.csv") == expected
            history = read_csv(folder / "training_history.csv")
            assert history[0] == ["epoch", "learning_rate", "loss", "test_accuracy"]
            assert len(history) == 51  # 表头 + 50 轮，不会跨次追加。
            lr = 0.3
            for epoch, record in enumerate(history[1:]):
                assert int(record[0]) == epoch + 1
                assert math.isclose(float(record[1]), lr)
                assert math.isfinite(float(record[2])) and float(record[2]) >= 0
                assert 0 <= float(record[3]) <= 1
                if epoch > 0 and epoch % 10 == 0:
                    lr *= 0.5
            assert (folder / "model.bin").stat().st_size > 0
            # 第二次直接使用导出的数据集，验证原来的读取函数能再次训练。
            shutil.copyfile(folder / "training_dataset.csv", folder / "mnist_train.csv")

        # 数据集整体放在 MNIST 子目录时，也应能训练并正确导出。
        (folder / "MNIST").mkdir()
        for filename in ("mnist_train.csv", "mnist_test.csv"):
            (folder / filename).rename(folder / "MNIST" / filename)
        output = run(folder)
        assert "MNIST/mnist_train.csv" in output
        assert "MNIST/mnist_test.csv" in output
        assert read_csv(folder / "training_dataset.csv") == expected

        # 用同名目录模拟输出无法创建，确认不会继续训练保存模型。
        (folder / "model.bin").unlink()
        (folder / "training_dataset.csv").unlink()
        (folder / "training_dataset.csv").mkdir()
        output = run(folder, expected_code=3)
        assert "训练已取消" in output
        assert not (folder / "model.bin").exists()

    with tempfile.TemporaryDirectory(prefix="bp-missing-data-") as name:
        folder = Path(name)
        output = run(folder, expected_code=3)
        assert "数据加载失败" in output
        assert not (folder / "training_dataset.csv").exists()
    print("PASS: 样本还原、重复训练读回、50 轮记录、学习率、覆盖行为和失败分支。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=EXE, help="待验证的 BP.exe 路径")
    EXE = parser.parse_args().exe.resolve()
    main()
