# BP Studio · 手写数字识别

基于 **C++17、Win32 和 OpenCV** 的 Windows 桌面数字识别项目，使用自行实现的 BP 神经网络，支持鼠标手绘、图片导入、单行多位数字识别和本地模型训练。

项目将矩阵运算、前向传播、反向传播、图像分割与桌面交互串联起来，适合课程设计、教学演示和算法学习。识别及训练均在本机执行，无需云端 API。

## 功能特点

- **手绘输入**：鼠标绘制数字，支持撤销笔画和清空画布。
- **图片导入**：通过文件选择或输入路径载入图片，支持 PNG、JPG、BMP 等格式。
- **多位识别**：按数字间的空白列分割，从左到右输出字符串，保留前导零，例如 `00123`。
- **结果核对**：展示分割框与逐位模型得分；无法确认的数字以 `?` 占位，保留位置。
- **拒识与复制控制**：低分、类别得分接近或疑似粘连时拒识；存在未确认位时，桌面界面禁用复制按钮。
- **模型训练**：支持 MNIST CSV 输入、全量训练、从现有权重继续训练、每轮检查点及训练记录导出。
- **自动备份**：通过全量训练脚本启动时，先备份已有模型与记录，再进入训练。

当前主要面向**背景简单、数字间隔清晰的单行手写数字**。多行排版、复杂背景、严重倾斜或粘连数字不属于可靠支持范围。

## 项目结构

```text
BP/
├── README.md
├── .gitignore
├── BP.sln                       # Visual Studio 解决方案
├── BP.vcxproj                   # C++ 工程配置
├── main.cpp                     # 程序入口、训练菜单、命令行训练与评估
├── DesktopUI.cpp                # Windows 中文桌面界面
├── Recognizer.h / .cpp          # 识别、拒识与结果汇总
├── ImageSegmenter.h / .cpp      # 预处理、数字分割与分割框绘制
├── Network.h / .cpp             # 神经网络、反向传播、模型读写
├── Layer.h / .cpp               # 网络层、激活函数与参数更新
├── matrix.h / .cpp              # 基础矩阵运算
├── DatasetWriter.h / .cpp       # 训练数据与逐轮指标导出
├── model.bin                    # 随源码一并提供时，可直接用于识别
├── Start_BP.cmd                 # 桌面启动入口
├── Start_Training.cmd           # 原控制台菜单入口
├── Start_Full_Training.cmd      # 全量训练入口
├── 全量训练指南.md
├── tools/
│   ├── build.py                 # MSBuild 构建辅助脚本
│   └── train_full.py            # 全量训练、参数检查与备份
└── tests/
    ├── test_product.py          # 多位数字顺序与前导零
    ├── test_rejection.py        # 拒识行为
    ├── test_training.py         # 原训练流程与记录导出
    ├── test_full_training.py    # 全量训练、检查点与继续训练
    └── evaluate_recognition.py  # 经图像分割的识别评估
```

以下目录由开发者在本地准备或由程序生成，默认不纳入源码仓库：

```text
dependencies/opencv/   # OpenCV 开发依赖
MNIST/                # 训练与测试数据
bin/                  # 编译输出
obj/                  # 编译中间文件
training_backups/     # 历次训练备份
outputs/              # 本地报告、演示材料及评估输出
```

## 从源码构建

### 环境要求

| 组件 | 当前工程配置 |
| --- | --- |
| 操作系统 | Windows，x64 |
| 编译环境 | Visual Studio，C++ 桌面开发组件与 Windows SDK |
| 平台工具集 | MSVC `v145` |
| C++ 标准 | C++17，UTF-8 源码 |
| 图像处理库 | OpenCV 4.12.0，MSVC x64 版本 |
| Python | Python 3，用于辅助脚本和测试；桌面程序本身不依赖 Python |

Python 脚本仅使用标准库，无需安装 PyTorch、TensorFlow 或 Python 版 OpenCV。

### 1. 准备 OpenCV

将 OpenCV 开发文件放在以下位置，目录名称需与 `BP.vcxproj` 一致：

```text
dependencies/opencv/build/
├── include/opencv2/...
└── x64/vc16/
    ├── lib/opencv_world4120.lib
    ├── lib/opencv_world4120d.lib
    ├── bin/opencv_world4120.dll
    └── bin/opencv_world4120d.dll
```

Release 使用无 `d` 后缀的库，Debug 使用带 `d` 后缀的库。只构建 Release 时，仅需对应的 Release 库和 DLL。不要混用 MinGW 与 MSVC 的二进制库。

若使用其他 OpenCV 版本或安装位置，需同步调整 `BP.vcxproj` 中的路径、库名和 DLL 复制配置。若未安装 `v145`，请安装对应工具集，或在 Visual Studio 中修改平台工具集后重新验证构建。

### 2. 生成程序

在项目根目录打开 PowerShell：

```powershell
python tools/build.py Release
```

或打开 `BP.sln`，选择 **Release | x64**，生成解决方案。

辅助脚本通过 Visual Studio Installer 的 `vswhere.exe` 查找 MSBuild，不会自动安装 Visual Studio 或 OpenCV。工程会将匹配的 OpenCV DLL 复制到可执行文件旁。

```text
bin/x64/Release/BP.exe
bin/x64/Release/opencv_world4120.dll
```

Debug 构建命令为 `python tools/build.py Debug`，输出目录为 `bin/x64/Debug/`。三个启动脚本默认使用 Release 程序。

### 3. 启动识别

确保项目根目录有有效的 `model.bin`，然后双击 `Start_BP.cmd`，或执行：

```powershell
.\Start_BP.cmd
```

桌面识别不需要 MNIST 数据。模型缺失、损坏或结构不匹配时，界面会提示并禁用识别；此时需要提供匹配的模型，或按下文训练生成。

使用已有可执行文件时，还需安装与 MSVC 构建匹配的 x64 C++ 运行库。直接下载源码不会自动得到 `BP.exe` 与 OpenCV DLL。

## 使用方法

1. 选择“手绘数字”，在画布中书写；或者选择图片、填写路径并载入。
2. 多位数字从左到右排列，保留明显间距。
3. 点击“开始识别”，查看结果文本、分割框和逐位得分。
4. 如出现 `?`，重写对应数字或更换图片，再次识别。
5. 全部位确认后，可点击“复制结果”；复制前仍应人工核对。

桌面导入限制为单张不超过 **32 MiB、2400 万像素**，分割数量上限为 **256 位**。这些是输入处理上限，不代表极限输入仍能保证准确识别。

### 得分与拒识

网络输出使用 Sigmoid，各类别分数不保证总和为 1。界面百分数是**模型得分，不是识别正确率或经过校准的概率**。

当前识别代码仅在以下条件全部满足时输出数字：

- 各类别输出均为有限数值；
- 最高得分不低于 `0.65`；
- 最高分与次高分之差不小于 `0.25`；
- 分割区域未被标记为疑似粘连。

否则输出 `?`。粘连判断使用区域宽高比启发式规则，可能误判或漏判；拒识能够减少直接猜测，但不能保证所有接受结果都正确。

## 算法流程

```text
输入图片 / 手绘画布
    ↓
灰度化、Otsu 二值化、背景方向判断
    ↓
连通域噪点过滤、按空白列分割单行数字
    ↓
保留灰度、亮度归一化、缩放到 20×20 范围
    ↓
填充到 28×28 并按图像矩居中
    ↓
784 → 128 → 10 的全连接 BP 网络
    ↓
得分比较、拒识判断、文本与分割框输出
```

矩阵计算、网络层、前向传播及反向传播由项目 C++ 代码实现；OpenCV 用于图像读取与预处理。训练输入像素除以 255，标签编码为 10 类目标向量，逐样本更新参数。

## 模型训练

### 数据格式

自行准备以下 CSV，放在 `MNIST/` 下，也可直接放在项目根目录；程序优先读取根目录同名文件。

```text
MNIST/mnist_train.csv
MNIST/mnist_test.csv
```

CSV **不带表头**，每行恰好 785 列：第一列为 `0–9` 标签，后 784 列为按行展开的 28×28 灰度像素，像素为 `0–255` 整数。

```text
标签,像素0,像素1,...,像素783
```

上面仅说明结构，不要将它写为表头。MNIST 原始 IDX 文件或 `.gz` 文件不能直接传入当前读取器，须先转换为上述 CSV；仓库目前未包含转换脚本。读取器会校验数据格式，非法数据会使训练停止。

### 全量训练入口

先检查数据文件的非空行数，此步骤不训练或修改模型，也不替代 C++ 读取器的逐字段校验：

```powershell
python tools/train_full.py --check-data
```

关闭桌面程序，然后双击 `Start_Full_Training.cmd`，或运行：

```powershell
# 默认从头训练，读取文件中的全部样本
python tools/train_full.py --epochs 50 --lr 0.3

# 从 model.bin 的权重继续训练，重新开始轮数和学习率计划
python tools/train_full.py --epochs 10 --lr 0.03 --resume
```

脚本在训练前将已有模型、检查点及训练 CSV 备份到 `training_backups/时间戳/`，并保存本次参数。训练完成后重新打开桌面程序，以加载新模型。

| 设置 | 默认行为 |
| --- | --- |
| 网络结构 | `784 → 128 → 10` |
| 激活函数 | Sigmoid |
| 训练轮数 | 50 |
| 初始学习率 | 0.3 |
| 数据范围 | 全量入口读取 CSV 中的全部样本 |
| 更新方式 | 每轮洗牌后逐样本更新 |
| 学习率衰减 | 第 11、21、31、41 轮结束后减半，从下一轮生效 |
| 测试数据用途 | 逐轮计算准确率，不参与梯度更新 |

`--resume` 读取的是 `model.bin`，不会自动选择 `model.checkpoint.bin`，也不会恢复原来的轮数或随机数状态。检查点恢复及更多操作见 [全量训练指南](全量训练指南.md)。

原 `Start_Training.cmd` 保留交互菜单：训练最多读取 60,000 条训练样本和 10,000 条测试样本，固定样本顺序；有有效模型时继续训练，**该入口不自动备份**。

### 输出文件

| 文件 | 说明 |
| --- | --- |
| `model.bin` | 全部轮次完成后保存的模型参数 |
| `model.checkpoint.bin` | 全量训练每轮更新的模型检查点 |
| `training_dataset.csv` | 实际训练样本，第一轮导出一次，无表头 |
| `training_history.csv` | 每轮学习率、损失与测试准确率 |
| `training_backups/` | 全量训练脚本生成的历史备份及参数 |

训练历史的表头为：

```csv
epoch,learning_rate,loss,test_accuracy
```

`loss` 为每个样本各输出误差平方和的样本平均值；`test_accuracy` 为 `0–1` 比例。每次训练会重新创建训练 CSV，最终保存会更新模型；中途失败可能留下不完整记录。

## 命令行接口

以下命令均在项目根目录执行：

```powershell
# 桌面界面
.\bin\x64\Release\BP.exe

# 原控制台菜单
.\bin\x64\Release\BP.exe --console

# 识别图片：替换为实际文件路径
.\bin\x64\Release\BP.exe --recognize "C:\images\digits.png"

# 在测试 CSV 的前 2000 条样本上评估模型
.\bin\x64\Release\BP.exe --evaluate 2000

# 使用标准 MNIST 测试集时，评估 10000 条
.\bin\x64\Release\BP.exe --evaluate 10000
```

识别输出示例（仅表示格式）：

```text
RESULT=00?23
COUNT=5
UNCERTAIN=1
```

`COUNT` 包含拒识占位，`UNCERTAIN` 是未确认位数。识别命令返回 `0` 表示产生了分割结果，仍可能含有 `?`；调用方需要检查 `UNCERTAIN`。模型无效返回 `2`，无数字返回 `3`，图片处理异常返回 `4`。

也可直接调用 `BP.exe --train-full 50 0.3`，但这会绕过 Python 脚本的自动备份。日常训练建议使用上面的 `tools/train_full.py` 入口。

## 测试与评估

构建 Release 后，可在根目录运行现有测试：

```powershell
python tests/test_rejection.py
python tests/test_training.py --exe bin/x64/Release/BP.exe
python tests/test_full_training.py
python tests/test_product.py
```

前三项使用临时目录中的受控模型或人工数据验证流程；产品测试还需要根目录模型与 `MNIST/mnist_test.csv`。产品测试会先选择模型能单独识别的样本再拼接，因此用于检查数字顺序、长串和前导零，不用于估计总体准确率。

若需评估经过图片分割后的效果，可运行：

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
python tests/evaluate_recognition.py --count 300 --output outputs/recognition_eval.json
```

该脚本使用测试 CSV 中固定顺序的样本，分别检查原亮度和降低亮度的图片，区分正确、错误和拒识，并报告接受覆盖率与接受结果的准确率。

**评估口径：** `--evaluate` 直接将 CSV 像素输入网络，不经过图片分割及拒识；其准确率不能等同于鼠标手绘或任意外部图片的准确率。模型文件会随训练变化，应对实际发布的 `model.bin` 重新评估，并同时记录测试样本范围。本文不将旧模型的历史数值作为当前模型的性能承诺。

## 常见问题

| 问题 | 处理方法 |
| --- | --- |
| 启动提示找不到程序 | 先构建 Release / x64，检查 `bin/x64/Release/BP.exe` |
| 缺少 OpenCV DLL | 检查可执行文件旁的 DLL 是否存在且与配置匹配 |
| 工具集不存在或构建失败 | 核对 `v145`、Windows SDK、OpenCV 路径和库版本 |
| 模型缺失或无效 | 将匹配网络结构的 `model.bin` 放在项目根目录并重启 |
| 找不到训练文件 | 检查根目录或 `MNIST/` 下的 CSV 文件名与工作目录 |
| 结果出现 `?` 或无法复制 | 重写或更换图片，避免粘连与模糊；不要将其当作已确认数字 |
| 数字数量不正确 | 输入单行数字，增加数字间距并裁剪复杂背景 |
| 训练后界面没有变化 | 关闭并重新打开界面，重新加载模型 |
