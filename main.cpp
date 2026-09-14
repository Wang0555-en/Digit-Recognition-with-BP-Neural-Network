#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#ifdef _WIN32
#include <windows.h>
#endif
#include "Network.h"
#include "Recognizer.h"
#include "DatasetWriter.h"
#include "ImageSegmenter.h"
#include <opencv2/opencv.hpp>

using namespace std;

// ============================================================
// 1. MNIST 数据读取 & 测试准确率
// ============================================================

vector<pair<Matrix, Matrix>> readMNIST(const string& filename, int max_count = -1) {
	vector<pair<Matrix, Matrix>> data;
	ifstream file(filename);
	// 兼容将 MNIST 数据集整个文件夹放入项目目录的情况。
	string actual_path = filename;
	if (!file.is_open()) {
		actual_path = "MNIST/" + filename;
		file.open(actual_path);
	}
	if (!file.is_open()) {
		cerr << "无法打开文件: " << filename << " 或 " << actual_path << endl;
		return data;
	}
	cout << "读取数据文件: " << actual_path << endl;

	string line;
	int count = 0;
	while (getline(file, line) && (max_count == -1 || count < max_count)) {
		stringstream ss(line);
		string token;
		vector<double> vals;
        // 原读取器静默跳过列数错误，并可能把非法标签用于越界写入。
        // 全量训练不能悄悄漏样本：验证全部字段，报错后终止本次加载。
        try {
            while (getline(ss, token, ',')) {
                size_t used=0;double value=stod(token,&used);
                if(token.find_first_not_of(" \t\r",used)!=string::npos || !std::isfinite(value))
                    throw std::invalid_argument("invalid number");
                vals.push_back(value);
            }
            if(vals.size()!=785 || vals[0]<0 || vals[0]>9 || vals[0]!=std::floor(vals[0]))
                throw std::invalid_argument("expected label 0-9 and 784 pixels");
            for(size_t i=1;i<vals.size();++i)
                if(vals[i]<0 || vals[i]>255 || vals[i]!=std::floor(vals[i]))
                    throw std::invalid_argument("pixels must be integers in 0-255");
        } catch(const std::exception& e) {
            cerr << actual_path << " 第 " << count+1 << " 行数据无效: " << e.what() << endl;
            return {};
        }
        int label = (int)vals[0];
		Matrix input(1, 784);
		for (int i = 0; i < 784; ++i) {
			input.setData(0, i, vals[i + 1] / 255.0);
		}

		Matrix target(1, 10, 0.0);
		target.setData(0, label, 1.0);
		data.push_back({ input, target });
		count++;
	}
	file.close();
	return data;
}

double testAccuracy(Network& net, const vector<pair<Matrix, Matrix>>& test_data) {
	int correct = 0;
	for (const auto& sample : test_data) {
		const Matrix& input = sample.first;
		const Matrix& target = sample.second;
		Matrix output = net.forward(input);

		int pred_label = 0;
		double max_val = output.getData(0, 0);
		for (int j = 1; j < 10; ++j) {
			if (output.getData(0, j) > max_val) {
				max_val = output.getData(0, j);
				pred_label = j;
			}
		}

		int true_label = 0;
		for (int j = 0; j < 10; ++j) {
			if (target.getData(0, j) == 1.0) {
				true_label = j;
				break;
			}
		}
		if (pred_label == true_label) correct++;
	}
	return (double)correct / test_data.size();
}

// ============================================================
// 2. 训练函数
// ============================================================

// 全量训练每轮保存可校验的检查点，原来仅结束时保存，中途关闭会丢失全部进度。
bool saveTrainingModel(Network& net, const string& path) {
    const string temporary=path+".tmp";
    // 清除上次遗留临时文件，避免本次写入失败却误校验旧文件。
    std::error_code cleanupError;std::filesystem::remove(temporary,cleanupError);
    if(cleanupError) return false;
    net.save(temporary);
    Network verify({784,128,10});
    if(!verify.load(temporary)) {cerr << "模型写入校验失败: " << temporary << endl;return false;}
#ifdef _WIN32
    if(!MoveFileExW(std::filesystem::path(temporary).c_str(),std::filesystem::path(path).c_str(),
                    MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)) return false;
#else
    std::error_code error;std::filesystem::rename(temporary,path,error);if(error) return false;
#endif
    return true;
}

bool trainModel(Network& net, const string& train_file, const string& test_file, int epochs, double lr, bool full=false) {
    // 用户已将旧上限改为60000/10000；全量入口进一步用-1读取文件全部样本，避免静默截断。
	auto train_data = readMNIST(train_file, full?-1:60000);
	auto test_data = readMNIST(test_file, full?-1:10000);

	if (train_data.empty() || test_data.empty()) {
		cerr << "数据加载失败！" << endl;
		return false;
	}

	cout << "训练样本数: " << train_data.size() << endl;
	cout << "测试样本数: " << test_data.size() << endl;

	// 新增：把实际参与训练的样本和每轮结果分别保存为 CSV。
	DatasetWriter dataset;
	if (!dataset.open("training_dataset.csv", "training_history.csv")) {
		cerr << "无法创建数据集或训练记录，请检查运行目录的写入权限。训练已取消。" << endl;
		return false;
	}
	cout << "同步保存到 training_dataset.csv 和 training_history.csv" << endl;

	// 全量入口每轮洗牌索引，避免固定样本顺序；网络结构和 BP 更新公式保持不变。
    std::vector<size_t> order(train_data.size());std::iota(order.begin(),order.end(),0);
    std::mt19937 shuffleRandom(20260914);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        if(full) std::shuffle(order.begin(),order.end(),shuffleRandom);
        size_t completed=0;
		double epoch_lr = lr;  // 记录本轮实际使用的学习率（衰减前）。
		double total_loss = 0.0;
		for (size_t index : order) {
            const auto& sample=train_data[index];
			const Matrix& input = sample.first;
			const Matrix& target = sample.second;

			// 同一批样本会训练多轮，只在第一轮保存，避免重复记录。
			if (epoch == 0 && !dataset.writeSample(input, target)) {
				cerr << "样本同步保存失败，训练已停止；导出文件可能不完整。" << endl;
				return false;
			}

			Matrix output = net.forward(input);
			Matrix diff = output - target;

			double loss = 0.0;
			for (int j = 0; j < 10; ++j) {
				double e = diff.getData(0, j);
				loss += e * e;
			}
			total_loss += loss;

			net.backwardAndUpdate(target, lr);
            // 原先整轮无输出，全量计算时像卡住；每1000张报告实际处理进度。
            ++completed;
            if(full && (completed%1000==0 || completed==train_data.size()))
                cout << "Epoch " << epoch+1 << "/" << epochs << "  Samples " << completed << "/" << train_data.size() << endl;
		}

		// 学习率衰减（每10轮减半）
		if (epoch > 0 && epoch % 10 == 0) {
			lr *= 0.5;
			cout << "学习率衰减至: " << lr << endl;
		}

		double avg_loss = total_loss / train_data.size();
		double acc = testAccuracy(net, test_data);
        if(!std::isfinite(avg_loss)) {cerr << "训练损失非有限值，停止保存。" << endl;return false;}
        if(full && !saveTrainingModel(net,"model.checkpoint.bin")) {
            cerr << "检查点保存失败，训练已停止。" << endl;return false;
        }
		if (!dataset.writeEpoch(epoch + 1, epoch_lr, avg_loss, acc)) {
			cerr << "训练记录保存失败，训练已停止；导出文件可能不完整。" << endl;
			return false;
		}
		cout << "Epoch " << epoch + 1 << "/" << epochs
			<< "  Loss = " << avg_loss
			<< "  Test Acc = " << acc * 100 << "%" << endl;
	}

	if(!saveTrainingModel(net,"model.bin")) {cerr << "最终模型保存失败。" << endl;return false;}
	cout << "模型已保存为 model.bin" << endl;

	double final_acc = testAccuracy(net, test_data);
	cout << "最终测试准确率: " << final_acc * 100 << "%" << endl;
    return true;
}

// ============================================================
// 3. 统一预测函数（自动分割 + 识别）
// ============================================================

void predictImage(Network& net, const string& imagePath) {
    // 原控制台独立执行 argmax，绕过低分检查；统一调用拒识流程。
    std::ifstream file(std::filesystem::u8path(imagePath), std::ios::binary);
    std::vector<uchar> bytes((std::istreambuf_iterator<char>(file)), {});
    try {
        auto result = recognize(net, cv::imdecode(bytes, cv::IMREAD_UNCHANGED));
        cout << "识别结果: " << result.text << "\n未确认位数: " << result.uncertain << endl;
        if(result.uncertain) cout << "? 表示未确认，请重新书写或更换图片。" << endl;
        if(result.digits.empty()) cout << "未检测到数字。" << endl;
        cv::imwrite("segmented_result.jpg",result.preview);
    } catch(const std::exception& e) { cerr << "识别失败: " << e.what() << endl; }
}

// ============================================================
// 4. 主函数
// ============================================================

int main(int argc, char** argv) {
    std::cout << CV_VERSION << std::endl;
    // 原入口被注释后，无参数启动会对空 argv[1] 构造字符串而崩溃。
    // 恢复双击/F5桌面入口；训练使用独立参数，不再靠注释 UI 入口切换。
    if (argc == 1) return runDesktop();
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);SetConsoleCP(CP_UTF8);
#endif
    if (std::string(argv[1]) == "--train-full") {
        try {
            int epochs=50;double lr=0.3;bool resume=false;
            if(argc>5) throw std::invalid_argument("too many arguments");
            size_t used=0;
            if(argc>2) {epochs=std::stoi(argv[2],&used);if(used!=std::string(argv[2]).size()) throw std::invalid_argument("epochs");}
            if(argc>3) {lr=std::stod(argv[3],&used);if(used!=std::string(argv[3]).size()) throw std::invalid_argument("learning rate");}
            if(argc>4) {if(std::string(argv[4])!="--resume") throw std::invalid_argument("expected --resume");resume=true;}
            if(epochs<1 || !std::isfinite(lr) || lr<=0 || lr>1) throw std::invalid_argument("epochs >= 1; 0 < learning rate <= 1");
            Network model({784,128,10});
            // 原菜单总是加载 model.bin，实际是继续训练；全量入口默认从头初始化。
            if(resume && !model.load("model.bin")) {cerr << "继续训练需要有效的 model.bin。" << endl;return 2;}
            cout << (resume?"继续训练（重新开始学习率计划）":"从头训练") << "，读取全部训练集及测试集。" << endl;
            return trainModel(model,"mnist_train.csv","mnist_test.csv",epochs,lr,true)?0:3;
        } catch(const std::exception& e) {cerr << "全量训练失败: " << e.what() << endl;return 4;}
    }
    // 可自动验证的接口与 UI 共用同一识别函数；保留原训练菜单。
    if (std::string(argv[1]) == "--recognize" && argc == 3) {
        Network model({784,128,10});
        if (!model.load("model.bin")) { std::cerr << "Invalid or missing model"; return 2; }
        try {
            // 使用 UTF-8 文件流读取路径，避免中文路径导致图片读取失败。
            std::ifstream file(std::filesystem::u8path(argv[2]),std::ios::binary);
            std::vector<uchar> bytes((std::istreambuf_iterator<char>(file)),{});
            auto image = cv::imdecode(bytes, cv::IMREAD_COLOR);
            auto result = recognize(model, image);
            std::cout << "RESULT=" << result.text << "\nCOUNT=" << result.digits.size() << std::endl;
            std::cout << "UNCERTAIN=" << result.uncertain << std::endl;
            return result.digits.empty() ? 3 : 0;
        } catch (const std::exception& e) {std::cerr << e.what(); return 4;}
    }
    if (std::string(argv[1]) == "--evaluate") {
        Network model({784,128,10});
        if (!model.load("model.bin")) return 2;
        auto data = readMNIST("mnist_test.csv", argc > 2 ? std::atoi(argv[2]) : 2000);
        if (data.empty()) return 3;
        std::cout << "ACCURACY=" << testAccuracy(model,data) << std::endl;
        return 0;
    }
    if (std::string(argv[1]) != "--console") {
        std::cerr << "Usage: BP.exe [--console | --train-full [EPOCHS LR [--resume]] | --recognize IMAGE | --evaluate COUNT]"; return 1;
    }

#ifdef _WIN32
	// 源文件和 MSVC 编译均使用 UTF-8，让 Windows 控制台正确显示中文。
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
#endif
	srand((unsigned)time(0));

	// 创建网络
	vector<int> layer_sizes = { 784, 128, 10 };
	Network net(layer_sizes);

	// 尝试加载已训练模型
	net.load("model.bin");

	cout << "\n========== 手写数字识别系统 ==========" << endl;
	cout << "1. 训练模式 (已有模型则继续训练并保存)" << endl;
	cout << "2. 预测模式 (输入图片路径，自动分割+识别)" << endl;
	cout << "========================================" << endl;
	cout << "请选择模式 (1/2): ";

	int mode = 0;
	if (!(cin >> mode)) {
		cerr << "请输入数字 1 或 2。" << endl;
		return 1;
	}
	cin.ignore(numeric_limits<streamsize>::max(), '\n');

	if (mode == 1) {
		string train_file = "mnist_train.csv";
		string test_file = "mnist_test.csv";
		int epochs = 50;
		double lr = 0.3;
		return trainModel(net, train_file, test_file, epochs, lr)?0:3;
	}
	else if (mode == 2) {
		string imagePath;
		cout << "请输入图片路径: ";
		getline(cin, imagePath);
		predictImage(net, imagePath);
	}
	else {
		cout << "无效选择，程序退出。" << endl;
	}
	return 0;
}