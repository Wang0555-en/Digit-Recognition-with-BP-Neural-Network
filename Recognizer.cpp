#include "Recognizer.h"
#include <algorithm>
#include <cmath>
Recognition recognize(Network& net, const cv::Mat& image) {
	ImageSegmenter segmenter;
	auto segments=segmenter.segment(image);
	Recognition result;
	int row=0;
	for(const auto& segment:segments) {
		Matrix input(1,784);
		for(int y=0;y<28;++y) for(int x=0;x<28;++x) input.setData(0,y*28+x,segment.image.at<uchar>(y,x)/255.0);
		auto output=net.forward(input);
		int best=0;double second=0;
		bool valid=true;
		for(int i=0;i<10;++i) valid &= std::isfinite(output.getData(0,i));
		for(int i=1;i<10;++i) if(output.getData(0,i)>output.getData(0,best)) best=i;
		double score=output.getData(0,best);
		for(int i=0;i<10;++i) if(i!=best) second=std::max(second,output.getData(0,i));
		// 原代码只增加 uncertain，仍无条件输出 argmax；所有类别得分极低也会猜数字。
		// 修改：沿用原阈值执行真正拒识，疑似粘连不能作为单个数字输出。
		// Sigmoid 得分不是校准后的正确概率，不应通过归一化虚增可信度。
		bool accepted=valid && score>=0.65 && score-second>=0.25 && !segment.touching;
		if(!accepted) ++result.uncertain;
		if(segment.line!=row && !result.text.empty()) result.text+='\n';
		row=segment.line;
		// 拒识用 ? 和 -1 占位，原先直接猜测会误导用户，删除则会导致数字串错位。
		result.text+=accepted?char('0'+best):'?';
		result.digits.push_back(accepted?best:-1);result.scores.push_back(valid?score:0.0);
		result.touching |= segment.touching;
	}
	result.preview=segmenter.drawBoundingBoxes(segments);
	return result;
}
