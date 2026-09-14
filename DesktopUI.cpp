#include "Recognizer.h"
#include <windows.h>
#include <windowsx.h>
#include <commdlg.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <sstream>
#include <iomanip>

namespace {
constexpr int DRAW=101, IMPORT=102, CLEAR=103, RUN=104, COPY=105, LOAD=106, UNDO=107;
HWND window, canvas, pathEdit, resultEdit, statusLabel, detailEdit, hintLabel;
HFONT font, titleFont, resultFont;
HBRUSH background;
std::unique_ptr<Network> network;
cv::Mat drawing(440,1600,CV_8UC3,cv::Scalar(255,255,255)), imported, overlay;
std::vector<cv::Mat> history;
bool drawMode=true, pressed=false, busy=false, ready=false, hasInk=false;
cv::Point previous;
std::future<Recognition> job;
std::wstring recognized;

std::wstring wide(const std::string& s) {
	if(s.empty()) return {};
	int n=MultiByteToWideChar(CP_UTF8,0,s.data(),int(s.size()),nullptr,0);
	std::wstring out(n,0);MultiByteToWideChar(CP_UTF8,0,s.data(),int(s.size()),out.data(),n);return out;
}
void status(const std::wstring& text) {SetWindowTextW(statusLabel,text.c_str());}
void invalidateResult() {
	recognized.clear();overlay.release();SetWindowTextW(resultEdit,L"");SetWindowTextW(detailEdit,L"");
	EnableWindow(GetDlgItem(window,COPY),FALSE);InvalidateRect(canvas,nullptr,FALSE);
}
void enableControls(bool value) {
	for(int id:{DRAW,IMPORT,CLEAR,LOAD,UNDO}) EnableWindow(GetDlgItem(window,id),value);
	EnableWindow(pathEdit,value);EnableWindow(GetDlgItem(window,RUN),value&&ready);EnableWindow(canvas,value);
}
cv::Mat current() {return drawMode?drawing:imported;}


//
RECT imageRect()
{
	RECT r;GetClientRect(canvas,&r);auto image=overlay.empty()?current():overlay;
	if(image.empty()) return r;
	double scale=std::min(double(r.right)/image.cols,double(r.bottom)/image.rows);
	int w=int(image.cols*scale),h=int(image.rows*scale);
	return {(r.right-w)/2,(r.bottom-h)/2,(r.right+w)/2,(r.bottom+h)/2};
}
cv::Point mapPoint(LPARAM lp) {
	auto r=imageRect();
	int x = int((GET_X_LPARAM(lp)-r.left)*drawing.cols/std::max<LONG>(1,r.right-r.left));
	int y = int((GET_Y_LPARAM(lp)-r.top)*drawing.rows/std::max<LONG>(1,r.bottom-r.top));
	return {std::clamp(x,0,drawing.cols-1),std::clamp(y,0,drawing.rows-1)};
}
LRESULT CALLBACK canvasProc(HWND hwnd,UINT msg,WPARAM wp,LPARAM lp) {
	if(msg==WM_ERASEBKGND) return 1;
	if(msg==WM_PAINT) {
		PAINTSTRUCT ps;HDC screen=BeginPaint(hwnd,&ps);RECT client;GetClientRect(hwnd,&client);
        // 原代码直接在屏幕上先 FillRect 再 StretchDIBits，鼠标连续移动时
        // 用户会看到背景与图像交替出现。修改为内存双缓冲，完整画好后一次提交。
        if(client.right<=0 || client.bottom<=0) {EndPaint(hwnd,&ps);return 0;}
        HDC dc=CreateCompatibleDC(screen);
        HBITMAP frame=CreateCompatibleBitmap(screen,client.right,client.bottom);
        if(!dc || !frame) {
            if(frame) DeleteObject(frame);
            if(dc) DeleteDC(dc);
            EndPaint(hwnd,&ps);return 0;
        }
        HGDIOBJ oldBitmap=SelectObject(dc,frame);
		FillRect(dc,&client,background);
		auto image=overlay.empty()?current():overlay;
		if(!image.empty()) {
			cv::Mat pixels;
			if(image.channels()==4) cv::cvtColor(image,pixels,cv::COLOR_BGRA2BGR);
			else if(image.channels()==1) cv::cvtColor(image,pixels,cv::COLOR_GRAY2BGR);
			else pixels=image;
			cv::Mat bgra;cv::cvtColor(pixels,bgra,cv::COLOR_BGR2BGRA);
			BITMAPINFO info{};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=bgra.cols;
			info.bmiHeader.biHeight=-bgra.rows;info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;info.bmiHeader.biCompression=BI_RGB;
			RECT r=imageRect();SetStretchBltMode(dc,HALFTONE);SetBrushOrgEx(dc,0,0,nullptr);
			StretchDIBits(dc,r.left,r.top,r.right-r.left,r.bottom-r.top,0,0,bgra.cols,bgra.rows,bgra.data,&info,DIB_RGB_COLORS,SRCCOPY);
		}
		if(drawMode&&!hasInk) {
			SetBkMode(dc,TRANSPARENT);SetTextColor(dc,RGB(130,141,155));SelectObject(dc,font);
			DrawTextW(dc,L"在这里按住鼠标左键书写数字  ·  可连续写多个数字",-1,&client,DT_CENTER|DT_VCENTER|DT_SINGLELINE);
		}
		// 屏幕只接收成品帧；释放 GDI 资源，避免长时间书写时句柄泄漏。
        BitBlt(screen,ps.rcPaint.left,ps.rcPaint.top,ps.rcPaint.right-ps.rcPaint.left,
               ps.rcPaint.bottom-ps.rcPaint.top,dc,ps.rcPaint.left,ps.rcPaint.top,SRCCOPY);
        SelectObject(dc,oldBitmap);DeleteObject(frame);DeleteDC(dc);
        EndPaint(hwnd,&ps);return 0;
	}
	if(msg==WM_LBUTTONDOWN && drawMode&&!busy) {
		RECT r=imageRect();POINT p{GET_X_LPARAM(lp),GET_Y_LPARAM(lp)};if(!PtInRect(&r,p)) return 0;
		if(history.size()==20) history.erase(history.begin());history.push_back(drawing.clone());
		invalidateResult();pressed=true;hasInk=true;SetCapture(hwnd);previous=mapPoint(lp);
		cv::circle(drawing,previous,7,cv::Scalar(25,29,37),-1,cv::LINE_AA);InvalidateRect(hwnd,nullptr,FALSE);status(L"书写完成后，点击“开始识别”。数字之间请留出空隙。");return 0;
	}



	if(msg==WM_MOUSEMOVE && pressed) {
		auto point=mapPoint(lp);
        if(point==previous) return 0;
        cv::line(drawing,previous,point,cv::Scalar(25,29,37),14,cv::LINE_AA);
        // 原先每次移动都刷新全画布；仅标记新增笔画范围（含笔刷及缩放边缘）。
        RECT r=imageRect();double sx=double(r.right-r.left)/drawing.cols,sy=double(r.bottom-r.top)/drawing.rows;
        RECT dirty{LONG(r.left+(std::min(previous.x,point.x)-10)*sx)-2,
                   LONG(r.top+(std::min(previous.y,point.y)-10)*sy)-2,
                   LONG(r.left+(std::max(previous.x,point.x)+10)*sx)+3,
                   LONG(r.top+(std::max(previous.y,point.y)+10)*sy)+3};
        previous=point;InvalidateRect(hwnd,&dirty,FALSE);return 0;
	}



	if(msg==WM_LBUTTONUP) {pressed=false;ReleaseCapture();return 0;}
	if(msg==WM_CAPTURECHANGED) {pressed=false;return 0;}
	return DefWindowProcW(hwnd,msg,wp,lp);
}
HWND control(const wchar_t* type,const wchar_t* text,DWORD style,int id) {
	HWND child=CreateWindowExW(type==std::wstring(L"EDIT")?WS_EX_CLIENTEDGE:0,type,text,WS_CHILD|WS_VISIBLE|style,0,0,0,0,window,reinterpret_cast<HMENU>(INT_PTR(id)),nullptr,nullptr);
	SendMessageW(child,WM_SETFONT,reinterpret_cast<WPARAM>(font),TRUE);return child;
}
void layout() {
	RECT r;GetClientRect(window,&r);int w=r.right,h=r.bottom,margin=28;
	auto move=[&](int id,int x,int y,int cw,int ch){MoveWindow(GetDlgItem(window,id),x,y,cw,ch,TRUE);};
	move(201,margin,22,w-56,40);move(202,margin,65,w-56,26);
	move(DRAW,margin,106,136,36);move(IMPORT,176,106,136,36);move(CLEAR,324,106,96,36);move(UNDO,432,106,96,36);
	move(203,margin,156,68,30);MoveWindow(pathEdit,98,152,w-272,34,TRUE);move(LOAD,w-160,152,132,34);
	MoveWindow(hintLabel,margin,198,w-56,26,TRUE);
	int canvasHeight=std::max(150,h-490);MoveWindow(canvas,margin,230,w-56,canvasHeight,TRUE);
	int y=240+canvasHeight;
	move(RUN,margin,y,160,42);move(COPY,w-152,y,124,42);MoveWindow(statusLabel,204,y+1,w-374,50,TRUE);
	move(204,margin,y+57,w-56,26);MoveWindow(resultEdit,margin,y+87,w-56,70,TRUE);
	MoveWindow(detailEdit,margin,y+165,w-56,53,TRUE);move(205,margin,h-29,w-56,24);
}
void loadPath() {
	int len=GetWindowTextLengthW(pathEdit);std::wstring path(len+1,0);GetWindowTextW(pathEdit,path.data(),len+1);path.resize(len);
	if(path.size()>1&&path.front()==L'"'&&path.back()==L'"') path=path.substr(1,path.size()-2);
	try {
		if(path.empty()) throw std::runtime_error("empty");
		std::ifstream file(std::filesystem::path(path),std::ios::binary|std::ios::ate);
		if(!file || file.tellg()<=0 || file.tellg()>32*1024*1024) throw std::runtime_error("size");
		file.seekg(0);std::vector<uchar> bytes((std::istreambuf_iterator<char>(file)),{});
		auto image=cv::imdecode(bytes,cv::IMREAD_COLOR);
		if(image.empty()||image.total()>24000000) throw std::runtime_error("decode");
		imported=image;drawMode=false;invalidateResult();
		SetWindowTextW(hintLabel,L"图片预览 · 请使用清晰、正向的数字图片；识别后显示分割框。");
		status(L"图片已载入，点击“开始识别”。");
	} catch(...) {invalidateResult();imported.release();drawMode=false;InvalidateRect(canvas,nullptr,TRUE);status(L"无法读取图片。请检查路径与格式（PNG/JPG/BMP，≤32 MB、2400 万像素）。");}
}
void chooseFile() {
	wchar_t path[32768]{};OPENFILENAMEW dialog{};dialog.lStructSize=sizeof(dialog);dialog.hwndOwner=window;
	dialog.lpstrFilter=L"数字图片 (*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.webp)\0*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.webp\0所有文件\0*.*\0";
	// 每次都从用户指定的项目上级目录打开，避免暴露其他个人目录。
	dialog.lpstrInitialDir=L"C:\\Users\\27698\\Desktop\\personal files\\cpp group task";
	dialog.lpstrFile=path;dialog.nMaxFile=32768;
	dialog.Flags=OFN_FILEMUSTEXIST|OFN_PATHMUSTEXIST|OFN_NOCHANGEDIR|OFN_DONTADDTORECENT;
	if(GetOpenFileNameW(&dialog)) {SetWindowTextW(pathEdit,path);loadPath();}
}
void beginRecognition() {
	if(busy||!ready) return;
	auto image=current();invalidateResult();
	if(image.empty() || (drawMode&&!hasInk)) {status(L"请先手绘数字，或选择一张数字图片。");return;}
	busy=true;enableControls(false);status(L"正在分割和识别，请稍候…");
	job=std::async(std::launch::async,[image=image.clone()](){return recognize(*network,image);});SetTimer(window,1,80,nullptr);
}
void finishRecognition() {
	if(!job.valid()||job.wait_for(std::chrono::seconds(0))!=std::future_status::ready) return;
	KillTimer(window,1);busy=false;enableControls(true);
	try {
		auto result=job.get();overlay=result.preview;InvalidateRect(canvas,nullptr,FALSE);
		if(result.text.empty()) {status(L"未检测到数字。请增加笔画对比度，或裁剪图片中的数字区域。");return;}
		recognized=wide(result.text);
		// Windows 编辑框需要 CRLF；复制时也保留行顺序和前导零。
		for(size_t p=0;(p=recognized.find(L'\n',p))!=std::wstring::npos;p+=2) recognized.replace(p,1,L"\r\n");
		// 原先低分也允许复制，容易当作确认结果；有拒识位时禁止复制。
		SetWindowTextW(resultEdit,recognized.c_str());EnableWindow(GetDlgItem(window,COPY),result.uncertain==0);
		std::wstring message=L"识别完成，共 "+std::to_wstring(result.digits.size())+L" 位。请核对结果。";
		if(result.uncertain) message=L"有 "+std::to_wstring(result.uncertain)+L" 位未确认（?），请重写或更换图片；暂不可复制。";
		if(result.touching) message=L"疑似粘连位置已拒识（?），请分开书写后重试；暂不可复制。";
		status(message);
		std::wostringstream details;details<<L"逐位模型得分（不是正确率）：  ";
		for(size_t i=0;i<result.digits.size();++i) details<<i+1<<L":"<<(result.digits[i]<0?L"?（未确认）":std::to_wstring(result.digits[i]))<<L" ["<<std::fixed<<std::setprecision(0)<<result.scores[i]*100<<L"%]   ";
		SetWindowTextW(detailEdit,details.str().c_str());
	} catch(...) {status(L"识别失败。请使用较小、清晰的图片，每张不超过 256 位数字。");}
}
LRESULT CALLBACK windowProc(HWND hwnd,UINT msg,WPARAM wp,LPARAM lp) {
	switch(msg) {
	case WM_CREATE: {
		window=hwnd;
		font=CreateFontW(-17,0,0,0,FW_NORMAL,FALSE,FALSE,FALSE,DEFAULT_CHARSET,0,0,CLEARTYPE_QUALITY,0,L"Microsoft YaHei UI");
		titleFont=CreateFontW(-29,0,0,0,FW_BOLD,FALSE,FALSE,FALSE,DEFAULT_CHARSET,0,0,CLEARTYPE_QUALITY,0,L"Microsoft YaHei UI");
		resultFont=CreateFontW(-29,0,0,0,FW_SEMIBOLD,FALSE,FALSE,FALSE,DEFAULT_CHARSET,0,0,CLEARTYPE_QUALITY,0,L"Consolas");
		auto title=control(L"STATIC",L"数字识别  /  BP Studio",0,201);SendMessageW(title,WM_SETFONT,(WPARAM)titleFont,TRUE);
		control(L"STATIC",L"01  选择输入方式        02  写入或导入数字        03  开始识别并核对结果",0,202);
		control(L"BUTTON",L"手绘数字",WS_TABSTOP,DRAW);control(L"BUTTON",L"选择图片…",WS_TABSTOP,IMPORT);
		control(L"BUTTON",L"清空",WS_TABSTOP,CLEAR);control(L"BUTTON",L"撤销笔画",WS_TABSTOP,UNDO);
		control(L"STATIC",L"文件路径",0,203);pathEdit=control(L"EDIT",L"",WS_TABSTOP|ES_AUTOHSCROLL,210);
		SendMessageW(pathEdit,EM_SETLIMITTEXT,32767,0);control(L"BUTTON",L"载入路径",WS_TABSTOP,LOAD);
		hintLabel=control(L"STATIC",L"手绘画布 · 按住鼠标左键书写 0–9；长串请从左到右书写，数字之间留空隙。",0,211);
		canvas=control(L"BPCanvas",L"",WS_BORDER,212);
		control(L"BUTTON",L"开始识别",WS_TABSTOP|BS_DEFPUSHBUTTON,RUN);control(L"BUTTON",L"复制结果",WS_TABSTOP,COPY);
		statusLabel=control(L"STATIC",L"准备就绪，请手绘或导入数字。",0,213);
		control(L"STATIC",L"识别结果",0,204);
		resultEdit=control(L"EDIT",L"",WS_TABSTOP|ES_MULTILINE|ES_READONLY|WS_VSCROLL|WS_HSCROLL|ES_AUTOHSCROLL,214);
		SendMessageW(resultEdit,WM_SETFONT,(WPARAM)resultFont,TRUE);SendMessageW(resultEdit,EM_SETLIMITTEXT,4096,0);
		detailEdit=control(L"EDIT",L"",ES_MULTILINE|ES_READONLY|WS_VSCROLL,215);
		control(L"STATIC",L"本地离线识别 · 支持多位数字及前导零 · 手写模型对印刷体、粘连或复杂背景可能误判",0,205);
		EnableWindow(GetDlgItem(hwnd,COPY),FALSE);
		EnableWindow(GetDlgItem(hwnd,RUN),ready);
		if(!ready) status(L"模型缺失或损坏。请将有效的 model.bin 放到 BP 项目目录后重启。");
		return 0;
	}
	case WM_SIZE:layout();return 0;
	case WM_GETMINMAXINFO:reinterpret_cast<MINMAXINFO*>(lp)->ptMinTrackSize={840,720};return 0;
	case WM_CTLCOLORSTATIC:case WM_CTLCOLOREDIT:
		SetTextColor((HDC)wp,RGB(36,49,66));SetBkColor((HDC)wp,RGB(245,247,250));return (LRESULT)background;
	case WM_COMMAND:
		if(LOWORD(wp)==210&&HIWORD(wp)==EN_CHANGE&&!busy) {invalidateResult(); if(!drawMode) {imported.release();InvalidateRect(canvas,nullptr,FALSE);} status(L"路径已修改，点击“载入路径”后再识别。");return 0;}
		switch(LOWORD(wp)) {
		case DRAW:drawMode=true;invalidateResult();SetWindowTextW(hintLabel,L"手绘画布 · 按住鼠标左键书写 0–9；长串请从左到右书写，数字之间留空隙。");status(L"手绘模式：数字之间请留出空隙。");break;
		case IMPORT:chooseFile();break;
		case LOAD:loadPath();break;
		case CLEAR:invalidateResult();if(drawMode){drawing.setTo(cv::Scalar(255,255,255));history.clear();hasInk=false;}else{imported.release();SetWindowTextW(pathEdit,L"");}InvalidateRect(canvas,nullptr,FALSE);status(L"已清空，可以重新输入。");break;
		case UNDO:if(drawMode&&!history.empty()){drawing=history.back();history.pop_back();hasInk=!history.empty();invalidateResult();status(L"已撤销上一笔。");}break;
		case RUN:beginRecognition();break;
		case COPY:
			if(!recognized.empty()&&recognized.find(L'?')==std::wstring::npos&&OpenClipboard(hwnd)) {
				SIZE_T bytes=(recognized.size()+1)*sizeof(wchar_t);HGLOBAL block=GlobalAlloc(GMEM_MOVEABLE,bytes);
				if(block) {void* dest=GlobalLock(block);if(dest){memcpy(dest,recognized.c_str(),bytes);GlobalUnlock(block);EmptyClipboard();if(SetClipboardData(CF_UNICODETEXT,block)) status(L"识别结果已复制。");else GlobalFree(block);}else GlobalFree(block);}CloseClipboard();
			}break;
		}return 0;
	case WM_TIMER:finishRecognition();return 0;
	case WM_DESTROY:
		if(job.valid()) job.wait();PostQuitMessage(0);return 0;
	}
	return DefWindowProcW(hwnd,msg,wp,lp);
}
}
int runDesktop() {
	// 从 exe 向上寻找项目根，双击、快捷方式和 VS F5 均可定位模型。
	wchar_t exe[32768];GetModuleFileNameW(nullptr,exe,32768);
	auto folder=std::filesystem::path(exe).parent_path();
	for(int i=0;i<5;++i) {
		if(std::filesystem::exists(folder/L"BP.vcxproj")||std::filesystem::exists(folder/L"model.bin")) {std::filesystem::current_path(folder);break;}
		folder=folder.parent_path();
	}
	network=std::make_unique<Network>(std::vector<int>{784,128,10});ready=network->load("model.bin");
	// 普通运行只展示产品窗口；--console 保留原训练入口。
	FreeConsole();background=CreateSolidBrush(RGB(245,247,250));
	WNDCLASSW cls{};cls.hInstance=GetModuleHandleW(nullptr);cls.hCursor=LoadCursorW(nullptr,IDC_ARROW);cls.hbrBackground=background;
	cls.lpszClassName=L"BPCanvas";cls.lpfnWndProc=canvasProc;RegisterClassW(&cls);
	cls.lpszClassName=L"BPStudio";cls.lpfnWndProc=windowProc;cls.hIcon=LoadIconW(nullptr,IDI_APPLICATION);RegisterClassW(&cls);
	HWND hwnd=CreateWindowExW(0,L"BPStudio",L"BP Studio · 数字识别",WS_OVERLAPPEDWINDOW|WS_CLIPCHILDREN,CW_USEDEFAULT,CW_USEDEFAULT,1040,820,nullptr,nullptr,cls.hInstance,nullptr);
	if(!hwnd) return 1;ShowWindow(hwnd,SW_SHOW);UpdateWindow(hwnd);
	MSG message{};while(GetMessageW(&message,nullptr,0,0)>0) {if(!IsDialogMessageW(hwnd,&message)) {TranslateMessage(&message);DispatchMessageW(&message);}}
	DeleteObject(font);DeleteObject(titleFont);DeleteObject(resultFont);DeleteObject(background);return 0;
}
