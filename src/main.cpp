#include "calibrator.h"
#include "calibrator_aruco.h"
#include <filesystem>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <sstream>


template<typename T>
void CalibInternal(T& calibrator, const std::filesystem::path& dataPath, const std::filesystem::path& jsonPath, 
	const bool& display = false, const cv::Size& tarSize = cv::Size()) {
	// calib internal param
	std::map<std::string, Camera> cams;
	for (const auto& folder : std::filesystem::directory_iterator(dataPath)) {
		if (folder.is_directory()) {
			calibrator.Clear();
			for (const auto& imgFile : std::filesystem::directory_iterator(folder.path())) {
				if (imgFile.status().type() == std::filesystem::file_type::regular) {
					cv::Mat img = cv::imread(imgFile.path().string());
					if (tarSize != cv::Size() && tarSize != cv::Size(img.cols, img.rows))
						cv::resize(img, img, tarSize);
					bool flag = calibrator.Push(img, display);
					std::cout << "Push image " + std::string(flag ? "successful " : "failed ") + imgFile.path().string() << std::endl;
				}
			}
			calibrator.Calib();
			cams.insert(std::make_pair(folder.path().filename().string(), calibrator.cam));
		}
	}
	SerializeCameras(cams, jsonPath.string());
}


template<typename T>
void CalibExternal(T& calibrator, const std::filesystem::path& dataPath, const std::filesystem::path& jsonPath,
	const bool& display = false, const cv::Size& tarSize = cv::Size()) {
	calibrator.cams = ParseCameras(jsonPath.string());
	std::filesystem::path markerCache = dataPath / "markerCache.txt";
	if (!std::filesystem::exists(markerCache)) {
		std::map<int, std::map<std::string, cv::Mat>> imgs_seq;
		for (const auto& folder : std::filesystem::directory_iterator(dataPath)) {
			if (folder.is_regular_file()) {
				//if(folder.path().extension().string() == 'jpeg')

				std::string filePath = folder.path().string();
				std::istringstream sstream(folder.path().filename().replace_extension().string());

				std::string frameIdxStr, deviceID;
				std::getline(sstream, frameIdxStr, '_');
				std::getline(sstream, deviceID);
				const int frameIdx = std::stoi(frameIdxStr);

				if (calibrator.cams.find(deviceID) == calibrator.cams.end()) {
					std::cerr << deviceID << " without internal calibration!" << std::endl;
					std::abort();
				}

				auto seqIter = imgs_seq.find(frameIdx);
				if (seqIter == imgs_seq.end())
					seqIter = imgs_seq.insert(std::make_pair(frameIdx, std::map<std::string, cv::Mat>())).first;

				auto&& imgs = seqIter->second;

				cv::Mat img = cv::imread(filePath);
				if (tarSize != cv::Size() && tarSize != cv::Size(img.cols, img.rows))
					cv::resize(img, img, tarSize);

				imgs.insert(std::make_pair(deviceID, img));
				if (imgs.size() == calibrator.cams.size()) {
					std::cout << "Frame " << seqIter->first << " ready for calibration" << std::endl;
					calibrator.Push(imgs, display);
					imgs_seq.erase(frameIdx);
				}
			}
		}
		std::ofstream ofs(markerCache.string());
		ofs << calibrator.p2ds.size() << std::endl;
		for (int i = 0; i < calibrator.p2ds.size(); i++) {
			ofs << calibrator.p2ds[i].size() << std::endl;
			for (const auto& iter : calibrator.p2ds[i]) {
				ofs << iter.first << std::endl;
				for (int j = 0; j < iter.second.size(); j++)
					ofs << iter.second[j].x << " " << iter.second[j].y << std::endl;
			}
		}
	}
	else {
		std::ifstream ifs(markerCache.string());
		int frameSize;
		ifs >> frameSize;
		for (int i = 0; i < frameSize; i++) {
			int camSize;
			ifs >> camSize;
			std::map<std::string, std::vector<cv::Point2f>> pairs;
			for (int camIdx = 0; camIdx < camSize; camIdx++) {
				std::string camStr;
				ifs >> camStr;
				std::vector<cv::Point2f> p2ds(calibrator.patternSize.width * calibrator.patternSize.height);
				for (int j = 0; j < p2ds.size(); j++)
					ifs >> p2ds[j].x >> p2ds[j].y;
				pairs.insert(std::make_pair(camStr, p2ds));
			}
			calibrator.p2ds.emplace_back(pairs);
		}
	}

	calibrator.Init();
	calibrator.Evaluate();
	calibrator.Bundle();
	calibrator.Evaluate();
	SerializeCameras(calibrator.cams, jsonPath.string());
}


int main()
{
	const cv::Size imgSize(5320, 4600);

	const std::filesystem::path internalPath("../data/internal");
	const std::filesystem::path externalPath("../data/external");
	const std::filesystem::path jsonPath("../data/calibration.json");

	//ArucoInternalCalibrator calibrator(cv::Size(9,6), 0.08, 0.062);
	//CalibInternal(calibrator, internalPath, jsonPath, true);

	ExternalCalibrator calibrator({10, 7}, 0.07);
	CalibExternal(calibrator, externalPath, jsonPath, true, imgSize/4);


	return 0;
}


//		cv::cuda::Stream cvStream;
//		cv::cuda::GpuMat grayImg(cap->second.bayer.rows, cap->second.bayer.cols, CV_8UC1);
//		cv::Mat hostImg(cap->second.bayer.rows, cap->second.bayer.cols, CV_8UC1);
//		cv::Mat drawImg(monitorSizeyBox->value(), monitorSizexBox->value(), CV_8UC3);
//		cv::cuda::GpuMat resizedImg(monitorSizeyBox->value(), monitorSizexBox->value(), CV_8UC3);
//
//		calibrator.patternSize = cv::Size(chessboardGridxBox->value(), chessboardGridyBox->value());
//		calibrator.squareSize = cv::Size2f(1e-3f * chessboardSizeBox->value(), 1e-3f * chessboardSizeBox->value());
//		calibrator.imgSize = cv::Size(cap->second.bayer.cols, cap->second.bayer.rows);
//
//		std::chrono::time_point<std::chrono::steady_clock> stamp = std::chrono::steady_clock::now();
//		cap->second.ptr->BeginAcquisition();
//		while (true) {
//			cap->second.AcquireImage(cv::cuda::StreamAccessor::getStream(cvStream));
//			cv::cuda::resize(cap->second.bgr, resizedImg, resizedImg.size(), 0.0, 0.0, 1, cvStream);
//			resizedImg.download(drawImg, cvStream);
//			cv::cuda::cvtColor(cap->second.bgr, grayImg, cv::COLOR_BGR2GRAY, 0, cvStream);
//			grayImg.download(hostImg, cvStream);
//			cvStream.waitForCompletion();
//
//			cv::imshow("calib", drawImg);
//			const int key = cv::waitKey(1);
//			if (key == 32 || (grabIntervalBox->value() > 0 &&
//					std::chrono::duration_cast<std::chrono::milliseconds>(
//						std::chrono::steady_clock::now() - stamp).count() > grabIntervalBox->value())) {
//				cap->second.ptr->EndAcquisition();
//				if (calibrator.Push(hostImg)) {
//					std::vector<cv::Point2f> corners;
//					for (const auto& p : calibrator.p2ds.back())
//						corners.emplace_back(cv::Point2f(p.x / float(hostImg.cols) * float(drawImg.cols), 
//							p.y / float(hostImg.rows) * float(drawImg.cols)));
//					cv::drawChessboardCorners(drawImg, calibrator.patternSize, corners, true);
//					cv::imshow("calib", drawImg);
//					cv::waitKey(100);
//				}
//				cap->second.ptr->BeginAcquisition();
//				stamp = std::chrono::steady_clock::now();
//			}
//			else if (key == 27)
//				break;
//		}
//		cap->second.ptr->EndAcquisition();
//		cv::destroyWindow("calib");
//		calibrator.Calib();
//		calibration[cap->first] = calibrator.cam;


//
//
//std::vector<std::string> deviceIDs;
//std::ifstream ifs("../data/cams.txt");
//
//std::map<std::string, Camera> cams;
//
//std::string deviceID;
//while (std::getline(ifs, deviceID))
//{
//	Camera cam;
//	cam.imgSize = cv::Size(1330, 1150);
//
//	Json::Value json;
//	std::ifstream fs("../data/" + deviceID + "_intr.json");
//	std::string errs;
//	Json::parseFromStream(Json::CharReaderBuilder(), fs, &json, &errs);
//	fs.close();
//
//	Json::Value var = json["K"];
//	for (int row = 0; row < 3; row++)
//		for (int col = 0; col < 3; col++)
//			cam.originK(row, col) = var[row * 3 + col].asFloat();
//
//
//	var = json["distortion"];
//	cam.distCoeff.resize(var.size(), 1);
//	for (int i = 0; i < int(var.size()); i++)
//		cam.distCoeff(i) = var[i].asFloat();
//
//	cams.insert(std::make_pair(deviceID, cam));
//}
//
//
//SerializeCameras(cams, "../data/calibration.json");
//
//
//
//
//


//
//
//
//std::vector<std::string> deviceIDs;
//std::ifstream ifs("../data/cams.txt");
//
//std::map<std::string, Camera> cams;
//
//std::string deviceID;
//while (std::getline(ifs, deviceID))
//{
//	Camera cam;
//	cam.imgSize = cv::Size(1330, 1150);
//
//	Json::Value json;
//	std::ifstream fs("../data/" + deviceID + "_intr.json");
//	std::string errs;
//	Json::parseFromStream(Json::CharReaderBuilder(), fs, &json, &errs);
//	fs.close();
//
//	Json::Value var = json["K"];
//	for (int row = 0; row < 3; row++)
//		for (int col = 0; col < 3; col++)
//			cam.originK(row, col) = var[row * 3 + col].asFloat();
//
//
//	var = json["distortion"];
//	cam.distCoeff.resize(var.size(), 1);
//	for (int i = 0; i < int(var.size()); i++)
//		cam.distCoeff(i) = var[i].asFloat();
//
//	cams.insert(std::make_pair(deviceID, cam));
//}
//
//
//SerializeCameras(cams, "../data/calibration.json");
//
//
