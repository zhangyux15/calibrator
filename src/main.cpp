#include "calibrator.h"
#include <filesystem>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>


void CalibInternal(const cv::Size& patternSize, const cv::Size2f& squareSize,
	const std::filesystem::path& dataPath, const std::filesystem::path& jsonPath, const bool& display = false) {

	// calib internal param
	std::map<std::string, Camera> cams;
	for (const auto& folder : std::filesystem::directory_iterator(dataPath)) {
		if (folder.is_directory()) {
			InternalCalibrator calibrator;
			calibrator.patternSize = patternSize;
			calibrator.squareSize = squareSize;
			for (const auto& imgFile : std::filesystem::directory_iterator(folder.path())) {
				if (imgFile.status().type() == std::filesystem::file_type::regular) {
					cv::Mat img = cv::imread(imgFile.path().string());
					calibrator.Push(img, display);
				}
			}
			calibrator.Calib();
			cams.insert(std::make_pair(folder.path().filename().string(), calibrator.cam));
		}
	}
	SerializeCameras(cams, jsonPath.string());
}


void CalibExternal(const cv::Size& patternSize, const cv::Size2f& squareSize,
	const std::filesystem::path& dataPath, const std::filesystem::path& jsonPath, const bool& display = false) {

	// calib external param
	ExternalCalibrator calibrator;
	calibrator.patternSize = patternSize;
	calibrator.squareSize = squareSize;
	calibrator.cams = ParseCameras(jsonPath.string());
	for (const auto& folder : std::filesystem::directory_iterator(dataPath)) {
		std::map<std::string, cv::Mat> imgs;
		if (folder.is_directory())
			for (const auto& imgFile : std::filesystem::directory_iterator(folder.path()))
				if (imgFile.status().type() == std::filesystem::file_type::regular)
					imgs.insert(std::make_pair(imgFile.path().filename().replace_extension().string(), cv::imread(imgFile.path().string())));
		calibrator.Push(imgs, display);
	}


	std::ofstream ofs("tmp.txt");
	ofs << calibrator.p2ds.size()<<std::endl;
	for (int i = 0; i < calibrator.p2ds.size(); i++) {
		ofs << calibrator.p2ds[i].size() << std::endl;
		for (const auto& iter : calibrator.p2ds[i]) {
			ofs << iter.first << std::endl;
			for (int j = 0; j < iter.second.size(); j++)
				ofs << iter.second[j].x << " " << iter.second[j].y << std::endl;
		}
	}

	//std::ifstream ifs("tmp.txt");
	//int frameSize;
	//ifs >> frameSize;
	//for (int i = 0; i < frameSize; i++) {
	//	int camSize;
	//	ifs >> camSize;
	//	std::map<std::string, std::vector<cv::Point2f>> pairs;
	//	for (int camIdx = 0; camIdx < camSize; camIdx++) {
	//		std::string camStr;
	//		ifs >> camStr;
	//		std::vector<cv::Point2f> p2ds(88);
	//		for (int j = 0; j < p2ds.size(); j++)
	//			ifs >> p2ds[j].x >> p2ds[j].y;
	//		pairs.insert(std::make_pair(camStr, p2ds));
	//	}

	//	if (i != 1 && i != 10)
	//		calibrator.p2ds.emplace_back(pairs);
	//}

	calibrator.Init();
	calibrator.Bundle();
	calibrator.Evaluate();
	SerializeCameras(calibrator.cams, jsonPath.string());

}


int main()
{
	const cv::Size imgSize(2048, 2048);
	const cv::Size patternSize(8, 11);
	const cv::Size2f squareSize(0.0835, 0.0835);
	const std::filesystem::path internalPath("../data/internal");
	const std::filesystem::path externalPath("../data/external");
	const std::filesystem::path jsonPath("../data/calibration.json");

	
	// CalibInternal(patternSize, squareSize, internalPath, jsonPath, true);
	CalibExternal(patternSize, squareSize, externalPath, jsonPath, true);


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