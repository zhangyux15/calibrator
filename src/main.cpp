#include "calibrator.h"
#include <filesystem>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <sstream>


void Load(Calibrator& calibrator, const std::filesystem::path& dataPath, const cv::Size& tarSize = cv::Size()) {
	std::filesystem::path markerCache = dataPath / "markerCache.txt";
	if (std::filesystem::exists(markerCache)) 
		calibrator.LoadCache(markerCache.string());
	else {
		for (const auto& folder : std::filesystem::directory_iterator(dataPath)) {
			if (folder.is_regular_file()) {
				const std::string extension = folder.path().extension().string();
				if (extension != ".jpeg" && extension != ".jpg" && extension != ".png" && extension != ".png")
					continue;

				std::string filePath = folder.path().string();
				std::istringstream sstream(folder.path().filename().replace_extension().string());

				std::string frameIdxStr, sn;
				std::getline(sstream, frameIdxStr, '_');
				std::getline(sstream, sn);
				const int frameIdx = std::stoi(frameIdxStr);

				cv::Mat img = cv::imread(filePath);
				if (tarSize != cv::Size() && tarSize != cv::Size(img.cols, img.rows))
					cv::resize(img, img, tarSize);

				calibrator.Push(img, frameIdx, sn);
			}
		}
		calibrator.SaveCache(markerCache.string());
	}
}


int main()
{
	const cv::Size imgSize(5320, 4600);

	const std::filesystem::path internalPath("../data/internal");
	const std::filesystem::path externalPath("../data/external_aruco");
	const std::filesystem::path jsonPath("../data/calibration_full.json");

	Calibrator calibrator;
	//calibrator.InitChessboard({ 10, 7 }, 0.07f);
	calibrator.InitArucoboard({ 8,5 }, 0.08, 0.062);

	//Load(calibrator, internalPath);
	//calibrator.CalibInternal();

	calibrator.cams = ParseCameras(jsonPath.string());
	Load(calibrator, externalPath);
	calibrator.CalibExternal();
	SerializeCameras(calibrator.cams, jsonPath.string());

	//ArucoInternalCalibrator calibrator(cv::Size(9,6), 0.08, 0.062);
	//CalibInternal(calibrator, internalPath, jsonPath, true);

	//ExternalCalibrator calibrator({10, 7}, 0.07);
	//ArucoExternalCalibrator calibrator(cv::Size(9, 6), 0.08, 0.062);


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
