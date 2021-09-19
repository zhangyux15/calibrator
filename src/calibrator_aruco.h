#pragma once
#include "camera.h"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <json/json.h>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>


struct ArucoInternalCalibrator
{
	Camera cam;
	cv::Size2f patternSize;
	float squareLen, markerLen;
	long dictDef;
	bool refindStrategy;
	long calibFlags;
	cv::Ptr<cv::aruco::Dictionary> dictionary;
	cv::Ptr<cv::aruco::CharucoBoard> charucoboard;
	cv::Ptr<cv::aruco::Board> board;
	std::vector<cv::Mat> allCharucoCorners;
	std::vector<cv::Mat> allCharucoIds;

	ArucoInternalCalibrator(const cv::Size& _patternSize, const float& _squareLen, const float& _markerLen,
		const long& _dictDef = cv::aruco::DICT_4X4_50, const bool& _refindStrategy = false, const long& _calibFlags = NULL) {
		patternSize = _patternSize;
		squareLen = _squareLen;
		markerLen = _markerLen;
		refindStrategy = _refindStrategy;
		calibFlags = _calibFlags;
		dictionary = cv::aruco::getPredefinedDictionary(dictDef);
		charucoboard = cv::aruco::CharucoBoard::create(patternSize.width, patternSize.height, squareLen, markerLen, dictionary);
		board = charucoboard.staticCast<cv::aruco::Board>();
	}

	void Clear() { allCharucoCorners.clear(); allCharucoIds.clear(); cam = Camera(); }
	bool Push(const cv::Mat& img, const bool& display = false);
	void Calib();
};



//
//
//struct ExternalCalibrator
//{
//	cv::Size patternSize;
//	cv::Size2f squareSize;
//	std::map<std::string, Camera> cams;
//	std::vector<std::map<std::string, std::vector<cv::Point2f>>> p2ds;
//	long calibFlags = 0;
//	void Clear() { p2ds.clear(); }
//	bool Push(const std::map<std::string, cv::Mat>& imgs, const bool& display = false);
//	void Init();
//	void Bundle();
//	void Evaluate();
//};
//
