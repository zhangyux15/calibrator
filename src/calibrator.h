#pragma once
#include "camera.h"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <json/json.h>


struct Calibrator
{
	enum CalibratorType
	{
		CALIBRATOR_NULL = -1,
		CHESSBOARD,
		ARUCOBOARD
	};

	struct CornerPack
	{
		std::vector<cv::Point2f> points;
		std::vector<int> ids;
		Eigen::Matrix3Xf pointsAligned;
	};

	std::map<std::string, Camera> cams;
	bool display = true;
	long calibFlags = NULL;
	int displayHeight = 800;
	CalibratorType type;
	
	// for chessboard and arucoboard
	cv::Size patternSize;
	float squareLen;

	// for aruco
	float arucoMarkerLen;
	long arucoDictDef;
	bool arucoRefindStrategy;
	cv::Ptr<cv::aruco::Dictionary> arucoDictionary;
	cv::Ptr<cv::aruco::CharucoBoard> charucoboard;
	cv::Ptr<cv::aruco::Board> arucoBoard;

	// data
	std::map<int, std::map<std::string, CornerPack>> seqCorners;

	void InitChessboard(const cv::Size& _patternSize, const float& _squareLen);
	void InitArucoboard(const cv::Size& _patternSize, const float& _squareLen, const float& _markerLen,
		const long& _dictDef = cv::aruco::DICT_4X4_50, const bool& _refindStrategy = false);

	bool Push(const cv::Mat& img, const int& frameIdx, const std::string& sn);
	void SaveCache(const std::string& path);
	void LoadCache(const std::string& path);
	void CalibInternal();
	void CalibExternal();
	void InitExternal();
	void BundleExternal();
	void EvaluateExternal();
};


