#pragma once
#include "camera.h"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <json/json.h>


struct InternalCalibrator
{
	Camera cam;
	cv::Size patternSize;
	float squareLen;
	long calibFlags = NULL;

	InternalCalibrator(const cv::Size& _patternSize, const float& _squareLen, const long& _calibFlags=NULL) {
		patternSize = _patternSize;
		squareLen = _squareLen;
		calibFlags = _calibFlags;
	}

	std::vector<std::vector<cv::Point2f>> p2ds;
	void Clear() { p2ds.clear(); cam = Camera(); }
	bool Push(const cv::Mat& img, const bool& display = false);
	void Calib();
};


struct ExternalCalibrator
{
	cv::Size patternSize;
	float squareLen;
	std::map<std::string, Camera> cams;
	std::vector<std::map<std::string, std::vector<cv::Point2f>>> p2ds;
	long calibFlags = NULL;

	ExternalCalibrator(const cv::Size& _patternSize, const float& _squareLen, const long& _calibFlags = NULL) {
		patternSize = _patternSize;
		squareLen = _squareLen;
		calibFlags = _calibFlags;
	}

	void Clear() { p2ds.clear(); }
	bool Push(const std::map<std::string, cv::Mat>& imgs, const bool& display = false);
	void Init();
	void Bundle();
	void Evaluate();
};

