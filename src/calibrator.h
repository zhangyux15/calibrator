#pragma once
#include "camera.h"
#include <opencv2/core.hpp>
#include <json/json.h>


struct InternalCalibrator
{
	Camera cam;
	cv::Size patternSize;
	cv::Size2f squareSize;
	std::vector<std::vector<cv::Point2f>> p2ds;

	void Clear() { p2ds.clear(); }
	bool Push(const cv::Mat& img, const bool& display = false);
	void Calib();
};


struct ExternalCalibrator
{
	cv::Size patternSize;
	cv::Size2f squareSize;
	std::map<std::string, Camera> cams;
	std::vector<std::map<std::string, std::vector<cv::Point2f>>> p2ds;
	void Clear() { p2ds.clear(); }
	bool Push(const std::map<std::string, cv::Mat>& imgs, const bool& display = false);
	void Init();
	void Bundle();
	void Evaluate();
};

