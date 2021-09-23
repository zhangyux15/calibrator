#pragma once
#include <cmath>
#include <fstream>
#include <type_traits>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <iostream>


namespace MathUtil {
	template<typename T>
	inline Eigen::Matrix<T, 3, 3> Skew(const Eigen::Matrix<T, 3, 1>& vec)
	{
		Eigen::Matrix<T, 3, 3> skew;
		skew << 0, -vec.z(), vec.y(),
			vec.z(), 0, -vec.x(),
			-vec.y(), vec.x(), 0;
		return skew;
	}

	template<typename T>
	Eigen::Matrix<T, 3, 3> Rodrigues(const Eigen::Ref<const Eigen::Matrix<T, 3, 1>> r) {
		const T theta = r.norm();
		Eigen::Matrix<T, 3, 3> R;
		if (theta < T(1e-5))
			R = Eigen::Matrix3d::Identity().cast<T>();
		else
			R = Eigen::AngleAxis<T>(theta, r / theta).matrix();
		return R;
	}

	template <typename T>
	Eigen::Matrix<T, -1, -1> LoadMat(const std::string& filename) {
		std::ifstream ifs(filename);
		if (!ifs.is_open()) {
			std::cerr << "can not open fie: " << filename;
			std::abort();
		}
		int rows, cols;
		ifs >> rows >> cols;
		Eigen::Matrix<T, -1, -1> mat(rows, cols);
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				ifs >> mat(i, j);
		return mat;
	}

	template <typename T>
	inline void SaveMat(const Eigen::Matrix<T, -1, -1>& mat, const std::string& filename) {
		std::ofstream ofs(filename);
		ofs << mat.rows() << " " << mat.cols() << std::endl;
		ofs << mat << std::endl;
		ofs.close();
	}
}

