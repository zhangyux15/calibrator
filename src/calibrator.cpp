#include <fstream>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Eigen>
#include "calibrator.h"


bool InternalCalibrator::Push(const cv::Mat& img, const bool& display)
{
	if (cam.cvImgSize.empty())
		cam.cvImgSize = cam.cvNewImgSize = cv::Size(img.cols, img.rows);
	
	std::vector<cv::Point2f> corners;
	bool patternFound = cv::findChessboardCornersSB(img, patternSize, corners, cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_EXHAUSTIVE);
	if (patternFound) {
		p2ds.emplace_back(corners);
		if (display) {
			cv::Mat temp;
			const float rate = 1024.f / float(img.cols);
			cv::resize(img, temp, cv::Size(1024, int(std::ceil(float(img.rows) * rate))));
			cv::drawChessboardCorners(temp, patternSize, cv::Mat(patternSize.width * patternSize.height, 2, CV_32FC1, &corners.data()->x) * rate, true);
			cv::imshow("chessboard", temp);
			cv::waitKey(1);
		}
	}
	return patternFound;
}


void InternalCalibrator::Calib()
{
	std::vector<std::vector<cv::Point3f>> p3ds(p2ds.size());
	for (auto&& points : p3ds)
		for (int row = 0; row < patternSize.height; row++)
			for (int col = 0; col < patternSize.width; col++)
				points.emplace_back(cv::Point3f(col*squareSize.width, row*squareSize.height, 0.f));

	std::vector<cv::Mat> tvecsMat;
	std::vector<cv::Mat> rvecsMat;

	// calibrate
	calibrateCamera(p3ds, p2ds, cam.cvImgSize, cam.cvK, cam.cvDistCoeff, rvecsMat, tvecsMat);
	cam.Update();

	// evaluate
	for (int imgIdx = 0; imgIdx < p2ds.size(); imgIdx++) {
		std::vector<cv::Point2f> _p2ds;
		cv::projectPoints(p3ds[imgIdx], rvecsMat[imgIdx], tvecsMat[imgIdx], cam.cvK, cam.cvDistCoeff, _p2ds);

		float err = 0.f;
		for (int pIdx = 0; pIdx < _p2ds.size(); pIdx++)
			err += cv::norm(_p2ds[pIdx] - p2ds[imgIdx][pIdx]);
		err /= _p2ds.size();
		std::cout << "image " << imgIdx << " err: " << err << "pixel" << std::endl;
	}
}


bool ExternalCalibrator::Push(const std::map<std::string, cv::Mat>& imgs, const bool& display)
{
	std::map<std::string, std::vector<cv::Point2f>> corrs;
	for (const auto& iter : imgs) {
		std::vector<cv::Point2f> corners;
		const Camera& cam = cams.find(iter.first)->second;
		cv::Mat temp;
		cv::remap(iter.second, temp, cam.cvRectifyMapX, cam.cvRectifyMapY, cv::INTER_LINEAR);
		bool patternFound = cv::findChessboardCornersSB(temp, patternSize, corners, cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_EXHAUSTIVE);
		if (patternFound) {
			corrs.insert(std::make_pair(iter.first, corners));
			if (display) {
				const float rate = 1024.f / float(iter.second.cols);
				cv::resize(temp, temp, cv::Size(1024, int(std::ceil(float(iter.second.rows) * rate))));
				cv::drawChessboardCorners(temp, patternSize, cv::Mat(patternSize.width * patternSize.height, 2, CV_32FC1, &corners.data()->x) * rate, true);
				cv::imshow("chessboard", temp);
				cv::waitKey(1);
			}
		}
	}
	if (corrs.size() >= 2) {
		p2ds.emplace_back(corrs);
		return true;
	}
	else
		return false;
}


void ExternalCalibrator::Init()
{
	std::set<std::string> initCams;
	
	bool setAnchor = false;
	while (initCams.size() != cams.size()) {
		// find next camera to solve pnp
		Eigen::VectorXi candidates = Eigen::VectorXi::Zero(int(p2ds.size()));
		for (int i = 0; i < p2ds.size(); i++) {
			int validCnt = 0;
			for (const auto& iter : p2ds[i])
				if (std::find(initCams.begin(), initCams.end(), iter.first) != initCams.end())
					validCnt++;
			if (!setAnchor)
				candidates[i] = p2ds[i].size() - validCnt;
			else if (validCnt >= 2 && validCnt != p2ds[i].size())
				candidates[i] = validCnt;
		}

		int maxIdx;
		const int maxVal = candidates.maxCoeff(&maxIdx);
		if (maxVal == 0) {
			std::cerr << "Init Camera Failed" << std::endl;
			std::abort();
		}

		std::cout << (setAnchor ? "Init camera" : "Set Anchor") << "\t";
		for (const auto& iter : p2ds[maxIdx])
			if (std::find(initCams.begin(), initCams.end(), iter.first) == initCams.end())
				std::cout << iter.first << "\t";
		std::cout << " by frame " << maxIdx << std::endl;

		std::vector<cv::Point3f> p3ds(patternSize.height * patternSize.width);
		if (!setAnchor) {
			for (int row = 0; row < patternSize.height; row++)
				for (int col = 0; col < patternSize.width; col++)
					p3ds[row* patternSize.width + col] = cv::Point3f(col*squareSize.width, row*squareSize.height, 0.f);
		}
		else {
			Triangulator triangulator;
			std::vector<Eigen::Matrix<float, 3, 4>> projs;
			for (const auto& iter : p2ds[maxIdx])
				if (std::find(initCams.begin(), initCams.end(), iter.first) != initCams.end())
					projs.emplace_back(cams.find(iter.first)->second.proj);
			triangulator.projs = Eigen::Map<Eigen::Matrix3Xf>(projs.data()->data(), 3, 4 * projs.size());
			for (int i = 0; i < patternSize.height * patternSize.width; i++) {
				std::vector<Eigen::Vector3f> points;
				for (const auto& iter : p2ds[maxIdx]) 
					if (std::find(initCams.begin(), initCams.end(), iter.first) != initCams.end()) 
						points.emplace_back(Eigen::Vector3f(iter.second[i].x, iter.second[i].y, 1.f));

				triangulator.points = Eigen::Map<Eigen::Matrix3Xf>(points.data()->data(), 3, points.size());
				triangulator.Solve();
				p3ds[i] = cv::Point3f(triangulator.pos.x(), triangulator.pos.y(), triangulator.pos.z());
			}
		}

		for (const auto& iter : p2ds[maxIdx]) {
			if (std::find(initCams.begin(), initCams.end(), iter.first) == initCams.end()) {
				Camera& cam = cams.find(iter.first)->second;
				cv::Mat rvec, rmat;
				cv::solvePnP(p3ds, iter.second, cam.cvNewK, cv::Mat(), rvec, cam.cvT);
				cv::Rodrigues(rvec, rmat);
				cam.cvR = rmat;
				cam.Update();
				initCams.insert(iter.first);
			}
		}

		if (!setAnchor)
			setAnchor = true;
	}
}


struct ReprojCostFunctor
{
	ReprojCostFunctor(const Eigen::Matrix3d& _K, const Eigen::Vector2d& _p2d) {
		m_K = _K;
		m_p2d = _p2d;
	}

	template<typename T>
	bool operator()(const T* const _r, const T* const _t, const T* _p3d, T* residuals) const {
		const Eigen::Map<const Eigen::Matrix<T, 3, 1>> r(_r);
		const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(_t);
		const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p3d(_p3d);

		const T theta = r.norm();
		Eigen::Matrix<T, 3, 3> R;
		if (theta < T(DBL_EPSILON))
			R = Eigen::Matrix3d::Identity().cast<T>();
		else
			R = Eigen::AngleAxis<T>(theta, r / theta).matrix();

		Eigen::Matrix<T, 3, 1> p = m_K.cast<T>()*(R*p3d + t);
		Eigen::Matrix<T, 2, 1> uv(p[0] / p[2], p[1] / p[2]);
		residuals[0] = (uv - m_p2d.cast<T>()).norm();
		return true;
	}

private:
	Eigen::Matrix3d m_K;
	Eigen::Vector2d m_p2d;
};


void ExternalCalibrator::Bundle()
{
	std::map<std::string, Eigen::Vector3d> rs;
	std::map<std::string, Eigen::Vector3d> ts;
	for (const auto& cam : cams) {
		Eigen::AngleAxisf angleAxis(cam.second.R);
		rs.insert(std::make_pair(cam.first, Eigen::Vector3f(angleAxis.axis() * angleAxis.angle()).cast<double>()));
		ts.insert(std::make_pair(cam.first, cam.second.T.cast<double>()));
	}

	std::vector<Eigen::Matrix3Xd> p3ds(p2ds.size(), Eigen::Matrix3Xd(3, patternSize.height * patternSize.width));
	ceres::Problem problem;
	for (int frameIdx = 0; frameIdx < p2ds.size(); frameIdx++) {
		Triangulator triangulator;
		std::vector<Eigen::Matrix<float, 3, 4>> projs;
		for (const auto& iter : p2ds[frameIdx])
			projs.emplace_back(cams.find(iter.first)->second.proj);
		triangulator.projs = Eigen::Map<Eigen::Matrix3Xf>(projs.data()->data(), 3, 4 * projs.size());

		for (int i = 0; i < patternSize.height * patternSize.width; i++) {
			std::vector<Eigen::Vector3f> points;
			for (const auto& iter : p2ds[frameIdx])
				points.emplace_back(Eigen::Vector3f(iter.second[i].x, iter.second[i].y, 1.f));
			triangulator.points = Eigen::Map<Eigen::Matrix3Xf>(points.data()->data(), 3, points.size());
			triangulator.Solve();
			p3ds[frameIdx].col(i) = triangulator.pos.cast<double>();
		}
		
		for (auto&& iter : p2ds[frameIdx]) {
			const Camera& cam = cams.find(iter.first)->second;
			double* _r = rs.find(iter.first)->second.data();
			double* _t = ts.find(iter.first)->second.data();
			for (int pIdx = 0; pIdx < iter.second.size(); pIdx++) {
				const Eigen::Vector2d p2d(iter.second[pIdx].x, iter.second[pIdx].y);
				ceres::CostFunction* func = new ceres::AutoDiffCostFunction<ReprojCostFunctor, 1, 3, 3, 3>(new ReprojCostFunctor(cam.K.cast<double>(), p2d));
				problem.AddResidualBlock(func, NULL, _r, _t, p3ds[frameIdx].col(pIdx).data());
			}
		}
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_type = ceres::LINE_SEARCH;
	options.max_num_iterations = 5000;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	// update
	for (auto&& cam : cams) {
		cv::eigen2cv(MathUtil::Rodrigues(Eigen::Vector3f(rs.find(cam.first)->second.cast<float>())), cam.second.cvR);
		cv::eigen2cv(Eigen::Vector3f(ts.find(cam.first)->second.cast<float>()), cam.second.cvT);
		cam.second.Update();
	}

	std::cout << summary.BriefReport() << std::endl;
}


void ExternalCalibrator::Evaluate()
{
	// evaluate
	for (int frameIdx = 0; frameIdx < p2ds.size(); frameIdx++) {
		Triangulator triangulator;
		std::vector<Eigen::Matrix<float, 3, 4>> projs;
		for (const auto& iter : p2ds[frameIdx])
			projs.emplace_back(cams.find(iter.first)->second.proj);
		triangulator.projs = Eigen::Map<Eigen::Matrix3Xf>(projs.data()->data(), 3, 4 * projs.size());

		std::vector<float> error;
		for (int i = 0; i < patternSize.height * patternSize.width; i++) {
			std::vector<Eigen::Vector3f> points;
			for (const auto& iter : p2ds[frameIdx])
				points.emplace_back(Eigen::Vector3f(iter.second[i].x, iter.second[i].y, 1.f));
			triangulator.points = Eigen::Map<Eigen::Matrix3Xf>(points.data()->data(), 3, points.size());
			triangulator.Solve();

			for (int j = 0; j < projs.size(); j++) {
				Eigen::Vector2f p2dEst = (projs[j] * triangulator.pos.homogeneous()).hnormalized();
				error.emplace_back((p2dEst - points[j].head(2)).norm());
			}
		}
		std::cout << frameIdx << " error: " << std::accumulate(error.begin(), error.end(), 0.f) / error.size() << std::endl;
	}
}

//
//struct AlignRTCostFunctor
//{
//	AlignRTCostFunctor(const MathUtil::Matrix34d& _proj, const Eigen::Vector3d& _p3d, const Eigen::Vector2d& _p2d) {
//		m_proj = _proj;
//		m_p3d = _p3d;
//		m_p2d = _p2d;
//	}
//
//	template<typename T>
//	bool operator()(const T* const _r, const T* const _t, T* residuals) const {
//		const Eigen::Map<const Eigen::Matrix<T, 3, 1>> r(_r);
//		const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(_t);
//
//		const T theta = r.norm();
//		Eigen::Matrix<T, 3, 1> m_p3dAffined;
//		if (theta > T(DBL_EPSILON))
//			m_p3dAffined = Eigen::AngleAxis<T>(theta, r / theta).matrix() * m_p3d.cast<T>();
//		m_p3dAffined += t;
//
//		Eigen::Matrix<T, 3, 1> uvHomo = m_proj.cast<T>()* m_p3dAffined.homogeneous();
//		Eigen::Matrix<T, 2, 1> uv(uvHomo[0] / uvHomo[2], uvHomo[1] / uvHomo[2]);
//		residuals[0] = (uv - m_p2d.cast<T>()).norm();
//		return true;
//	}
//
//private:
//	MathUtil::Matrix34d m_proj;
//	Eigen::Vector3d m_p3d;
//	Eigen::Vector2d m_p2d;
//};


//
//void Calibrator::AlignRT(std::map<std::string, Camera>& cameras,
//	const std::vector<std::tuple<std::string, cv::Point3f, cv::Point2f>>& corres)
//{
//	Eigen::Vector3d R = Eigen::Vector3d::Constant(1e-3f);
//	Eigen::Vector3d T = Eigen::Vector3d::Zero();
//	ceres::Problem problem;
//
//	for (const auto& corr : corres) {
//		const Eigen::Matrix34f& proj = cameras.find(std::get<0>(corr))->second.proj;
//		ceres::CostFunction* func = new ceres::AutoDiffCostFunction<AlignRTCostFunctor, 1, 3, 3>(
//			new AlignRTCostFunctor(proj, std::get<1>(corr), std::get<2>(corr)));
//		problem.AddResidualBlock(func, NULL, R.data(), T.data());
//	}
//
//	ceres::Solver::Options options;
//	options.linear_solver_type = ceres::DENSE_QR;
//	// options.minimizer_type = ceres::LINE_SEARCH;
//	options.max_num_iterations = 5000;
//	options.minimizer_progress_to_stdout = true;
//	ceres::Solver::Summary summary;
//	ceres::Solve(options, &problem, &summary);
//	std::cout << summary.BriefReport() << std::endl;
//
//	// update
//	for (auto&& camera : cameras) {
//		camera.second.T = camera.second.T + camera.second.R*T.cast<float>();
//		camera.second.R = camera.second.R * MathUtil::Rodrigues(R.cast<float>());
//		camera.second.Update();
//	}
//}

