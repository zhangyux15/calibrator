#include <fstream>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Eigen>
#include "calibrator.h"



void Calibrator::InitChessboard(const cv::Size& _patternSize, const float& _squareLen) {
	type = CHESSBOARD;
	patternSize = _patternSize;
	squareLen = _squareLen;
}


void Calibrator::InitArucoboard(const cv::Size& _patternSize, const float& _squareLen, const float& _markerLen, const long& _dictDef, const bool& _refindStrategy) {
	type = ARUCOBOARD;
	patternSize = _patternSize;
	squareLen = _squareLen;
	arucoMarkerLen = _markerLen;
	arucoRefindStrategy = _refindStrategy;
	arucoDictDef = _dictDef;
	arucoDictionary = cv::aruco::getPredefinedDictionary(_dictDef);
	charucoboard = cv::aruco::CharucoBoard::create(patternSize.width + 1, patternSize.height + 1, squareLen, arucoMarkerLen, arucoDictionary);
	arucoBoard = charucoboard.staticCast<cv::aruco::Board>();
}


bool Calibrator::Push(const cv::Mat& img, const int& frameIdx, const std::string& sn)
{
	auto camIter = cams.find(sn);
	if (camIter == cams.end()) {
		Camera cam;
		cam.imgSize = cv::Size(img.cols, img.rows);
		camIter = cams.insert(std::make_pair(sn, cam)).first;
	}
	else 
		assert(camIter->second.imgSize == cv::Size(img.cols, img.rows));

	Camera& cam = camIter->second;
	CornerPack pack;
	bool patternFound = false;
	if (type == CHESSBOARD) {
		for (int i = 0; i < patternSize.width * patternSize.height; i++)
			pack.ids.emplace_back(i);
		patternFound = cv::findChessboardCornersSB(img, patternSize, pack.points, calibFlags);
	}
	else if (type == ARUCOBOARD) {
		std::vector<int> markerIds;
		std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
		cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
		cv::aruco::detectMarkers(img, arucoDictionary, markerCorners, markerIds, parameters, rejectedCandidates);
		if (arucoRefindStrategy)
			cv::aruco::refineDetectedMarkers(img, arucoBoard, markerCorners, markerIds, rejectedCandidates);

		if (!markerCorners.empty()) {
			cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, img, charucoboard, 
				pack.points, pack.ids);
			patternFound = pack.points.size() > 4;
		}
	}

	if (patternFound) {
		pack.pointsAligned.setZero(3, patternSize.width * patternSize.height);
		for (int idx = 0; idx < pack.ids.size(); idx++)
			pack.pointsAligned.col(pack.ids[idx]) = Eigen::Vector3f(pack.points[idx].x, pack.points[idx].y, 1.f);

		// undistort if with calibrated internal
		if(!cam.rectifyMapX.empty())
			cv::undistortPoints(pack.points, pack.points, cam.originK, cam.distCoeff, cv::noArray(), cam.cvK);

		auto frameSetIter = seqCorners.find(frameIdx);
		if (frameSetIter == seqCorners.end())
			frameSetIter = seqCorners.insert(std::make_pair(frameIdx, std::map<std::string, CornerPack>())).first;

		auto cornerIter = frameSetIter->second.find(sn);
		if (cornerIter == frameSetIter->second.end())
			frameSetIter->second.insert(std::make_pair(sn, pack));
		else {
			std::cerr << "Duplicate Push for frame:" << frameIdx << " sn:" << sn << std::endl;
			std::abort();
		}
	}

	if (display) {
		cv::Mat temp;
		const float rate = float(displayHeight) / float(img.cols);
		const int displayWidth = int(std::ceil(float(img.rows) * rate));

		if (cam.rectifyMapY.empty())
			cv::resize(img, temp, cv::Size(displayHeight, displayWidth));
		else {
			cv::remap(img, temp, cam.rectifyMapX, cam.rectifyMapY, cv::INTER_LINEAR);
			cv::resize(temp, temp, cv::Size(displayHeight, displayWidth));
		}
		if (patternFound) {
			auto cornersClip = pack.points;
			for (auto&& p : cornersClip)
				p *= rate;
			if (type == CHESSBOARD)
				cv::drawChessboardCorners(temp, patternSize, cornersClip, true);
			else if (type == ARUCOBOARD)
				cv::aruco::drawDetectedCornersCharuco(temp, cornersClip, pack.ids);
		}
		cv::imshow("chessboard", temp);
		cv::waitKey(1);
	}
	return patternFound;
}


void Calibrator::CalibInternal()
{
	assert(type != CALIBRATOR_NULL);
	if (type == CHESSBOARD) {
		for (auto&& camIter : cams) {
			const std::string& sn = camIter.first;
			Camera& cam = camIter.second;
			std::vector<std::vector<cv::Point2f>> p2ds;
			for (const auto frameIter : seqCorners) {
				const auto cornerIter = frameIter.second.find(sn);
				if (cornerIter != frameIter.second.end())
					p2ds.emplace_back(cornerIter->second.points);
			}
			if (p2ds.empty())
				continue;

			std::vector<std::vector<cv::Point3f>> p3ds(p2ds.size());
			for (auto&& points : p3ds)
				for (int row = 0; row < patternSize.height; row++)
					for (int col = 0; col < patternSize.width; col++)
						points.emplace_back(cv::Point3f(col * squareLen, row * squareLen, 0.f));

			std::vector<cv::Mat> tvecsMat;
			std::vector<cv::Mat> rvecsMat;

			// calibrate
			calibrateCamera(p3ds, p2ds, cam.imgSize, cam.originK, cam.distCoeff, rvecsMat, tvecsMat);
			cam.Rectify();
			cam.CV2Eigen();

			// evaluate
			std::cout << "sn: " << sn << std::endl;
			for (int idx = 0; idx < p2ds.size(); idx++) {
				std::vector<cv::Point2f> _p2ds;
				cv::projectPoints(p3ds[idx], rvecsMat[idx], tvecsMat[idx], cam.originK, cam.distCoeff, _p2ds);
				float err = 0.f;
				for (int pIdx = 0; pIdx < _p2ds.size(); pIdx++)
					err += cv::norm(_p2ds[pIdx] - p2ds[idx][pIdx]);
				err /= _p2ds.size();
				std::cout << "err: " << err << "pixel" << std::endl;
			}
		}
	}
	else if (type == ARUCOBOARD) {
		for (auto&& camIter : cams) {
			const std::string& sn = camIter.first;
			Camera& cam = camIter.second;

			std::vector<std::vector<cv::Point2f>> allArucoCorners;
			std::vector<std::vector<int>> allArucoIds;
			for (const auto frameIter : seqCorners) {
				const auto cornerIter = frameIter.second.find(sn);
				if (cornerIter != frameIter.second.end()) {
					allArucoCorners.emplace_back(cornerIter->second.points);
					allArucoIds.emplace_back(cornerIter->second.ids);
				}
			}
			if (allArucoCorners.size() < 4) {
				std::cerr << "Not enough aruco corners to calibrate internal parameters" << std::endl;
				continue;
			}

			std::vector<cv::Mat> tvecsMat;
			std::vector<cv::Mat> rvecsMat;
			cv::Mat stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors;

			// calibrate
			cv::aruco::calibrateCameraCharuco(allArucoCorners, allArucoIds, charucoboard,
				cam.imgSize, cam.originK, cam.distCoeff, rvecsMat, tvecsMat,
				stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors);

			cam.Rectify();
			cam.CV2Eigen();

			// evaluate
			std::cout << "sn: " << sn << std::endl;
			std::cout << perViewErrors << std::endl;
		}

	}
}


void Calibrator::InitExternal()
{
	std::vector<std::map<std::string, CornerPack>> seqCornersVec;
	for (const auto& iter : seqCorners)
		seqCornersVec.emplace_back(iter.second);

	std::set<std::string> initCams;
	bool firstFlag = true;
	while (initCams.size() != cams.size()) {
		// find next camera to solve pnp
		Eigen::VectorXi candidates = Eigen::VectorXi::Zero(int(seqCornersVec.size()));
		for (int i = 0; i < seqCornersVec.size(); i++) {
			int validCnt = 0;
			int validView = 0;
			for (const auto& iter : seqCornersVec[i]) {
				const auto initCam = std::find(initCams.begin(), initCams.end(), iter.first);
				if (firstFlag || initCam != initCams.end()){
					validView++;
					validCnt += iter.second.ids.size();
				}
			}
			if (firstFlag || validView != seqCornersVec[i].size())
				candidates[i] = validCnt;
		}
		int maxIdx;
		const int maxVal = candidates.maxCoeff(&maxIdx);
		if (maxVal == 0) {
			std::cerr << "Init External Failed When Iterate Next Cam" << std::endl;
			std::abort();
		}

		std::cout << (!firstFlag ? "Init camera" : "Set Anchor") << "\t";
		for (const auto& iter : seqCornersVec[maxIdx])
			if (std::find(initCams.begin(), initCams.end(), iter.first) == initCams.end())
				std::cout << iter.first << "\t";
		std::cout << " by frame " << maxIdx << std::endl;

		for (const auto& iter : seqCornersVec[maxIdx]) {
			if (std::find(initCams.begin(), initCams.end(), iter.first) == initCams.end()) {
				const CornerPack& pack = iter.second;
				std::vector<cv::Point3f> p3ds;
				std::vector<cv::Point2f> p2ds;
				for (int idx = 0; idx < pack.ids.size(); idx++) {
					const int cornerId = pack.ids[idx];
					const int row = cornerId / patternSize.width;
					const int col = cornerId % patternSize.width;
					if (firstFlag) {
						p2ds.emplace_back(pack.points[idx]);
						p3ds.emplace_back(cv::Point3f(col * squareLen, row * squareLen, 0.f));
					}
					else {
						Triangulator triangulator;
						std::vector<Eigen::Matrix<float, 3, 4>> projs;
						std::vector<Eigen::Vector3f> points;

						for (const auto& iter : seqCornersVec[maxIdx]) {
							if (std::find(initCams.begin(), initCams.end(), iter.first) != initCams.end()) {
								for (int tarIdx = 0; tarIdx < iter.second.ids.size(); tarIdx++) {
									if (iter.second.ids[tarIdx] == cornerId) {
										projs.emplace_back(cams.find(iter.first)->second.eiProj);
										points.emplace_back(Eigen::Vector3f(iter.second.points[tarIdx].x, iter.second.points[tarIdx].y, 1.f));
									}
								}
							}
						}
						
						if (projs.size() >= 2) {
							triangulator.projs = Eigen::Map<Eigen::Matrix3Xf>(projs.data()->data(), 3, 4 * projs.size());
							triangulator.points = Eigen::Map<Eigen::Matrix3Xf>(points.data()->data(), 3, points.size());
							triangulator.Solve();

							p2ds.emplace_back(pack.points[idx]);
							p3ds.emplace_back(cv::Point3f(triangulator.pos.x(), triangulator.pos.y(), triangulator.pos.z()));
						}
					}
				}

				if (p3ds.size() < 4) {
					std::cerr << "Init External Failed When SolvePnP" << std::endl;
					std::abort();
				}

				Camera& cam = cams.find(iter.first)->second;
				cv::Mat rvec, rmat;
				cv::solvePnP(p3ds, p2ds, cam.cvK, cv::Mat(), rvec, cam.cvT);
				cv::Rodrigues(rvec, rmat);
				cam.cvR = rmat;
				cam.Rectify();
				cam.CV2Eigen();
				initCams.insert(iter.first);
			}
		}

		if (firstFlag)
			firstFlag = false;
	}
}


struct PlaneReprojCostFunctor
{
	PlaneReprojCostFunctor(const Eigen::Matrix3d& _K, const Eigen::Vector2d& _pGT, const Eigen::Vector2d& _pPlane) {
		m_K = _K;
		m_pGT = _pGT;
		m_pPlane = _pPlane;
		m_fixCam = false;
		m_fixPlane = false;
	}

	void FixCam(const Eigen::Matrix3d& _R, const Eigen::Vector3d& _t) {
		m_fixCam = true;
		m_R = _R;
		m_t = _t;
	}

	void FixPlane(const Eigen::Matrix3d& _RPlane, const Eigen::Vector3d& _tPlane) {
		m_fixPlane = true;
		m_RPlane = _RPlane;
		m_tPlane = _tPlane;
	}

	template<typename T>
	bool operator()(const T* const _r, const T* const _t, const T* const _rPlane, const T* const _tPlane, T* residuals) const {
		Eigen::Matrix<T, 3, 1> p3d = Eigen::Vector3f(m_pPlane.x(), m_pPlane.y(), 0.0).cast<T>();
		const Eigen::Map<const Eigen::Matrix<T, 3, 1>> rPlane(_rPlane), r(_r);
		const Eigen::Map<const Eigen::Matrix<T, 3, 1>> tPlane(_tPlane), t(_t);
		Eigen::Matrix<T, 3, 3> RPlane = MathUtil::Rodrigues<T>(rPlane);
		Eigen::Matrix<T, 3, 3> R = MathUtil::Rodrigues<T>(r);

		if (!m_fixPlane)
			p3d = RPlane * p3d + tPlane;
		else 
			p3d = m_RPlane.cast<T>() * p3d + m_tPlane.cast<T>();
		
		if (!m_fixCam) 
			p3d = R * p3d + t;
		else
			p3d = m_R.cast<T>() * p3d + m_t.cast<T>();

		Eigen::Matrix<T, 3, 1> p = m_K.cast<T>() * p3d;
		Eigen::Matrix<T, 2, 1> uv(p[0] / p[2], p[1] / p[2]);
		residuals[0] = (uv - m_pGT.cast<T>()).norm();
		return true;
	}

private:
	Eigen::Matrix3d m_K;
	Eigen::Vector2d m_pGT, m_pPlane;
	bool m_fixCam, m_fixPlane;
	Eigen::Matrix3d m_R, m_RPlane;
	Eigen::Vector3d m_t, m_tPlane;
};



void Calibrator::BundleExternal()
{
	assert(!seqCorners.empty());

	std::map<std::string, Eigen::Vector3d> rs;
	std::map<std::string, Eigen::Vector3d> ts;
	for (const auto& cam : cams) {
		ts.insert(std::make_pair(cam.first, cam.second.eiT.cast<double>()));
		Eigen::AngleAxisd angleAxis(cam.second.eiR.cast<double>());
		rs.insert(std::make_pair(cam.first, angleAxis.axis() * angleAxis.angle()));
	}
	Eigen::Matrix3Xd rPlanes = Eigen::Matrix3Xd::Random(3, seqCorners.size());
	Eigen::Matrix3Xd tPlanes = Eigen::Matrix3Xd::Random(3, seqCorners.size());

	std::cout << "Bundle Step 1: Align initial board RT" << std::endl;
	Eigen::VectorXi validBoard = Eigen::VectorXi::Zero(seqCorners.size());

	{ int boardIdx = 0;
	for (const auto& allCorners : seqCorners) {
		std::vector<Eigen::Vector3f> _tarPoints, _srcPoints;
		for (int cornerId = 0; cornerId < patternSize.width * patternSize.height; cornerId++) {
			Triangulator triangulator;
			std::vector<Eigen::Matrix<float, 3, 4>> projs;
			std::vector<Eigen::Vector3f> points;
			for (const auto& iter : allCorners.second) {
				for (int tarIdx = 0; tarIdx < iter.second.ids.size(); tarIdx++) {
					if (iter.second.ids[tarIdx] == cornerId) {
						projs.emplace_back(cams.find(iter.first)->second.eiProj);
						points.emplace_back(Eigen::Vector3f(iter.second.points[tarIdx].x, iter.second.points[tarIdx].y, 1.f));
					}
				}
			}

			if (projs.size() >= 2) {
				triangulator.projs = Eigen::Map<Eigen::Matrix3Xf>(projs.data()->data(), 3, 4 * projs.size());
				triangulator.points = Eigen::Map<Eigen::Matrix3Xf>(points.data()->data(), 3, points.size());
				triangulator.Solve();
				_srcPoints.emplace_back(Eigen::Vector3f(float(cornerId % patternSize.width) * squareLen, float(cornerId / patternSize.width) * squareLen, 0.f));
				_tarPoints.emplace_back(Eigen::Vector3f(triangulator.pos.x(), triangulator.pos.y(), triangulator.pos.z()));
			}
		}
		if (_srcPoints.size() <= std::max(patternSize.width, patternSize.height)) {
			std::cout << "Discard Frame " << allCorners.first << " for too less triangulated points" << std::endl;
			boardIdx++;
			continue;
		}
		Eigen::Matrix3Xf srcPoints = Eigen::Map<Eigen::Matrix3Xf>(_srcPoints.data()->data(), 3, _srcPoints.size());
		Eigen::Matrix3Xf tarPoints = Eigen::Map<Eigen::Matrix3Xf>(_tarPoints.data()->data(), 3, _tarPoints.size());
		const Eigen::Vector3f srcMean = srcPoints.rowwise().mean();
		const Eigen::Vector3f tarMean = tarPoints.rowwise().mean();
		const Eigen::Matrix3Xf srcCoeff = srcPoints.colwise() - srcMean;
		const Eigen::Matrix3Xf tarCoeff =  tarPoints.colwise() - tarMean;

		Eigen::Matrix3f covMat;
		covMat = (srcCoeff * tarCoeff.transpose()) / int(srcCoeff.cols());
		Eigen::JacobiSVD<Eigen::Matrix3f> svd(covMat, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3f V = svd.matrixV();
		Eigen::Matrix3f U = svd.matrixU();
		Eigen::Matrix3f rot = V * U.transpose();
		if (rot.determinant() < 0) {
			V.col(2) *= -1;
			rot = V * U.transpose();
		}
		Eigen::Vector3f t = tarMean - rot * srcMean;

		Eigen::Matrix3Xf error = tarPoints - ((rot * srcPoints).colwise() + t);
		std::cout << "Frame " << allCorners.first << " warp loss: " << error.colwise().norm().mean() << std::endl;

		const Eigen::AngleAxisd angleAxis(rot.cast<double>());
		rPlanes.col(boardIdx) = angleAxis.angle() * angleAxis.axis();
		tPlanes.col(boardIdx) = t.cast<double>();
		validBoard[boardIdx] = true;
		boardIdx++;
	}}

	// warp to anchor frame RT
	int anchorFrame = 100;
	if (!validBoard[anchorFrame]) {
		std::cout << "Warning! Cannot align RT for the first frame, jump RT align" << std::endl;
		anchorFrame = -1;
	}
	else {
		const Eigen::Matrix3d anchorRot = MathUtil::Rodrigues<double>(rPlanes.col(anchorFrame));
		const Eigen::Vector3d anchorT = tPlanes.col(anchorFrame);
		for (int boardIdx = 0; boardIdx < seqCorners.size(); boardIdx++) {
			if (validBoard[boardIdx]) {
				tPlanes.col(boardIdx) = anchorRot.transpose() * (tPlanes.col(boardIdx) - anchorT);
				Eigen::Matrix3d rot = MathUtil::Rodrigues<double>(rPlanes.col(boardIdx));
				Eigen::AngleAxisd angleAxis(anchorRot.transpose() * rot);
				rPlanes.col(boardIdx) = (angleAxis.angle() * angleAxis.axis());
			}
		}

		for (auto&& cam : cams) {
			const Eigen::Matrix3d camRot = cam.second.eiR.cast<double>();
			const Eigen::Vector3d camT = cam.second.eiT.cast<double>();
			ts.find(cam.first)->second = camT + camRot * anchorT;
			Eigen::AngleAxisd angleAxis(camRot * anchorRot);
			rs.find(cam.first)->second = angleAxis.axis() * angleAxis.angle();

			cam.second.eiT = (camT + camRot * anchorT).cast<float>();
			cam.second.eiR = (camRot * anchorRot).cast<float>();
			cam.second.Eigen2CV();
		}
	}

	std::cout << "Bundle Step 2: Solve board RT and camera RT" << std::endl;
	ceres::Problem problem;
	{ int boardIdx = 0;
	for (const auto& allCorners : seqCorners) {
		if (!validBoard[boardIdx]) {
			boardIdx++;
			continue;
		}
		for (auto&& iter : allCorners.second) {
			const Camera& cam = cams.find(iter.first)->second;
			Eigen::Vector3d& r = rs.find(iter.first)->second;
			Eigen::Vector3d& t = ts.find(iter.first)->second;
			for (int pIdx = 0; pIdx < iter.second.ids.size(); pIdx++) {
				const Eigen::Vector2d p2d(iter.second.points[pIdx].x, iter.second.points[pIdx].y);
				const int cornerId = iter.second.ids[pIdx];
				const Eigen::Vector2d pPlane(double(cornerId % patternSize.width) * squareLen,
					double(cornerId / patternSize.width) * squareLen);
				auto cost = new PlaneReprojCostFunctor(cam.eiK.cast<double>(), p2d, pPlane);
				if (boardIdx == anchorFrame)
					cost->FixPlane(MathUtil::Rodrigues<double>(rPlanes.col(boardIdx)), tPlanes.col(boardIdx));
				ceres::CostFunction* func = new ceres::AutoDiffCostFunction<PlaneReprojCostFunctor, 1, 3, 3, 3, 3>(cost);
				problem.AddResidualBlock(func, NULL, r.data(), t.data(), rPlanes.col(boardIdx).data(), tPlanes.col(boardIdx).data());
			}
		}
		boardIdx++;
	}}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_type = ceres::LINE_SEARCH;
	options.max_num_iterations = 5000;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << std::endl;

	// update
	for (auto&& cam : cams) {
		cv::eigen2cv(MathUtil::Rodrigues<float>(Eigen::Vector3f(rs.find(cam.first)->second.cast<float>())), cam.second.cvR);
		cv::eigen2cv(Eigen::Vector3f(ts.find(cam.first)->second.cast<float>()), cam.second.cvT);
		cam.second.CV2Eigen();
	}
}


void Calibrator::EvaluateExternal()
{
	// evaluate
	for (const auto& corners : seqCorners) {
		std::vector<float> error;
		for (int cornerId = 0; cornerId < patternSize.width * patternSize.height; cornerId++) {
			const int row = cornerId / patternSize.width;
			const int col = cornerId % patternSize.width;
			Triangulator triangulator;
			std::vector<Eigen::Matrix<float, 3, 4>> projs;
			std::vector<Eigen::Vector3f> points;

			for (const auto& iter : corners.second) {
				for (int tarIdx = 0; tarIdx < iter.second.ids.size(); tarIdx++) {
					if (iter.second.ids[tarIdx] == cornerId) {
						projs.emplace_back(cams.find(iter.first)->second.eiProj);
						points.emplace_back(Eigen::Vector3f(iter.second.points[tarIdx].x, iter.second.points[tarIdx].y, 1.f));
					}
				}
			}

			if (projs.size() >= 2) {
				triangulator.projs = Eigen::Map<Eigen::Matrix3Xf>(projs.data()->data(), 3, 4 * projs.size());
				triangulator.points = Eigen::Map<Eigen::Matrix3Xf>(points.data()->data(), 3, points.size());
				triangulator.Solve();

				for (int j = 0; j < projs.size(); j++) {
					Eigen::Vector2f p2dEst = (projs[j] * triangulator.pos.homogeneous()).hnormalized();
					error.emplace_back((p2dEst - points[j].head(2)).norm());
				}
			}
		}
		if(!error.empty())
			std::cout << "frame: " << corners.first << " error: " << std::accumulate(error.begin(), error.end(), 0.f) / error.size() << std::endl;
	}
}



void Calibrator::CalibExternal()
{
	InitExternal();
	EvaluateExternal();
	BundleExternal();
	EvaluateExternal();
}


void Calibrator::SaveCache(const std::string& path) {
	std::ofstream ofs(path);
	ofs << patternSize.width << " " << patternSize.height << " " << squareLen << std::endl;
	ofs << cams.size() << std::endl;
	for (const auto& cam : cams)
		ofs << cam.first << " " << cam.second.imgSize.width << " " << cam.second.imgSize.height << std::endl;

	ofs << seqCorners.size() << std::endl;
	for (const auto& corners : seqCorners) {
		ofs << corners.first << std::endl << corners.second.size() << std::endl;
		for (const auto& iter : corners.second) {
			ofs << iter.first << std::endl << iter.second.pointsAligned << std::endl;
		}
	}

}


void Calibrator::LoadCache(const std::string& path) {
	std::ifstream ifs(path);
	cv::Size _patternSize; 
	float _squareLen;
	ifs >> _patternSize.width >> _patternSize.height >> _squareLen;
	assert(_patternSize == patternSize && _squareLen == squareLen);

	int snSize;
	ifs >> snSize;
	for (int i = 0; i < snSize; i++) {
		std::string sn;
		cv::Size imgSize;
		ifs >> sn >> imgSize.width >> imgSize.height;
		auto camIter = cams.find(sn);
		if (camIter == cams.end()) {
			Camera cam;
			cam.imgSize = imgSize;
			cams.insert(std::make_pair(sn, cam)).first;
		}
		else
			assert(camIter->second.imgSize == imgSize);
	}

	int frameSize;
	ifs >> frameSize;
	for (int i = 0; i < frameSize; i++) {
		int frameIdx, camSize;
		ifs >> frameIdx >> camSize;
		std::map<std::string, CornerPack> corners;
		for (int camIdx = 0; camIdx < camSize; camIdx++) {
			std::string camStr;
			ifs >> camStr;
			CornerPack pack;
			pack.pointsAligned.setZero(3, patternSize.width * patternSize.height);
			for (int row = 0; row < pack.pointsAligned.rows(); row++) {
				for (int col = 0; col < pack.pointsAligned.cols(); col++) {
					ifs >> pack.pointsAligned(row, col);
					if (row == 2 && pack.pointsAligned(row, col) > FLT_EPSILON) {
						pack.ids.emplace_back(col);
						pack.points.emplace_back(cv::Point2f(pack.pointsAligned(0, col), pack.pointsAligned(1, col)));
					}
				}
			}
			corners.insert(std::make_pair(camStr, pack));
		}
		seqCorners.insert(std::make_pair(frameIdx, corners));
	}
}




//
//struct ReprojCostFunctor
//{
//	ReprojCostFunctor(const Eigen::Matrix3d& _K, const Eigen::Vector2d& _p2d) {
//		m_K = _K;
//		m_p2d = _p2d;
//	}
//
//	template<typename T>
//	bool operator()(const T* const _r, const T* const _t, const T* _p3d, T* residuals) const {
//		const Eigen::Map<const Eigen::Matrix<T, 3, 1>> r(_r);
//		const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(_t);
//		const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p3d(_p3d);
//
//		const T theta = r.norm();
//		Eigen::Matrix<T, 3, 3> R;
//		if (theta < T(DBL_EPSILON))
//			R = Eigen::Matrix3d::Identity().cast<T>();
//		else
//			R = Eigen::AngleAxis<T>(theta, r / theta).matrix();
//
//		Eigen::Matrix<T, 3, 1> p = m_K.cast<T>() * (R * p3d + t);
//		Eigen::Matrix<T, 2, 1> uv(p[0] / p[2], p[1] / p[2]);
//		residuals[0] = (uv - m_p2d.cast<T>()).squaredNorm();
//		return true;
//	}
//
//private:
//	Eigen::Matrix3d m_K;
//	Eigen::Vector2d m_p2d;
//};




//
//void Calibrator::BundleExternal()
//{
//	std::map<std::string, Eigen::Vector3d> rs;
//	std::map<std::string, Eigen::Vector3d> ts;
//	Eigen::Matrix3Xd rPlanes = Eigen::Matrix3Xd::Zero(3, seqCorners.size());
//	Eigen::Matrix3Xd tPlanes = Eigen::Matrix3Xd::Zero(3, seqCorners.size());
//
//	// init plane RT
//	for (const auto& allCorners : seqCorners) {
//		cv::aruco::estimatePoseSingleMarkers()
//
//			Eigen::Matrix3f tri;
//		Triangulator triangulator;
//		std::vector<Eigen::Matrix<float, 3, 4>> projs;
//		for (const auto& iter : allCornersVec[idx])
//			projs.emplace_back(cams.find(iter.first)->second.eiProj);
//		triangulator.projs = Eigen::Map<Eigen::Matrix3Xf>(projs.data()->data(), 3, 4 * projs.size());
//
//		for (int triIdx = 0; triIdx < 3; triIdx++) {
//			std::vector<Eigen::Vector3f> points;
//			const int pIdx = triIdx == 0 ? 0 : triIdx == 1 ? patternSize.width : 1;
//			for (const auto& iter : allCornersVec[idx])
//				points.emplace_back(Eigen::Vector3f(iter.second[pIdx].x, iter.second[pIdx].y, 1.f));
//			triangulator.points = Eigen::Map<Eigen::Matrix3Xf>(points.data()->data(), 3, points.size());
//			triangulator.Solve();
//			tri.col(triIdx) = triangulator.pos;
//		}
//		// set T
//		tPlanes.col(idx) = tri.col(0).cast<double>();
//		// set R
//		Eigen::Matrix3f rot;
//		rot.col(0) = (tri.col(1) - tri.col(0)).normalized();
//		rot.col(1) = (tri.col(2) - tri.col(0)).normalized();
//		rot.col(2) = rot.col(0).cross(rot.col(1));
//		Eigen::AngleAxisd angleAxis(rot.cast<double>());
//		rPlanes.col(idx) = angleAxis.angle() * angleAxis.axis();
//	}
//
//	// warp to anchor frame RT
//	assert(anchorFrame < allCornersVec.size() && anchorFrame >= 0);
//	const Eigen::Matrix3d anchorRot = MathUtil::Rodrigues<double>(rPlanes.col(anchorFrame));
//	const Eigen::Vector3d anchorT = tPlanes.col(anchorFrame);
//	for (int idx = 0; idx < allCornersVec.size(); idx++) {
//		Eigen::Matrix3d rot = MathUtil::Rodrigues<double>(rPlanes.col(idx));
//
//		// set T
//		tPlanes.col(idx) = anchorRot.transpose() * (tPlanes.col(idx) - anchorT);
//
//		// set R
//		Eigen::AngleAxisd angleAxis(anchorRot.transpose() * rot);
//		rPlanes.col(idx) = (angleAxis.angle() * angleAxis.axis());
//	}
//
//	for (const auto& cam : cams) {
//		Eigen::Matrix3d camRot = cam.second.eiR.cast<double>();
//		Eigen::Vector3d camT = cam.second.eiT.cast<double>();
//
//		// set T
//		ts.insert(std::make_pair(cam.first, camT + camRot * anchorT));
//
//		// set R
//		Eigen::AngleAxisd angleAxis(camRot * anchorRot);
//		rs.insert(std::make_pair(cam.first, angleAxis.axis() * angleAxis.angle()));
//	}
//
//	ceres::Problem problem;
//	for (int frameIdx = 0; frameIdx < allCornersVec.size(); frameIdx++) {
//		for (auto&& iter : allCornersVec[frameIdx]) {
//			const Camera& cam = cams.find(iter.first)->second;
//			double* _r = rs.find(iter.first)->second.data();
//			double* _t = ts.find(iter.first)->second.data();
//			for (int pIdx = 0; pIdx < iter.second.size(); pIdx++) {
//				const Eigen::Vector2d p2d(iter.second[pIdx].x, iter.second[pIdx].y);
//				const Eigen::Vector2d pPlane(float(pIdx / patternSize.width) * squareLen, float(pIdx % patternSize.width) * squareLen);
//				ceres::CostFunction* func = new ceres::AutoDiffCostFunction<PlaneReprojCostFunctor, 1, 3, 3, 3, 3>(
//					new PlaneReprojCostFunctor(cam.eiK.cast<double>(), p2d, pPlane, frameIdx == anchorFrame));
//				problem.AddResidualBlock(func, NULL, _r, _t, rPlanes.col(frameIdx).data(), tPlanes.col(frameIdx).data());
//			}
//		}
//	}
//
//	ceres::Solver::Options options;
//	options.linear_solver_type = ceres::DENSE_QR;
//	options.minimizer_type = ceres::LINE_SEARCH;
//	options.max_num_iterations = 5000;
//	options.minimizer_progress_to_stdout = true;
//	ceres::Solver::Summary summary;
//	ceres::Solve(options, &problem, &summary);
//
//	// update
//	for (auto&& cam : cams) {
//		cv::eigen2cv(MathUtil::Rodrigues(Eigen::Vector3f(rs.find(cam.first)->second.cast<float>())), cam.second.cvR);
//		cv::eigen2cv(Eigen::Vector3f(ts.find(cam.first)->second.cast<float>()), cam.second.cvT);
//		cam.second.CV2Eigen();
//	}
//
//	std::cout << summary.BriefReport() << std::endl;
//}
//
