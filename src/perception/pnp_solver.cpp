#include "rm_aim/perception/pnp_solver.hpp"

#include <cmath>
#include <vector>

#ifdef RM_AIM_WITH_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#endif

namespace rm_aim::perception {

PnpSolver::PnpSolver(const CameraConfig& cam_cfg) : cfg_(cam_cfg) {}

bool PnpSolver::solve(Armor& armor) const {
#ifdef RM_AIM_WITH_OPENCV
    const double hw = armor.is_large ? kLargeArmorHalfW : kSmallArmorHalfW;
    const double hh = kArmorHalfH;

    // 装甲板局部坐标系下的 3D 四角点：左上、右上、右下、左下（Z = 0 共面）
    const std::vector<cv::Point3f> obj_pts = {
        {static_cast<float>(-hw), static_cast<float>(-hh), 0.0f},
        {static_cast<float>(hw), static_cast<float>(-hh), 0.0f},
        {static_cast<float>(hw), static_cast<float>(hh), 0.0f},
        {static_cast<float>(-hw), static_cast<float>(hh), 0.0f},
    };

    const std::vector<cv::Point2f> img_pts = {
        {armor.corners[0].x, armor.corners[0].y},
        {armor.corners[1].x, armor.corners[1].y},
        {armor.corners[2].x, armor.corners[2].y},
        {armor.corners[3].x, armor.corners[3].y},
    };

    const cv::Mat K = (cv::Mat_<double>(3, 3) <<
        cfg_.camera_matrix[0], cfg_.camera_matrix[1], cfg_.camera_matrix[2],
        cfg_.camera_matrix[3], cfg_.camera_matrix[4], cfg_.camera_matrix[5],
        cfg_.camera_matrix[6], cfg_.camera_matrix[7], cfg_.camera_matrix[8]);

    const cv::Mat D = (cv::Mat_<double>(1, 5) <<
        cfg_.distort_coeffs[0], cfg_.distort_coeffs[1], cfg_.distort_coeffs[2],
        cfg_.distort_coeffs[3], cfg_.distort_coeffs[4]);

    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    cv::Mat reproj_errs;
    const int n_sols = cv::solvePnPGeneric(
        obj_pts, img_pts, K, D, rvecs, tvecs, false, cv::SOLVEPNP_IPPE,
        cv::noArray(), cv::noArray(), reproj_errs);

    if (n_sols <= 0) return false;

    // 选取满足正向深度（Z > 0.1m）且重投影误差最小的解
    int best = -1;
    double best_err = 1e18;
    for (int i = 0; i < n_sols; ++i) {
        const double z = tvecs[static_cast<size_t>(i)].at<double>(2);
        if (!std::isfinite(z) || z < 0.1) continue;
        const double err = reproj_errs.empty() ? 0.0 : reproj_errs.at<double>(i);
        if (err < best_err) {
            best_err = err;
            best = i;
        }
    }
    if (best < 0) return false;

    const cv::Mat& tvec = tvecs[static_cast<size_t>(best)];
    const cv::Mat& rvec = rvecs[static_cast<size_t>(best)];

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    armor.position_cam = {
        static_cast<float>(tvec.at<double>(0)),
        static_cast<float>(tvec.at<double>(1)),
        static_cast<float>(tvec.at<double>(2)),
    };
    // 装甲板表面法向量在相机坐标系下为 R * [0, 0, 1]^T（即 R 的第 2 列）
    const double nx = R.at<double>(0, 2);
    const double nz = R.at<double>(2, 2);
    armor.yaw_cam = static_cast<float>(std::atan2(nx, -nz));
    return true;
#else
    (void)armor;
    return false;
#endif
}

#ifdef RM_AIM_WITH_EIGEN

double PnpSolver::optimize_yaw_1d(const Armor& armor, const Eigen::Vector3d& pos_world,
                                  const Eigen::Matrix3d& R_world2cam,
                                  const Eigen::Vector3d& t_world2cam,
                                  double armor_pitch_rad) const {
    const double hw = armor.is_large ? kLargeArmorHalfW : kSmallArmorHalfW;
    const double hh = kArmorHalfH;
    const double fx = cfg_.camera_matrix[0];
    const double cx = cfg_.camera_matrix[2];
    const double fy = cfg_.camera_matrix[4];
    const double cy = cfg_.camera_matrix[5];

    // 装甲板在自身坐标系下的 4 个角点（局部系：X 沿外法线、Y 向左、Z 向上）
    // 对应图像顺序：左上(+Y,+Z)、右上(-Y,+Z)、右下(-Y,-Z)、左下(+Y,-Z)
    const double cp = std::cos(armor_pitch_rad);
    const double sp = std::sin(armor_pitch_rad);
    const std::array<Eigen::Vector3d, 4> local_pts = {
        Eigen::Vector3d{-hh * sp,  hw,  hh * cp},
        Eigen::Vector3d{-hh * sp, -hw,  hh * cp},
        Eigen::Vector3d{ hh * sp, -hw, -hh * cp},
        Eigen::Vector3d{ hh * sp,  hw, -hh * cp},
    };

    auto reproj_cost = [&](double yaw_world) -> double {
        const double cyaw = std::cos(yaw_world);
        const double syaw = std::sin(yaw_world);
        Eigen::Matrix3d R_a2w;
        R_a2w << cyaw, -syaw, 0.0,
                 syaw,  cyaw, 0.0,
                  0.0,   0.0, 1.0;

        double err = 0.0;
        for (size_t i = 0; i < 4; ++i) {
            const Eigen::Vector3d pw = pos_world + R_a2w * local_pts[i];
            const Eigen::Vector3d pc = R_world2cam * pw + t_world2cam;
            if (pc.z() <= 0.05) return 1e12;
            const double u = fx * (pc.x() / pc.z()) + cx;
            const double v = fy * (pc.y() / pc.z()) + cy;
            const double du = u - static_cast<double>(armor.corners[i].x);
            const double dv = v - static_cast<double>(armor.corners[i].y);
            err += du * du + dv * dv;
        }
        return err;
    };

    // 搜索区间以“装甲板正对相机”方向为中心，向两侧各展 75 度
    const double view_yaw = std::atan2(-pos_world.y(), -pos_world.x());
    constexpr double kHalfSpan = 75.0 * 3.14159265358979323846 / 180.0;
    double a = view_yaw - kHalfSpan;
    double b = view_yaw + kHalfSpan;

    // 黄金分割搜索（18 次迭代区间缩小至 150° * 0.618^18 < 0.03°）
    constexpr double kInvPhi = 0.6180339887498949;
    double c = b - kInvPhi * (b - a);
    double d = a + kInvPhi * (b - a);
    double fc = reproj_cost(c);
    double fd = reproj_cost(d);

    for (int iter = 0; iter < 18; ++iter) {
        if (fc < fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - kInvPhi * (b - a);
            fc = reproj_cost(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + kInvPhi * (b - a);
            fd = reproj_cost(d);
        }
    }

    const double best = 0.5 * (a + b);
    // 归一化到 (-pi, pi]
    constexpr double kPi = 3.14159265358979323846;
    double res = std::fmod(best + kPi, 2.0 * kPi);
    if (res <= 0.0) res += 2.0 * kPi;
    return res - kPi;
}

#endif  // RM_AIM_WITH_EIGEN

}  // namespace rm_aim::perception
