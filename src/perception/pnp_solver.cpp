#include "rm_aim/perception/pnp_solver.hpp"

#include <cmath>
#include <vector>

#ifdef RM_AIM_WITH_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#endif

#ifdef RM_AIM_WITH_EIGEN
#include <Eigen/Dense>
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
        double err = 0.0;
        if (!reproj_errs.empty()) {
            err = (reproj_errs.depth() == CV_64F)
                      ? reproj_errs.at<double>(i)
                      : static_cast<double>(reproj_errs.at<float>(i));
        }
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

void PnpSolver::refine_pose_4dof(const Armor& armor, Eigen::Vector3d& pos_world, double& yaw_world,
                                 const Eigen::Matrix3d& R_world2cam,
                                 const Eigen::Vector3d& t_world2cam,
                                 double armor_pitch_rad) const {
    const double hw = armor.is_large ? kLargeArmorHalfW : kSmallArmorHalfW;
    const double hh = kArmorHalfH;

    // 步骤 1：将 4 个检测角点去畸变到无畸变归一化相机平面 (x/z, y/z)
    std::array<Eigen::Vector2d, 4> obs_norm{};
#ifdef RM_AIM_WITH_OPENCV
    const cv::Mat K = (cv::Mat_<double>(3, 3) <<
        cfg_.camera_matrix[0], cfg_.camera_matrix[1], cfg_.camera_matrix[2],
        cfg_.camera_matrix[3], cfg_.camera_matrix[4], cfg_.camera_matrix[5],
        cfg_.camera_matrix[6], cfg_.camera_matrix[7], cfg_.camera_matrix[8]);
    const cv::Mat D = (cv::Mat_<double>(1, 5) <<
        cfg_.distort_coeffs[0], cfg_.distort_coeffs[1], cfg_.distort_coeffs[2],
        cfg_.distort_coeffs[3], cfg_.distort_coeffs[4]);
    std::vector<cv::Point2f> raw_pts(4), undist_pts(4);
    for (size_t i = 0; i < 4; ++i) {
        raw_pts[i] = {armor.corners[i].x, armor.corners[i].y};
    }
    cv::undistortPoints(raw_pts, undist_pts, K, D);
    for (size_t i = 0; i < 4; ++i) {
        obs_norm[i] = {undist_pts[i].x, undist_pts[i].y};
    }
#else
    const double fx = cfg_.camera_matrix[0];
    const double cx = cfg_.camera_matrix[2];
    const double fy = cfg_.camera_matrix[4];
    const double cy = cfg_.camera_matrix[5];
    for (size_t i = 0; i < 4; ++i) {
        obs_norm[i] = {(armor.corners[i].x - cx) / fx, (armor.corners[i].y - cy) / fy};
    }
#endif

    // 步骤 2：先用 1D 黄金分割提供全局无翻转 yaw 初值
    yaw_world = optimize_yaw_1d(armor, pos_world, R_world2cam, t_world2cam, armor_pitch_rad);

    // 步骤 3：固定物理倾角 (pitch = ±15°, roll = 0°)，对 4 自由度 [x, y, z, yaw] 做阻尼 Gauss-Newton 精修
    const double cp = std::cos(armor_pitch_rad);
    const double sp = std::sin(armor_pitch_rad);
    const std::array<Eigen::Vector3d, 4> local_pts = {
        Eigen::Vector3d{-hh * sp,  hw,  hh * cp},
        Eigen::Vector3d{-hh * sp, -hw,  hh * cp},
        Eigen::Vector3d{ hh * sp, -hw, -hh * cp},
        Eigen::Vector3d{ hh * sp,  hw, -hh * cp},
    };

    Eigen::Vector3d p = pos_world;
    double yaw = yaw_world;
    double lambda = 1e-4;

    auto eval_residuals = [&](const Eigen::Vector3d& pv, double yv,
                              Eigen::Matrix<double, 8, 1>& r_out,
                              Eigen::Matrix<double, 8, 4>* J_out) -> bool {
        const double cyaw = std::cos(yv);
        const double syaw = std::sin(yv);
        Eigen::Matrix3d R_a2w;
        R_a2w << cyaw, -syaw, 0.0,
                 syaw,  cyaw, 0.0,
                  0.0,   0.0, 1.0;
        Eigen::Matrix3d dR_dyaw;
        dR_dyaw << -syaw, -cyaw, 0.0,
                    cyaw, -syaw, 0.0,
                     0.0,   0.0, 0.0;

        for (int i = 0; i < 4; ++i) {
            const Eigen::Vector3d& lp = local_pts[static_cast<size_t>(i)];
            const Eigen::Vector3d pw = pv + R_a2w * lp;
            const Eigen::Vector3d pc = R_world2cam * pw + t_world2cam;
            if (pc.z() <= 0.05) return false;

            const double inv_z = 1.0 / pc.z();
            const double nx = pc.x() * inv_z;
            const double ny = pc.y() * inv_z;
            r_out(2 * i) = nx - obs_norm[static_cast<size_t>(i)].x();
            r_out(2 * i + 1) = ny - obs_norm[static_cast<size_t>(i)].y();

            if (J_out != nullptr) {
                // 归一化平面投影对相机坐标 pc 的 2x3 雅可比
                Eigen::Matrix<double, 2, 3> J_proj;
                J_proj << inv_z, 0.0, -nx * inv_z,
                          0.0, inv_z, -ny * inv_z;
                // pc 对世界系中心位置 pv 的导数为 R_world2cam
                J_out->block<2, 3>(2 * i, 0) = J_proj * R_world2cam;
                // pc 对 yaw 的导数为 R_world2cam * dR_dyaw * lp
                J_out->block<2, 1>(2 * i, 3) = J_proj * (R_world2cam * (dR_dyaw * lp));
            }
        }
        return true;
    };

    Eigen::Matrix<double, 8, 1> r;
    Eigen::Matrix<double, 8, 4> J;
    if (!eval_residuals(p, yaw, r, &J)) return;
    double cost = r.squaredNorm();

    for (int iter = 0; iter < 6; ++iter) {
        const Eigen::Matrix4d H = J.transpose() * J + lambda * Eigen::Matrix4d::Identity();
        const Eigen::Vector4d g = J.transpose() * r;
        const Eigen::Vector4d delta = -H.ldlt().solve(g);
        if (!delta.allFinite()) break;

        // 限制单步更新幅度，保证物理稳健性
        const Eigen::Vector3d p_new = p + delta.head<3>();
        const double yaw_new = yaw + std::clamp(delta(3), -0.35, 0.35);

        Eigen::Matrix<double, 8, 1> r_new;
        if (eval_residuals(p_new, yaw_new, r_new, nullptr) && r_new.squaredNorm() < cost) {
            p = p_new;
            yaw = yaw_new;
            cost = r_new.squaredNorm();
            lambda = std::max(1e-6, lambda * 0.5);
            if (delta.norm() < 1e-5) break;
            eval_residuals(p, yaw, r, &J);
        } else {
            lambda = std::min(1e2, lambda * 4.0);
        }
    }

    // 校验精修结果相对初值未发生异常跳变后再写回
    if ((p - pos_world).norm() < 0.5) {
        pos_world = p;
    }
    constexpr double kPi = 3.14159265358979323846;
    double res_yaw = std::fmod(yaw + kPi, 2.0 * kPi);
    if (res_yaw <= 0.0) res_yaw += 2.0 * kPi;
    yaw_world = res_yaw - kPi;
}

#endif  // RM_AIM_WITH_EIGEN

}  // namespace rm_aim::perception
