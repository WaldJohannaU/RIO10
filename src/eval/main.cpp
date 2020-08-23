#include <iostream>
#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <ctime>

using MatrixMap = std::map<std::string, Eigen::Matrix4f, std::less<std::string>,
        Eigen::aligned_allocator<std::pair<const std::string, Eigen::Matrix4f>>>;

constexpr char frame_prefix[] = "frame-";
constexpr char frame_depth_suffix[] = ".rendered.depth.png";
constexpr double original_depth_width = 540.0f;
constexpr double original_depth_height = 960.0f;

class EvaluationServer {
    MatrixMap poses_GT;
    MatrixMap poses_predicted;

    struct Intrinsics {
        float cx;
        float cy;
        float fx;
        float fy;
    };
    std::map<std::string, Intrinsics> intrinsics_map;

    double DCRE(const Intrinsics& intrinsics, 
                const std::string& depth_image_file,
                const Eigen::Matrix4f& GT_camera2world, 
                const Eigen::Matrix4f& prediction_camera2world) {
        cv::Mat depth_image = cv::imread(depth_image_file, -1);
        if (depth_image.empty()) {
            std::cout << "couldn't find " << depth_image_file << std::endl;
            return -1;
        }
        const float diagonal = sqrt(depth_image.cols * depth_image.cols + depth_image.rows * depth_image.rows);
        if (depth_image.cols > depth_image.rows)
            return -1;

        double abs_DCRE = 0;
        unsigned int valid_points = 0;
        const float scale_factor_width = original_depth_width / depth_image.cols;
        const double c_x = intrinsics.cx / scale_factor_width;
        const double c_y = intrinsics.cy / scale_factor_width;
        const double f_x = intrinsics.fx / scale_factor_width;
        const double f_y = intrinsics.fy / scale_factor_width;

    #pragma omp parallel for
        for (int row = 0; row < depth_image.rows; row++) {
            for (int col = 0; col < depth_image.cols; col++) {
                unsigned short depth = depth_image.at<unsigned short>(row, col);
                if (depth != 0) {
                    const Eigen::Vector4f point4D_cam(((col - c_x) / f_x * depth) / 1000.0f, ((row - c_y) / f_y * depth) / 1000.0f, depth / 1000.0f, 1.0f);
                    const Eigen::Vector4f point4D_prediction = prediction_camera2world.inverse() * GT_camera2world * point4D_cam;
                    const Eigen::Vector2f iuv(point4D_prediction.x() / point4D_prediction.z() * f_x + c_x, point4D_prediction.y() / point4D_prediction.z() * f_y + c_y);
                    const Eigen::Vector2f DCR(iuv(1) - row, iuv(0) - col);
                    abs_DCRE += std::fmin((DCR.norm()) / diagonal, 1.0f);
                    valid_points++;
                }
            }
        }
        if (valid_points == 0)
            return -1;
        return abs_DCRE / valid_points;
    }

    bool IsValidMatrix(const Eigen::Matrix4f& matrix) const {
        const Eigen::Matrix4f rot_inverse = matrix.inverse();
        for (int y = 0; y < matrix.rows(); y++) {
            for (int x = 0; x < matrix.cols(); x++) {
                if (std::isinf(matrix(y, x)) || std::isinf(rot_inverse(y, x)) || 
                    std::isnan(matrix(y, x)) || std::isnan(rot_inverse(y, x)))
                    return false;
            }
        }
        return true;
    }
    
    const float QuaternionAngularError(const Eigen::Quaternionf& q1, const Eigen::Quaternionf& q2) const {
        const float d1 = std::fabs(q1.dot(q2));
        const float d2 = std::fmin(1.0f, std::fmax(-1.0f, d1));
        return 2 * acos(d2) * 180 / M_PI;
    }

    bool ComputeErrors(const std::string& file_path, const std::string& output_filename) {
        std::cout << "saving results in " << output_filename << std::endl;
        std::ofstream file;
        file.open(output_filename);
        for (const auto& gt: poses_GT) {
            const Eigen::Matrix4f& GT_pose = gt.second;
            const std::string identifier = gt.first;
            if (poses_predicted.find(identifier) != poses_predicted.end()) {
                const Eigen::Matrix4f predicted_pose = poses_predicted[identifier];
                if (IsValidMatrix(predicted_pose)) {
                    const std::string seq = identifier.substr(0, identifier.find("_"));
                    const std::string seq_identifier = identifier.substr(0, identifier.find("/"));
                    const std::string depth_path = file_path + "/" + seq + "/" + identifier + frame_depth_suffix;
                    const float trans_error = Eigen::Vector3f(GT_pose.block<3, 1>(0, 3) - predicted_pose.block<3,1>(0,3)).norm();
                    const Eigen::Quaternionf quaternion_pred(predicted_pose.block<3,3>(0,0));
                    const float rot_error = QuaternionAngularError(Eigen::Quaternionf(GT_pose.block<3,3>(0,0)), quaternion_pred);
                    const float DCRError = DCRE(intrinsics_map.at(seq_identifier), depth_path, GT_pose, predicted_pose);
                    file << identifier << " " << trans_error << " " << rot_error << " " << DCRError  << std::endl;
                }
            }
        }
        file.close();
        return true;
    }

    
    void ReadLine(MatrixMap& poses, const std::string& line) {
        std::string file_name;
        std::istringstream iss(line);
        Eigen::Vector3f translation(0,0,0);
        Eigen::Quaternionf rotation;
        iss >> file_name >> rotation.w() >> rotation.x() >> rotation.y() >> rotation.z()
            >> translation(0) >> translation(1) >> translation(2);
        Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
        matrix.block<3,3>(0,0) = Eigen::Matrix3f(rotation);
        matrix.block<3,1>(0,3) = translation;
        if (IsValidMatrix(matrix))
            poses[file_name] = matrix;
    }
    
    void LoadPoses(MatrixMap& poses, const std::string& path) {
        std::ifstream file(path);
        std::string line;
        if (file.is_open()) {
            while (std::getline(file, line))
                ReadLine(poses, line);
        } else std::cerr << "poses not found: " << path << std::endl;
    }

    void LoadIntrinsics(const std::string& path) {
        std::ifstream file(path);
        std::string line;
        if (file.is_open()) {
            while (std::getline(file, line)) {
                std::istringstream ss(line);
                std::string seq_str;
                Intrinsics intrinsics;
                ss >> seq_str >> intrinsics.fx >> intrinsics.fy >> intrinsics.cx >> intrinsics.cy;
                intrinsics_map[seq_str] = intrinsics;
            }
        } else std::cerr << "intrinsics not found: " << path << std::endl;
    }
    
public:
    void Run(const std::string& datapath, const std::string& prediction, 
        const std::string& output_filename) {
        LoadPoses(poses_GT, datapath+"/GT_RIO10.txt");
        LoadPoses(poses_predicted, prediction);
        std::cout << "#ground truth: " << poses_GT.size() << ", #predictions: " << poses_predicted.size() << std::endl;
        LoadIntrinsics(datapath +"/intrinsics.txt");
        ComputeErrors(datapath, output_filename);
    }
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "./eval <data_path> <prediction_file> <output_file>" << std::endl;
        return -1;
    }
    const std::string data_path{argv[1]};
    const std::string pred_path{argv[2]};
    const std::string output_file{argv[3]};

    const std::clock_t start = std::clock();
    EvaluationServer eval;
    eval.Run(data_path, pred_path, output_file);
    std::cout << "took: "<< (std::clock() - start) / (double) CLOCKS_PER_SEC << "seconds \n";
    return 0;
}