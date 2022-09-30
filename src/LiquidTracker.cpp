#include <uima/api.hpp>

#include <pcl/point_types.h>
#include <robosherlock/types/all_types.h>
//RS
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include "robosherlock/DrawingAnnotator.h"
#include "robosherlock/io/TFListenerProxy.h"
#include <opencv2/optflow.hpp>

using namespace uima;


class LiquidTracker : public DrawingAnnotator {

    struct Gripper
    {
        tf::Transform transform;
        std::string reference_frame;
        std::string name, type;
        double y_dimension, z_dimension, x_dimension;
    };
private:
    float test_param;
    cv::Mat image_;
    cv::Mat lastImage, oldGray, frameGray;
    std::vector<cv::Point2f> p0, p1, p3;

    sensor_msgs::CameraInfo camera_info_;
    rs::TFListenerProxy listener_;


public:

    LiquidTracker(): DrawingAnnotator(__func__)
    {}

    TyErrorId initialize(AnnotatorContext &ctx) {
        outInfo("initialize");
        ctx.extractValue("test_param", test_param);
        return UIMA_ERR_NONE;
    }

    TyErrorId destroy() {
        outInfo("destroy");
        return UIMA_ERR_NONE;
    }

    TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec) {
        outInfo("process start");
        rs::StopWatch clock;
        rs::SceneCas cas(tcas);
        rs::Scene scene = cas.getScene();
        cas.get(VIEW_COLOR_IMAGE_HD, image_);
        cas.get(VIEW_CAMERA_INFO_HD, camera_info_);

        applyRegion();
        denseFarneback(image_);

        return UIMA_ERR_NONE;
    }

    void drawImageWithLock(cv::Mat &disp) {
        disp = image_.clone();
    }

    void fillVisualizerWithLock(pcl::visualization::PCLVisualizer& visualizer, const bool firstRun) {
    }

private:

    void applyRegion() {

        tf::StampedTransform gripperToHead;
        cv::Point2f gripper;

        if (rs::TFListenerProxy::listener->frameExists("r_gripper_tool_frame")) {
            listener_.listener->waitForTransform(camera_info_.header.frame_id, "r_gripper_tool_frame",
                                                 ros::Time(0),ros::Duration(2.0));
            listener_.listener->lookupTransform(camera_info_.header.frame_id, "r_gripper_tool_frame",
                                                ros::Time(0),gripperToHead);

            //Pose from gripper
            cv::Point3f gripperPose = cv::Point3f((float) gripperToHead.getOrigin().getX(), (float) gripperToHead.getOrigin().getY(), (float) gripperToHead.getOrigin().getZ());
            std::vector<cv::Point3f> roiPoints;
            roiPoints.push_back(gripperPose);

            std::vector<cv::Point2f> pointsImage;

            //cameraMatrix
            cv::Mat cameraMatrix = cv::Mat(3, 3, CV_64F);
            cameraMatrix.at<double>(0, 0) = camera_info_.K[0];
            cameraMatrix.at<double>(0, 1) = camera_info_.K[1];
            cameraMatrix.at<double>(0, 2) = camera_info_.K[2];
            cameraMatrix.at<double>(1, 0) = camera_info_.K[3];
            cameraMatrix.at<double>(1, 1) = camera_info_.K[4];
            cameraMatrix.at<double>(1, 2) = camera_info_.K[5];
            cameraMatrix.at<double>(2, 0) = camera_info_.K[6];
            cameraMatrix.at<double>(2, 1) = camera_info_.K[7];
            cameraMatrix.at<double>(2, 2) = camera_info_.K[8];

            //empty matrix for rvec and tvec
            cv::Mat vec(3, 1, cv::DataType<double>::type); // Rotation vector
            vec.at<double>(0) = 0.0;
            vec.at<double>(1) = 0.0;
            vec.at<double>(2) = 0.0;

            cv::projectPoints(roiPoints, vec, vec,
                              cameraMatrix, camera_info_.D, pointsImage);

            gripper = pointsImage[0];

            cv::Rect roi((int) gripper.x - 100, (int) gripper.y + 250, 200, 300);

            image_ = image_(roi);

        } else {
            outInfo("gripper frame not found");
        }


    }

    void sparseLucasKanade(cv::Mat image) {

        std::vector<cv::Point2f> good_new;

        if (lastImage.empty()) {
            outInfo("first frame");
            lastImage = image;
            cv::cvtColor(lastImage, oldGray, cv::COLOR_RGB2GRAY);
            cv::goodFeaturesToTrack(oldGray, p0, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);

        } else {
            cv::cvtColor(image, frameGray, cv::COLOR_RGB2GRAY);
            cv::goodFeaturesToTrack(oldGray, p3, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
            for (uint i = 0; i < p3.size(); i++) {
                p0.push_back(p3[i]);
            }
            std::vector<uchar> status;
            std::vector<float> err;
            cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
            calcOpticalFlowPyrLK(oldGray, frameGray, p0, p1, status, err, cv::Size(15, 15), 1, criteria);
            for (uint i = 0; i < p1.size(); i++) {
                // Select good points
                if (status[i] == 1) {
                    good_new.push_back(p1[i]);
                }
            }
            outInfo(p0.size());
            p0 = good_new;
            oldGray = frameGray.clone();
        }
#pragma omp parallel for
        for (auto &i: p0) {
            for (int j = round(i.x) - 2; j < round(i.x) + 2; j++) {
                for (int k = round(i.y) - 2; k < round(i.y) + 2; k++) {
                    image_.at<cv::Vec3b>(round(k), round(j)) = cv::Vec3b(255, 0, 255);
                }
            }
        }
    }

    void denseLucasKanade(cv::Mat image) {

        cv::cvtColor(image_, frameGray, cv::COLOR_RGB2GRAY);
        if (lastImage.empty()) {
            outInfo("first frame");
        } else {
            cv::Mat flow(lastImage.size(), CV_32FC2);
            cv::optflow::calcOpticalFlowSparseToDense(oldGray, frameGray, flow, 8, 128, 0.05f, false, 500.0, 1.5f);

            /*cv::Mat flow_parts[2];
            split(flow, flow_parts);
            cv::Mat magnitude, angle, magn_norm;
            cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
            normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
            angle *= ((1.f / 360.f) * (180.f / 255.f));
            //build hsv image
            cv::Mat _hsv[3], hsv, hsv8, bgr;
            _hsv[0] = angle;
            _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
            _hsv[2] = magn_norm;
            merge(_hsv, 3, hsv);
            hsv.convertTo(hsv8, CV_8U, 255.0);
            cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
            image_ = bgr.clone();*/
            outInfo(isLiquidFlow(flow));
        }
        lastImage = image_.clone();
        oldGray = frameGray.clone();
    }

    void denseFarneback(cv::Mat image) {

        cv::cvtColor(image_, frameGray, cv::COLOR_RGB2GRAY);
        if (lastImage.empty()) {
            outInfo("first frame");
        } else {
            cv::Mat flow(lastImage.size(), CV_32FC2);
            cv::calcOpticalFlowFarneback(oldGray, frameGray, flow, 0.75, 15, 30, 10, 7, 1.6, 0);

            cv::Mat flow_parts[2];
            split(flow, flow_parts);
            cv::Mat magnitude, angle, magn_norm;
            cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
            normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
            angle *= ((1.f / 360.f) * (180.f / 255.f));
            //build hsv image
            cv::Mat _hsv[3], hsv, hsv8, bgr;
            _hsv[0] = angle;
            _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
            _hsv[2] = magn_norm;
            merge(_hsv, 3, hsv);
            hsv.convertTo(hsv8, CV_8U, 255.0);
            cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
            image_ = bgr.clone();
            if (isLiquidFlow(flow) == 1) {
                outInfo("FFFFFFFFFFFFFFFFFFLLLLLLLLLLLLLLLOOOOOOOOOOOWWWWWWWWWWWWWWWWWWWW");
            } else {
                outInfo("no flow");
            }
        }
        lastImage = image_.clone();
        oldGray = frameGray.clone();
    }

    void denseRlof(cv::Mat image) {

        if (lastImage.empty()) {
            outInfo("first frame");
        } else {
            cv::Mat flow(lastImage.size(), CV_32FC2);
            cv::optflow::calcOpticalFlowDenseRLOF(lastImage, image_, flow, cv::Ptr<cv::optflow::RLOFOpticalFlowParameter>(),
                    0, cv::Size(2, 6),
                    cv::optflow::InterpolationType::INTERP_EPIC, 128, 0.05f, 100.0f,
                    15,100, true, 500.0f, 1.5f, false);

            cv::Mat flow_parts[2];
            split(flow, flow_parts);
            cv::Mat magnitude, angle, magn_norm;
            cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
            normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
            angle *= ((1.f / 360.f) * (180.f / 255.f));
            //build hsv image
            cv::Mat _hsv[3], hsv, hsv8, bgr;
            _hsv[0] = angle;
            _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
            _hsv[2] = magn_norm;
            merge(_hsv, 3, hsv);
            hsv.convertTo(hsv8, CV_8U, 255.0);
            cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
            image_ = bgr.clone();
        }
        lastImage = image_.clone();
    }

    bool isLiquidFlow(cv::Mat flow) {
        cv::Mat flow_parts[2];
        cv::split(flow, flow_parts);
        int flowCount;

#pragma omp parallel for
        for (int x = 0; x <= flow.rows; x++) {
            for (int y = 0; y <= flow.cols; y++) {
                if ((flow_parts[1].at<float>(x, y) >= 0.5f || flow_parts[1].at<float>(x, y) <= -0.5f) && flow_parts[0].at<float>(x, y) >= 0.0f) {
                    flowCount++;
                } else {
                    //image_.at<cv::Vec3b>(x, y) = cv::Vec3b(0, 0, 0);
                }

            }
        }
        outInfo(flowCount);
        if (flowCount > 36000) {
            return true;
        } else {
            return false;
        }
    }




};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(LiquidTracker)