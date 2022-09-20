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
        //denseLucasKanade(image_);

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

        if (rs::TFListenerProxy::listener->frameExists("r_gripper_palm_link")) {
            listener_.listener->waitForTransform("r_gripper_palm_link", camera_info_.header.frame_id,
                                                 ros::Time(0),
                                                 ros::Duration(2.0));
            listener_.listener->lookupTransform("r_gripper_palm_link", camera_info_.header.frame_id, ros::Time(0),
                                                gripperToHead);

            cv::Point3f gripperPose = cv::Point3f((float) gripperToHead.getOrigin().getX(), (float) gripperToHead.getOrigin().getY(), (float) gripperToHead.getOrigin().getZ());
            std::vector<cv::Point3f> roiPoints;
            roiPoints.push_back(gripperPose);
            cv::Mat distortionCoefficients = cv::Mat(1, (int) camera_info_.D.size(), CV_64F);
            std::vector<cv::Point2f> pointsImage;
            cv::Mat cameraMatrix = cv::Mat(3, 3, CV_64F);
            cv::Mat vec(3, 1, cv::DataType<double>::type); // Rotation vector
            vec.at<double>(0) = 0.0;
            vec.at<double>(1) = 0.0;
            vec.at<double>(2) = 0.0;
            cv::projectPoints(roiPoints, vec, vec,
                              cameraMatrix, distortionCoefficients, pointsImage);


            gripper = pointsImage[0];
            //outInfo((float) gripperToHead.getOrigin().getX());
            //outInfo((float) gripperToHead.getOrigin().getY());
            //[1.735442140139042e+238, 2.347269349079779e+251, 8.238772645159547e-67;
            // 4.437500133906592e-38, 1.723720460929229e-47, 3.170958682177391e+180;
            // 2.440460595065171e-152, 2.316339904798213e-152, 1.278502225360474e-152]
//            [1.677918434315618e+243, 9.021338756902148e+217, 1.591352950536471e-76;
//            6.702880279567067e-177, 6.509443960381337e+252, 1.944135954238013e+233;
//            9.364104449322752e-76, 6.572753743897687e+16, 4.619647653228141e+281]


            outInfo(pointsImage[0].x);
            outInfo(pointsImage[0].y);
            for (int x = (int) gripper.x - 5; x < (int) gripper.x + 5; x++) {
                for (int y = (int) gripper.y - 5; y < (int) gripper.y + 5; y++) {
                        image_.at<cv::Vec3b>(x, y) = cv::Vec3b(255, 0, 255);
                }
            }
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
            cv::optflow::calcOpticalFlowSparseToDense(oldGray, frameGray, flow, 8, 128, 0.05f, true, 500.0, 1.5f);

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
        oldGray = frameGray.clone();
    }

    void denseFarneback(cv::Mat image) {

        cv::cvtColor(image_, frameGray, cv::COLOR_RGB2GRAY);
        if (lastImage.empty()) {
            outInfo("first frame");
        } else {
            cv::Mat flow(lastImage.size(), CV_32FC2);
            cv::calcOpticalFlowFarneback(oldGray, frameGray, flow, 0.5, 3, 15, 6, 5, 1.2, 0);

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
        oldGray = frameGray.clone();
    }

    void denseRlof(cv::Mat image) {

        if (lastImage.empty()) {
            outInfo("first frame");
        } else {
            cv::Mat flow(lastImage.size(), CV_32FC2);
            cv::optflow::calcOpticalFlowDenseRLOF(lastImage, image_, flow, cv::Ptr<cv::optflow::RLOFOpticalFlowParameter>(),
                    1.f, cv::Size(6, 6),
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



};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(LiquidTracker)