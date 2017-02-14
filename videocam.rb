require 'ropencv'
include OpenCV

video_file = cv::VideoCapture.new( 0)

id = 0

cv::namedWindow("edges",1)
frame = cv::Mat.new
detector = cv::FeatureDetector::create("SURF")
keypoints = Std::Vector.new(cv::KeyPoint)

while true do
	video_file.read( frame)
	detector.detect( frame, keypoints)

#puts "found #{keypoints.size} keypoints"
#puts "first keypoint is at #{keypoints[0].pt.x}/#{keypoints[0].pt.y}"

	cv::draw_keypoints(frame,keypoints,
						frame)
	cv::imshow("edges", frame)
	cv::wait_key( 1)
end

