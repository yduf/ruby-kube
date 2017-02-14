# http://www.ropencv.aduda.eu/examples/#loading
require 'ropencv'
include OpenCV

# detect cube according to XXX
class RubikFinder

	def initialize()
	end

	def analyze_frame( frame)
		# create  some buffer
		@sg  = cv::Mat.new( frame.size, cv::CV_8UC3)
		# @sgc = cv::Mat.new( frame.size, cv::CV_8UC3)

		@grey = cv::Mat.new #( frame.size, cv::CV_8U)

		# cv::resize( frame, @sg)
		cv::resize( frame, @sg, @sg.size(), 0, 0, interpolation = 1);
        # cv::copy( @sg, @sgc)

        cv::cvtColor( @sg, @grey, CV_RGB2GRAY)

        detect( @grey)
	end

	# detect lines
	# return an array of [rho, theta]
	def hough( edge)
        # http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html#houghlinesp
        lines = cv::Mat.new

        pi = 3.1415926
        cv::HoughLines( edge, lines, 1, pi/180.0, 10, 0, 0 )
        puts "rows / cols #{lines.rows}, #{lines.cols}"
        lines = lines.to_a

        # puts "lines #{lines[0,1]}"

        a = []
        b = []
        lines[0].each_with_index do |l, i|
        	#puts l

        	if (i % 2) == 0
        		a << b if b.size > 0
        		b = []
        	end

        	b << l
        end

        a
	end

	# draw line from hough, onto image
	def drawPolar( lines, dst, count = 50)
		lines.each_with_index do |l,i|
        	# puts l.to_s
        	rho   = l[0]
        	theta = l[1]
        	a = Math::cos(theta)
        	b = Math::sin(theta)
        	x0 = a*rho
        	y0 = b*rho

			pt1 = cv::Point.new( (x0 + 2000*(-b)),
                  				 (y0 + 2000*(a)))
        	pt2 = cv::Point.new( (x0 - 2000*(-b)),
                  				 (y0 - 2000*(a)))

 			# pt1 = cv::Point.new( 0, 0)
    #     	pt2 = cv::Point.new( 100, 100)
       	

        	#puts [ x0, y0, pt1, pt2 ].to_s
        	cv::line( dst, pt1, pt2,
            			   cv::Scalar.new(0,0,255), 3, 8 )


       		# cv::line( dst2, pt1, pt2,
         #    			   cv::Scalar.new(255), 3, 8 )
        	break if i > count
        end

	end

	def detect( grey)  
      	# edge detection
      	# http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/laplace_operator/laplace_operator.html?highlight=laplacian       
        i = 1
  		dst2 = cv::Mat.new
        cv::GaussianBlur( grey, dst2, cv::Size.new( 0, 0 ), i, i)

        d = cv::Mat.new
        # cv::Laplacian( dst2, d, cv::CV_16S)
        ddepth = cv::CV_16S
        # cv::Laplacian( dst2, d, ddepth, kernel_size = 3, scale = 1, delta = 0 )
 
        # http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html#houghlinesp
        # http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
        lowThreshold = 80
        ratio = 3
        cv::Canny( dst2, dst2, lowThreshold, lowThreshold*ratio, kernel_size  = 3);

        # std hough
        lines = hough( dst2)
        drawPolar( lines, @sg)

        #cv::cmpS( d, 8, d2, cv.CV_CMP_GT)
        d2 = cv::Mat.new
        # cv::convertScaleAbs( d, d2 )

        # li = cv.HoughLines2(self.d2, cv.CreateMemStorage(), cv.CV_HOUGH_PROBABILISTIC, 1, 3.1415926 / 45, self.THR, 10, 5)
	    # vector<Vec4i> lines;
	 #    lines = cv::Mat.new # Std::Vector.new( cv::Vec4i)
	 #    cv::HoughLinesP( d2, lines, 1, 3.1415926/45, thr = 70, 10, 5 );

		# color_dst = cv::Mat.new( grey.size, cv::CV_8UC3)
		# l = lines
	 #    lines.reshape( 1, lines.cols).each_row do |l|
	 #    	puts [ l[0], l[1], l[2], l[3] ].to_s
	 #    	cv::line( @sg, cv::Point.new(l[0], l[1]),
  #           			   cv::Point.new(l[2], l[3]),
  #           			   cv::Scalar.new(0,0,255), 3, 8 );

	 #    end
        cv::imshow("cube", dst2)

  		cv::imshow("lines", @sg)

	end
end

filename = '/home/yves/Pictures/rubik/2017-02-13-182254.jpg'
img = cv.imread( filename)

RubikFinder.new.analyze_frame( img)
cv::wait_key(-1)
