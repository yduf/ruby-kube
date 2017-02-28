# http://www.ropencv.aduda.eu/examples/#loading
require 'ropencv'
include OpenCV

require 'matrix'
require 'pp'

# distance between 2 points
def ptdst(p1, p2)
    Math::sqrt((p1[0] - p2[0]).abs2 + (p1[1] - p2[1]).abs2)
end

def avg(p1, p2)
    [ 0.5 * (p1[0] + p2[0]), 0.5 * (p2[1] + p2[1]) ]
end

# return the pts in correct order based on quadrants
def winded(p1, p2, p3, p4)
    avg = [ 0.25 * (p1[0] + p2[0] + p3[0] + p4[0]), 
    		0.25 * (p1[1] + p2[1] + p3[1] + p4[1])]
    ps = [p1, p2, p3, p4].map { |p| [ Math::atan2(p[1] - avg[1], p[0] - avg[0]), p] }
    ps.sort!.reverse!

    ps.map { |p| p[1] }
end

def cvAvg( img, roi)
	p1, p2 = roi
	roi = cv::Rect.new(	p1[0], p1[1], 
                    	p2[0] - p1[0], p2[1] - p1[1] )
	image_roi = img.block( roi )

	col = cv::mean( image_roi)
end

def cvLine( img, p1, p2, col, width = 3)
 	cv::line( img, cv::Point.new(p1[0], p1[1]),
     			   cv::Point.new(p2[0], p2[1]),
     			   cv::Scalar.new( col[0], col[1], col[2]), width, 8 )
end

def cvCircle( img, center, radius, col, width = 1)
	cv.circle( img, cv::Point.new( center[0], center[1]), radius, cv::Scalar.new(col[0], col[1], col[2]), width)
end

# detect cube according to XXX
class RubikFinder
	DECTS = 100						# ideal number of number of lines detections

	class Tracking
		attr_accessor 	:prevface,
						:pt,
						:lastpt,
						:tracking,
						:detected,
						:undetectednum,
						:succ

		attr_accessor 	:p0,
						:p1, :p2,
						:q1, :q2,
						:v1, :v2,
						:p3

		def initialize( )			
	        # stores the coordinates that make up the face. in order: p,p1,p3,p2 (i.e.) counterclockwise winding
	        @prevface = [ [0, 0], [5, 0], [0, 5]]
	        # @prevface = [[349.0, 46.0], [531, 52], [340, 237]]
	        @pt     = []
	        @lastpt = []

	        @tracking = false
	        @succ     = 0					# count success of recognizing face acros multiple match

	        @detected = false
	        @undetectednum = 100
	    end
	end # class Tracking


	def initialize( resize = 1)
		@resize_factor = resize		# resize down factor for image processing

		# automatic threshold
		@thr = 70                   # adaptative threshold for detecting such number of lines
		@lastdetected = 0

		@tracker = Tracking.new		# internal state for tracking

		# option
		@dodetection = true
	end

	# analyse one frame (with adaptive threshold from previous frame analysis)
	def analyze_frame( frame)
		# create  some buffer
		# cv::resize( frame, @sg)
 		@width, @height = [ frame.size.width, frame.size.height].map { |x| x / @resize_factor }
		resized = cv::Size.new( @width, @height )
		@sg  = cv::Mat.new( resized, cv::CV_8UC3)
		cv::resize( frame, @sg, @sg.size(), 0, 0, interpolation = 1);

		#
		@sgc  = @sg.clone

		@grey = cv::Mat.new #( frame.size, cv::CV_8U)
        cv::cvtColor( @sg, @grey, CV_RGB2GRAY)
            
        # detection mode
        if ! @tracker.tracking
	        res = detect( @grey) 
 			@tracker = init_tracker( res, @tracker )
 		end

 		# tracking mode
 		@tracker.tracking = false
        if  false && @tracker.tracking
        	# we are in tracking mode, we need to fill in pt[] array
            # calculate the pt array for drawing from features
            p  = @tracker.features[0]
            p1 = @tracker.features[1]
            p2 = @tracker.features[2]

            v1 = [ p1[0] - p[0], p1[1] - p[1]]
            v2 = [ p2[0] - p[0], p2[1] - p[1]]

            @tracker.pt = [[p[0] - v1[0] - v2[0],     p[1] - v1[1] - v2[1] ],
	                       [p[0] + 2 * v2[0] - v1[0], p[1] + 2 * v2[1] - v1[1]],
	                       [p[0] + 2 * v1[0] - v2[0], p[1] + 2 * v1[1] - v2[1]]]

            @tracker.prevface = [@tracker.pt[0], @tracker.pt[1], @tracker.pt[2]]

	        # tracking mode
	        @tracker = verify_still_tracking( @tracker)
 		end

 		# use pt[] array to do drawing
 		if ( @tracker.detected || @tracker.undetectednum < 1) && @dodetection
	        draw_circles_and_lines( @tracker)
	    end

	    @tracker.detected = false

	    cv::imshow("cube", @dst2)
  		cv::imshow("lines", @sg)
	end

	def draw_circles_and_lines( tracker)
        # undetectednum 'fills in' a few detection to make
        # things look smoother in case we fall out one frame
        # for some reason
        if !tracker.detected
            tracker.undetectednum += 1
            tracker.pt = tracker.lastpt
        end

        if tracker.detected
            tracker.undetectednum = 0
            tracker.lastpt = tracker.pt
        end

        # convert to HSV
        @hsv = cv::Mat.new
        cv::cvtColor( @sgc, @hsv, CV_RGB2HSV)

        channel = Std::Vector::Cv_Mat.new
        cv::split( @hsv, channel )
        @hue, @sat, @val = [ channel[0], channel[1], channel[2] ]

        pt_int = []
        tracker.pt.each do |p|
        	foo, bar = p
            pt_int << [ foo, bar]
        end

        # do the drawing. pt array should store p,p1,p2
        tracker.p3 = [	tracker.pt[2][0] + tracker.pt[1][0] - tracker.pt[0][0], 
        				tracker.pt[2][1] + tracker.pt[1][1] - tracker.pt[0][1] ]
        p2_int = [ tracker.p2[0], tracker.p2[1] ]
        p3_int = [ tracker.p3[0], tracker.p3[1] ]

        col = [0, 255, 0]
        cvLine( @sg, pt_int[0], pt_int[1], col, 2)
        cvLine( @sg, pt_int[1], p3_int,    col, 2)
        cvLine( @sg, p3_int,    pt_int[2], col, 2)
        cvLine( @sg, pt_int[2], pt_int[0], col, 2)

        # first sort the points so that 0 is BL 1 is UL and 2 is BR
        pt = winded( tracker.pt[0], tracker.pt[1], tracker.pt[2], tracker.p3)

        # find the coordinates of the 9 places we want to extract over
        tracker.v1 = [ pt[1][0] - pt[0][0], pt[1][1] - pt[0][1] ]
        tracker.v2 = [ pt[3][0] - pt[0][0], pt[3][1] - pt[0][1] ]
        tracker.p0 = [ pt[0][0], pt[0][1]]

        ep = []
        i = 1
        j = 5
        for k in 0..8
            ep << [ tracker.p0[0] + i * tracker.v1[0] / 6.0 + j * tracker.v2[0] / 6.0,
                    tracker.p0[1] + i * tracker.v1[1] / 6.0 + j * tracker.v2[1] / 6.0]
            i = i + 2
            if i == 7
                i = 1
                j = j - 2
            end
        end

        rad = ptdst(tracker.v1, [ 0.0, 0.0] ) / 6.0
        cs = []
        center_pixels = []
        hsvcs = []
        den = 2

        ep.each_with_index do |p, i|
        	# puts i
            if p[0] > rad && p[0] < @width - rad &&
               p[1] > rad && p[1] < @height - rad

                # valavg=val[int(p[1]-rad/3):int(p[1]+rad/3),int(p[0]-rad/3):int(p[0]+rad/3)]
                # mask=cv.CreateImage(cv.GetDims(valavg), 8, 1 )
                # avg x2 times
                roi = [ (p[0] - rad / den), (p[1] - rad / den)], 
                      [ (p[0] + rad / den), (p[1] + rad / den)]
                col = cvAvg( @sgc, roi)
                col = cvAvg( @sgc, roi)

                p_int = [ p[0], p[1] ]
                # puts "Circle #{p_int}-#{rad}  #{[ col[0], col[1], col[2]] }"
                cvCircle( @sg, p_int, rad, [ col[0], col[1], col[2]], -1)

                if i == 4
                    cvCircle( @sg, p_int, rad, [ 0, 255, 255], 2)
                else
                    cvCircle( @sg, p_int, rad, [255, 255, 255], 2)
                end

                hueavg = cvAvg( @hue, roi)
                satavg = cvAvg( @sat, roi)

                # cv.PutText(self.sg, repr(int(hueavg[0])), [ p_int[0] + 70, p_int[1]], self.ff, [255, 255, 255])
                # cv.PutText(self.sg, repr(int(satavg[0])), [ p_int[0] + 70, p_int[1] + 10], self.ff, [255, 255, 255])

                if @extract
                    cs << col
                    center_pixels << [p_int[0] * den, p_int[1] * den]
                    hsvcs << [ hueavg[0], satavg[0]]
                end
        	end
        end

        if @extract
            @extract = false
            @colors[ @selected] = cs
            @center_pixels[ @selected] = center_pixels
            @hsvs[ @selected] = hsvcs
            @selected = [ @selected + 1, 5].min
        end
    end

	# detect lines
	# return an array of [rho, theta]
	def hough( edge)
        # http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html#houghlinesp
        lines = cv::Mat.new

        pi = 3.1415926
        cv::HoughLines( edge, lines, 1, pi/180.0*1, threshold=10, srn=2, stn=2 )
        #puts "rows / cols #{lines.rows}, #{lines.cols}"
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

	# return an array of points [ pt1.x, pt1.y, pt2.x, pt2.y ]
	def houghP( edge, threshold = 70)
        # http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html#houghlinesp
        lines = cv::Mat.new

 	    # li = cv.HoughLines2(self.d2, cv.CreateMemStorage(), cv.CV_HOUGH_PROBABILISTIC, 1, 3.1415926 / 45, self.THR, 10, 5)
 	    # cv::HoughLinesP( d2, lines, 1, 3.1415926/45, thr = 70, 10, 5 );
        pi = 3.1415926
        cv::HoughLinesP( edge, lines, 1, pi/45, threshold, minLineLength=10, maxLineGap=20 )
        #puts "rows / cols #{lines.rows}, #{lines.cols}"
        lines = lines.to_a

        # puts "lines #{lines[0,1]}"

        # convert result back to array of point
        a = []
        b = []
        ( lines[0] || []).each_with_index do |l, i|
        	#puts l

        	if (i % 4) == 0
        		a << b if b.size > 0
        		b = []
        	end

        	b << l
        end

        a
	end


	# draw line from hough, onto image
	def drawPolar( lines, dst, count = 100)
		lines = lines[1..count].reverse
		lines.each_with_index do |l,i|
        	puts l.to_s
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
       #  	pt2 = cv::Point.new( 100, 100)
       	

        	#puts [ x0, y0, pt1, pt2 ].to_s
        	cv::line( dst, pt1, pt2,
            			   cv::Scalar.new(0,i,255 -2*i), 3, 8 )


       		# cv::line( dst2, pt1, pt2,
         #    			   cv::Scalar.new(255), 3, 8 )
        	break if i > count
        end
	end

	def drawSegment( lines, dst, count = 50)
		lines.each_with_index do |l,i|
	     	# puts [ l[0], l[1], l[2], l[3] ].to_s
	     	cvLine( @sg, [l[0], l[1]], 
	     				 [l[2], l[3]], 
	     				 [0,0,255])
		end
	end

	def drawCorner( to_try)
  		# cv.Circle(sg, IT[0], 5, (255,255,255))
  		to_try.each do | it|
  			l = it[0]
  			cvCircle(@sg, [ l[0], l[1]], 5, [255,255,255])
  		end
	end

	# edge filtering preparation for hough transform = 1
	CV_CMP_GT = 1
	CV_CMP_LT = 2

	def laplacian( grey)
		out = cv::Mat.new

      	# edge detection
      	# http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/laplace_operator/laplace_operator.html?highlight=laplacian       
        i = 1
        cv::GaussianBlur( grey, out, cv::Size.new( 0, 0 ), i, i)

        # cv::Laplacian( dst2, d, cv::CV_16S)
        cv::Laplacian( out, out, ddepth = cv::CV_16S) #, kernel_size = 1, scale = 10, delta = 0 )

        # cv.CmpS(self.d, 8, self.d2, cv.CV_CMP_GT)
        cmp = cv::Mat.new [ threshold = 4] 
        cv::compare( out, cmp, out, CV_CMP_GT)
 
        # if false # self.onlyBlackCubes:
        #     # can also detect on black lines for improved robustness
        #     b = cv::Mat.new
        #     cmp = cv::Mat.new [ threshold = 100] 
        #     cv::compare( grey, cmp, b, CV_CMP_LT)
        #     # cv.And(b, d2, d2)
        #     cv::bitwise_and( b, out, out)
        # end

        #d2 = cv::Mat.new
        #cv::convertScaleAbs( dst2, d2 )

        out
	end

	def canny( dst2)
        # http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html#houghlinesp
        # http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
        lowThreshold = 50
        ratio = 3
        cv::Canny( dst2, dst2, lowThreshold, lowThreshold*ratio, kernel_size  = 3);
 
 		dst2
	end

	def detect( grey)  
        @dst2 = laplacian( grey)

        # these weights should be adaptive. We should always detect 100 lines
        @thr = @thr + 1 if @lastdetected > DECTS
        @thr = [ 2, @thr - 1].max if @lastdetected < DECTS

	    detected = houghP( @dst2, @thr)
	    @lastdetected = detected.size
	    puts "\#lines #{@lastdetected } - thr : #{@thr}"

	    lines = detected[0..99]
    	# drawSegment( lines, @sg)
  		to_try = find_corner( lines)
  		# drawCorner( to_try)
 
  		res = check_corner( to_try, lines)
        res.sort_by! { |x| x[0] }.reverse!
 		# puts res.map { |x| x.to_s }
 		
 		if res[0] && res[0][0] > 0
 			p, p1, p2, p3 = p3v( res[0][1])
 			drawX( p, p1, p2, p3 )
 		end

  		res
	end

	def seg2points( l)
		[ [ l[0], l[1]], 
		  [ l[2], l[3]] ]
	end

	def intersect_seg(x1, x2, x3, x4, y1, y2, y3, y4)
	    den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

	    return [ false, [0, 0], [0, 0]] if den.abs < 0.1

	    ua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
	    ub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
	    ua = ua / den
	    ub = ub / den
	    x = x1 + ua * (x2 - x1)
	    y = y1 + ua * (y2 - y1)

	    [ true, [ua, ub], [x, y]]
	end

	# is t1 close to t2 within t?
	def areclose(t1, t2, t)
		(t1[0] - t2[0]).abs < t && (t1[1] - t2[1]).abs < t
	end

	# not touching at corner... try also inner grid segments hypothesis?
	def matched_0( li_i, li_j)
    	p1, p2 = seg2points( li_i)
		q1, q2 = seg2points( li_j)		

	    @tracker.p1 = [ p1[0], p1[1] ]
	    @tracker.p2 = [ p2[0], p2[1] ]
	    @tracker.q1 = [ q1[0], q1[1] ]
	    @tracker.q2 = [ q2[0], q2[1] ]
	    success, (ua, ub), (x, y) = intersect_seg(	@tracker.p1[0], @tracker.p2[0], 
	    											@tracker.q1[0], @tracker.q2[0], 
	    											@tracker.p1[1], @tracker.p2[1], 
	    											@tracker.q1[1], @tracker.q2[1])

	    if success && ua > 0 && ua < 1 && ub > 0 && ub < 1
	        # if they intersect
	        # cv.Line(sg, p1, p2, (255,255,255))
	        ok1 = 0
	        ok2 = 0

	        ok1 = 1 if abs(ua - 1.0 / 3) < 0.05
	        ok1 = 2 if abs(ua - 2.0 / 3) < 0.05

	        ok2 = 1 if abs(ub - 1.0 / 3) < 0.05
	        ok2 = 2 if abs(ub - 2.0 / 3) < 0.05

	        if ok1 > 0 and ok2 > 0
	            # ok these are inner lines of grid
	            # flip if necessary
	            @tracker.p1, @tracker.p2 = @tracker.p2, @tracker.p1 if ok1 == 2
	            @tracker.q1, @tracker.q2 = @tracker.q2, @tracker.q1 if ok2 == 2

	            # both lines now go from p1->p2, q1->q2 and
	            # intersect at 1/3
	            # calculate IT
	            z1 = [ @tracker.q1[0] + 2.0 / 3 * ( @tracker.p2[0] - @tracker.p1[0]), @tracker.q1[1] + 2.0 / 3 * (@tracker.p2[1] - @tracker.p1[1])]
	            z2 = [ @tracker.p1[0] + 2.0 / 3 * ( @tracker.q2[0] - @tracker.q1[0]), @tracker.p1[1] + 2.0 / 3 * (@tracker.q2[1] - @tracker.q1[1])]
	            z  = [ @tracker.p1[0] - 1.0 / 3 * ( @tracker.q2[0] - @tracker.q1[0]), @tracker.p1[1] - 1.0 / 3 * (@tracker.q2[1] - @tracker.q1[1])]
	            it = [z, z1, z2, dd1]
	            matched = 1

	            return [ matched, it]
	        end
	    end

	    [ 0, nil]
	end # matched_0

	# only single one matched!! Could be corner
	def matched_1( li_i, li_j)
		p1, p2 = seg2points( li_i)
		q1, q2 = seg2points( li_j)		

	    # also test angle
        a1 = Math::atan2(p2[1] - p1[1], p2[0] - p1[0])
        a2 = Math::atan2(q2[1] - q1[1], q2[0] - q1[0])

        a1 += Math::PI if a1 < 0
        a2 += Math::PI if a2 < 0

        ang = ((a2 - a1).abs - Math::PI / 2).abs

        if ang < 0.5
            return true 		# totry.append(IT)
        end

        return false
	end

	def test_lines( li_i, li_j, t)
    	p1, p2 = seg2points( li_i)
		q1, q2 = seg2points( li_j)

        # test lengths are approximately consistent
        dd1 = Math::sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]))
        dd2 = Math::sqrt((q2[0] - q1[0]) * (q2[0] - q1[0]) + (q2[1] - q1[1]) * (q2[1] - q1[1]))

        return nil  if [ dd1, dd2].max / [ dd1, dd2].min > 1.3
            
        matched = 0
        it = nil
        if areclose(p1, q2, t)
            it = [ avg(p1, q2), p2, q1, dd1 ]
            matched = matched + 1
        end

        if areclose(p2, q2, t)
            it = [ avg(p2, q2), p1, q1, dd1 ]
            matched = matched + 1
        end

        if areclose(p1, q1, t)
            it = [ avg(p1, q1), p2, q2, dd1 ]
            matched = matched + 1
        end

        if areclose(p2, q1, t)
            it = [ avg(p2, q1), q2, p1, dd1 ]
            matched = matched + 1
        end

		# not touching at corner... try also inner grid segments hypothesis?
        if matched == 0
            matched, it = matched_0( li_i, li_j)
        end

        # only single one matched!! Could be corner
        if matched == 1
        	return it if matched_1( li_i, li_j)
        end

        return nil
	end

	# line processing
	# lets look for lines that share a common end point
	def find_corner( li)
        t = 10
        to_try = []

        for i in 0..(li.size - 1)
        	for j in (i + 1)..(li.size - 1)
        		try = test_lines( li[i], li[j], t)
        		to_try << try if try
            end  # for j
        end # for i

        to_try
	end

    # check likelihood of this coordinate system. iterate all lines
    # and see how many align with grid
	def test_grid( angs, li, m_Ainv, a1, a2)
        evidence = 0

        for j in 0..(li.size - 1)
            # test angle consistency with either one of the two angles
            a = angs[j]
            ang1 = ((a - a1).abs - Math::PI / 2).abs
            ang2 = ((a - a2).abs - Math::PI / 2).abs

            next if ang1 > 0.1 && ang2 > 0.1
                
            # test position consistency.
            q1, q2 = seg2points( li[j] )
            qwe = 0.06

            # test one endpoint
            v = Matrix[[q1[0]], [q1[1]], [1]]
            vp = m_Ainv * v

            # project it
            next if vp[0, 0] > 1.1 || vp[0, 0] < -0.1
        
            next if vp[1, 0] > 1.1 || vp[1, 0] < -0.1        

            next if (vp[0, 0] - 1 / 3.0).abs > qwe && 
        			(vp[0, 0] - 2 / 3.0).abs > qwe &&
                	(vp[1, 0] - 1 / 3.0).abs > qwe &&
                	(vp[1, 0] - 2 / 3.0).abs > qwe
                

            # the other end point
            v = Matrix[[q2[0]], [q2[1]], [1]]
            vp = m_Ainv * v

            next if vp[0, 0] > 1.1 || vp[0, 0] < -0.1
                
            next if vp[1, 0] > 1.1 || vp[1, 0] < -0.1

            next if (vp[0, 0] - 1 / 3.0).abs > qwe &&
        			(vp[0, 0] - 2 / 3.0).abs > qwe &&
                	(vp[1, 0] - 1 / 3.0).abs > qwe &&
                	(vp[1, 0] - 2 / 3.0).abs > qwe
                

            # cv.Circle(sg, q1, 3, (255,255,0))
            # cv.Circle(sg, q2, 3, (255,255,0))
            # cv.Line(sg,q1,q2,(0,255,255))
            evidence += 1
        end

        evidence
	end

    # now check if any points in totry are consistent!
	def check_corner( totry, li)
        # store angles for later
        angs = []
        li.each do |l|
        	p1, p2 = seg2points( l)
            # cvLine( @sg, p1, p2, [255,255,0])
            a = Math::atan2(p2[1] - p1[1], p2[0] - p1[0])
            a += Math::PI if a < 0
                
            angs << a
        end

        # t=4
        res = []
        for i in 0..(totry.size() -1)

            p, p1, p2, dd = totry[i]
            a1 = Math::atan2(p1[1] - p[1], p1[0] - p[0])
            a2 = Math::atan2(p2[1] - p[1], p2[0] - p[0])

            a1 += Math::PI if a1 < 0
            a2 += Math::PI if a2 < 0

            dd = 1.7 * dd

            #
            #cvLine( @sg, p, p2, [0,255,0])
            #cvLine( @sg, p, p1, [0,255,0])

            # affine transform to local coords
            m_A = Matrix[ [p2[0] - p[0], p1[0] - p[0], p[0]], 
            			  [p2[1] - p[1], p1[1] - p[1], p[1]], 
            			  [0, 0, 1]]
            m_Ainv = m_A.inv

            # check likelihood of this coordinate system. iterate all lines
            # and see how many align with grid
            evidence = test_grid( angs, li, m_Ainv,
            				      a1, a2)

            res << [ evidence, [ p, p1, p2]]
		end

		res
	end

	# compute point 2 point distance for the 4 points of each faces (without taking account order)
	def compfaces(f1, f2)
	    totd = 0
	    f1.each do |p1|
	        mind = 10000
	        f2.each do |p2|
	            d = ptdst(p1, p2)
	            mind = d if d < mind
	        end

	        totd += mind
	    end

	    return totd / 4
	end

	def drawX( p, p1, p2, p3, col1 = [0,255,0], col2 = [0,0,255])
	    cvLine( @sg,p,p1, col1,2)
	    cvLine( @sg,p,p2, col1,2)
	    cvLine( @sg,p2,p3,col1,2)
	    cvLine( @sg,p3,p1,col1,2)

	    cen=[0.5*p2[0]+0.5*p1[0],0.5*p2[1]+0.5*p1[1]]  
	    cvCircle( @sg, cen, 20, col2,5)
	    #cvLine( @sg, [0,cen[1]], [320,cen[1]], [0,255,0],2)
	    #cvLine( @sg, [cen[0],0], [cen[0],240], [0,255,0],2)
	end

	# convert [p, p1, p2] to [p, p1, p2, p3]
	def p3v( vecp12)
        p, p1, p2 = vecp12
        p3 = [ p2[0] + p1[0] - p[0], p2[1] + p1[1] - p[1] ]
        [ p, p1, p2, p3]
	end
		
	# among good observations find best one that fits with last one
	# => tracking: update tracker state
	# return tracker state
	def init_tracker( res, tracker)
        minch = 10000
        # log.info("dects %s, res:\n%s" % (self.dects, pformat(res)))

        if res.size > 0
            minps = []
            pt = []

            # among good observations find best one that fits with last one
            for i in 0..(res.size() -1)
                if res[i][0] > 0.05 * DECTS
                    # OK WE HAVE GRID
                    p, p1, p2, p3 = p3v( res[i][1])
                    # p3 = [ p2[0] + p1[0] - p[0], p2[1] + p1[1] - p[1] ]

                    #
                    #drawX( p, p1, p2, p3, [128,255,128], [128,128,255])

                    w = [p, p1, p2, p3]
                    # p3 = [@prevface[2][0] + @prevface[1][0] - @prevface[0][0],
                    #       @prevface[2][1] + @prevface[1][1] - @prevface[0][1]]
                    tc = p3v( tracker.prevface) 	# [@prevface[0], @prevface[1], @prevface[2], p3]
                    ch = compfaces(w, tc)

                    # log.info("ch %s, minch %s" % (ch, minch))
                    # puts "\##{i} - ch #{ch}, minch #{minch}"
                    if ch < minch
                        minch = ch
                        minps = [p, p1, p2]
                    end
                end
            end

            # log.info("minch %d, minps:\n%s" % (minch, pformat(minps)))
            if minps.size > 0
            	puts "succ #{tracker.succ}/tracking: #{tracker.tracking} - minch #{minch}, minps #{minps}"

	            p, p1, p2, p3 = p3v( minps)
	            drawX( p, p1, p2, p3)

                tracker.prevface = minps

                if minch < 20
                    # good enough!
                    tracker.succ += 1
                    tracker.pt = tracker.prevface
                    tracker.detected = true
                    # log.info("detected %d, succ %d" % (self.detected, self.succ))
                end
            else
                tracker.succ = 0
            end

            # log.info("succ %d\n\n" % self.succ)

            # we matched a few times same grid
            # coincidence? I think NOT!!! Init LK tracker
            if tracker.succ > 2
                # initialize features for LK
                pt = []
                [ 1.0/3, 2.0/3].each do |i|
                	[ 1.0/3, 2.0/3].each do |j|
                         # pt << [ @p0[0] + i * @v1[0] + j * @v2[0], 
                         		 # @p0[1] + i * @v1[1] + j * @v2[1] ]
                 	end
              	end

                @features = pt
                tracker.tracking = true
                tracker.succ     = 0
                # log.info("non-tracking -> tracking: succ %d" % self.succ)
            end
        end

        # return state object
	    tracker
    end

	def verify_still_tracking( tracker)
		tracker
	end
end # class RubikFinder

# analyze file as if it comes from a video stream (to allow adaptative threshold)
ATTEMPTS = 10

def analyse_file( filename)

	img = cv::imread( filename)

	rf = RubikFinder.new( 2)

	for i in 1..ATTEMPTS
		rf.analyze_frame( img)
		cv::wait_key( 1)
	end
end

def track_webcam( )
	# video mode
	video_file = cv::VideoCapture.new( 0)
	frame = cv::Mat.new
	rubik = RubikFinder.new

	while true do
		video_file.read( frame)
		rubik.analyze_frame( frame)
		cv::wait_key( 1)
	end
end

# main test
if false
	# file mode
	filename = '/home/yves/Pictures/rubik/2017-02-13-182254.jpg'
	#filename = '/home/yves/Pictures/rubik/2017-02-26-124830.jpg'
	analyse_file( filename)
	cv::wait_key(-1)

else
	track_webcam( )
end

