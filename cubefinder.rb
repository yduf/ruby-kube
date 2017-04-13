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

def drawBox( pt_int)
    col = [0, 255, 0]
    cvLine( @sg, pt_int[0], pt_int[1], col, 2)
    cvLine( @sg, pt_int[1], pt_int[3], col, 2)
    cvLine( @sg, pt_int[3], pt_int[2], col, 2)
    cvLine( @sg, pt_int[2], pt_int[0], col, 2)
end

def drawSegment( lines, img, count = 50)
    lines.each_with_index do |l,i|
        # puts [ l[0], l[1], l[2], l[3] ].to_s
        cvLine( img, [l[0], l[1]], 
                     [l[2], l[3]], 
                     [0,255,0])
    end
end

def drawCorner( img, points, size = 5, color = [255,255,255])
    # cv.Circle(sg, IT[0], 5, (255,255,255))
    points.each do | l|
        cvCircle( img, l, 5, color, size)
    end
end


# convert input rgb image to grey
def to_grey( rgb)
    grey = cv::Mat.new #( frame.size, cv::CV_8U)
    cv::cvtColor( rgb, grey, CV_RGB2GRAY)
    grey
end

# remap one dimension image to rgb space
def to_color( grey)
    rgb = cv::Mat.new
    cv::cvtColor( grey, rgb, CV_GRAY2RGB)
    rgb
end

# detect cube according to XXX
class RubikFinder
	DECTS = 100						# ideal number of lines detections

    attr_reader :tracker            # get current tracker state

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

		attr_accessor	:prev_grey,
						:pyramid,
						:prev_pyramid

		attr_accessor	:features,
                        :colors,
                        :center_pixels

        attr_accessor   :minLineLength,
                        :maxLineGap       

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

            # hough parameter
            @minLineLength = 30 
            @maxLineGap    = 12
	    end

        # tell if cube was detected and current face is analyzed
        def detected?
            @detected || @undetectednum < 1
        end

        def good?
            @tracking 
        end
	end # class Tracking


	def initialize( resize = 1)
		@resize_factor = resize		# resize down factor for image processing

		# automatic threshold
		@thr = 70                   # adaptative threshold for detecting such number of lines
		@lastdetected = 0

		@tracker = Tracking.new		# internal state for tracking

		# tracking option
        @win_size = 21
    	@flags    = 0
	end

    # resized input
    def resize( frame, resize_factor)
        @width, @height = [ frame.size.width, frame.size.height].map { |x| x / resize_factor }
        resized = cv::Size.new( @width, @height )
        sg     = cv::Mat.new( resized, cv::CV_8UC3)
        cv::resize( frame, sg, sg.size(), 0, 0, interpolation = 1);

        sg
    end

	# analyse one frame (with adaptive threshold from previous frame analysis)
    # it can be used to analyze the same frame several taime
    # producing different output
	def analyze_frame( frame, tracker = nil)
        @tracker = tracker if tracker     # import tracking status from extern

		# resized input before processing
        @sg     = resize( frame, @resize_factor)        # will be used for display
		@sgc    = @sg.clone               # copy for latter color extraction

        # Denoising - very consuming
        # cv::fastNlMeansDenoisingColored( @sg, @sg)
		@grey = to_grey( @sg)

        # this could be move in dectect function, but we use it for display
        @dst2 = laplacian( @grey)       
        @lap  = to_color( @dst2)         # convert back to color for display
                  
        # if was tracking contine tracking on current frame
        # this work if tracking was initialized
        if @tracker.tracking
       		@tracker = verify_still_tracking( @grey, @tracker)
        end

        # on init or if we lost tracking => detection mode
        if ! @tracker.tracking
        	@tracker.detected = false
	        res = detect( @dst2) 
 			@tracker = init_tracker( res, @tracker )
 		end

 		# tracking mode
        if  @tracker.tracking
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
 		end

        # prepare for next tracking call
        @tracker.prev_grey    = @grey
        @tracker.prev_pyramid = @tracker.pyramid


 		# use pt[] array to do drawing
 		if @tracker.detected?
            # this init tracker
	        @tracker = draw_circles_and_lines( @tracker, extract = true)
	    end

        # display images, with content
	    cv::imshow("lines", @lap)
        #cv::imshow("corner",  @corner)
  		cv::imshow("cube", @sg)



        # return tracker state
        @tracker
	end

    # convert to HSV
    def hsv( color)
        @hsv = cv::Mat.new
        cv::cvtColor( color, @hsv, CV_RGB2HSV)

        channel = Std::Vector::Cv_Mat.new
        cv::split( @hsv, channel )
        [ channel[0], channel[1], channel[2] ]
    end

    def rgb( color)
        channel = Std::Vector::Cv_Mat.new
        cv::split( color, channel )
        [ channel[0], channel[1], channel[2] ]
    end

    def extraction_point( tracker)
        # find the coordinates of the 9 places we want to extract over
        # first sort the points so that 0 is BL 1 is UL and 2 is BR
        pt = winded( tracker.pt[0], tracker.pt[1], tracker.pt[2], tracker.p3)

        tracker.v1 = [ pt[1][0] - pt[0][0], pt[1][1] - pt[0][1] ]
        tracker.v2 = [ pt[3][0] - pt[0][0], pt[3][1] - pt[0][1] ]
        tracker.p0 = [ pt[0][0],  pt[0][1]]

        ep = []         # extraction point
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

        [ ep, tracker]
    end

    def extract_color( tracker, extract)
        cs = []
        center_pixels = []

        den = @resize_factor
        ep, tracker = extraction_point( tracker)
        rad = ptdst( tracker.v1, [ 0.0, 0.0] ) / 6.0

        drawCorner( @lap, ep, rad, [ 255, 0, 0] )

        ep.each_with_index do |p, i|
            # puts i
            if p[0] > rad && p[0] < @width  - rad &&
               p[1] > rad && p[1] < @height - rad

                # valavg=val[int(p[1]-rad/3):int(p[1]+rad/3),int(p[0]-rad/3):int(p[0]+rad/3)]
                # mask=cv.CreateImage(cv.GetDims(valavg), 8, 1 )
                # avg x2 times
                roi = [ (p[0] - rad / den), (p[1] - rad / den)], 
                      [ (p[0] + rad / den), (p[1] + rad / den)]
                col = cvAvg( @sgc, roi).to_a

                # p_int = [ p[0], p[1] ]
                # puts "Circle #{p_int}-#{rad}  #{[ col[0], col[1], col[2]] }"
                cvCircle( @sg, p, rad, [ col[0], col[1], col[2]], -1)

                if i == 4
                    cvCircle( @sg, p, rad, [ 0, 255, 255], 2)
                else
                    cvCircle( @sg, p, rad, [255, 255, 255], 2)
                end

                # extract face 
                b,g,r = col
                cs << [ r, g, b]           # rgb color

                if extract
                    center_pixels << [p[0] * den, p[1] * den] # pixel coordinate in frame
                    # hsvcs << [ hueavg, satavg]    # hue , sat
                end
            end
        end

        [ cs, center_pixels ]
    end

	# return updtated tracker object
	def draw_circles_and_lines( tracker, extract )
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

        pt_int = []
        tracker.pt.each do |p|
        	foo, bar = p
            pt_int << [ foo, bar]
        end

        # do the drawing. pt array should store p,p1,p2
        tracker.p3 = [	tracker.pt[2][0] + tracker.pt[1][0] - tracker.pt[0][0], 
        				tracker.pt[2][1] + tracker.pt[1][1] - tracker.pt[0][1] ]
        #p2_int = [ tracker.p2[0], tracker.p2[1] ]
        p3_int = [ tracker.p3[0], tracker.p3[1] ]
        pt_int << p3_int

        drawBox( pt_int)
        cs, center_pixels = extract_color( tracker, extract)

        if extract
            tracker.colors        = cs
            tracker.center_pixels = center_pixels
        end

        tracker
    end

	# detect lines
	# return an array of points [ pt1.x, pt1.y, pt2.x, pt2.y ]
	def houghP( edge, threshold = 70)
        # http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html#houghlinesp
        lines = cv::Mat.new

 	    # li = cv.HoughLines2(self.d2, cv.CreateMemStorage(), cv.CV_HOUGH_PROBABILISTIC, 1, 3.1415926 / 45, self.THR, 10, 5)
 	    # cv::HoughLinesP( d2, lines, 1, 3.1415926/45, thr = 70, 10, 5 );
        pi = 3.1415926
        cv::HoughLinesP( edge, lines, 2, 1*pi/180, threshold, 
                         minLineLength=@tracker.minLineLength, 
                         maxLineGap=@tracker.maxLineGap )
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

	# edge filtering preparation for hough transform = 1
	CV_CMP_GT = 1
	CV_CMP_LT = 2

    def laplacian3
        i1 = laplacian( @red)
        i2 = laplacian( @green)
        i3 = laplacian( @blue)

        i1+i2+i3
    end

	def laplacian( grey)
        i = 1

		out = cv::Mat.new

        # Apply Histogram Equalization
        # 

        #cv::medianBlur( out, out, 3)
        cv::GaussianBlur( grey, out, cv::Size.new( 0, 0 ), i, i)
        #cv::equalizeHist( out, out )

      	# edge detection
      	# http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/laplace_operator/laplace_operator.html?highlight=laplacian       
        #cv::GaussianBlur( out, out, cv::Size.new( 0, 0 ), i, i)
        cv::Laplacian( out, out, ddepth = cv::CV_16S, 
                       kernel_size = 1, scale = 1, delta = 0 )

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

    # detect line in images
    # and try to map a grid on them
	def detect( grey)
        # these weights should be adaptive. We should always detect 100 lines
        @thr = @thr + 1           if @lastdetected > DECTS + 20
        @thr = [ 2, @thr - 1].max if @lastdetected < DECTS

	    detected = houghP( grey, @thr)
	    @lastdetected = detected.size
	    #puts "\#lines #{@lastdetected } - thr : #{@thr}"

	    lines = detected[0..DECTS]
    	drawSegment( lines, @lap)

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

    # given to line [p1, p2], [qA, q2]
    # 
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
        # disable for the
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
                         pt << [ tracker.p0[0] + i * tracker.v1[0] + j * tracker.v2[0], 
                         		 tracker.p0[1] + i * tracker.v1[1] + j * tracker.v2[1] ]
                 	end
              	end

                tracker.features = pt
                tracker.tracking = true
                tracker.succ     = 0
                # log.info("non-tracking -> tracking: succ %d" % self.succ)
            end
        end

        # return state object
	    tracker
    end

	def verify_still_tracking( grey, tracker)
		# puts "tracking"
        tracker.detected = 2

        # compute optical flow
        status = cv::Mat.new
        err    = cv::Mat.new

        mat_features = cv::Mat.new( tracker.features.size, 2, cv::CV_32F)
        tracker.features.each_with_index do |p,i| 
        	mat_features[i,0] = p[0]
        	mat_features[i,1] = p[1]
        end

        # puts mat_features.checkVector(2, cv::CV_32F, true) 
        winSize = cv::Size.new( @win_size, @win_size)
        maxLevel = 3

        #
		tracker.pyramid = Std::Vector::Cv_Mat.new
		cv::buildOpticalFlowPyramid( grey, tracker.pyramid,
									 winSize, maxLevel)

		if not tracker.prev_pyramid
			tracker.prev_pyramid = Std::Vector::Cv_Mat.new
			cv::buildOpticalFlowPyramid( tracker.prev_grey, tracker.prev_pyramid,
										 winSize, maxLevel)			
		end

        # puts mat_features.to_a.to_s
		new_features = cv::Mat.new

        cv::calcOpticalFlowPyrLK(
            tracker.prev_grey, grey, 			
            # tracker.prev_pyramid, tracker.pyramid,
            mat_features,  new_features, 
            status,	err,						# output
            winSize, 
            maxLevel,
            cv::TermCriteria.new( cv::TermCriteria::MAX_ITER + 
            					  cv::TermCriteria::EPS, max_iter = 20, accuracy= 0.03),
            @flags)

        # set back the points we keep
        # puts new_features.to_a.to_s
        #puts status.zip( new_features.to_a).to_s
        tracker.features = status.zip( new_features.to_a)
        						.keep_if { |x| st, p = x;  st > 0 }
        						.map { |x| st, p = x; p}
        # puts tracker.features.to_s
        drawCorner( @lap, tracker.features, 5, [ 0, 255, 0] )


        if tracker.features.size < 4
            tracker.tracking = false  # we lost it, restart search
            puts "tracking -> not tracking: len features #{tracker.features.size } < 4"
        else
            # make sure that in addition the distances are consistent
            ds1 = ptdst( tracker.features[0], tracker.features[1])
            ds2 = ptdst( tracker.features[2], tracker.features[3])

            if [ ds1, ds2].max / [ ds1, ds2].min > 1.4
                tracker.tracking = false
                puts "tracking -> not tracking: max/min ds1 #{ds1}, ds2 #{ds2} > 1.4"
            end

            ds3 = ptdst( tracker.features[0], tracker.features[2])
            ds4 = ptdst( tracker.features[1], tracker.features[3])

            if [ ds3, ds4].max / [ ds3, ds4].min > 1.4
                tracker.tracking = false
                puts "tracking -> not tracking: max/min ds3 #{ds3}, ds4 #{ds4} > 1.4"
            end

            if ds1 < 10 || ds2 < 10 || ds3 < 10 || ds4 < 10
                tracker.tracking = false
                puts "tracking -> not tracking: ds1 #{ds1}, ds2 #{ds2}, ds3 #{ds3}, ds4 #{ds4}"
            end

            tracker.detected = false if not tracker.tracking
		end

		tracker
	end
	    
end # class RubikFinder

# analyze file as if it comes from a video stream (to allow adaptative threshold)
ATTEMPTS = 10

require 'tco'

# cube as colored string
def cube_to_s( colors)
 colors.map { |c|"  ".bg c }
        .each_slice(3)
        .to_a.map { |x| x.join("") }
        .join("\n")
end

# convert RGB to HSV 
# return [ h, s, v]
def RGBtoHSV( rgb )
   r, g, b = rgb

   rc = r / 255.0
   gc = g / 255.0
   bc = b / 255.0
   max = [ rc, gc, bc].max
   min = [ rc, gc, bc].min
   delta = max - min;
   v = max

   s = if max != 0.0
      delta / max
   else
      0.0
   end

   if (s == 0.0) 
      h = 0.0
   else 
      if (rc == max)
        h = (gc - bc) / delta
      elsif (gc == max)
        h = 2 + (bc - rc) / delta
      elsif (bc == max)
        h = 4 + (rc - gc) / delta
      end

      h *= 60.0

      h += 360.0 if h < 0
   end

   [ h, s, v]
end

# find nearest color 
class ColorMatcher
    @@rgb_keys = {
        red:    [ 255,   0,   0],
        orange: [ 255, 165,   0],
        blue:   [   0,   0, 255],
        green:  [   0, 255,   0],
        yellow: [ 255, 255,   0],
        white:  [ 255, 255, 255],
        black:  [   0,   0,   0]
    }

    def initialize()
        # convert RGB definition to HSV for matching
        @hsv_keys = {}
        @@rgb_keys.each { |k, v|
            @hsv_keys[k] = RGBtoHSV( v)
        }

        puts @hsv_keys.to_s
    end

    # find color name
    def name( color)
        h,s,v = RGBtoHSV( color)

        # distinguish white /black/color
        if s < 0.3  # probably white or black
            if v > 0.3
                h = [ :white, @hsv_keys[ :white] ]
                puts h.to_s
                h
            else
                h = [ :black, @hsv_keys[ :black] ]
                h
            end
        else
            puts @hsv_keys.min_by { |k,v| (v[0] - h).abs }.to_s 
            @hsv_keys.min_by { |k,v| (v[0] - h).abs }
        end
        # find color
    end

    # convert name to RGB
    def getRGB( names)
        names.map { |x|
            @@rgb_keys[ x]
        }
    end

end

def getTargetColor( colors)
    cm = ColorMatcher.new

    colors.map { |x|
        cm.name( x).first
    }
end

def good!( tracker)

    pp tracker
    puts cube_to_s( tracker.colors)

    cm = ColorMatcher.new
    tracker.colors.each { |x|
        puts RGBtoHSV( x).to_s
     }

    names = getTargetColor( tracker.colors)
    rgb   = cm.getRGB( names)

    puts cube_to_s( rgb)    
end

def analyse_file( filename)
    @hsvs = []
	img = cv::imread( filename)

	rf = RubikFinder.new( 2)

	for i in 1..ATTEMPTS
        puts "\##{i}"
		tracker = rf.analyze_frame( img)

        if tracker.good?
            good!( tracker)
            return
        end

		cv::wait_key( 1)
	end
end

# realtime cube tracker for given camera
# sync threshold with UI
def track_webcam( cam = 0)
	# video mode
	video_file = cv::VideoCapture.new( cam)
	frame = cv::Mat.new
	rubik = RubikFinder.new

    #
    # int createTrackbar(const String& trackbarname, const String& winname, int* value, int count, TrackbarCallback onChange=0, void* userdata=0)
    # thr = 0
    # cv::setTrackbarPos       

    # first step 
    video_file.read( frame)
    puts [ frame.size.width, frame.size.height].to_s

    # continuous 
    tracker = rubik.tracker
    cv::setTrackbarPos( "minLineLength", "LLines", tracker.minLineLength)
    cv::setTrackbarPos( "maxLineGap", "LLines", tracker.maxLineGap)

	while true do
		video_file.read( frame)
        
        # manipulate state 
        if tracker
            tracker.minLineLength = cv::getTrackbarPos( "minLineLength", "LLines")
            tracker.maxLineGap = cv::getTrackbarPos( "maxLineGap", "LLines")

            # tracker.tracking = false        # force detect mode always
        end

		tracker = rubik.analyze_frame( frame, tracker)

		key_press = cv::wait_key( 1)
        action = input_key2action key_press, frame

        action = handle_top_action( action, frame)

        # forward info
        yield tracker, action if block_given?
	end
end

# direct FFI version for 
# c interface 
# int cvCreateTrackbar(const char* trackbar_name, const char* window_name, int* value, int count, CvTrackbarCallback on_change=NULL )
require 'ffi'

module OpenCV
  extend FFI::Library
  ffi_lib 'libopencv_highgui.so'
  attach_function :cvCreateTrackbar, :cvCreateTrackbar,[ :string, :string, :pointer, :int, :pointer ], :int
end

def init_ui()
    # att UI for threshold
    cv::namedWindow("LLines", 1);
    cvCreateTrackbar( "minLineLength", "LLines", valuep = nil, count = 200, onChange=nil)
    cvCreateTrackbar( "maxLineGap", "LLines", valuep = nil, count = 200, onChange=nil)
end

# main test
if __FILE__ == $0

def input_key2action( key_press, frame )
    puts "input_key2action #{key_press}"
    case key_press.chr
    
    when 'a'                # reset search
        { reset: {} }

    when 'c'                # save detected color
        { save_color: {} }

    when 's'                # save frame
        puts "save image"

        $frame_counter  = ( $frame_counter ||= 0 ) + 1
        { save_frame: { frame: frame,
                      file_name: "frame_#{$frame_counter}.jpg" }
        }
    # store rgb snap for given face
    when 'u', 'l', 'f', 'r', 'b', 'd'
        puts "save face #{key_press.chr}"
        { store_face: { face: key_press.chr } }
        # @colors[ key_press.chr] = tracker.colors

    # resolve color attributuion for whole cube
    when 'p'
        puts "resolve whole faces color"
        { resolve_color: {} }

    else
        {}
    end

rescue      # invalid key
    {}
end

def handle_top_action( action, frame)
    # jandle general action internally
    drop_action = true

    case action.keys.first
    when :save_frame
        params = action.values.first
        cv::imwrite( params[:file_name], params[:frame])      

    else
        drop_action = false
    end

    action = {} if drop_action
    action
end


if false
	# file mode
	filename = '/home/yves/Pictures/rubik/2017-02-13-182254.jpg'
	#filename = '/home/yves/Pictures/rubik/2017-02-26-124830.jpg'
    # filename = '/home/yves/Pictures/rubik/2017-03-12-163532.jpg'
	analyse_file( filename)
	cv::wait_key(-1)

else
    init_ui

    # @selected = 0
    @colors   = {}
    # @center_pixels = []

    # get camera as last video device
    video_dev = Dir.glob( "/dev/video*")
    cam = video_dev.map { |p| /(\d+)$/ =~ p
                            $1.to_i }
                    .sort
                    .last

    puts "using camera #{cam}"

	track_webcam( cam) do |tracker, action|
        puts "#{tracker.good?} - detected #{tracker.detected} - undetectednum #{tracker.undetectednum}"

        if tracker.good?
            # pp tracker
            puts cube_to_s( tracker.colors)

            case action.keys.first
            when :save_color
                puts "save color"
                IO.write( "./colors.txt", "#{tracker.colors}\n", mode: 'a')

            when :store_face
                puts "save face #{key_press.chr}"
                @colors[ key_press.chr] = tracker.colors

                puts @colors.to_s
            end

            # @center_pixels[ @selected] = tracker.center_pixels

            good!( tracker)

            # inc faces
            #@selected = [ @selected + 1, 5].min
        end

    end
end

end # __FILE__ == $0
