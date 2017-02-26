# http://www.ropencv.aduda.eu/examples/#loading
require 'ropencv'
include OpenCV

require 'matrix'


# detect cube according to XXX
class RubikFinder

	def initialize()
		@resize_factor = 2			# resize down factor for image processing
		@dects = 50					# ideal number of number of lines detections

		# internal state
        # stores the coordinates that make up the face. in order: p,p1,p3,p2 (i.e.) counterclockwise winding
        @prevface = [ [0, 0], [5, 0], [0, 5]]
        @prevface = [[349.0, 46.0], [531, 52], [340, 237]]

        @succ = 0					# count success of recognizing face acros multiple match
	end

	def analyze_frame( frame)
		# create  some buffer
		# cv::resize( frame, @sg)
 		w, h = [ frame.size.width, frame.size.height].map { |x| x / @resize_factor }
		resized = cv::Size.new( w, h )
		@sg  = cv::Mat.new( resized, cv::CV_8UC3)
		cv::resize( frame, @sg, @sg.size(), 0, 0, interpolation = 1);

		@grey = cv::Mat.new #( frame.size, cv::CV_8U)
        cv::cvtColor( @sg, @grey, CV_RGB2GRAY)

        detect( @grey)
	end

	# detect lines
	# return an array of [rho, theta]
	def hough( edge)
        # http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html#houghlinesp
        lines = cv::Mat.new

        pi = 3.1415926
        cv::HoughLines( edge, lines, 1, pi/180.0*1, threshold=10, srn=2, stn=2 )
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

	# return an array of points [ pt1.x, pt1.y, pt2.x, pt2.y ]
	def houghP( edge)
        # http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html#houghlinesp
        lines = cv::Mat.new

 	    # li = cv.HoughLines2(self.d2, cv.CreateMemStorage(), cv.CV_HOUGH_PROBABILISTIC, 1, 3.1415926 / 45, self.THR, 10, 5)
 	    # cv::HoughLinesP( d2, lines, 1, 3.1415926/45, thr = 70, 10, 5 );
        pi = 3.1415926
        cv::HoughLinesP( edge, lines, 1, pi/45, threshold=70, minLineLength=30, maxLineGap=15 )
        puts "rows / cols #{lines.rows}, #{lines.cols}"
        lines = lines.to_a

        # puts "lines #{lines[0,1]}"

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

	def cvLine( img, p1, p2, col, width = 3)
     	cv::line( img, cv::Point.new(p1[0], p1[1]),
         			   cv::Point.new(p2[0], p2[1]),
         			   cv::Scalar.new( col[0], col[1], col[2]), width, 8 )
	end

	def cvCircle( img, center, radius, col, width = 1)
		cv.circle( img, cv::Point.new( center[0], center[1]), radius, cv::Scalar.new(col[0], col[1], col[2]), width)
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
        cv::compare( out, cmp, out, cmp_op = 1)
 
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
        dst2 = laplacian( grey)
	    lines = houghP( dst2)

    	# drawSegment( lines, @sg)

  		to_try = find_corner( lines)
  		# drawCorner( to_try)
 
  		res = check_corner( to_try, lines)
        res.sort_by! { |x| x[0] }.reverse!
 		puts res.map { |x| x.to_s }

 		# puts find_self_consistent( res )
 		if res[0]
 			p, p1, p2, p3 = p3v( res[0][1])
 			drawX( p, p1, p2, p3 )
 		end

        cv::imshow("cube", dst2)
  		cv::imshow("lines", @sg)
	end


	def avg(p1, p2)
	    [ 0.5 * p1[0] + 0.5 * p2[0], 0.5 * p2[1] + 0.5 * p2[1] ]
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

	    @p1 = [ p1[0], p1[1] ]
	    @p2 = [ p2[0], p2[1] ]
	    @q1 = [ q1[0], q1[1] ]
	    @q2 = [ q2[0], q2[1] ]
	    success, (ua, ub), (x, y) = intersect_seg( @p1[0], @p2[0], @q1[0], @q2[0], @p1[1], @p2[1], @q1[1], @q2[1])

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
	            @p1, @p2 = @p2, @p1 if ok1 == 2
	            @q1, @q2 = @q2, @q1 if ok2 == 2

	            # both lines now go from p1->p2, q1->q2 and
	            # intersect at 1/3
	            # calculate IT
	            z1 = [ @q1[0] + 2.0 / 3 * ( @p2[0] - @p1[0]), @q1[1] + 2.0 / 3 * (@p2[1] - @p1[1])]
	            z2 = [ @p1[0] + 2.0 / 3 * ( @q2[0] - @q1[0]), @p1[1] + 2.0 / 3 * (@q2[1] - @q1[1])]
	            z  = [ @p1[0] - 1.0 / 3 * ( @q2[0] - @q1[0]), @p1[1] - 1.0 / 3 * (@q2[1] - @q1[1])]
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

	# distance between 2 points
	def ptdst(p1, p2)
	    Math::sqrt((p1[0] - p2[0]).abs2 + (p1[1] - p2[1]).abs2)
	end

	# compute point 2 point distance for the 4 points of each faces (without taking account order)
	def compfaces(f1, f2)
		puts f1.to_s
		puts f2.to_s
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
	# => tracking
	def find_self_consistent( res)
        minch = 10000
        # log.info("dects %s, res:\n%s" % (self.dects, pformat(res)))

        if res.size > 0
            minps = []
            pt = []

            # among good observations find best one that fits with last one
            for i in 0..(res.size() -1)
                if res[i][0] > 0.05 * @dects
                    # OK WE HAVE GRID
                    p, p1, p2, p3 = p3v( res[i][1])
                    # p3 = [ p2[0] + p1[0] - p[0], p2[1] + p1[1] - p[1] ]

                    #
                    drawX( p, p1, p2, p3, [128,255,128], [128,128,255])

                    w = [p, p1, p2, p3]
                    # p3 = [@prevface[2][0] + @prevface[1][0] - @prevface[0][0],
                    #       @prevface[2][1] + @prevface[1][1] - @prevface[0][1]]
                    tc = p3v( @prevface) 	# [@prevface[0], @prevface[1], @prevface[2], p3]
                    ch = compfaces(w, tc)

                    # log.info("ch %s, minch %s" % (ch, minch))
                    puts "\##{i} - ch #{ch}, minch #{minch}"
                    if ch < minch
                        minch = ch
                        minps = [p, p1, p2]
                    end
                end
            end

            # log.info("minch %d, minps:\n%s" % (minch, pformat(minps)))
            puts "minch #{minch}, minps #{minps}"
            p, p1, p2, p3 = p3v( minps)
            drawX( p, p1, p2, p3)

            if minps.size > 0
                @prevface = minps

                if minch < 10
                    # good enough!
                    @succ += 1
                    @pt = @prevface
                    @detected = 1
                    # log.info("detected %d, succ %d" % (self.detected, self.succ))
                end
            else
                @succ = 0
            end

            # log.info("succ %d\n\n" % self.succ)

            # we matched a few times same grid
            # coincidence? I think NOT!!! Init LK tracker
            # if @succ > 2

            #     # initialize features for LK
            #     pt = []
            #     for i in [1.0 / 3, 2.0 / 3]
            #         for j in [1.0 / 3, 2.0 / 3]
            #             pt << [self.p0[0] + i * self.v1[0] + j * self.v2[0], self.p0[1] + i * self.v1[1] + j * self.v2[1] ]
            #     	end
            #   	end

            #     self.features = pt
            #     self.tracking = True
            #     @succ = 0
            #     log.info("non-tracking -> tracking: succ %d" % self.succ)
            # end
        end

	    @succ > 0
    end

end # class RubikFinder

filename = '/home/yves/Pictures/rubik/2017-02-13-182254.jpg'
filename = '/home/yves/Pictures/rubik/2017-02-26-124830.jpg'


img = cv.imread( filename)

RubikFinder.new.analyze_frame( img)
cv::wait_key(-1)
