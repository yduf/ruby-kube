# http://stackoverflow.com/questions/1313/followup-finding-an-accurate-distance-between-colors
# module for color manipulation

# char as rgb color for console display
require 'tco'

def to_char( rgb)
	"  ".bg rgb
end


# RGB input in 0.255
# color conversion code from http://www.easyrgb.com/en/math.php
class RGB < Array
	def initialize( rgb)
		super( rgb)
	end

	# sR, sG and sB (Standard RGB) input range = 0 ÷ 255
	# X, Y and Z output refer to a D65/2° standard illuminant.
	def to_xyz()
		sR, sG, sB = self.to_a

		var_R = ( sR / 255.0 )
		var_G = ( sG / 255.0 )
		var_B = ( sB / 255.0 )

		if ( var_R > 0.04045 ) 
								var_R = ( ( var_R + 0.055 ) / 1.055 ) ** 2.4
		else                   	var_R = var_R / 12.92
		end
		if ( var_G > 0.04045 ) 
								var_G = ( ( var_G + 0.055 ) / 1.055 ) ** 2.4
		else                   	var_G = var_G / 12.92
		end
		if ( var_B > 0.04045 ) 
								var_B = ( ( var_B + 0.055 ) / 1.055 ) ** 2.4
		else                    var_B = var_B / 12.92
		end

		var_R = var_R * 100
		var_G = var_G * 100
		var_B = var_B * 100

		x = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
		y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
		z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

		XYZ.new([x,y,z])
	end

	# R, G and B input range = 0 ÷ 255
	# H, S and V output range = 0 ÷ 1.0
	def to_hsv
		sR, sG, sB = self.to_a

		var_R = ( sR / 255.0 )
		var_G = ( sG / 255.0 )
		var_B = ( sB / 255.0 )

		var_Min = [var_R, var_G, var_B].min    # Min. value of RGB
		var_Max = [var_R, var_G, var_B].max    # Max. value of RGB
		del_Max = var_Max - var_Min            # Delta RGB value

		v = var_Max

		if ( del_Max == 0 )                     # This is a gray, no chroma...
		    h = 0
		    s = 0
		else                                    # Chromatic data...
		   s = del_Max / var_Max

		   del_R = ( ( ( var_Max - var_R ) / 6.0 ) + ( del_Max / 2.0 ) ) / del_Max
		   del_G = ( ( ( var_Max - var_G ) / 6.0 ) + ( del_Max / 2.0 ) ) / del_Max
		   del_B = ( ( ( var_Max - var_B ) / 6.0 ) + ( del_Max / 2.0 ) ) / del_Max

		   if    ( var_R == var_Max ) 
		   								h = del_B - del_G
		   elsif ( var_G == var_Max ) 
		   								h = ( 1 / 3.0 ) + del_R - del_B
		   elsif ( var_B == var_Max ) 
		   								h = ( 2 / 3.0 ) + del_G - del_R
		   end

		   h += 1 if ( h < 0 )
		   h -= 1 if ( h > 1 ) 
		end

		HSV.new([h,s,v])
	end

	def to_HSV
		RGBtoHSV( self)
	end

	def to_c
		"  ".bg self.to_a
	end

	# def dist2( rgb)
	# 	dist2( self, rgb)
	# end

	def delta_C( rgb)
	    a_lab = self.to_xyz.to_lab
	    b_lab =  rgb.to_xyz.to_lab
	    a_lab.delta_C( b_lab)
	end

	def deltaE2000( rgb )
	    a_lch = self.to_xyz.to_lab.to_lch
	    b_lch =  rgb.to_xyz.to_lab.to_lch
	    a_lch.deltaE2000( b_lch)

	rescue => x
		puts "2 - #{x}"
	end

	def face_dist( rgb)
		h1,s1,v1 = p1 = RGBtoHSV( self )
		h2,s2,v2 = p2 = RGBtoHSV( rgb )

		if s1 < 0.2 || s2 < 0.2	
		    # a_lab = self.to_xyz.to_lab
		    # b_lab =  rgb.to_xyz.to_lab
		    # 30 + dist2( a_lab, b_lab)
		    30 + 1
		else
			hsv_dist( p1, p2)
		end

		hsv_dist( p1, p2)

	rescue => x
		puts "3 - #{x}"
	end
end # class RGB


class XYZ < Array
	def initialize( xyz)
		super( xyz)
	end

	def to_lab( ref = nil)
		# Reference-X, Y and Z refer to specific illuminants and observers.
		# Common reference values are available below in this same page.
		ref ||= [ 94.811, 100.000, 107.304] 	# D65 / Daylight, sRGB, Adobe-RGB
		ref_X, ref_Y, ref_Z = ref

		x, y, z = self.to_a

		var_X = x / ref_X
		var_Y = y / ref_Y
		var_Z = z / ref_Z

		if ( var_X > 0.008856 ) 
								var_X = var_X ** ( 1.0/3 )
		else                    var_X = ( 7.787 * var_X ) + ( 16.0 / 116 )
		end
		if ( var_Y > 0.008856 ) 
								var_Y = var_Y ** ( 1.0/3 )
		else                    var_Y = ( 7.787 * var_Y ) + ( 16.0 / 116 )
		end
		if ( var_Z > 0.008856 ) 
								var_Z = var_Z ** ( 1.0/3 )
		else                    var_Z = ( 7.787 * var_Z ) + ( 16.0 / 116 )
		end

		cie_L = ( 116 * var_Y ) - 16
		cie_a = 500 * ( var_X - var_Y )
		cie_b = 200 * ( var_Y - var_Z )

		LAB.new([cie_L,cie_a,cie_b])
	end
end # class XYZ

class HSV < Array
	def initialize( hsv)
		super( hsv)
	end
end

class LAB < Array
	def initialize( xyz)
		super( xyz)
	end

	def to_lch
		l,a,b = self.to_a

		var_H = Math::atan2( b, a ) 			# Quadrant by signs

		if ( var_H > 0 ) 
			  var_H = ( var_H / Math::PI ) * 180
		else  var_H = 360 - ( var_H.abs / Math::PI ) * 180
		end

	    ciel_l = l
	    ciel_c = Math::sqrt( a*a + b*b )
	    ciel_h = var_H

	    LCH.new([ ciel_l, ciel_c, ciel_h])
	end

	def delta_C( lab_2)
		l1,a1,b1 = self
		l2,a2,b2 = lab_2

		Math::sqrt( ( a2 ** 2 ) + ( b2 ** 2 ) ) - 
			Math::sqrt( ( a1 ** 2 ) + ( b1 ** 2 ) )
	end
end # class LAB

class LCH < Array
	CV_PI = Math::PI

	def initialize( xyz)
		super( xyz)
	end

	def deltaE2000( lch2 )
		lch1 = self.to_a

	    avg_L = ( lch1[0] + lch2[0] ) * 0.5;
	    delta_L = lch2[0] - lch1[0];
	    avg_C = ( lch1[1] + lch2[1] ) * 0.5;
	    delta_C = lch1[1] - lch2[1];
	    avg_H = ( lch1[2] + lch2[2] ) * 0.5;

	    if( ( lch1[2] - lch2[2] ).abs > CV_PI )
        	avg_H += CV_PI;
        end

    	delta_H = lch2[2] - lch1[2];
	    if( ( delta_H ).abs > CV_PI )
	        if( lch2[2] <= lch1[2] )
	            delta_H += CV_PI * 2.0;
	        else
	            delta_H -= CV_PI * 2.0;
	        end
	    end

	    delta_H = Math::sqrt( lch1[1] * lch2[1] ) * Math::sin( delta_H ) * 2.0;
	    t = 1.0 -
	            0.17 * Math::cos( avg_H - CV_PI / 6.0 ) +
	            0.24 * Math::cos( avg_H * 2.0 ) +
	            0.32 * Math::cos( avg_H * 3.0 + CV_PI / 30.0 ) -
	            0.20 * Math::cos( avg_H * 4.0 - CV_PI * 7.0 / 20.0 );

	    sl = avg_L - 50.0;
	    sl *= sl;
	    sl = sl * 0.015 / Math::sqrt( sl + 20.0 ) + 1.0;

    	sc = avg_C * 0.045 + 1.0;
    	sh = avg_C * t * 0.015 + 1.0;
    	delta_Theta = avg_H / 25.0 - CV_PI * 11.0 / 180.0;
    	delta_Theta = Math::exp( delta_Theta * -delta_Theta ) * ( CV_PI / 6.0 );
    	rt = ( avg_C ** 7.0 ); # Math::pow
    	rt = Math::sqrt( rt / ( rt + 6103515625.0 ) ) * Math::sin( delta_Theta ) * -2.0; # 6103515625 = 25^7

	    delta_L /= sl;
	    delta_C /= sc;
	    delta_H /= sh;
	    return Math::sqrt(	 delta_L*delta_L + 
			    			 delta_C*delta_C + 
			    			 delta_H*delta_H + 
			    			 rt * delta_C*delta_H );

	rescue => x
		puts "1 - #{x}"
	end
end


# 
# convert RGB to HSV 
# return [ h, s, v] => [0..360, 0..1, 0..1]
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

def hsv_dist( p1, p2)
	h1,s1,v1 = p1
	h2,s2,v2 = p2

	a = [ h1, h2].min
	b = [ h1, h2].max

	if s1 < 0.2 && s2 < 0.2		# low saturation black / or white
		dist = 	10 + (v1 - v2).abs2
	else
		dist = 	( [ b - a, a + 360 - b].min / 180 ).abs2 + (s1 - s2).abs2  + (v1 - v2).abs2
	end

	dist
end

def dist2( a, b)
	a.zip(b).map { |x| x[0] - x[1] }
			.map { |x| x.abs2 }
			.inject(0, :+)
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

        # puts @hsv_keys.to_s
    end

    # find color name
    def name( color)
        h,s,v = RGBtoHSV( color)

        # distinguish white /black/color
        if s < 0.3  # probably white or black
            if v > 0.3
                h = [ :white, @hsv_keys[ :white] ]
                # puts h.to_s
                h
            else
                h = [ :black, @hsv_keys[ :black] ]
                h
            end
        else
            # puts @hsv_keys.min_by { |k,v| (v[0] - h).abs }.to_s 
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
end # class ColorMatcher


# main test
if __FILE__ == $0


	c_y = [223.32899886377982, 237.88751420275216, 174.12952909986112]
	c_w = [207.0494887009216,  208.6025754323949,  207.69031687918192]
	##
	f_w = [210.34818836005553, 207.8665572528721, 202.81567983840424]



    ###
	f_lab = RGB.new(f_w).to_xyz.to_lab

	sort_lab = [ c_y, c_w].map { |c| [ to_char(c), RGB.new(c).to_xyz.to_lab] }
				.sort_by { |c,lab| dist2( lab, f_lab)}
				.map { |c,lab| c}

	puts "lab:   #{to_char(f_w)} -> #{sort_lab.join("")}"

	####
	f_hsv = RGB.new(f_w).to_hsv

	sort_hsv = [ c_y, c_w].map { |c| [ to_char(c), RGB.new(c).to_hsv] }
				.sort_by { |c,hsv| dist2( hsv, f_hsv)}
				.map { |c,hsv| c}

	puts "hsv:   #{to_char(f_w)} -> #{sort_hsv.join("")}"

	####
	f_hsv = RGBtoHSV(f_w)

	sort_hsv = [ c_y, c_w].map { |c| [ to_char(c), RGBtoHSV(c)] }
				.sort_by { |c,hsv| hsv_dist( hsv, f_hsv)}
				.map { |c,hsv| c}

	puts "hsv_d: #{to_char(f_w)} -> #{sort_hsv.join("")}"

end