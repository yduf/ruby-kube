# faces = [] <<
# [[138.50119934351721, 234.3087993940159, 173.58060850902663], [220.49021588183308, 94.2335563691453, 90.52897361444262], [121.31283928796869, 76.62555232925135, 183.1527584900896], [209.02512309051886, 245.2310314354248, 206.3489458401717], [124.92134831460673, 233.53339224845345, 179.26436056053527], [228.53288726170936, 99.0098472415099, 101.30892564070193], [124.15528342381012, 227.68438328493875, 176.77452341876025], [209.02676429743718, 224.7855068804444, 196.75861633632115], [113.63186466355258, 84.87690948112612, 170.48604974119428]] <<
# [[241.26208812018683, 113.7057189748769, 109.72213104406009], [137.26789546774395, 91.13470521398813, 190.95492993308923], [227.90089635147075, 229.00277742709252, 228.43757101376087], [248.98371417750283, 120.6773134705214, 113.36611538947102], [240.90960737280645, 186.82729453351848, 184.24504481757353], [152.25602827925766, 246.36624163615704, 204.99823254639566], [216.78601186718848, 215.14972856962504, 202.97235197576063], [133.5644489332155, 93.83651054159827, 184.0954424946345], [226.17447292008583, 118.0165383158692, 114.59689433152379]] <<
# [[246.52670117409417, 173.94003282413834, 158.84433783613179], [233.69410427976266, 229.67036990279004, 227.93523545006943], [217.87665698775405, 243.40436813533643, 195.58389092286328], [213.1728317131675, 229.53339224845345, 179.31536422168918], [111.12195429869965, 95.51142532508521, 183.84067668223707], [231.62479484913518, 232.48693346799647, 232.40032824138365], [226.30917813407396, 166.1886125489206, 146.31687918192148], [103.86870344653452, 219.5758111349577, 151.72187855068805], [239.07650549173084, 172.29920464587804, 155.57972478222445]] <<
# [[239.08029289231158, 237.32268652947857, 234.48011614695113], [248.0295417245297, 195.6170938012877, 190.00871102133567], [132.13129655346546, 99.29718469890165, 194.64575179901527], [235.881580608509, 238.16980179270294, 220.17472541345788], [239.85759373816435, 114.37242772377225, 108.81365989142785], [117.2392374700164, 85.28329756343895, 184.03219290493624], [212.9912889786643, 224.07208685771997, 205.44691326852669], [122.99293018558262, 232.01439212220677, 182.3937634137104], [122.51634894584016, 229.7567226360308, 186.36371670243656]] <<
# [[243.95492993308923, 173.0908976139376, 155.2490847115263], [221.47771745991668, 89.13584143416234, 80.77856331271303], [108.58553212978158, 231.45789673021082, 156.35184951395024], [115.0325716449943, 230.2078020451963, 159.86895593990656], [207.0494887009216, 208.6025754323949, 207.69031687918192], [105.74738038126499, 94.19656609014012, 179.75546016917056], [205.46736523166265, 98.53414972856962, 88.20956949880065], [229.52190380002523, 221.58894079030424, 220.16096452468122], [220.14253250852164, 219.95328872617094, 221.33531119808103]] <<
# [[224.58944577704833, 95.1371039010226, 83.28620123721751], [249.10655220300467, 149.21790178007825, 130.62858224971595], [225.72415099103648, 229.42090645120564, 227.4568867567226], [245.68577199848502, 152.57757858856203, 135.53124605479104], [223.32899886377982, 237.88751420275216, 174.12952909986112], [224.47153137230146, 239.10655220300467, 236.128897866431], [210.34818836005553, 207.8665572528721, 202.81567983840424], [228.9964650927913, 139.9428102512309, 126.52884736775658], [102.54967807095063, 80.24062618356268, 155.7903042545133]]

# include faces and target
load "example2.rb"
faces = $faces

# setup face orientation
def rotate_l( face)
	a,b,c = 0,1,2
	d,e,f = 3,4,5
	g,h,i = 6,7,8

	index = [ c,f,i,
	          b,e,h,
	          a,d,g]

	index.map { |i| face[i] }
end

# compute upward orientation from
# given diretion
# 0 => up
# 1 => right
# 2 => dpwn
# 3 => left
def set_f( o, f)
	case o
	when 0 then f
	when 1 then rotate_l( f)
	when 2 then rotate_l( rotate_l(f))
	when 3 then rotate_l( rotate_l( rotate_l(f)))
	else
		raise "error"
	end
end


require_relative 'color_dist.rb'

#
# cube as colored string
def cube_to_s( colors)
 colors.map { |c| "  ".bg c }
        .each_slice(3)
        .to_a.map { |x| x.join("") }
        .join("\n")
end

def concat_2d( array, sep = "")
	aa = array.map { |a|  a.size 
						  a.split("\n") }
	# aa.map { |c| puts "#{c.size} #{c.join("-")}"

	a = aa.shift
	b = a.zip( *aa)

	b.map { |c| c.join( sep )
			}
	 			.join("\n")
end

### swap faces 
#   U              4                 yellow
# L F R B   --   0 1 2 3   --   blue   red  green orange
#   D              5                  white
# faces[0], faces[2] = faces[2], faces[0]
# faces[1], faces[3] = faces[3], faces[1]
# faces[4], faces[5] = faces[5], faces[4]

# puts "Rgb input:"
# ff = faces.map { |f| cube_to_s( f) }
# puts concat_2d( ff, "  " )


### orient sample faces according to our assumption
# snap_or = [2,2,2,2, 2,2]

# faces = (0..5).to_a.map { |i| 
# 	set_f( snap_or[i], faces[i])
# }

# puts "RGB Oriented input:"
# puts "faces = [] << "
# puts faces.map { |f| "#{f}"}
# 		  .join(" <<\n")

# extrac center
centers = faces.map { |f| f[4] }

# extract rgb colors
# x_rgb = []	<< centers[4] << centers[5] << faces[4][2]

# x_rgb.each { |rgb|
# 	puts "#{to_char(rgb)} - rgb: #{rgb}"
# }

# puts faces.to_s
# puts centers.to_s
# puts "RGB Oriented input:"
# faces.each { |f| puts "#{f}"}


puts "Oriented input:"
ff = faces.map { |f| cube_to_s( f) }
puts concat_2d( ff, "  " )


puts "centers:"
puts centers.map { |c|
					"  " + "  ".bg( c) + "  " }
					.join("  ")


tiles = faces.each_with_index
			 .map { |f,i| f.each_with_index
			 			   .map { |rgb,j| [ to_char(rgb), RGBtoHSV(rgb), i, j ] }
			 		}
			 .flatten( 1)
			 .sort_by { |c,hsv,i,j|  hsv[1] }
			 .reverse


# for each remaining facets
# order match by best centers
# check that cube remain consistent
# - yes => continue with facets - 1
# - no  =>
#    - if center remaining => use this center, continue with facets -1
#    - no => backtrack
#

# def ptdstw(p1, p2):
#     # return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

#     # test if hue is reliable measurement
#     if p1[1] < 100 or p2[1] < 100:
#         # hue measurement will be unreliable. Probably white stickers are present
#         # leave this until end
#         return 300.0 + abs(p1[0] - p2[0])
#     else:
#         return abs(p1[0] - p2[0])


# compute index of opposite face
# compatible with all representation that put 4/5 upper/down
def opposite?( f)
	opposite ={ 0 => 2, 
				1 => 3, 
				2 => 0, 
				3 => 1, 
				4 => 5, 
				5 => 4 } 

	opposite[ f]
end

f_index = (0..8).to_a
puts f_index.to_s

# xyz upper face bottom line
# abc down  face upper  line

class ABC
 	def initialize
		f_index = (0..8).to_a
		# puts f_index.to_s

		@abc = []
		@xyz = []

		(0..3).to_a.each { |i|
			# puts i
			rot = set_f( i, f_index)

			upper  = rot.take( 3)
			bottom = rot.last( 3)

			# puts upper.to_s
			# puts bottom.to_s

			@abc[i] = upper
			@xyz[i] = bottom 
		}
 	end

 	# given face index 0..3, return upper line index for face 5
 	def abc( fi)
 		orientation = (fi - 1) % 4
 		@abc[ orientation]
 	end

 	# given face index 0..3, return bottom line index for face 4
 	def xyz( fi)
 		orientation = (1 - fi) % 4
 		@xyz[ orientation]
 	end
 end

K =  ABC.new
# puts "=="

# puts K.xyz( 0).to_s
# puts K.xyz( 1).to_s
# exit 0


# return tuple of neighbors given face and sticker indeces
# Assumption (not much about color but faces relation)
# 0..3 faces have assumption on top/Down orientation
#   U              4                 yellow
# L F R B   --   0 1 2 3   --   blue   red  green orange
#   D              5                  white
def neighbors(f, s)
	# xyz upper face bottom line
	# abc down  face upper  line
	m = { 0 => {  udlr:  [ 4, 5, 3, 1],
				abc:   K.abc( 0),			# [0, 3, 6]
				xyz:   K.xyz( 0),			# [ 8, 5, 2]
			 },
		
		  1 => {  udlr:  [ 4, 5, 0, 1],
				abc:   K.abc( 1),			# [ 0, 1, 2],
				xyz:   K.xyz( 1)		    # [ 6, 7, 8] 
				},
		
		  2 => {  udlr:  [ 4, 5, 1, 3],
				abc:   K.abc( 2),			# [ 2, 5, 8],
				xyz:   [ 0, 3, 6] },

		  3 => {  udlr:  [ 4, 5, 2, 0],
				abc:   K.abc( 3),			# [ 8, 7, 6],
				xyz:   [ 6, 7, 8] },
		}

    if (0..3).to_a.include?( f) 
    	u,d,l,r = m[f][ :udlr]
    	a, b, c = K.abc( f)
    	x, y, z = K.xyz( f)

        return [[l, 2], [u, x]]         if s == 0
        return         [[u, y]]         if s == 1
        return         [[u, z], [r, 0]] if s == 2
        return [[l, 5]]                 if s == 3
        return                 [[r, 3]] if s == 5
        return [[l, 8], [d, a]]         if s == 6
        return         [[d, b]]         if s == 7
        return         [[d, c], [r, 6]] if s == 8
    end

    if f == 4
    	u,d,l,r = 3,1, 0,2

        return [[l, 0], [u, 2]]         if s == 0
        return         [[u, 1]]         if s == 1
        return         [[u, 0], [r, 2]] if s == 2
        return [[l, 1]]                 if s == 3
        return                 [[r, 1]] if s == 5
        return [[l, 2], [d, 0]]         if s == 6
        return         [[d, 1]]         if s == 7
        return         [[d, 2], [r, 0]] if s == 8
    end

    if f == 5
    	u,d,l,r = 1,3, 0,2

        return [[l, 8], [u, 6]]         if s == 0
        return         [[u, 7]]         if s == 1
        return         [[u, 8], [r, 6]] if s == 2
        return [[l, 7]]                 if s == 3
        return                 [[r, 7]] if s == 5
        return [[l, 6], [d, 8]]         if s == 6
        return         [[d, 7]]         if s == 7
        return         [[d, 6], [r, 8]] if s == 8
    end
end


# => return cube or exception ?
def match( facets = [], centers = [], cube = {} )
	if facets.size > 0
		n_facets  = facets.dup
		f_rgb,i,j = n_facets.shift
		f = f_rgb.to_c

		# are we an edge or corner
		is_edge   = (j % 2) == 1
		is_corner = !is_edge

		# get possible centers, and order them by color distance
		# counter    = cube[:f_per_c]
		counter    = cube[:edge_color].zip(cube[:corner_color])
						.map { |e,c| if is_edge then e 
									 else c end 
							 }
		constraint = cube[:constraint][ [i,j] ] ||= (0..5).to_a
		# puts constraint.to_s

		closest = centers.select { |c_rgb,i,j| constraint.include?( i) }
						 .select { |c_rgb,i,j| counter[i] > 0 }
						 .sort_by{ |c_rgb,i,j| c_rgb.face_dist( f_rgb ) }
		compact = closest.map { |c_rgb,i,j| c_rgb.to_c }
						 .join("")
		# puts f ;  puts compact

		closest.each { |c_rgb,ci,cj|
		begin
			new_counter = counter.dup
			new_counter[ ci] = new_counter[ ci] - 1

			# setting this impose some contraint on neighbour faces
			# so update the constraints
			new_cube = cube.merge( { edge_color: new_counter} )   if is_edge
			new_cube = cube.merge( { corner_color: new_counter} ) if is_corner

			c = c_rgb.to_c
			new_cube[:match] = new_cube[:match].dup << [f,i,j,ci,c, compact]

			# new_cube[:match].each { |k| puts k.join(" ") }

			puts "#{f} #{[i,j]} => #{ci} #{c} - #{compact}"
			neighbors(i,j).each { |f,s|
				new_cube[:constraint][ [f,s] ] ||= (0..5).to_a
				# cur color cannot be used on neighbours
				new_cube[:constraint][ [f,s] ].delete( ci)
				# oposite color cannot be used either
				new_cube[:constraint][ [f,s] ].delete( opposite?(ci))	

				puts "#{[f,s]} -> #{new_cube[:constraint][ [f,s] ]}"
			}

			cube = match( n_facets, centers, new_cube)
			return cube
		rescue => x
			puts x
		end }

		# no match found
		exit 0

		raise "backtracking"
	else
		cube
	end
end

# => return cube
def do_match( faces)
	tiles = faces.each_with_index
				 .map { |f,i| f.each_with_index
				 			   .map { |rgb,j| 
				 			   		  [ RGB.new(rgb), i, j ] }
				 		}
				 .flatten( 1)

	centers = tiles.select  { |rgb,i,j| j == 4 }
	               .sort_by { |rgb,i,j| i }

	# facets ordered by saturation
	# to process them by likelineess of finding the right color
	sat_thr = 0.2

	# high sat descending
	facets = tiles.select   { |rgb,i,j| j != 4 }
				   .sort_by { |rgb,i,j| 
				   			  closest = centers.map { |c_rgb,k,l|
				   									   # hsv_dist(c_rgb.to_HSV, rgb.to_HSV)
				   									   dist2( c_rgb, rgb)
				   									   } 
				   			  # (closest[0] - closest[1]).abs * ( closest[0] - closest[2])
				   			  rgb.to_HSV[1]
				   			}
				   .reverse


	puts "centers face order: "
	puts centers.map   { |rgb,i,j| rgb.to_c + " [#{i},#{j}] " }

	puts "facets H[S]V order: "
	puts facets.map    { |rgb,i,j| rgb.to_c + " [#{i},#{j}] "  + rgb.to_hsv.to_s }


	cube = { f_per_c:     [8]*6,		# total number of colors available
			 edge_color:  [4]*6,		# split by edge/corner
			 corner_color:[4]*6,
			 constraint:  {},
			 match:       []
			}

	match( facets, centers, cube)
rescue => x
	puts "#{x}"
	puts "== Nothing found"
end

#exit 0

puts "cube:"
cube = do_match( faces)



cube[:match].each { |k| puts k.join(" ") }

puts "Rgb input:"
ff = faces.map { |f| cube_to_s( f) }
puts concat_2d( ff, "  " )


# display result
centers = tiles.select { |c,hsv,i,j| j == 4 }
               .sort_by { |c,hsv,i,j|  i }

final = []
cube[:match].each { |f,i,j,ci,c, compact|
	final[i] ||= []
	final[i] << [f,i,j,ci,c]
}
centers.each  { |c,hsv,i,j| 
				final[i] << [c,i,j,i,c] }



puts "with centers colors"
new_faces = final.map { |a| 
						a.sort_by { |f,i,j,ci,c| j }
						  .map    { |f,i,j,ci,c| c }
						    .each_slice(3)
					        .map { |x| x.join("") }
					        .join("\n")
						}

puts concat_2d( new_faces, "  " )

# express as face number
digit = final.map { |a| 
						a.sort_by { |f,i,j,ci,c| j }
						  .map    { |f,i,j,ci,c| ci }
						}

puts "expected:"
e_digit = $expected.map { |a| 
						a.each_slice(3)
					        .map { |x| x.join(" ") }
					        .join("\n")
						}
puts concat_2d( e_digit, "   " )

# puts final.map { |a| 
# 						a.sort_by { |f,i,j,ci,c| j }
# 						  .map    { |f,i,j,ci,c| ci }
# 				}
# 			.map { |a| "#{a}"}
#    		    .join(" <<\n")

puts "checked:"
testc = $expected.zip( digit)	
		 .map { |f1,f2| 
		 		f1.zip( f2)
		 			.map { |ff1,ff2|
		 					red = [ 200, 100, 100 ]
		 					if ff1 == ff2 then "o"
		 					else "X".bg red end } 
		 		}

c_digit = testc.map { |a| 
						a.each_slice(3)
					        .map { |x| x.join(" ") }
					        .join("\n")
						}
puts concat_2d( c_digit, "   " )