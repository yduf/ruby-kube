# Rest API for cubefinder
require 'celluloid'

#


# start an actor to collect detection state
class CubeStatus
	include Celluloid

	attr_reader :face

	def initialize()
		@face = []
	end

	# take a snapshot of cube status detection
	# return any pending action
	def set_status!( s)
		@s = s
	end

	# retrieve last status
	def get_status()
		@s
	end

	# move current cube status / info cube face structure
	def snap_face!( face_id)
		@face[face_id] = @s
	end

	# infer face color from 
	# face registered
	def cube_color? 

	end


end # CubeStatus


$cube = CubeStatus.new

# start a thread for cube finder detection
require_relative 'cubefinder.rb'
Thread.new do 
	init_ui
	track_webcam( cam = 0) { |tracker,key_press|
		action = $cube.set_status tracker
	}
end

# make sure we have a first state set
sleep 0.1

# start a rest server to distribute state
require 'grape'

class API < Grape::API
  get :hello do
    { hello: $cube.get_status.good? }
  end
end

Rack::Handler::WEBrick.run API
