# Rubik cube finder

Base on Andrej Karpathy python code and article.
Big thank to him for sharing his work.

I move some (not all) part of initial script to ruby.
Mainly because I was unable to make the python script work with opencv 2.4,
and also because I wanted to better understand how it works (and finally because
I am not much fan of python itself).

The port only cover detection and still has some bug.
I try to split the code to have cleaner  dependencies between
parts.

# Install

this miss a bundler config
gem used: 
ropencv

# installing opencv

ropencv need version 2.4 of opencv
need to have library in path

then:
gem install ropencv

### TODO:

- fix detection code matched_0
(probably a mistake during the port)

- fisnish color detection
current work is incomplete

### Sources:

cubefinder.rb -> rough equivalent of cubefinder.py
cube_server.rb -> rest API to query the detection from an other program

