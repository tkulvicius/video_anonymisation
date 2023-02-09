# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 08:50:41 2021

@author: Tomas Kulvicius
"""

# From Python
# It requires OpenCV installed for Python
import cv2
import sys
from sys import platform
import os
import subprocess as sp
import numpy as np
import time
# import argparse

### BEGIN PARAMETERS ==========================================================

# Folder where OpenPose is installed
OP_FOLDER = 'C:/Program Files/openpose'
# OP_FOLDER = '/home/tkulvicius/openpose/build'

# Folder where OpenPose model are stored
OP_MODELS = 'C:/Program Files/openpose/models'
# OP_MODELS = '/home/tkulvicius/openpose/models'

# Folder where FFMPEG is installed
# FFMPEG_BIN = 'C:/Program Files/ffmpeg/bin/ffmpeg.exe' # on Windows
FFMPEG_BIN = 'ffmpeg' # on Linux

# Input folder where video data is stored
video_path = 'samples/'

# Output folder where OpenPose skeleton will be stored
out_path_skeleton = 'skeleton/'

# Output folder where anonymised video will be stored
out_path_video = 'video/'

# Video resolution, frame rate and number of frames in the video
WIDTH = 1920
HEIGHT = 1080
FRAME_RATE = 50
NUM_FRAMES = 250

# Video codec for output video
# VIDEO_CODEC = 'XVID'
VIDEO_CODEC = 'mp4v'

# Parameters for face blurring mask
BLUR_LEVEL = 25
NOISE_LEVEL = 25
BLUR_WND_SIZE = 75 # half of the width of the elipse mask in pixels
ASPECT_RATIO = 0.45 # 1:X (height of the elipse mask with respect to whidth)

### END PARAMETERS ============================================================


# Import OpenPose
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        # sys.path.append(dir_path + '/../bin/python/openpose/Release');
        # os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../x64/Release;' +  dir_path + '/../bin;'
        
        sys.path.append(OP_FOLDER + '/bin/python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + OP_FOLDER + '/x64/Release;' +  OP_FOLDER + '/bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(OP_FOLDER + '/python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


def read_frame_from_pipe(pipe):
    raw_img = pipe.stdout.read(WIDTH*HEIGHT*3)
    img = np.frombuffer(raw_img, dtype='uint8')
    img = img.reshape((HEIGHT,WIDTH,3))
    # img = img[...,::-1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

def get_body_keypoints(opWrapper, img):
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum

def get_face_center_point(datum, x_old, y_old):
    x = datum.poseKeypoints[0,(0,15,16,17,18),0]
    y = datum.poseKeypoints[0,(0,15,16),1]
    p = datum.poseKeypoints[0,(0,15,16),2]
    #print(x,y,np.mean(p))
    if np.mean(p)<0.35:
    	x = x_old
    	y = y_old
    else:
    	x = int(np.mean(x[x>0]))
    	y = int(np.mean(y[y>0]))
    return x, y

def blur_face(img, x, y):
    img_blurred = img.copy()
    img_final = img.copy()
    w = BLUR_WND_SIZE
    h = int(ASPECT_RATIO*BLUR_WND_SIZE)
    img_blurred[y-h:y+h, x-w:x+w] = cv2.blur(img_blurred[y-h:y+h,x-w:x+w],(BLUR_LEVEL, BLUR_LEVEL)) + NOISE_LEVEL*np.random.random((int(2*h),int(2*w),3))
    mask = np.zeros((HEIGHT,WIDTH), dtype="uint8")
    cv2.ellipse(mask,(int(x),int(y)),(w,h),0,0,360,255,-1)
    img_final[mask==255] = img_blurred[mask==255]
    return img_final
    

def main():

	# Custom Params (refer to include/openpose/flags.hpp for more parameters)
	params = dict()
	# params["model_folder"] = "../../../models/"
	params["model_folder"] = OP_MODELS
	params["number_people_max"] = 1

	# Starting OpenPose
	opWrapper = op.WrapperPython()
	opWrapper.configure(params)
	opWrapper.start()

	video_files = os.listdir(video_path)
	print(video_files)

	# Process all videos in the folder "video_path"
	for i in range(len(video_files)):

		print('\n Video file: ', i, video_files[i], '\n')

		fname = video_path + video_files[i]
		command = [FFMPEG_BIN, '-i', fname, '-r', str(FRAME_RATE), '-f',
		'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-']

		# Open pipe
		pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)

		body = np.zeros((NUM_FRAMES,int(25*3)), dtype='float32')

		out = cv2.VideoWriter(out_path_video + video_files[i],
		cv2.VideoWriter_fourcc(*VIDEO_CODEC),
		FRAME_RATE, (WIDTH,HEIGHT))
        
		x_old, y_old = (960,200)
		
		# Process frames in video
		for f in range(NUM_FRAMES):
		    
			#print('Frame number: ', f+1)

			# Read frame
			img = read_frame_from_pipe(pipe)

			# Get body keypoints
			datum = get_body_keypoints(opWrapper, img)

			# # Print keypoints
			# print("Body keypoints: \n" + str(datum.poseKeypoints))
		    
			if datum.poseKeypoints is not None:
				# Store keypoints
				body[f,:] = np.reshape(datum.poseKeypoints.squeeze(),(1,int(25*3)))

				# # Display skeleton image
				#cv2.imshow("Skeleton", datum.cvOutputData)
				#cv2.waitKey(200)
					    
				# Find eye-nose-ears center point (we assume BODY_25 model)
				x, y = get_face_center_point(datum, x_old, y_old)
			else:
				x, y = (x_old, y_old)
				
			# Apply filter
			if f>0:
				x = int(0.5*x + 0.5*x_old)
				y = int(0.5*y + 0.5*y_old)
			x_old, y_old = (x, y)
		    
			# Blur face
			img_blurred = blur_face(img, x, y)

			# Write frame to video
			out.write(img_blurred)

			# # Display blurred face image
			#cv2.imshow("Face", img_blurred)
			#cv2.waitKey(100)
		    
		# Throw away the data in the pipe's buffer
		pipe.stdout.flush()
		pipe.terminate()

		# Release VideoWriter
		out.release()
		
		# Save skeleton
		np.savetxt(out_path_skeleton + os.path.splitext(video_files[i])[0] + '.txt', body)


if __name__ == '__main__':
	t0 = time.time()
	main()
	print(time.time()-t0)
	
	


    
