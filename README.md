# Video anonymisation

## Description of the program:

This program is used to anonymise faces in videos. It will apply blurred mask on the face (around the eye region) in each video frame. NOTE that this program uses OpenPose to find face keypoints. You have to use your own videos. It can run on both Windows and Linux OS.

This program was tested on Windows 11 OS with Python 3.7.0, Numpy 1.21.5, and OpenCV2 4.5.5.

## Requirements:
### OpenPose
You can download OpenPose from

https://github.com/CMU-Perceptual-Computing-Lab/openpose

### OpenCV 2
You can download OpenCV 2 from

https://opencv.org/releases/

### FFMPEG
You can download FFMPEG from

https://ffmpeg.org/download.html

## Parameters:
To run the program you will first need to adapt parameters in the Python script:

### Folder where OpenPose is installed

OP_FOLDER = 'xxx'

### Folder where OpenPose model are stored

OP_MODELS = 'xxx'

### Folder where FFMPEG is installed (change to your path)

FFMPEG_BIN = 'C:/Program Files/ffmpeg/bin/ffmpeg.exe' (on Windows)

FFMPEG_BIN = 'ffmpeg' (on Linux)

### Input folder where video data is stored

video_path = 'xxx'

### Output folder where OpenPose skeleton will be stored

out_path_skeleton = 'skeleton/'

### Output folder where anonymised video will be stored

out_path_video = 'video/'

### Video resolution, frame rate and number of frames in the video

WIDTH = 1920

HEIGHT = 1080

FRAME_RATE = 50

NUM_FRAMES = 250

### Video codec for output video

VIDEO_CODEC = 'mp4v'

### Parameters for face blurring mask

BLUR_LEVEL = 25

NOISE_LEVEL = 25

BLUR_WND_SIZE = 75 (half of the width of the elipse mask in pixels)

ASPECT_RATIO = 0.45 # (height of the elipse mask with respect to whidth)


