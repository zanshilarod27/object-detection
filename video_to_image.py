#libraries
import cv2
import os

#input video - absolute path of the video
# OR paste video in the folder in which you are running this file
vidcap = cv2.VideoCapture('example.mp4')

image_path = 'path to save images'

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        cv2.imwrite(os.path.join(image_path, "frame"+str(count)+".jpg"), image)   # save frame/image as JPG file
    return hasFrames

sec = 0
frameRate = 0.01 #//it will capture image in each 0.01 second
count=1
success = getFrame(sec)

while success:
    #print("frame success: ", success)
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    #print('second: ', sec)
    success = getFrame(sec)