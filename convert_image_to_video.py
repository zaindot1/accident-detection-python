import cv2
import numpy as np
import glob
from PIL import Image
import datetime
import calendar

img_array = []

for filename in glob.glob('./object_detector/results/frames/*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

date = datetime.datetime.utcnow()
utc_time = calendar.timegm(date.utctimetuple())
video_name=str(utc_time)+'.avi'

out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
for i in range(len(img_array)):
    out.write(img_array[i])

out.release()