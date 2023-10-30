import cv2
import os
from natsort import natsorted

# path
input_folder = "./output/"
output_video = "./output/output_video.mp4"

# search for .png files
image_files = natsorted([f for f in os.listdir(input_folder) if f.endswith(".png")])

# get info about first image
first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
height, width, layers = first_image.shape

# mp4 file
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))

# 逐一将图像写入视频
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    frame = cv2.imread(image_path)
    out.write(frame)

out.release()
cv2.destroyAllWindows()
