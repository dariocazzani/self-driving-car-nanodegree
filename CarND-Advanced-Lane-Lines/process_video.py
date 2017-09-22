from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
import cv2
import glob
import numpy as np

import sys
def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[{}] {}{} ...{}\r'.format(bar, percents, '%', suffix))
    sys.stdout.flush()

clip = VideoFileClip("project_video.mp4")
frames = int(clip.fps * clip.duration)
image_folder = "frames/"
video_file = 'processed_video.mp4'

for idx, frame in enumerate(clip.iter_frames()):
    progress(idx+1, frames)
    cv2.imwrite('{}frame_{:010d}.jpg'.format(image_folder, idx), frame)
print('')
