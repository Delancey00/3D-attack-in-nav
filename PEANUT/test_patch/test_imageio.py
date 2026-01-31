import imageio
import glob
import os
import re
image_folder = '/home/wmy/project/Physical-Attacks-in-Embodied-Nav/PEANUT/test_patch/result/dump/debug/episodes/thread_0/eps_0'
video_name = '/home/wmy/project/Physical-Attacks-in-Embodied-Nav/video水池花瓶.mp4'
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

def numerical_sort(value):
    # Extract all numbers and return as a tuple for sorting
    numbers = re.findall(r'\d+', value)
    return [int(num) for num in numbers] if numbers else [0]

images.sort(key=numerical_sort)
writer = imageio.get_writer(video_name, fps=5)
for image in images:
    print(os.path.join(image_folder, image))
    writer.append_data(imageio.imread(os.path.join(image_folder, image)))
writer.close()
