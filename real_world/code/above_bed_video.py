import pyrealsense2 as rs
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import os.path as osp
import time

parser = argparse.ArgumentParser(description='')
parser.add_argument('--pose-dir', type=str, default='TEST')
args = parser.parse_args()


t0 = time.time()

# subject_dir = osp.join('/home/kpputhuveetil/git/vBM-GNNdev/real_world/STUDY_DATA', f'subject_{args.subject_id}') 
# pose_dir = osp.join(subject_dir, f'pose_{args.pose_num}')

pipeline = rs.pipeline()
config = rs.config()
config.enable_device('141722070195')
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

color_path = osp.join(args.pose_dir, f'eval_real_video.mp4')
print(color_path)
colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'mp4v'), 6, (1280, 720), 1)

pipeline.start(config)

try:
    print('preparing to capture video...')
    for i in tqdm(range(30)):
        pipeline.wait_for_frames()

    print()
    print('---------- VIDEO RECORDING IN PROGRESS ----------')
    
    while True:
        # k = cv2.waitKey(0)

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        #convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        
        colorwriter.write(color_image)
        
        cv2.imshow("Image",color_image)
        if cv2.waitKey(1) == ord("q"):
            break

finally:
    cv2.imwrite(osp.join(args.pose_dir, f'eval_real_rgb.png'), color_image)
    colorwriter.release()
    pipeline.stop()
    print('video and final eval image saved')
    print()
    print()