import cv2
import cv2.aruco as aruco
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from datetime import date
from sympy import Point, Line
import sys
sys.path.insert(0, '/home/kpputhuveetil/git/vBM-GNNdev/bm-gnns')
from assistive_gym.envs.bu_gnn_util import *
import matplotlib.pyplot as plt
import argparse
import os.path as osp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 

human_pose_joints = [
    mp_pose.PoseLandmark.RIGHT_INDEX, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_HIP,
    None, mp_pose.PoseLandmark.NOSE
]


def transform_pose_to_origin_px(origin_px, human_pose_px, image):
    transformed_joints = []
    pose_trans_img = image.copy()
    for joint in human_pose_px:
        joint_trans = joint - origin_px
        transformed_joints.append(joint_trans)
        cv2.circle(pose_trans_img, (int(joint_trans[0]), int(joint_trans[1])), 4, (0, 0, 255), -1)
    
    return np.array(transformed_joints), pose_trans_img

def get_human_pose_px(mp_detection, w, h, image):
    joint_coordinates = []
    for joint in human_pose_joints:
        if joint is not None:
            x = mp_detection.pose_landmarks.landmark[joint].x * w
            y = mp_detection.pose_landmarks.landmark[joint].y * h
            joint_coordinates.append((x,y))
        else:
            x1 = joint_coordinates[2][0]    # right shoulder
            y1 = joint_coordinates[2][1]
            x2 = joint_coordinates[11][0]   # left hip
            y2 = joint_coordinates[11][1]

            x3 = joint_coordinates[8][0]    # left shoulder
            y3 = joint_coordinates[8][1]
            x4 = joint_coordinates[5][0]    # right hip
            y4 = joint_coordinates[5][1]

            line1 = Line(Point(x1, y1), Point(x2, y2))
            line2 = Line(Point(x3, y3), Point(x4, y4))
            upperchest = line1.intersection(line2)[0] # upperchest
            joint_coordinates.append(upperchest)
    
    pose_img = image.copy()
    for c in joint_coordinates:
        cv2.circle(pose_img, (int(c[0]), int(c[1])), 4, (0, 0, 255), -1)
    
    return pose_img, joint_coordinates

def show_estimated_pose(image):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # Convert the BGR image to RGB and process it with MediaPipe Pose.
        mp_detection = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not mp_detection.pose_landmarks:
            return None
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            mp_detection.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)
        return annotated_image, mp_detection

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--subject-dir', type=str, default='TEST')
    parser.add_argument('--pose-dir', type=str, default='TEST')
    args = parser.parse_args()

    image = cv2.imread(osp.join(args.pose_dir, 'uncovered_rgb.png'))
    if image is None:
        print('IMAGE NOT FOUND')
    
    with open(osp.join(args.pose_dir,'sim_origin_data.pkl'),'rb') as f:
        data = pickle.load(f)
        dist = data['dist']
        mtx = data['mtx']
        centers_px = data['centers_px']
        centers_m = data['centers_m']
        origin_px = data['origin_px']
        origin_m = data['origin_m']
        m2px_scale = data['m2px_scale']


    image_height, image_width, _ = image.shape

    #pose landmarks are w.r.t bottom left corner
    est, mp_detection = show_estimated_pose(image)
    cv2.imshow("Image",est)
    cv2.imwrite(osp.join(args.pose_dir,'mp_estimated_pose.png'), est)
    cv2.waitKey(0)

    pose_img, human_pose_px = get_human_pose_px(mp_detection, image_width, image_height, image)
    human_pose_px = np.array(human_pose_px).reshape(-1,2)
    # cv2.imshow("Image",pose_img)
    # cv2.waitKey(0)

    human_pose_px_transformed, pose_trans_img = transform_pose_to_origin_px(origin_px, human_pose_px, image)
    human_m = human_pose_px_transformed*m2px_scale
    human_m_correct_axis = human_m.copy()
    human_m_correct_axis[:, [1, 0]] = human_m_correct_axis[:, [0, 1]]
    print(human_m_correct_axis)

    with open(osp.join(args.pose_dir,'human_pose.pkl'), 'wb') as f:
        pickle.dump(human_m_correct_axis, f)
    
    # with open(osp.join(args.pose_dir,'human_pose_TEST.pkl'), 'wb') as f:
    #     pickle.dump(human_m, f)


    with open(osp.join(args.subject_dir,'body_info.pkl'),'rb') as f:
        body_info = pickle.load(f)
    all_body_points = get_body_points_from_obs(human_pose=human_m_correct_axis.astype(float), target_limb_code=4, body_info=body_info)[:,:2]

    # plt.figure()
    # # plt.scatter(human_m[:, 0], human_m[:, 1])
    # # plt.scatter(human_m_correct_axis[:, 0], human_m_correct_axis[:, 1])
    # plt.scatter(all_body_points[:,1], all_body_points[:,0])
    # plt.show()

    all_body_points[:, [1, 0]] = all_body_points[:, [0, 1]]
    all_body_points_px = all_body_points*(1/m2px_scale) + origin_px

    # plt.figure()
    # # plt.scatter(human_m[:, 0], human_m[:, 1])
    # # plt.scatter(human_m_correct_axis[:, 0], human_m_correct_axis[:, 1])
    # plt.scatter(all_body_points[:,0], all_body_points[:,1])
    # plt.show()

    # plt.figure()
    # # plt.scatter(human_m[:, 0], human_m[:, 1])
    # # plt.scatter(human_m_correct_axis[:, 0], human_m_correct_axis[:, 1])
    # plt.scatter(all_body_points_px[:,0], all_body_points_px[:,1])
    # plt.show()


    for point in all_body_points_px:
        cv2.circle(image, (int(point[0]), int(point[1])), 4, (255, 186, 71), -1)
    cv2.imshow("Image",image)
    cv2.imwrite(osp.join(args.pose_dir,'all_body_points_over_rgb.png'), image)
    cv2.waitKey(0)