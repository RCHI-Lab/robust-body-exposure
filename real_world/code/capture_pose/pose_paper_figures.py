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
import tkinter as tk
from PIL import Image, ImageTk

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 

human_pose_joints = {
    18:mp_pose.PoseLandmark.RIGHT_INDEX, 
    16:mp_pose.PoseLandmark.RIGHT_ELBOW, 
    14:mp_pose.PoseLandmark.RIGHT_SHOULDER,
    39:mp_pose.PoseLandmark.RIGHT_ANKLE,
    36:mp_pose.PoseLandmark.RIGHT_KNEE, 
    35:mp_pose.PoseLandmark.RIGHT_HIP,
    28:mp_pose.PoseLandmark.LEFT_INDEX,
    26:mp_pose.PoseLandmark.LEFT_ELBOW,
    24:mp_pose.PoseLandmark.LEFT_SHOULDER,
    46:mp_pose.PoseLandmark.LEFT_ANKLE,
    43:mp_pose.PoseLandmark.LEFT_KNEE,
    42:mp_pose.PoseLandmark.LEFT_HIP,
    8:None, 
    32:mp_pose.PoseLandmark.NOSE
}

connected_joints = {
    18:[16], 
    16:(18, 14), 
    14:(16, 8),
    39:[36],
    36:(39, 35), 
    35:(36, 8),
    28:[26],
    26:(28, 24),
    24:(26, 8),
    46:[43],
    43:(46, 42),
    42:(42, 8),
    8:(14, 35, 24, 42), 
    32:None
}

limbs = {
    'r_fore_arm':[18, 16], 
    'r_up_arm':[16, 14], 
    'r_chest':[14, 8],
    'r_low_leg':[39, 36],
    'r_thigh':[36, 35], 
    'r_pelvis':[35, 8],
    'l_fore_arm':[28, 26],
    'l_up_arm':[26, 24],
    'l_chest':[24, 8],
    'l_low_leg':[46, 43],
    'l_thigh':[43, 42],
    'l_pelvis':[42, 8],
}



def transform_pose_to_origin_px(origin_px, human_pose_px, image):
    transformed_joints = []
    pose_trans_img = image.copy()
    for joint in human_pose_px:
        joint_trans = joint - origin_px
        transformed_joints.append(joint_trans)
        cv2.circle(pose_trans_img, (int(joint_trans[0]), int(joint_trans[1])), 4, (0, 0, 255), -1)
    
    return np.array(transformed_joints), pose_trans_img

def get_human_pose_px(mp_detection, w, h, image):
    joint_coordinates = {}
    for id, joint in human_pose_joints.items():
        if joint is not None:
            x = mp_detection.pose_landmarks.landmark[joint].x * w
            y = mp_detection.pose_landmarks.landmark[joint].y * h
            joint_coordinates[id] = (x,y)
        else:
            x1 = joint_coordinates[14][0]    # right shoulder
            y1 = joint_coordinates[14][1]
            x2 = joint_coordinates[42][0]   # left hip
            y2 = joint_coordinates[42][1]

            x3 = joint_coordinates[24][0]    # left shoulder
            y3 = joint_coordinates[24][1]
            x4 = joint_coordinates[35][0]    # right hip
            y4 = joint_coordinates[35][1]

            line1 = Line(Point(x1, y1), Point(x2, y2))
            line2 = Line(Point(x3, y3), Point(x4, y4))
            upperchest = line1.intersection(line2)[0] # upperchest
            joint_coordinates[id] = upperchest
    
    pose_img = image.copy()
    # print(joint_coordinates)

    # print(adjusted_joint_coords)
    adjust_pose(joint_coordinates, image)
    with open('temp.pkl', 'rb') as f:
        adjusted_joint_coords = pickle.load(f)
    # print(adjusted_joint_coords)
    # for c in joint_coordinates:
    #     cv2.circle(pose_img, (int(c[0]), int(c[1])), 4, (0, 0, 255), -1)
    
    return pose_img, list(adjusted_joint_coords.values())

# --------------------------------------------------------------

def adjust_pose(initial_joint_coordinates, image):

    def on_closing():
        with open('temp.pkl', 'wb') as f:
            pickle.dump(canvas.adjusted_joint_coords, f)
        root.destroy()
    
    def draw_connections(id=None):
        for limb, ids in limbs.items():
            if id is None or id in ids:
                id1, id2 = ids
                x1, y1 = canvas.adjusted_joint_coords[id1]
                x2, y2 = canvas.adjusted_joint_coords[id2]
                canvas.delete(canvas.line_ids[limb])
                canvas.line_ids[limb] = canvas.create_line(int(x1), int(y1), int(x2), int(y2), width=10, fill='green2')

    def drag(event):
        x = event.x + event.widget.winfo_x()
        y = event.y + event.widget.winfo_y()
        event.widget.place(x=x, y=y, anchor="center")
        draw_connections(event.widget.id)
        canvas.adjusted_joint_coords[event.widget.id] = (x, y)


    root = tk.Tk() 

    scale = 1
    canvas = tk.Canvas(root, width=1280*scale, height=720*scale)
    canvas.pack()

    bg= ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)))
    canvas.create_image(0,0,image=bg, anchor="nw")
    canvas.adjusted_joint_coords = initial_joint_coordinates.copy()
    canvas.line_ids = {
        'r_fore_arm':None, 'r_up_arm':None, 'r_chest':None,
        'r_low_leg':None, 'r_thigh':None, 'r_pelvis':None,
        'l_fore_arm':None, 'l_up_arm':None, 'l_chest':None,
        'l_low_leg':None, 'l_thigh':None, 'l_pelvis':None
        }

    points = []
    # initial_joint_coordinates = [initial_joint_coordinates[0]]
    for id, joint in initial_joint_coordinates.items():
        point = tk.Canvas(root, width=30, height=30, bg="green2")
        x, y = int(joint[0]), int(joint[1])
        point.id = id
        point.place(x=x, y=y, anchor="center")
        point.bind("<B1-Motion>", drag)
        points.append(point)
    
    for limb in limbs:
        draw_connections()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

# --------------------------------------------------------------
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

    save_dir = '/home/kpputhuveetil/Desktop/paper_figures/real_pose_detect'
    subject_dir = '/home/kpputhuveetil/git/vBM-GNNdev/real_world/STUDY_DATA/subject_3'
    pose_dir = osp.join(subject_dir,'pose_0_TL2_1674850628')

    image = cv2.imread(osp.join(save_dir, 'pose_est_uncov.png'))
    # image = cv2.imread('/home/kpputhuveetil/git/vBM-GNNdev/real_world/STUDY_DATA/subject_TEST/pose_0/uncovered_rgb.png')
    if image is None:
        print('IMAGE NOT FOUND')
    
    with open(osp.join(pose_dir,'sim_origin_data.pkl'),'rb') as f:
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
    # cv2.imwrite(osp.join(args.pose_dir,'mp_estimated_pose.png'), est)
    # cv2.waitKey(0)

    pose_img, human_pose_px = get_human_pose_px(mp_detection, image_width, image_height, image)
    human_pose_px = np.array(human_pose_px).reshape(-1,2)
    # cv2.imshow("Image",pose_img)
    # cv2.waitKey(0)

    human_pose_px_transformed, pose_trans_img = transform_pose_to_origin_px(origin_px, human_pose_px, image)
    human_m = human_pose_px_transformed*m2px_scale
    human_m_correct_axis = human_m.copy()
    human_m_correct_axis[:, [1, 0]] = human_m_correct_axis[:, [0, 1]]
    print(human_m_correct_axis)

    # with open(osp.join(sim_origin_dir,'human_pose.pkl'), 'wb') as f:
    #     pickle.dump(human_m_correct_axis, f)
    
    # with open(osp.join(args.pose_dir,'human_pose_TEST.pkl'), 'wb') as f:
    #     pickle.dump(human_m, f)


    with open(osp.join(subject_dir,'body_info.pkl'),'rb') as f:
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
        cv2.circle(image, (int(point[0]), int(point[1])), 6, (255, 186, 71), -1)
    cv2.imshow("Image",image)
    cv2.imwrite(osp.join(save_dir,'all_body_points_over_rgb.png'), image)
    # cv2.waitKey(0)
