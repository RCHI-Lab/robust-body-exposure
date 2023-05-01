import subprocess
import argparse
import os.path as osp
from pathlib import Path

parser = argparse.ArgumentParser(description='Save spectral data')
parser.add_argument('--subject-dir', type=str, default='TEST')
parser.add_argument('--pose-dir', type=str, default='TEST')
parser.add_argument('--manikin', type=int, default=0)
args = parser.parse_args()


if args.manikin:
    if not osp.exists(osp.join(args.subject_dir, 'uncovered_rgb.png')):
        subprocess.call(['python', './code/capture_pose/get_uncovered_img_and_origin.py', '--save-dir', args.subject_dir])
    if not osp.exists(osp.join(args.subject_dir, 'body_info.pkl')):
        subprocess.call(['python', './code/capture_pose/get_body_info_from_img.py', '--subject-dir', args.subject_dir, '--pose-dir', args.subject_dir])
    if not osp.exists(osp.join(args.subject_dir, 'human_pose.pkl')):
        subprocess.call(['python', './code/capture_pose/mediapipe_pose_detect_gui.py', '--subject-dir', args.subject_dir, '--pose-dir', args.subject_dir])
else:
    subprocess.call(['python', './code/capture_pose/get_uncovered_img_and_origin.py', '--save-dir', args.pose_dir])
    # if osp.exists(osp.join(args.subject_dir, 'pose_0')) and not osp.exists(osp.join(args.subject_dir, 'body_info.pkl')):
    if not osp.exists(osp.join(args.subject_dir, 'body_info.pkl')):
        subprocess.call(['python', './code/capture_pose/get_body_info_from_img.py', '--subject-dir', args.subject_dir, '--pose-dir', args.pose_dir])
    subprocess.call(['python', './code/capture_pose/mediapipe_pose_detect_gui.py', '--subject-dir', args.subject_dir, '--pose-dir', args.pose_dir])

