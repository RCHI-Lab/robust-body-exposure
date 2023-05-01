import subprocess
import argparse
import os.path as osp
from pathlib import Path
import time
import sys
sys.path.insert(0, '/home/kpputhuveetil/git/vBM-GNNdev/assistive-gym-fem')
from assistive_gym.envs.bu_gnn_util import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--subject-id', type=str, default='TEST', required=True)
parser.add_argument('--pose-num', type=str, default='TEST', required=True)
parser.add_argument('--tl-code', type=str, required=True)
# to start at a different section of the pipeline
parser.add_argument('--start-here', type=int, default=0)
parser.add_argument('--sub-id', type=int, default=None)
parser.add_argument('--manikin', type=str, default='0')
parser.add_argument('--approach', type=str, default='dyn')
args = parser.parse_args()

t0 = time.time()

target_limb_code = args.tl_code if args.tl_code != 'random' else str(randomize_target_limbs([2, 4, 5, 12, 13, 14, 15]))
sub_id = int(t0) if args.sub_id is None else args.sub_id

print('===============================================================================')
print(f'          SUBJECT: {args.subject_id},  POSE: {args.pose_num}, TL CODE: {target_limb_code},  SUB_ID: {sub_id}')
print('===============================================================================')

subject_dir = osp.join('/home/kpputhuveetil/git/vBM-GNNdev/real_world/STUDY_DATA', f'subject_{args.subject_id}') 
pose_dir = osp.join(subject_dir, f'pose_{args.pose_num}_TL{target_limb_code}_{sub_id}')

Path(pose_dir).mkdir(parents=True, exist_ok=True)

if args.start_here == 0:
    print('---------------------------------------------')
    print()
    print('Make sure the subject is uncovered')
    print('Press ENTER to capture the subject\'s pose...')
    input()
    print('---------------------------------------------')

    subprocess.call(['python', './code/capture_human_pose.py', '--subject-dir', subject_dir, '--pose-dir', pose_dir, '--manikin', args.manikin])

if args.start_here <= 1:
    print('---------------------------------------------')
    print()
    print('Cover the subject with the blanket')
    print('Press ENTER to capture blanket point cloud...')
    input()
    print('---------------------------------------------')


    subprocess.call(['python', './code/capture_blanket.py', '--subject-dir', subject_dir, '--pose-dir', pose_dir, '--manikin', args.manikin])

if args.start_here <= 2:
    print('---------------------------------------------')
    print()
    print('Prep the robot for action')
    print('Press ENTER to compute and action to uncover the target...')
    input()
    print('---------------------------------------------')

    subprocess.call(['python', './code/get_action.py', '--subject-dir', subject_dir, '--pose-dir', pose_dir, '--tl-code', target_limb_code, '--manikin', args.manikin, '--approach', args.approach])


print((time.time() - t0)/60)

if args.start_here <= 3:
    print('---------------------------------------------')
    print()
    print('Robot is ready to execute the action')
    print('Recording will begin...')
    print()
    print('---------------------------------------------')

    subprocess.call(['python', './code/above_bed_video.py', '--pose-dir', pose_dir])