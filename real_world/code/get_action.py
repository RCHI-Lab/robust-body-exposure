import subprocess
import argparse
import os.path as osp
from pathlib import Path

parser = argparse.ArgumentParser(description='')
parser.add_argument('--subject-dir', type=str, default='TEST')
parser.add_argument('--pose-dir', type=str, default='TEST')
parser.add_argument('--tl-code', type=str)
parser.add_argument('--manikin', type=str, default='0')
parser.add_argument('--approach', type=str, default='dyn')
args = parser.parse_args()

if args.approach == 'ppo':
    subprocess.call(['python', './code/get_action/get_ppo_action.py',
        '--subject-dir', args.subject_dir,
        '--pose-dir', args.pose_dir, 
        '--tl-code', args.tl_code, 
        '--manikin', args.manikin])
elif args.approach == 'naive':
    subprocess.call(['python', './code/get_action/get_naive_action.py',
        '--subject-dir', args.subject_dir,
        '--pose-dir', args.pose_dir, 
        '--tl-code', args.tl_code, 
        '--manikin', args.manikin])
else:
    subprocess.call(['python', './code/get_action/run_cma_over_dyn_model.py',
        '--subject-dir', args.subject_dir,
        '--pose-dir', args.pose_dir, 
        '--tl-code', args.tl_code, 
        '--manikin', args.manikin])
