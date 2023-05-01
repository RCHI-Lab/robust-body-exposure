import subprocess
import argparse
import os.path as osp
from pathlib import Path

parser = argparse.ArgumentParser(description='')
parser.add_argument('--subject-dir', type=str, default='TEST')
parser.add_argument('--pose-dir', type=str, default='TEST')
parser.add_argument('--manikin', type=str, default='0')
args = parser.parse_args()

subprocess.call(['python', './code/capture_blanket/capture_and_merge_pcds.py', '--subject-dir', args.subject_dir, '--pose-dir', args.pose_dir, '--manikin', args.manikin])
