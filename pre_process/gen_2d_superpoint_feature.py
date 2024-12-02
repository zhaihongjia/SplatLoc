import sys
sys.path.append('../submodules/Hierarchical-Localization')

from hloc import extract_save_sp_feature
from tqdm import tqdm
from pathlib import Path
import argparse
from hloc.utils.parsers import parse_retrieval
from hloc.utils.io import get_keypoints, get_matches

import cv2
import numpy as np

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--out_dir', type=Path, required=True)
    args = parser.parse_args()

    images = args.image_dir
    outputs= args.out_dir 

    feature_conf = extract_save_sp_feature.confs['superpoint_inloc']
    features = extract_save_sp_feature.save_sp_feature(feature_conf, images, outputs)
