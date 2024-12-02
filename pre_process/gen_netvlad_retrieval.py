import argparse
import numpy as np
import os
from pathlib import Path

import torch

import sys
sys.path.append('../submodules/Hierarchical-Localization')

from hloc import extract_features, pairs_from_retrieval
from hloc.utils.io import list_h5_names


def generate_retrieval_file(descriptors, db_descriptors, out_path, num_matched=10):
    db_descriptors = [db_descriptors]
    name2db = {n: i for i, p in enumerate(db_descriptors) for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())

    # print('name2db:', name2db)
    # print('db_names_h5:', db_names_h5)

    query_names_h5 = list_h5_names(descriptors)
    db_names = pairs_from_retrieval.parse_names(None, None, db_names_h5)
    query_names = pairs_from_retrieval.parse_names(None, None, query_names_h5)

    # print('query_names:', query_names)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    db_desc = pairs_from_retrieval.get_descriptors(db_names, db_descriptors, name2db)
    query_desc = pairs_from_retrieval.get_descriptors(query_names, descriptors)
    similarity = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))

    sims, ind1 = similarity.topk(num_matched, dim=1, largest=True)

    with open(out_path, "w") as f:
        for i in range(ind1.shape[0]):
            f.write(query_names[i])
            for j in range(ind1.shape[1]):
                f.write(' ')
                f.write(db_names[ind1[i, j]])
            f.write('\n')
    
if __name__ == '__main__':
    '''
    Usage:
    python run_retrieval.py --scene_path /mnt/nas_7/datasets/nerf-loc/replica/room_0 --save_path /mnt/nas_10/group/hongjia
    python run_retrieval.py --scene_path /mnt/nas_54/datasets/nerf-loc/12scenes/office2/5b --save_path /mnt/nas_10/group/hongjia
    '''
    parser = argparse.ArgumentParser(description='Arguments for running the NICE-SLAM/iMAP*.')
    parser.add_argument('--scene_path', type=str, help='Path to query_path.')
    parser.add_argument('--save_path', type=str, help='Path to save query_path.')
    args = parser.parse_args()

    dataset_name = 'replica-nerf' # replica-nerf 12-scenes


    # NOTE:
    # 1. the database (db) images should be the sampled images.
    if dataset_name == 'replica-nerf':
        query_path = Path(os.path.join(args.scene_path, 'Sequence_2', 'rgb'))
        db_path = Path(os.path.join(args.scene_path, 'Sequence_1', 'rgb_skip5'))
        query_out_path = Path(os.path.join(args.save_path, 'Sequence_2'))
        db_out_path = Path(os.path.join(args.save_path, 'Sequence_1'))

        retrieval_conf = extract_features.confs['netvlad']
        query_retrieval_path = extract_features.main(retrieval_conf, query_path, query_out_path)
        db_retrieval_path = extract_features.main(retrieval_conf, db_path, db_out_path)
        query_retrieval_path = Path(os.path.join(args.save_path, 'Sequence_2', 'global-feats-netvlad.h5'))
        db_retrieval_path = Path(os.path.join(args.save_path, 'Sequence_1', 'global-feats-netvlad.h5'))

        db_out_path = Path(os.path.join(args.save_path, 'Sequence_2', 'netvlad_retrieval.txt'))
        generate_retrieval_file(query_retrieval_path, db_retrieval_path, db_out_path)

    elif dataset_name == '12-scenes':
        query_path = Path(os.path.join(args.scene_path, 'test', 'colors'))
        db_path = Path(os.path.join(args.scene_path, 'train', 'colors_skip5'))
        query_out_path = Path(os.path.join(args.save_path, 'test'))
        db_out_path = Path(os.path.join(args.save_path, 'train'))

        retrieval_conf = extract_features.confs['netvlad']
        query_retrieval_path = extract_features.main(retrieval_conf, query_path, query_out_path)
        db_retrieval_path = extract_features.main(retrieval_conf, db_path, db_out_path)
        query_retrieval_path = Path(os.path.join(args.save_path, 'test', 'global-feats-netvlad.h5'))
        db_retrieval_path = Path(os.path.join(args.save_path, 'train', 'global-feats-netvlad.h5'))

        db_out_path = Path(os.path.join(args.save_path, 'test', 'netvlad_retrieval.txt'))
        generate_retrieval_file(query_retrieval_path, db_retrieval_path, db_out_path)