import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from autoencoder.dataset import Autoencoder_dataset
from autoencoder.model import Autoencoder
import argparse
from utils.config_utils import load_config
import yaml
from utils.dataset import load_dataset
from datetime import datetime
import numpy as np

from models.decoders import FeatureDecoder
import open3d as o3d

torch.autograd.set_detect_anomaly(True)

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    sim = torch.cosine_similarity(network_output, gt, dim=1)
    return 1 - sim.mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=41)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)
    config = load_config(args.config)

    # load decoder，dataset
    num_epochs = args.num_epochs

    dataset = load_dataset(config=config)
    train_dataset = Autoencoder_dataset(dataset.sparse_ply, dataset.sparse_feature)
    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=16, drop_last=False)
    num_batch = len(train_loader)
    lr = 0.001 # 0.001 0.05
    
    decoder = FeatureDecoder(config).cuda()
    trainable_parameters = [{'params': decoder.feature_net.parameters(), 'weight_decay': 1e-6, 'lr': lr},
                        {'params': decoder.encoding.parameters(), 'eps': 1e-15, 'lr': lr}]
        
    optimizer = torch.optim.Adam(trainable_parameters, betas=(0.9, 0.99))

    path = config["Dataset"]["dataset_path"].split("/")
    if config['Dataset']['type'] == 'replica':
        save_dir = os.path.join(config["Results"]["save_dir"], path[-2], path[-1], 'train_feat')
    elif config['Dataset']['type'] == '12scenes':
        save_dir = os.path.join(config["Results"]["save_dir"], path[-3], path[-2] + '_' + path[-1], 'train_feat')
    else:
        print('Dataset type should be replica or 12scenes')
        exit()
        
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(num_epochs)):
        decoder.train()
        batch_i = 1
        for pts, features in train_loader:
            xyz, feat = pts, features.cuda()
            outputs = decoder(xyz)
            
            cosloss = cos_loss(outputs, feat)
            loss = cosloss
            
            print('epoch: ', epoch, ' batch: {}/{}'.format(batch_i, num_batch), ' cosloss: ', cosloss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_i += 1

        # if epoch % 10 == 0:
            # torch.save(decoder.state_dict(), f'{save_dir}/{epoch}_ckpt.pth')
    torch.save(decoder.state_dict(), f'{save_dir}/ckpt.pth')

            