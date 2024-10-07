import torch
import torch.nn as nn
import numpy as np

from models.encoding import get_encoder

class FeatureNet(nn.Module):
    def __init__(self, config, input_ch=4, hidden_dim=32, num_layers=3, final_dim=22):
        super(FeatureNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.final_dim = final_dim

        self.model = self.get_model()
    
    def forward(self, input_feat):
        return self.model(input_feat)
    
    def get_model(self):
        net =  []
        for l in range(self.num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = self.hidden_dim
            
            if l == self.num_layers - 1:
                out_dim = self.final_dim 
            else:
                out_dim = self.hidden_dim
            
            if l == self.num_layers - 1:
                net.append(nn.Linear(in_dim, out_dim, bias=False))
            else:
                net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != self.num_layers - 1:
                net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(net))

class FeatureDecoder(nn.Module):
    def __init__(self, config, input_ch=3):
        super(FeatureDecoder, self).__init__()
        self.config = config

        self.bounding_box = torch.from_numpy(np.array(self.config['scene']['bound']))
        dim_max = (self.bounding_box[:,1] - self.bounding_box[:,0]).max()
        self.resolution_sdf = int(dim_max / self.config['scene']['voxel_sdf'])

        print('bounding_box: ', self.bounding_box)
        print('resolution_sdf: ', self.resolution_sdf)

        self.encoding, self.embed_dim = get_encoder(config['decoder']['enc'], desired_resolution=self.resolution_sdf)
        self.feature_net = FeatureNet(config, 
                input_ch=self.embed_dim, 
                hidden_dim=config['decoder']['hidden_dim'], 
                num_layers=config['decoder']['num_layers'],
                final_dim=config['decoder']['final_dim'])
            
    def forward(self, pos):
        pos = (pos - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        embed = self.encoding(pos).cuda()
        feature = self.feature_net(embed) 
        feature = feature / feature.norm(dim=-1, keepdim=True)
        
        return feature