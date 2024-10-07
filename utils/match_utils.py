import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_solve(descriptors1, descriptors2,):
    assert descriptors1.shape[0] == descriptors2.shape[0]
    if descriptors1.shape[1] == 0 or descriptors2.shape[1] == 0:
        print('No descriptor for match')
        return np.zeros((3, 0))
    
    descriptors1, descriptors2 = descriptors1.cpu(), descriptors2.cpu()
    descriptors1 = torch.nn.functional.normalize(descriptors1, p=2, dim=0)
    descriptors2 = torch.nn.functional.normalize(descriptors2, p=2, dim=0)
    similarity = torch.matmul(descriptors1.t(), descriptors2)     # [N1, N2]

    similarity[similarity<0.4] = 0
    cost = 1 - similarity
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = torch.stack([torch.from_numpy(row_ind), torch.from_numpy(col_ind)], dim=1).permute([1, 0])
    sims = similarity[row_ind, col_ind]

    return matches, sims

class HungarianMatcher:
    def __init__(self):
        pass

    def __call__(self, data):
        required_keys = ['query_descs', 'train_descs']
        for key in required_keys:
            if key not in data:
                raise ValueError(key + ' not exist in input')

        matches, scores = hungarian_solve(data['query_descs'], data['train_descs'])
        return {
            'matches': matches,
            'scores': scores
        }