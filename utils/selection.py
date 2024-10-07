import numpy as np
import os
from tqdm import tqdm

def makePlyFile(xyzs, rgbs, fileName='makeply.ply'):
    '''Make a ply file for open3d.visualization.draw_geometries
    :param xyzs:    numpy array of point clouds 3D coordinate, shape (numpoints, 3).
    :param labels:  numpy array of point label, shape (numpoints, ).
    '''
    with open(fileName, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment PCL generated\n')
        f.write('element vertex {}\n'.format(len(xyzs)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(len(xyzs)):
            r, g, b = rgbs[i]
            x, y, z = xyzs[i]
            f.write('{} {} {} {} {} {}\n'.format(x, y, z, int(r), int(g), int(b)))

def inside_check(point_w, w2c, intrinsics):
    point_c = w2c[:3, :3] @ point_w + w2c[:3, 3]

    # depth check
    if point_c[2] < 0.01:
        return False, None
    # pixel check
    else:
        pixel = np.dot(intrinsics, point_c)
        pixel = pixel[:2] / pixel[2] # [w, h]
        if (pixel[0] < 640) and (pixel[0] > 0) and (pixel[1] < 480) and (pixel[1] > 0):
            return True, pixel
        else:
            return False, None

def ComputePerPointAngularSpan(pointInGlobal, wc2s, intrinsics):
    valids = 0
    H = np.zeros((3, 3))

    for i in range(wc2s.shape[0]):
        valid, pixel = inside_check(pointInGlobal, wc2s[i], intrinsics)
        if valid:
            valids += 1
            Ri = wc2s[i, :3, :3]
            ti = wc2s[i, :3, 3]

            bi = Ri.T @ (pointInGlobal - ti)
            bi = bi / np.linalg.norm(bi)
            H += (np.eye(3) - np.outer(bi, bi))
    
    # print('valids: ', valids)
    if valids < 1:
        return 0
    
    H /= valids
    eigH = np.linalg.eigvals(0.5*(H + H.T))
    
    return np.arccos(np.clip(1 - 2.0 * np.min(eigH)/np.max(eigH), 0, 1))

def Computedist2surfarce(pointInGlobal, wc2s, intrinsics, depths):
    diffs = []
    for i in range(wc2s.shape[0]):
        valid, pixel = inside_check(pointInGlobal, wc2s[i], intrinsics)
        if valid:
            pointInCamerai = wc2s[i, :3, :3] @ pointInGlobal + wc2s[i, :3, 3]
            diff = np.abs(pointInCamerai[2] - depths[i, int(pixel[1]), int(pixel[0])]) # pixel: [w,h]
            if (diff < 0.3) and (depths[i, int(pixel[1]), int(pixel[0])] > 0.02):
                diffs.append(diff)

    diffs = np.array(diffs)
    pointDepthMean, pointDepthStd = np.mean(diffs), np.std(diffs)
    # print('pointDepthMean: ', pointDepthMean)
    # print('pointDepthStd: ', pointDepthStd)

    return pointDepthMean, pointDepthStd

def random_down_sample(points3D, num_gs):
    n_points = points3D.shape[0]

    mask = np.random.choice(n_points, num_gs)
    subset = points3D[mask]
    return subset


def gaussian_selectition(points3D, wc2s, intrinsics, depths, num_gs=100):
    '''
    points3D: [N, 3], in world sys.
    poses:  [M, 4, 4]
    intrinsics: [3, 3]
    '''
    n_points = points3D.shape[0]
    points3D_scores = np.zeros(n_points)

    # compute saliency score
    progress_bar = tqdm(range(n_points), position=0)
    progress_bar.set_description("Compute saliency score")
    for it in progress_bar:
        point_3d = points3D[it]
        # points3D_scores[it] = 1
        # continue

        depthMean, depthStd = Computedist2surfarce(point_3d, wc2s, intrinsics, depths)        
        anglespan = ComputePerPointAngularSpan(point_3d, wc2s, intrinsics)
            
        # depthScore = min(1.0, depthStd / depthMean)
        depthScore = min(2, 0.05 / depthMean) + min(2, 0.05 / depthStd)
        points3D_scores[it] = depthScore + anglespan

        # print('depthScore: ', depthScore, '. anglespan: ', anglespan)
    
    ## Sort scores
    sorted_indices = np.argsort(points3D_scores)

    ## Greedy selection
    selected_landmarks = {'xyz': np.zeros((3, num_gs)), 
                          'score': np.zeros(num_gs)}

    ## Selecting first point
    selected_landmarks['xyz'][:, 0] = points3D[sorted_indices[-1]]
    selected_landmarks['score'][0] = points3D_scores[sorted_indices[-1]]

    nselected = 1
    radius = 18.0
    print('Select 3D Landmark.')
    while nselected < num_gs:
        for i in reversed(sorted_indices):
            xyz = points3D[i]        

            if np.sum(np.linalg.norm(xyz.reshape(3, 1) - selected_landmarks['xyz'][:, :nselected], axis=0) < radius):
                continue
            else:
                selected_landmarks['xyz'][:, nselected] = xyz
                selected_landmarks['score'][nselected] = points3D_scores[i]
                nselected += 1

            if nselected == num_gs:
                break
            
        radius *= 0.5

    ## Saving
    xyzs = []
    rgbs = []
    for i in range(num_gs):
        xyzs.append(list(selected_landmarks['xyz'][:, i]))
    
    # save selected ply
    # save_path = os.path.join('landmarks_{}.ply'.format(num_gs))
    # makePlyFile(xyzs, xyzs, save_path)

    return np.array(xyzs)
