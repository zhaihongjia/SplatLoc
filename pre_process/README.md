
### 1. 2D descriptor extraction

Move `extract_save_sp_feature.py` into `submodules/Hierarchical-Localization/hloc`.

run the following scripts:
```
# replica-nerf: room_0
python encode_feat.py --image_dir /mnt/nas_54/datasets/nerf-loc/replica/room_0/Sequence_1/rgb_skip5 \
                      --out_dir /mnt/nas_54/datasets/nerf-loc/replica/room_0/sp_inloc_skip5
```
params:
- image_dir: the dirs of your sampled database/training images
- out_dir: the output dirs for saving feature maps and keypoint scores 

you can obtain:
- `xx.pt`: 2D image feature maps with shape `[256, H, W]` for each item, 256 is the feature dimention.
- `xx_score.npy`: 2D keypoint scores with shape `[H, W]` in the range of [0, 1]


### 2. Image retrieval results
```
# replica-nerf: room_0
python gen_netvlad_retrieval.py --scene_path /mnt/nas_7/datasets/nerf-loc/replica/room_0 --save_path /mnt/nas_10/group/hongjia/replica/room_0
```
params:
- scene_path: the dir of sampled database/training images and query/test images
- save_path: the dit for saving generated results

NOTES: For more details about the database/query dirs for different datasets, please see the code in the `gen_netvlad_retrieval.py`

### 3. 3D feature volumen and cloud extraction

```
# replica-nerf: room_0
python run_fusion.py --config ./configs/replica_nerf/room_0.yaml
```
params:
- config: the config file of each scene