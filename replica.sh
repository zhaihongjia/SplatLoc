for scene in 'room_0' 'room_1' 'room_2' 'office_0' 'office_1' 'office_2' 'office_3' 'office_4'
do 
CUDA_VISIBLE_DEVICES=1 python train_decoder.py --config ./configs/replica_nerf/$scene.yaml
CUDA_VISIBLE_DEVICES=1 python train_gaussians.py --config ./configs/replica_nerf/$scene.yaml
CUDA_VISIBLE_DEVICES=1 python test.py --config ./configs/replica_nerf/$scene.yaml --eval_pose --eval_rendering 
CUDA_VISIBLE_DEVICES=1 python test.py --config ./configs/replica_nerf/$scene.yaml --eval_selection --landmark_num 5000
done