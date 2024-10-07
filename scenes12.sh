# # scenes12
for scene in 'apt1_kitchen' 'apt1_living' 'apt2_bed' 'apt2_kitchen' 'apt2_living' 'apt2_luke' \
             'of1_gates362' 'of1_gates381' 'of1_lounge' 'of1_manolis' \
             'of2_5a' 'of2_5b' 
do 
CUDA_VISIBLE_DEVICES=0 python train_decoder.py --config ./configs/scenes12/$scene.yaml
CUDA_VISIBLE_DEVICES=0 python train_gaussians.py --config ./configs/scenes12/$scene.yaml
CUDA_VISIBLE_DEVICES=0 python test.py --config ./configs/scenes12/$scene.yaml --eval_pose
done