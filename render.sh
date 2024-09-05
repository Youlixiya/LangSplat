# for level in 1 2 3
# do
#     CUDA_VISIBLE_DEVICES=1 python render.py -s data/gsgrouping/figurines -m output/figurines_${level} --feature_level ${level} --include_feature
# done

# for level in 1 2 3
# do
#     CUDA_VISIBLE_DEVICES=2 python render.py -s data/lerf_ovs/ramen -m output/ramen_${level} --feature_level ${level} --include_feature
# done

# for level in 1 2 3
# do
#     CUDA_VISIBLE_DEVICES=3 python render.py -s data/gsgrouping/teatime -m output/teatime_${level} --feature_level ${level} --include_feature
# done
# CUDA_VISIBLE_DEVICES=1 python render.py -s data/gsgrouping/figurines -m output/figurines_1 --dataset_name figurines --include_feature --skip_train 
# CUDA_VISIBLE_DEVICES=1 python render.py -s data/gsgrouping/figurines -m output/figurines_1 --dataset_name figurines --include_feature --skip_train  --reasoning

# CUDA_VISIBLE_DEVICES=2 python render.py -s data/gsgrouping/ramen -m output/ramen_1 --dataset_name ramen --include_feature --skip_train 
# CUDA_VISIBLE_DEVICES=2 python render.py -s data/gsgrouping/ramen -m output/ramen_1 --dataset_name ramen --include_feature --skip_train  --reasoning

# CUDA_VISIBLE_DEVICES=3 python render.py -s data/gsgrouping/teatime -m output/teatime_1 --dataset_name teatime --include_feature --skip_train 
# CUDA_VISIBLE_DEVICES=3 python render.py -s data/gsgrouping/teatime -m output/teatime_1 --dataset_name teatime --include_feature --skip_train  --reasoning

# CUDA_VISIBLE_DEVICES=1 python render_ovs3d.py -s data/ovs3d/bed -m output/bed_1 --dataset_name bed --include_feature --skip_train 
# CUDA_VISIBLE_DEVICES=1 python render_ovs3d.py -s data/ovs3d/bed -m output/bed_1 --dataset_name bed --include_feature --skip_train  --reasoning

CUDA_VISIBLE_DEVICES=2 python render_ovs3d.py -s data/ovs3d/bench -m output/bench_1 --dataset_name bench --include_feature --skip_train 
CUDA_VISIBLE_DEVICES=2 python render_ovs3d.py -s data/ovs3d/bench -m output/bench_1 --dataset_name bench --include_feature --skip_train  --reasoning

CUDA_VISIBLE_DEVICES=3 python render_ovs3d.py -s data/ovs3d/lawn -m output/lawn_1 --dataset_name lawn --include_feature --skip_train 
CUDA_VISIBLE_DEVICES=3 python render_ovs3d.py -s data/ovs3d/lawn -m output/lawn_1 --dataset_name lawn --include_feature --skip_train  --reasoning

CUDA_VISIBLE_DEVICES=1 python render_ovs3d.py -s data/ovs3d/room -m output/room_1 --dataset_name room --include_feature --skip_train 
CUDA_VISIBLE_DEVICES=1 python render_ovs3d.py -s data/ovs3d/room -m output/room_1 --dataset_name room --include_feature --skip_train  --reasoning

CUDA_VISIBLE_DEVICES=2 python render_ovs3d.py -s data/ovs3d/sofa -m output/sofa_1 --dataset_name sofa --include_feature --skip_train 
CUDA_VISIBLE_DEVICES=2 python render_ovs3d.py -s data/ovs3d/sofa -m output/sofa_1 --dataset_name sofa --include_feature --skip_train  --reasoning


CUDA_VISIBLE_DEVICES=2 python render_messy_rooms.py -s data/messy_rooms/large_corridor_25 -m output/large_corridor_25_1 --dataset_name large_corridor_25 --include_feature --skip_train 

