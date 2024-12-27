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

CUDA_VISIBLE_DEVICES=3 python render_ovs3d.py -s data/ovs3d/room -m output/room_1 --dataset_name room --include_feature --skip_train 
CUDA_VISIBLE_DEVICES=3 python render_ovs3d.py -s data/ovs3d/room -m output/room_1 --dataset_name room --include_feature --skip_train  --reasoning

CUDA_VISIBLE_DEVICES=2 python render_ovs3d.py -s data/ovs3d/sofa -m output/sofa_1 --dataset_name sofa --include_feature --skip_train 
CUDA_VISIBLE_DEVICES=2 python render_ovs3d.py -s data/ovs3d/sofa -m output/sofa_1 --dataset_name sofa --include_feature --skip_train  --reasoning


CUDA_VISIBLE_DEVICES=0 python render_messy_rooms.py -s data/messy_rooms/large_corridor_25 -m output/largecorridor25_1 --dataset_name large_corridor_25 --include_feature --skip_train 
CUDA_VISIBLE_DEVICES=1 python render_messy_rooms.py -s data/messy_rooms/large_corridor_50 -m output/largecorridor50_1 --dataset_name large_corridor_50 --include_feature --skip_train 
CUDA_VISIBLE_DEVICES=2 python render_messy_rooms.py -s data/messy_rooms/large_corridor_100 -m output/largecorridor100_1 --dataset_name large_corridor_100 --include_feature --skip_train 

CUDA_VISIBLE_DEVICES=3 python render_messy_rooms.py -s data/messy_rooms/large_corridor_25 -m output/largecorridor25_1 --dataset_name large_corridor_25 --include_feature --skip_train --reasoning
CUDA_VISIBLE_DEVICES=4 python render_messy_rooms.py -s data/messy_rooms/large_corridor_50 -m output/largecorridor50_1 --dataset_name large_corridor_50 --include_feature --skip_train --reasoning
CUDA_VISIBLE_DEVICES=5 python render_messy_rooms.py -s data/messy_rooms/large_corridor_100 -m output/largecorridor100_1 --dataset_name large_corridor_100 --include_feature --skip_train --reasoning



CUDA_VISIBLE_DEVICES=1 python render.py -s data/gsgrouping/figurines -m output/figurines_1 --dataset_name figurines --include_feature --skip_train  --reasoning
CUDA_VISIBLE_DEVICES=2 python render.py -s data/gsgrouping/ramen -m output/ramen_1 --dataset_name ramen --include_feature --skip_train  --reasoning
CUDA_VISIBLE_DEVICES=3 python render.py -s data/gsgrouping/teatime -m output/teatime_1 --dataset_name teatime --include_feature --skip_train  --reasoning

CUDA_VISIBLE_DEVICES=1 python render_ovs3d.py -s data/ovs3d/bed -m output/bed_1 --dataset_name bed --include_feature --skip_train  --reasoning
CUDA_VISIBLE_DEVICES=2 python render_ovs3d.py -s data/ovs3d/bench -m output/bench_1 --dataset_name bench --include_feature --skip_train  --reasoning
CUDA_VISIBLE_DEVICES=3 python render_ovs3d.py -s data/ovs3d/lawn -m output/lawn_1 --dataset_name lawn --include_feature --skip_train  --reasoning
CUDA_VISIBLE_DEVICES=4 python render_ovs3d.py -s data/ovs3d/room -m output/room_1 --dataset_name room --include_feature --skip_train  --reasoning
CUDA_VISIBLE_DEVICES=5 python render_ovs3d.py -s data/ovs3d/sofa -m output/sofa_1 --dataset_name sofa --include_feature --skip_train  --reasoning
CUDA_VISIBLE_DEVICES=0 python render_time.py -s data/gsgrouping/figurines -m output/figurines_1 --dataset_name figurines --include_feature --skip_train  --reasoning



CUDA_VISIBLE_DEVICES=1 python render_nerfiles.py -s data/hypernerf/chickchicken -m output/chickchicken_1 --dataset_name chickchicken --include_feature --skip_train
CUDA_VISIBLE_DEVICES=0 python render_nerfiles.py -s data/hypernerf/split-cookie -m output/split-cookie_1 --dataset_name split-cookie --include_feature --skip_train
CUDA_VISIBLE_DEVICES=2 python render_nerfiles.py -s data/hypernerf/torchocolate -m output/torchocolate_1 --dataset_name torchocolate --include_feature --skip_train
CUDA_VISIBLE_DEVICES=2 python render_nerfiles.py -s data/hypernerf/americano -m output/americano_1 --dataset_name americano --include_feature --skip_train
CUDA_VISIBLE_DEVICES=3 python render_nerfiles.py -s data/hypernerf/slice-banana -m output/slice-banana_1 --dataset_name slice-banana --include_feature --skip_train
CUDA_VISIBLE_DEVICES=3 python render_nerfiles.py -s data/NeRF-DS/as_novel_view -m output/asnovelview_1 --dataset_name as_novel_view --include_feature --skip_train
CUDA_VISIBLE_DEVICES=4 python render_nerfiles.py -s data/NeRF-DS/basin_novel_view -m output/basinnovelview_1 --dataset_name basin_novel_view --include_feature --skip_train
CUDA_VISIBLE_DEVICES=4 python render_nerfiles.py -s data/NeRF-DS/cup_novel_view -m output/cupnovelview_1 --dataset_name cup_novel_view --include_feature --skip_train
CUDA_VISIBLE_DEVICES=5 python render_nerfiles.py -s data/NeRF-DS/bell_novel_view -m output/bellnovelview_1 --dataset_name bell_novel_view --include_feature --skip_train
CUDA_VISIBLE_DEVICES=5 python render_nerfiles.py -s data/NeRF-DS/sieve_novel_view -m output/sievenovelview_1 --dataset_name sieve_novel_view --include_feature --skip_train