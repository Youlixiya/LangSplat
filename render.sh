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
CUDA_VISIBLE_DEVICES=1 python render.py -s data/gsgrouping/figurines -m output/figurines_1 --dataset_name figurines --include_feature --skip_train 
CUDA_VISIBLE_DEVICES=1 python render.py -s data/gsgrouping/figurines -m output/figurines_1 --dataset_name figurines --include_feature --skip_train  --reasoning

CUDA_VISIBLE_DEVICES=2 python render.py -s data/gsgrouping/ramen -m output/ramen_1 --dataset_name ramen --include_feature --skip_train 
CUDA_VISIBLE_DEVICES=2 python render.py -s data/gsgrouping/ramen -m output/ramen_1 --dataset_name ramen --include_feature --skip_train  --reasoning

CUDA_VISIBLE_DEVICES=3 python render.py -s data/gsgrouping/teatime -m output/teatime_1 --dataset_name teatime --include_feature --skip_train 
CUDA_VISIBLE_DEVICES=3 python render.py -s data/gsgrouping/teatime -m output/teatime_1 --dataset_name teatime --include_feature --skip_train  --reasoning