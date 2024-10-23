# chickchicken

CUDA_VISIBLE_DEVICES=1 python train.py --source_path data/hypernerf/chickchicken --images images --model_path data/hypernerf/chickchicken/output/chickchicken
mv data/hypernerf/chickchicken/output/chickchicken_-1 data/hypernerf/chickchicken/output/chickchicken
CUDA_VISIBLE_DEVICES=1 python preprocess.py --dataset_path data/hypernerf/chickchicken --dataset_type hypernerf

cd autoencoder
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path data/hypernerf/chickchicken --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name chickchicken
CUDA_VISIBLE_DEVICES=1 python test.py --dataset_path data/hypernerf/chickchicken --dataset_name chickchicken
cd .. 

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=0 python train.py --include_feature --source_path data/hypernerf/chickchicken -m output/chickchicken --start_checkpoint data/hypernerf/chickchicken/output/chickchicken/chkpnt30000.pth --feature_level ${level}
done


# split-cookie

CUDA_VISIBLE_DEVICES=2 python train.py --source_path data/hypernerf/split-cookie --images images --model_path data/hypernerf/split-cookie/output/split-cookie
mv data/hypernerf/split-cookie/output/split-cookie_-1 data/hypernerf/split-cookie/output/split-cookie
# CUDA_VISIBLE_DEVICES=2 python preprocess.py --dataset_path data/hypernerf/split-cookie --dataset_type hypernerf

cd autoencoder
CUDA_VISIBLE_DEVICES=2 python train.py --dataset_path data/hypernerf/split-cookie --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name split-cookie
CUDA_VISIBLE_DEVICES=2 python test.py --dataset_path data/hypernerf/split-cookie --dataset_name split-cookie
cd .. 

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=2 python train.py --include_feature --source_path data/hypernerf/split-cookie -m output/split-cookie --start_checkpoint data/hypernerf/split-cookie/output/split-cookie/chkpnt30000.pth --feature_level ${level}
done


# torchocolate

CUDA_VISIBLE_DEVICES=3 python train.py --source_path data/hypernerf/torchocolate --images images --model_path data/hypernerf/torchocolate/output/torchocolate
mv data/hypernerf/torchocolate/output/torchocolate_-1 data/hypernerf/torchocolate/output/torchocolate
# CUDA_VISIBLE_DEVICES=3 python preprocess.py --dataset_path data/hypernerf/torchocolate --dataset_type hypernerf

cd autoencoder
CUDA_VISIBLE_DEVICES=3 python train.py --dataset_path data/hypernerf/torchocolate --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name torchocolate
CUDA_VISIBLE_DEVICES=3 python test.py --dataset_path data/hypernerf/torchocolate --dataset_name torchocolate
cd .. 

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=3 python train.py --include_feature --source_path data/hypernerf/torchocolate -m output/torchocolate --start_checkpoint data/hypernerf/torchocolate/output/torchocolate/chkpnt30000.pth --feature_level ${level}
done


# americano

CUDA_VISIBLE_DEVICES=4 python train.py --source_path data/hypernerf/americano --images images --model_path data/hypernerf/americano/output/americano
mv data/hypernerf/americano/output/americano_-1 data/hypernerf/americano/output/americano
# CUDA_VISIBLE_DEVICES=4 python preprocess.py --dataset_path data/hypernerf/americano --dataset_type hypernerf

cd autoencoder
CUDA_VISIBLE_DEVICES=4 python train.py --dataset_path data/hypernerf/americano --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name americano
CUDA_VISIBLE_DEVICES=4 python test.py --dataset_path data/hypernerf/americano --dataset_name americano
cd .. 

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=4 python train.py --include_feature --source_path data/hypernerf/americano -m output/americano --start_checkpoint data/hypernerf/americano/output/americano/chkpnt30000.pth --feature_level ${level}
done


# slice-banana

CUDA_VISIBLE_DEVICES=5 python train.py --source_path data/hypernerf/slice-banana --images images --model_path data/hypernerf/slice-banana/output/slice-banana
mv data/hypernerf/slice-banana/output/slice-banana_-1 data/hypernerf/slice-banana/output/slice-banana
# CUDA_VISIBLE_DEVICES=5 python preprocess.py --dataset_path data/hypernerf/slice-banana --dataset_type hypernerf

cd autoencoder
CUDA_VISIBLE_DEVICES=5 python train.py --dataset_path data/hypernerf/slice-banana --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name slice-banana
CUDA_VISIBLE_DEVICES=5 python test.py --dataset_path data/hypernerf/slice-banana --dataset_name slice-banana
cd .. 

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=1 python train.py --include_feature --source_path data/hypernerf/slice-banana -m output/slice-banana --start_checkpoint data/hypernerf/slice-banana/output/slice-banana/chkpnt30000.pth --feature_level ${level}
done


# CUDA_VISIBLE_DEVICES=5 python preprocess.py --dataset_path data/hypernerf/americano --dataset_type hypernerf
# CUDA_VISIBLE_DEVICES=5 python preprocess.py --dataset_path data/hypernerf/slice-banana --dataset_type hypernerf
# CUDA_VISIBLE_DEVICES=5 python preprocess.py --dataset_path data/NeRF-DS/as_novel_view --dataset_type NeRF-DS
# CUDA_VISIBLE_DEVICES=5 python preprocess.py --dataset_path data/NeRF-DS/basin_novel_view --dataset_type NeRF-DS
# CUDA_VISIBLE_DEVICES=5 python preprocess.py --dataset_path data/NeRF-DS/cup_novel_view --dataset_type NeRF-DS
# CUDA_VISIBLE_DEVICES=5 python preprocess.py --dataset_path data/NeRF-DS/bell_novel_view --dataset_type NeRF-DS
# CUDA_VISIBLE_DEVICES=5 python preprocess.py --dataset_path data/NeRF-DS/sieve_novel_view --dataset_type NeRF-DS