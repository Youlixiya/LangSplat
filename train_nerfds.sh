# as

CUDA_VISIBLE_DEVICES=1 python train.py --source_path data/NeRF-DS/as_novel_view --images images --model_path data/NeRF-DS/as_novel_view/output/as_novel_view
mv data/NeRF-DS/as_novel_view/output/as_novel_view_-1 data/NeRF-DS/as_novel_view/output/as_novel_view
CUDA_VISIBLE_DEVICES=1 python preprocess.py --dataset_path data/NeRF-DS/as_novel_view --dataset_type NeRF-DS

cd autoencoder
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path data/NeRF-DS/as_novel_view --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name as_novel_view
CUDA_VISIBLE_DEVICES=1 python test.py --dataset_path data/NeRF-DS/as_novel_view --dataset_name as_novel_view
cd .. 

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=1 python train.py --source_path data/NeRF-DS/as_novel_view -m output/as_novel_view --start_checkpoint data/NeRF-DS/as_novel_view/output/as_novel_view/chkpnt30000.pth --feature_level ${level}
done


# basin

CUDA_VISIBLE_DEVICES=2 python train.py --source_path data/NeRF-DS/basin_novel_view --images images --model_path data/NeRF-DS/basin_novel_view/output/basin_novel_view
mv data/NeRF-DS/basin_novel_view/output/basin_novel_view_-1 data/NeRF-DS/basin_novel_view/output/basin_novel_view
CUDA_VISIBLE_DEVICES=2 python preprocess.py --dataset_path data/NeRF-DS/basin_novel_view --dataset_type NeRF-DS

cd autoencoder
CUDA_VISIBLE_DEVICES=2 python train.py --dataset_path data/NeRF-DS/basin_novel_view --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name basin_novel_view
CUDA_VISIBLE_DEVICES=2 python test.py --dataset_path data/NeRF-DS/basin_novel_view --dataset_name basin_novel_view
cd .. 

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=2 python train.py --source_path data/NeRF-DS/basin_novel_view -m output/basin_novel_view --start_checkpoint data/NeRF-DS/basin_novel_view/output/basin_novel_view/chkpnt30000.pth --feature_level ${level}
done


# cup_novel_view

CUDA_VISIBLE_DEVICES=3 python train.py --source_path data/NeRF-DS/cup_novel_view --images images --model_path data/NeRF-DS/cup_novel_view/output/cup_novel_view
mv data/NeRF-DS/cup_novel_view/output/cup_novel_view-1 data/NeRF-DS/cup_novel_view/output/cup_novel_view
CUDA_VISIBLE_DEVICES=3 python preprocess.py --dataset_path data/NeRF-DS/cup_novel_view --dataset_type NeRF-DS

cd autoencoder
CUDA_VISIBLE_DEVICES=3 python train.py --dataset_path data/NeRF-DS/cup_novel_view --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name cup_novel_view
CUDA_VISIBLE_DEVICES=3 python test.py --dataset_path data/NeRF-DS/cup_novel_view --dataset_name cup_novel_view
cd .. 

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=3 python train.py --source_path data/NeRF-DS/cup_novel_view -m output/cup_novel_view --start_checkpoint data/NeRF-DS/cup_novel_view/output/cup_novel_view/chkpnt30000.pth --feature_level ${level}
done


# bell_novel_view

CUDA_VISIBLE_DEVICES=4 python train.py --source_path data/NeRF-DS/bell_novel_view --images images --model_path data/NeRF-DS/bell_novel_view/output/bell_novel_view
mv data/NeRF-DS/bell_novel_view/output/bell_novel_view_-1 data/NeRF-DS/bell_novel_view/output/bell_novel_view
CUDA_VISIBLE_DEVICES=4 python preprocess.py --dataset_path data/NeRF-DS/bell_novel_view --dataset_type NeRF-DS

cd autoencoder
CUDA_VISIBLE_DEVICES=4 python train.py --dataset_path data/NeRF-DS/bell_novel_view --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name bell_novel_view
CUDA_VISIBLE_DEVICES=4 python test.py --dataset_path data/NeRF-DS/bell_novel_view --dataset_name bell_novel_view
cd .. 

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=4 python train.py --source_path data/NeRF-DS/bell_novel_view -m output/bell_novel_view --start_checkpoint data/NeRF-DS/bell_novel_view/output/bell_novel_view/chkpnt30000.pth --feature_level ${level}
done


# sieve_novel_view

CUDA_VISIBLE_DEVICES=5 python train.py --source_path data/NeRF-DS/sieve_novel_view --images images --model_path data/NeRF-DS/sieve_novel_view/output/sieve_novel_view
mv data/NeRF-DS/sieve_novel_view/output/sieve_novel_view_-1 data/NeRF-DS/sieve_novel_view/output/sieve_novel_view
CUDA_VISIBLE_DEVICES=5 python preprocess.py --dataset_path data/NeRF-DS/sieve_novel_view --dataset_type NeRF-DS

cd autoencoder
CUDA_VISIBLE_DEVICES=5 python train.py --dataset_path data/NeRF-DS/sieve_novel_view --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name sieve_novel_view
CUDA_VISIBLE_DEVICES=5 python test.py --dataset_path data/NeRF-DS/sieve_novel_view --dataset_name sieve_novel_view
cd .. 

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=5 python train.py --source_path data/NeRF-DS/sieve_novel_view -m output/sieve_novel_view --start_checkpoint data/NeRF-DS/sieve_novel_view/output/sieve_novel_view/chkpnt30000.pth --feature_level ${level}
done