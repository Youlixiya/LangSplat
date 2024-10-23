# large_corridor_25
CUDA_VISIBLE_DEVICES=1 python train.py --source_path data/messy_rooms/large_corridor_25 --images images --model_path data/messy_rooms/large_corridor_25/output/large_corridor_25
CUDA_VISIBLE_DEVICES=1 python preprocess.py --dataset_path data/messy_rooms/large_corridor_25

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path data/messy_rooms/large_corridor_25 --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name large_corridor_25 
CUDA_VISIBLE_DEVICES=1 python test.py --dataset_path data/messy_rooms/large_corridor_25 --dataset_name large_corridor_25

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=1 python train.py --source_path data/messy_rooms/large_corridor_25 -m output/largecorridor25 --start_checkpoint data/messy_rooms/large_corridor_25/output/large_corridor_25/chkpnt30000.pth --feature_level ${level}
    
done

# large_corridor_50
CUDA_VISIBLE_DEVICES=0 python train.py --source_path data/messy_rooms/large_corridor_50 --images images --model_path data/messy_rooms/large_corridor_50/output/large_corridor_50
CUDA_VISIBLE_DEVICES=0 python preprocess.py --dataset_path data/messy_rooms/large_corridor_50

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path data/messy_rooms/large_corridor_50 --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name large_corridor_50
CUDA_VISIBLE_DEVICES=1 python test.py --dataset_path data/messy_rooms/large_corridor_50 --dataset_name large_corridor_50

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=3 python train.py --source_path data/messy_rooms/large_corridor_50 -m output/largecorridor50 --start_checkpoint data/messy_rooms/large_corridor_50/output/large_corridor_50/chkpnt30000.pth --feature_level ${level}    
done

# large_corridor_100
CUDA_VISIBLE_DEVICES=3 python train.py --source_path data/messy_rooms/large_corridor_100 --images images --model_path data/messy_rooms/large_corridor_100/output/large_corridor_100
CUDA_VISIBLE_DEVICES=2 python preprocess.py --dataset_path data/messy_rooms/large_corridor_100

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path data/messy_rooms/large_corridor_100 --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name large_corridor_100 
CUDA_VISIBLE_DEVICES=1 python test.py --dataset_path data/messy_rooms/large_corridor_100 --dataset_name large_corridor_100

# for level in 1 2 3
# do
#     CUDA_VISIBLE_DEVICES=3 python train.py --source_path data/gsgrouping/teatime -m output/teatime --start_checkpoint data/gsgrouping/teatime/output/teatime/chkpnt30000.pth --feature_level ${level}
#     # e.g. python train.py -s data/sofa -m output/sofa --start_checkpoint data/sofa/sofa/chkpnt30000.pth --feature_level 3
# done

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=3 python train.py --source_path data/messy_rooms/large_corridor_100 -m output/largecorridor100 --start_checkpoint data/messy_rooms/large_corridor_100/output/large_corridor_100/chkpnt30000.pth --feature_level ${level}    
done