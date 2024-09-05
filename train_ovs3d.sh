CUDA_VISIBLE_DEVICES=1 python train.py --source_path data/ovs3d/bed --images images_4 --model_path data/ovs3d/bed/output/bed
CUDA_VISIBLE_DEVICES=2 python train.py --source_path data/ovs3d/bench --images images_4 --model_path data/ovs3d/bench/output/bench
CUDA_VISIBLE_DEVICES=3 python train.py --source_path data/ovs3d/lawn --images images_4 --model_path data/ovs3d/lawn/output/lawn
CUDA_VISIBLE_DEVICES=4 python train.py --source_path data/ovs3d/room --images images_4 --model_path data/ovs3d/room/output/room
CUDA_VISIBLE_DEVICES=3 python train.py --source_path data/ovs3d/sofa --images images_4 --model_path data/ovs3d/sofa/output/sofa


CUDA_VISIBLE_DEVICES=1 python preprocess.py --dataset_path data/ovs3d/bed --images images_4
CUDA_VISIBLE_DEVICES=2 python preprocess.py --dataset_path data/ovs3d/bench --images images_4
CUDA_VISIBLE_DEVICES=3 python preprocess.py --dataset_path data/ovs3d/lawn --images images_4
CUDA_VISIBLE_DEVICES=4 python preprocess.py --dataset_path data/ovs3d/room --images images_4
CUDA_VISIBLE_DEVICES=3 python preprocess.py --dataset_path data/ovs3d/sofa --images images_4

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path data/ovs3d/bed --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name bed 
CUDA_VISIBLE_DEVICES=2 python train.py --dataset_path data/ovs3d/bench --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name bench
CUDA_VISIBLE_DEVICES=3 python train.py --dataset_path data/ovs3d/lawn --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name lawn
CUDA_VISIBLE_DEVICES=4 python train.py --dataset_path data/ovs3d/room --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name room
CUDA_VISIBLE_DEVICES=5 python train.py --dataset_path data/ovs3d/sofa --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name sofa

CUDA_VISIBLE_DEVICES=1  python test.py --dataset_path data/ovs3d/bed --dataset_name bed
CUDA_VISIBLE_DEVICES=2  python test.py --dataset_path data/ovs3d/bench --dataset_name bench
CUDA_VISIBLE_DEVICES=3  python test.py --dataset_path data/ovs3d/lawn --dataset_name lawn
CUDA_VISIBLE_DEVICES=4  python test.py --dataset_path data/ovs3d/room --dataset_name room
CUDA_VISIBLE_DEVICES=5  python test.py --dataset_path data/ovs3d/sofa --dataset_name sofa

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=1 python train.py --source_path data/ovs3d/bed -m output/bed --images images_4 --start_checkpoint data/ovs3d/bed/output/bed/chkpnt30000.pth --feature_level ${level}
    # e.g. python train.py -s data/sofa -m output/sofa --start_checkpoint data/sofa/sofa/chkpnt30000.pth --feature_level 3
done

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=2 python train.py --source_path data/ovs3d/bench -m output/bench --images images_4 --start_checkpoint data/ovs3d/bench/output/bench/chkpnt30000.pth --feature_level ${level}
    # e.g. python train.py -s data/sofa -m output/sofa --start_checkpoint data/sofa/sofa/chkpnt30000.pth --feature_level 3
done

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=4 python train.py --source_path data/ovs3d/lawn -m output/lawn --images images_4 --start_checkpoint data/ovs3d/lawn/output/lawn/chkpnt30000.pth --feature_level ${level}
    # e.g. python train.py -s data/sofa -m output/sofa --start_checkpoint data/sofa/sofa/chkpnt30000.pth --feature_level 3
done

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=4 python train.py --source_path data/ovs3d/room -m output/room --images images_4 --start_checkpoint data/ovs3d/room/output/room/chkpnt30000.pth --feature_level ${level}
    # e.g. python train.py -s data/sofa -m output/sofa --start_checkpoint data/sofa/sofa/chkpnt30000.pth --feature_level 3
done

CUDA_VISIBLE_DEVICES=5 python train.py --dataset_path data/ovs3d/sofa --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name sofa
CUDA_VISIBLE_DEVICES=5  python test.py --dataset_path data/ovs3d/sofa --dataset_name sofa
cd ..
for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=5 python train.py --source_path data/ovs3d/sofa -m output/sofa --images images_4 --start_checkpoint data/ovs3d/sofa/output/sofa/chkpnt30000.pth --feature_level ${level}
    # e.g. python train.py -s data/sofa -m output/sofa --start_checkpoint data/sofa/sofa/chkpnt30000.pth --feature_level 3
done