CUDA_VISIBLE_DEVICES=1 python train.py --source_path data/gsgrouping/figurines --images images --model_path data/gsgrouping/figurines/output/figurines
CUDA_VISIBLE_DEVICES=2 python train.py --source_path data/gsgrouping/ramen --images images --model_path data/gsgrouping/ramen/output/ramen
CUDA_VISIBLE_DEVICES=3 python train.py --source_path data/gsgrouping/teatime --images images --model_path data/gsgrouping/teatime/output/teatime

CUDA_VISIBLE_DEVICES=1 python preprocess.py --dataset_path data/gsgrouping/figurines
CUDA_VISIBLE_DEVICES=2 python preprocess.py --dataset_path data/gsgrouping/ramen
CUDA_VISIBLE_DEVICES=3 python preprocess.py --dataset_path data/gsgrouping/teatime

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path data/gsgrouping/figurines --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name figurines 
CUDA_VISIBLE_DEVICES=2 python train.py --dataset_path data/gsgrouping/ramen --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name ramen
CUDA_VISIBLE_DEVICES=3 python train.py --dataset_path data/gsgrouping/teatime --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name teatime

CUDA_VISIBLE_DEVICES=1  python test.py --dataset_path data/gsgrouping/figurines --dataset_name figurines
CUDA_VISIBLE_DEVICES=2  python test.py --dataset_path data/gsgrouping/ramen --dataset_name ramen
CUDA_VISIBLE_DEVICES=3  python test.py --dataset_path data/gsgrouping/teatime --dataset_name teatime

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=1 python train.py --source_path data/gsgrouping/figurines -m output/figurines --start_checkpoint data/gsgrouping/figurines/output/figurines/chkpnt30000.pth --feature_level ${level}
    # e.g. python train.py -s data/sofa -m output/sofa --start_checkpoint data/sofa/sofa/chkpnt30000.pth --feature_level 3
done

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=2 python train.py --source_path data/gsgrouping/ramen -m output/ramen --start_checkpoint data/gsgrouping/ramen/output/ramen/chkpnt30000.pth --feature_level ${level}
    # e.g. python train.py -s data/sofa -m output/sofa --start_checkpoint data/sofa/sofa/chkpnt30000.pth --feature_level 3
done

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=3 python train.py --source_path data/gsgrouping/teatime -m output/teatime --start_checkpoint data/gsgrouping/teatime/output/teatime/chkpnt30000.pth --feature_level ${level}
    # e.g. python train.py -s data/sofa -m output/sofa --start_checkpoint data/sofa/sofa/chkpnt30000.pth --feature_level 3
done