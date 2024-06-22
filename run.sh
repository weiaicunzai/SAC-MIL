export CUDA_VISIBLE_DEVICES=2
#python3 train.py --name cifar10-100_500 --model_type Mixer-B_16 --train_batch_size 1 --eval_batch_size 4 --eval_every 100 --dataset_name cam16 --fold 0
python3 eval.py --name cifar10-100_500 --model_type Mixer-B_16 --eval_batch_size 1 --dataset_name cam16 --fold 0 --ckpt_dir  /data/hdd1/by/MLPforWSI/output/cifar10-100_500_checkpoint.bin

#cd DTFD-MIL/
#python  Main_DTFD_MIL.py --dataset_name cam16  --fold 0  --distill_type AFS --numGroup 196  --numGroup_test 196
