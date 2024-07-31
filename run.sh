export CUDA_VISIBLE_DEVICES=1
#python train.py --name cifar10-100_500 --model_type Mixer-B_16 --train_batch_size 1 --eval_batch_size 1  --dataset_name cam16 --fold 0  --num_steps 200
#python3 eval.py --name cifar10-100_500 --model_type Mixer-B_16 --eval_batch_size 1 --dataset_name cam16 --fold 0 --ckpt_dir  /data/hdd1/by/MLPforWSI/output/cifar10-100_500_checkpoint.bin

#cd DTFD-MIL/
#python  Main_DTFD_MIL.py --dataset_name cam16  --fold 0  --distill_type AFS --numGroup 4 --numGroup_test 4
# python  Main_DTFD_MIL.py --dataset_name lung --fold 0  --distill_type AFS --numGroup 4 --numGroup_test 4
#python  Main_DTFD_MIL.py --dataset_name brac --fold 0  --distill_type AFS --numGroup 4 --numGroup_test 4
cd TransMIL
python train.py --stage='train' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=0
