# python train.py --stage='train'   --gpus=2 --fold=0   --dataset='cam16' --model='mixer'
#python train.py --stage='train'   --gpus=2 --fold=0   --dataset='cam16' --model='mixer'
# python train.py --stage='train'   --gpus=2 --fold=0   --dataset='cam16' --model='dsmil'
# python train.py --stage='train'   --gpus=2 --fold=0   --dataset='cam16' --model='clam_sb'
# python train.py --stage='train'   --gpus=0 --fold=0   --dataset='cam16' --model='dsmil'
# python train.py --stage='train'   --gpus=3 --fold=0   --dataset='cam16' --model='abmil'
#python train.py --stage='train'   --gpus=1 --fold=0   --dataset='cam16' --model='mil'
#python train.py --stage='test'   --gpus=3 --fold=0   --dataset='cam16' --model='mixer' --ckpt   '/data/hdd1/by/MLPforWSI/ckpt/cam16/0/Monday_18_November_2024_00h_00m_39s/epoch=73-val_acc=0.5834.ckpt'
#python train.py --stage='train'   --gpus=3 --fold=0   --dataset='cam16' --model='transmil' --ckpt   '/data/hdd1/by/MLPforWSI/ckpt/cam16/0/Monday_18_November_2024_00h_00m_39s/epoch=73-val_acc=0.5834.ckpt'
#python train.py --stage='train'   --gpus=3 --fold=0   --dataset='cam16' --model='mixer' --ckpt   '/data/hdd1/by/MLPforWSI/ckpt/cam16/0/Monday_18_November_2024_00h_00m_39s/epoch=73-val_acc=0.5834.ckpt'
# python train.py --stage='train'   --gpus=3 --fold=0   --dataset='cam16' --model='mixer' --ckpt   '/data/hdd1/by/MLPforWSI/ckpt/cam16/0/Monday_18_November_2024_00h_00m_39s/epoch=73-val_acc=0.5834.ckpt'
# python train.py --stage='test'   --gpus=3 --fold=0   --dataset='cam16' --model='transmil' --ckpt   '/data/hdd1/by/MLPforWSI/ckpt/cam16/0/Monday_18_November_2024_00h_00m_39s/epoch=73-val_acc=0.5834.ckpt'
#python train.py --stage='test'   --gpus=3 --fold=0   --dataset='cam16' --model='mixer' --ckpt   '/data/hdd1/by/MLPforWSI/ckpt/cam16/0/Sunday_17_November_2024_23h_57m_54s/epoch=93-val_acc=0.7522.ckpt'
#python train.py --stage='test'   --gpus=3 --fold=0   --dataset='cam16' --model='mixer' --ckpt   '/data/ssd1/thw/project/Repetition/MLPforWSI/ckpt/cam16/0/Friday_06_December_2024_15h_53m_10s/epoch=130-val_acc=0.8730.ckpt' --feat res50
python train.py --stage='test'   --gpus=3 --fold=0   --dataset='cam16' --model='mixer' --ckpt   'epoch=130-val_acc=0.8730.ckpt' --feat res50
# '/data/hdd1/by/MLPforWSI/ckpt/cam16/0/Saturday_16_November_2024_15h_38m_25s/epoch=49-val_loss=0.0000.ckpt'



# # lung
# python train.py --stage='train'   --gpus=0 --fold=0   --dataset='lung' --model='dsmil'
# python train.py --stage='train'   --gpus=0 --fold=1   --dataset='lung' --model='dsmil'
# python train.py --stage='train'   --gpus=0 --fold=2   --dataset='lung' --model='dsmil'
# python train.py --stage='train'   --gpus=0 --fold=3   --dataset='lung' --model='dsmil'
# python train.py --stage='train'   --gpus=0 --fold=4   --dataset='lung' --model='dsmil'

# # Brac
# python train.py --stage='train'   --gpus=0 --fold=0   --dataset='brac' --model='dsmil'
# python train.py --stage='train'   --gpus=0 --fold=1   --dataset='brac' --model='dsmil'
# python train.py --stage='train'   --gpus=0 --fold=2   --dataset='brac' --model='dsmil'
# python train.py --stage='train'   --gpus=0 --fold=3   --dataset='brac' --model='dsmil'
# python train.py --stage='train'   --gpus=0 --fold=4   --dataset='brac' --model='dsmil'



# # lung
# python train.py --stage='train'   --gpus=1 --fold=0   --dataset='lung' --model='mixer'
# python train.py --stage='train'   --gpus=1 --fold=1   --dataset='lung' --model='mixer'
# python train.py --stage='train'   --gpus=1 --fold=2   --dataset='lung' --model='mixer'
# python train.py --stage='train'   --gpus=1 --fold=3   --dataset='lung' --model='mixer'
# python train.py --stage='train'   --gpus=1 --fold=4   --dataset='lung' --model='mixer'

# # Brac
# python train.py --stage='train'   --gpus=1 --fold=0   --dataset='brac' --model='mixer'
# python train.py --stage='train'   --gpus=1 --fold=1   --dataset='brac' --model='mixer'
# python train.py --stage='train'   --gpus=1 --fold=2   --dataset='brac' --model='mixer'
# python train.py --stage='train'   --gpus=1 --fold=3   --dataset='brac' --model='mixer'
# python train.py --stage='train'   --gpus=1 --fold=4   --dataset='brac' --model='mixer'


# # Brac
# python train.py --stage='train'   --gpus=2 --fold=0   --dataset='brac' --model='abmil'
# python train.py --stage='train'   --gpus=2 --fold=1   --dataset='brac' --model='abmil'
# python train.py --stage='train'   --gpus=2 --fold=2   --dataset='brac' --model='abmil'
# python train.py --stage='train'   --gpus=2 --fold=3   --dataset='brac' --model='abmil'
# python train.py --stage='train'   --gpus=2 --fold=4   --dataset='brac' --model='abmil'

# # lung
# python train.py --stage='train'   --gpus=2 --fold=0   --dataset='lung' --model='abmil'
# python train.py --stage='train'   --gpus=2 --fold=1   --dataset='lung' --model='abmil'
# python train.py --stage='train'   --gpus=2 --fold=2   --dataset='lung' --model='abmil'
# python train.py --stage='train'   --gpus=2 --fold=3   --dataset='lung' --model='abmil'
# python train.py --stage='train'   --gpus=2 --fold=4   --dataset='lung' --model='abmil'


# models=(dsmil transmil abmil mixer clam_sb clam_mb)

# gpu_id=0
# # ds={"brac", "lung"}
# datasets=("brac" "lung")

# for model in "${models[@]}"; do

#     # for i in ds:
#     for ds in "${datasets[@]}"; do
#     #     # echo $ds
#     #     python train.py --stage='train'   --gpus=$gpu_id   --fold=0   --dataset=$ds --model=$model &> $model_$ds_fold0
#     #     python train.py --stage='train'   --gpus=$gpu_id   --fold=1   --dataset=$ds --model=$model &> $model_$ds_fold1
#     #     python train.py --stage='train'   --gpus=$gpu_id   --fold=2   --dataset=$ds --model=$model &> $model_$ds_fold2
#     #     python train.py --stage='train'   --gpus=$gpu_id   --fold=3   --dataset=$ds --model=$model &> $model_$ds_fold3
#     #     python train.py --stage='train'   --gpus=$gpu_id   --fold=4   --dataset=$ds --model=$model &> $model_$ds_fold4
#         for fold in {0..4}; do
#             echo "Training fold $fold of dataset $ds with model $model"

#             # 使用大括号来正确拼接变量名
#             output_file="${model}_${ds}_fold${fold}_res50.log"
#             python train.py --stage='train' --gpus="$gpu_id" --fold="$fold" --dataset="$ds" --model="$model" &> "$output_file"
#         done
#     done

# done
# lung
# python train.py --stage='train'   --gpus=3 --fold=0   --dataset=$ds --model=$model
# python train.py --stage='train'   --gpus=3 --fold=1   --dataset=$ds --model=$model
# python train.py --stage='train'   --gpus=3 --fold=2   --dataset=$ds --model=$model
# python train.py --stage='train'   --gpus=3 --fold=3   --dataset=$ds --model=$model
# python train.py --stage='train'   --gpus=3 --fold=4   --dataset=$ds --model=$model

# python train.py --stage='train'   --gpus=0 --fold=0   --dataset='cam16' --model='dsmil' --feat res50
