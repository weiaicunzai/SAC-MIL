import os
import subprocess

def generate_params():
    models = [
        # 'dsmil',
        # 'transmil',
        # 'abmil',
        # 'mixer',
        # 'clam_sb',
        'clam_mb'
    ]

    # ds={"brac", "lung"}
    # datasets=("brac" "lung")
    datasets = ['brac', 'lung']
    fold = [0, 1, 2, 3, 4]
    feat_type = ['uni', 'res50']

    for mod in models:
        for ds in datasets:
            for feat in feat_type:
                for f_id in fold:
                    yield mod, ds, f_id, feat


N = 4
from queue import Queue
gpu_queue = Queue(maxsize=N)
for i in range(N):
    gpu_queue.put(i)  # 初始化队列，放入 GPU ID

def train_model(i):
    mod, ds, f_id, feat = i
    model_file = "{}_{}_fold{}_{}".format(mod, ds, f_id, feat)
    gpu_id = gpu_queue.get()
    print("Training fold {} of dataset {} with model {} on gpu {} using {} feature".format(f_id, ds, mod, gpu_id, feat))
    try:
        # 使用 subprocess.run 替代 os.system
        with open(model_file, 'w') as f:
            result = subprocess.run(
                [
                    'python', 'train.py',
                    '--stage=train',
                    '--gpus={}'.format(gpu_id),
                    '--fold={}'.format(f_id),
                    '--dataset={}'.format(ds),
                    '--model={}'.format(mod),
                    '--feat={}'.format(feat),
                ],
                stdout=f,
                stderr=subprocess.STDOUT  # 将 stderr 重定向到 stdout
            )

        # 检查子进程是否成功完成
        if result.returncode != 0:
            print(f"Training failed for {mod} on {ds} fold {f_id} feat {feat}")
    # try:
    #     os.system(
    #         'python train.py --stage=train   --gpus={gpu}  --fold={f_id}   --dataset={ds} --model={mod} &> {model_file}'.format(
    #             gpu=gpu_id,
    #             f_id=f_id,
    #             ds=ds,
    #             mod=mod,
    #             model_file=model_file
    #         )
    #     )
    finally:
        # 释放 GPU ID
        gpu_queue.put(gpu_id)

from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=N)
pool.map(train_model, generate_params())


# for i in generate_params():
#     train_model(i)





# for model in "${models[@]}"; do

    # for i in ds:
    # for ds in "${datasets[@]}"; do
    #     # echo $ds
    #     python train.py --stage='train'   --gpus=$gpu_id   --fold=0   --dataset=$ds --model=$model &> $model_$ds_fold0
    #     python train.py --stage='train'   --gpus=$gpu_id   --fold=1   --dataset=$ds --model=$model &> $model_$ds_fold1
    #     python train.py --stage='train'   --gpus=$gpu_id   --fold=2   --dataset=$ds --model=$model &> $model_$ds_fold2
    #     python train.py --stage='train'   --gpus=$gpu_id   --fold=3   --dataset=$ds --model=$model &> $model_$ds_fold3
    #     python train.py --stage='train'   --gpus=$gpu_id   --fold=4   --dataset=$ds --model=$model &> $model_$ds_fold4
    #     for fold in {0..4}; do
    #         echo "Training fold $fold of dataset $ds with model $model"

    #         # 使用大括号来正确拼接变量名
    #         output_file="${model}_${ds}_fold${fold}_res50.log"
    #         python train.py --stage='train' --gpus="$gpu_id" --fold="$fold" --dataset="$ds" --model="$model" &> "$output_file"
    #     done
    # done

# done