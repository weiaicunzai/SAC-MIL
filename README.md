python train.py --stage='train'  --gpus=0 --fold=0

DATSET_NAME: support 3 dataset names, cam16, brac, lung

Training:

```
python train.py --stage='train'   --gpus=3 --fold=0   --dataset=DATASET_NAME --model='mixer'
```


Generate Normalized Position:
```
python preprocess/create_coords.py --dataset DATASET_NAME  --k 64
```

Visualize Regions of each slide.
```
python utils/draw_image.py  --dataset DATASET_NAME --k 64
```

Visualize Patch 2D Coordinates
```
# draw normalized 2D coordinates
python utils/draw_pos.py --dataset DATASET_NAME --norm

# draw 2D index of each patch, 2D index is calculated as (X_coordinates / patch_size, Y_coordinates / patch_size)
python utils/draw_pos.py --dataset DATASET_NAME
```