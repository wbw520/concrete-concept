# Concept-based Recognition

## Usage

#### Data Set
1. Prepare the json file through Labeling box and set it under data/ direction.
2. run data/download_imgs.py (download images and corresponding masks).
3. run data/generate_dataset.py (crop and generate patches for training and evaluation).

#### Training and Visualization
(1) Pre-training the backbones
```
1. change --pre_train as True in file configs.py
2. run main.py
```

(2) Learning the concepts
```
1. change --pre_train as False in file configs.py
2. run main.py
```

(3) Extract concepts
```
1. run process.py
```

(4) Visualizing concepts
```
1. run vis_retri.py (all the concepts are shown in folder vis_pp)
```

#### Super Parameter Settings (in configs.py)
1. change --distinctiveness_bias and --consistence_bias for concept learning
2. change --att_bias for smaller attention areas
3. change --base_model for different backbone
4. change --num_classes according to classification needs
5. change --num_cpt for different concept number
