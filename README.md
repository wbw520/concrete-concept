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


#### Training with extra csv
1. Prepare the json file through Labeling box and set it under data/ direction.
2. run data/download_imgs.py (download images and corresponding masks, change root to "concrete_data2/").
3. run data/generate_dataset.py (crop and generate patches for training and evaluation, change root to "concrete_data2/", save_folder to "concrete_cropped_center2/").
4. Train fusion model
```
1. change --fusion and --fusion_loader as True in file configs.py
2. change --item_number to set number of item used in csv file (Using the same structure with first row of item name as 1, 2, 3, ...).
2. run main2.py
3. run inference2.py for run one image.
img_dir = "" (for your image root) 
set csv data in "csv_data = [2017, 58]", current two values are avaliable.
```