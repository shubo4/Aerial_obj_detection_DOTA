# DOTA-1.5 RetinaNet pytorch (mobilenet backbone)

Pytorch  implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002). I wanted to train it on kaggle
and couldn't use big resnet models. So I used mobile-net backbones. 

## Installation

1) Clone this repo

2) Install the python packages:
	
```
pip install pandas
pip install pycocotools
pip install opencv-python
pip install requests
pip install comet-ml

```
## Data Format
We need to folders:  
/Images -- containing all images training and validation   
/Annotations -- containing train.json and val.json (You can create using DOTA2COCO script)

## Training:
The network can be trained using 'main.py' script. 
```
--images_path - Path to images folder
--train_ann_path - Path to train.json
--val_ann_path   -  Path to val.json
--crop_height    - We use random crop and preprocess bounding boxes similarly. Default is 2048.
--crop_width     - Simialr to crop height.
--epochs, --batch_size - Training params
--comet, --comet_workspace  - I have used comet for visualizations. You need to login to comet(its free) and use Authorize key and comet_workspace name
--model_checkpoint - [optional] if you want to start training from checkpoint
```

## Results:
On commet I have logged Batch Loss, BBox Regression Loss, BBox Classification loss and gpu metrics.   
On comet project workspace in items and artifacts of specific experiment you will have 3 dataframes stored. One with class confusion matrices and AP.   
Second with all Images confusion matrices and AP.  Third with mAP forspecified ious. 
   
<img width="185" alt="Screenshot 2024-01-15 at 10 04 43 PM" src="https://github.com/shubo4/Aerial_obj_detection_DOTA/assets/90241581/4108d0fb-a453-422a-810f-22b3a1060080">


