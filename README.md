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
    
<img width="335" alt="Screenshot 2024-01-15 at 10 45 51 PM" src="https://github.com/shubo4/Aerial_obj_detection_DOTA/assets/90241581/93678ea6-ffd0-4e05-a65d-7aa1995350e3">
  


## Further work:
As given in (repo of dota-retiannet) they use resnet-50 backbone which is 146MB. We used mobile-net_v2_large which is 21MB. Our mAP is 17.7 and which is significantly less than Resnet 
accuracy. Goal is to try and close this gap. Below image shows the model is missing the ibjects with very small size. 

Prediction:
<img width="544" alt="Screenshot 2024-01-16 at 10 20 17 AM" src="https://github.com/shubo4/Aerial_obj_detection_DOTA/assets/90241581/3178519f-73c5-4627-9e0f-3f6bbf7701ec">

Ground Truth:
<img width="547" alt="Screenshot 2024-01-16 at 10 17 39 AM" src="https://github.com/shubo4/Aerial_obj_detection_DOTA/assets/90241581/18bc7943-c5e6-48b3-be9c-aa07108f8d15">
