
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast

from DOTA_dataset import CustomCocoDataset, custom_collate
from retinanet_model import RetinaNet_
from torchvision.models.detection.anchor_utils import AnchorGenerator
from comet_ml import Experiment
from mAP import mean_average_precision

def train(model, train_loader, val_loader ,experiment ,num_epochs, val_freq, load_from_checkpoint=False, model_checkpoint = None):
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  
  params = [p for p in model.parameters() if p.requires_grad]
  
  optimizer = torch.optim.SGD(
      params,
      lr=0.01,
      momentum=0.9,
      weight_decay=0.0001
  )
  
  if load_from_checkpoint:
      checkpoint = torch.load(model_checkpoint)
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      model.load_state_dict(checkpoint['model_state_dict'])
  
  lr_scheduler = torch.optim.lr_scheduler.StepLR(
      optimizer,
      step_size=2,
      gamma=0.1
  )
  
  # Batch accumulation settings
  accumulation_steps = 3  # Accumulate gradients over 5 batches before updating
  ##Training Loop
  for epoch in range(num_epochs):
      model.train()  # Set the model to training mode
  
      accumulated_loss = 0.0
  
      for batch_idx, (images, targets) in enumerate(train_loader):
          
          images = [image.to(device) for image in images]
          targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
          
          with autocast(): 
              # Forward pass
              outputs = model(images, targets)
  
              # Compute the total loss
              loss = outputs[list(outputs.keys())[0]] + outputs[list(outputs.keys())[1]]
              accumulated_loss += loss
  
          if (batch_idx + 1) % accumulation_steps == 0:
              # Backward pass and optimization
              optimizer.zero_grad()
              accumulated_loss.backward()
  
              clip_grad_norm_(model.parameters(), max_norm=20.0)
              optimizer.step()
  
              # Log metrics for the accumulated loss
              print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx + 1}, Accumulated Loss: {accumulated_loss.item() / accumulation_steps}")
  
              # Reset accumulated loss
              accumulated_loss = 0.0
            
              # Log other metrics if needed
          experiment.log_metric("Batch Loss", loss.item(), step=(epoch * len(train_loader) + batch_idx))
          experiment.log_metric("Classification Loss", outputs[list(outputs.keys())[0]], step=(epoch * len(train_loader) + batch_idx))
          experiment.log_metric("Regression Loss", outputs[list(outputs.keys())[1]], step=(epoch * len(train_loader) + batch_idx))
          # Log GPU memory usage to Comet.ml (optional)
          experiment.log_metric("GPU Memory Allocated (MB)", torch.cuda.memory_allocated() / (1024**2),
                                step=(epoch * len(train_loader) + batch_idx))
          
      #Validate model          
      if (epoch + 1) % val_freq == 0:
          validate(model,val_loader,((epoch+1)/val_freq),experiment,model_checkpoint=None)
          torch.save({
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                  }, 'checkpoint_state_dicts.pth')
      lr_scheduler.step()

    experiment.end()

def validate(model,loader,epoch,experiment,model_checkpoint=None):

    if model_checkpoint:
            checkpoint= torch.load(model_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])

    pred_boxes = []
    true_boxes = []

    model.eval()
    model.num_classes = 17
    num_classes = model.num_classes
    #precision_recall_class = np.zeros((len(loader)*2,2,num_classes))
    device = torch.device('cuda')
    model.to(device)
    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader):
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                out = model(images)
          
                for i in range(len(targets[0]['boxes'])):              
                  x1,y1,x2,y2 = list(targets[0]['boxes'].cpu().numpy())[i] # gt boxes coordinates xyxy format
                  true_boxes.append([ int(targets[0]['image_id'].cpu().numpy()),
                                       int(targets[0]['labels'].cpu().numpy()[i]),
                                       int(1),
                                       x1,y1,x2,y2])

                for i in range(len(list(out[0]['boxes'].cpu().numpy()))):
                  x1,y1,x2,y2 = list(out[0]['boxes'].cpu().numpy())[i]
                  pred_boxes.append([ int(targets[0]['image_id'].cpu().numpy()),
                                       list(out[0]['labels'].cpu().numpy())[i],
                                       list(out[0]['scores'].cpu().numpy())[i],
                                       x1,y1,x2,y2])

    mAP =[]
    mAP_ = {}
    ious = [0.35, 0.45,0.55] #np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    #ious = [0.45]
    for i in ious:
        if i == 0.45:
            df_c,df_i,avg_precisions = mean_average_precision(pred_boxes, true_boxes, iou_threshold=i, box_format="corners", num_classes=17,metric_df =True)
            mAP_[str(i)] = avg_precisions
            
        else:
            avg_precisions = mean_average_precision(pred_boxes, true_boxes, iou_threshold=i, box_format="corners", num_classes=17)
            mAP_[str(i)] = avg_precisions

        mAP.append(avg_precisions)
   # df_mAP = pd.DataFrame(mAP_, columns=list(mAP_.keys()))
    df_mAP = pd.DataFrame(list(mAP_.items()), columns=['Threshold', 'mAP'])
    f_mAP = sum(mAP)/len(mAP)
    experiment.log_metric("mAp:0.35:0.1:0.55",f_mAP , step=epoch)
    experiment.log_table(filename="classs_metric.csv", tabular_data=df_c)
    experiment.log_table(filename="image_mteric.csv",  tabular_data= df_i)
    experiment.log_table(filename="mAP_mteric.csv",  tabular_data= df_mAP)

    model.train()




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, help='Path where all images are stored including train and validation') ## Path where all images are stored
    parser.add_argument('--train_ann_path', type=str, help='Path to train annotation json file ')
    parser.add_argument('--val_ann_path', type=str, help='Path to val annotation json file ')
    parser.add_argument('--crop_height',  type=int, default = 2048,  help='Height of image after cropping') ## Since image sin DOTA have high resolution to avoid cuda out of memory in kaggle we use cropped images
    parser.add_argument('--crop_width',  type=int, default = 2048,    help='Width of image after cropping')
    parser.add_argument('--epochs',     type=int, default = 10,       help='epochs to run for') 
    parser.add_argument('--batch_size', type=int, default= 3,         help='batch size')
    parser.add_argument('--backbone', type=str, help='mobilenet small or large (options are "small" or "large")')
    parser.add_argument('--comet', type=str,    help='input API key')
    parser.add_argument('--comet_workspace', type=str,  help='comet workspace name')
    parser.add_argument('--model_check_path', type=str, default = None, help='model checkpint path(model will loadfrom this and start training)')
    
    args = parser.parse_args()

  ## Loading DataLoader
    train_dataset = CustomCocoDataset(args.images_path, annFile= args.train_ann_path, data_type = 'train')
    val_dataset = CustomCocoDataset(args.images_path, annFile= args.val_ann_path, data_type = 'val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,collate_fn=custom_collate)


## Loading Model
    if args.backbone== "large":
      backbone.out_channels = 960
      backbone = torchvision.models.mobilenet_v3_large(weights='DEFAULT').features
    elif args.backbone == "small":
      backbone.out_channels = 576
      backbone = torchvision.models.mobilenet_v3_small(weights='DEFAULT').features
      
    anchor_generator = AnchorGenerator(
    sizes=((4,8,16,32,64),),
    aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    num_classes = 17
    # put the pieces together inside a RetinaNet model
    
    model = RetinaNet_(backbone,
    num_classes,
    anchor_generator = anchor_generator, trainable_backbone_layers=0,detections_per_img =500,image_mean =[0.485, 0.456, 0.406] ,image_std=[0.229, 0.224, 0.225],
    g_iou_thresh =0.4 ,alpha=0.25)
    #This image mean and image_std is of imagent dataset since model is pre-trained on image-net
    # aplha parameter of focal loss(https://arxiv.org/abs/1708.02002) controls penalty for imbalance
    model.num_classes = num_classes

    
  
    experiment = Experiment(api_key=args.comet, project_name="finetune_obj_detection", workspace=args.comet_workspace)

    train(model, train_loader, val_loader ,experiment ,num_epochs=args.epochs, val_freq=int(args.epochs/2), model_checkpoint =args.model_check_path)
    
