import albumentations as A
import cv2
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import albumentations as A
import random


class CustomCocoDataset(CocoDetection):
    def __init__(self, root, annFile, data_type = 'train',crop_height=2048,crop_width = 2048):
        super(CustomCocoDataset, self).__init__(root, annFile)
        self.crop_height =crop_height
        self.crop_width = crop_width
        self.data_type  = data_type 

    def __getitem__(self, index):
        # Get the original data for the specified index
        original_data = super(CustomCocoDataset, self).__getitem__(index)
        if len(original_data[1])==0:
            return self.__getitem__(index+1)

        # Apply transformations if specified
#        if self.transform is not None:
        image, target = original_data

        image = np.array(image)
        target = {
            'boxes': [box['boxes'] for box in target],
            'labels': [box['category_id'] for box in target]
        }
        
        if self.data_type == 'train':
            # Get the dynamic width and height of the image
            height, width, _ = image.shape
            crop_height = self.crop_height
            crop_width = self.crop_width
            if height<=crop_height and width<=crop_width:
                crop_height= height
                crop_width = width

            elif height<crop_height and width>crop_width:
                crop_height = height
                crop_width  = crop_width

            elif width<crop_width and height>crop_height:
                crop_width = width
                crop_height =crop_height

            else:
                crop_height = crop_height
                crop_width  = crop_width

            # Adjust transformations based on image size
            bbox_params = A.BboxParams(
                format='albumentations',
                min_area=1024,
                min_visibility=0.1,
                label_fields=['labels']
            )

            dynamic_transform = A.Compose([
                A.RandomCrop(width=crop_width, height=crop_height),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2)
            ], bbox_params=bbox_params)

            crop_transforms = dynamic_transform(image=image, bboxes=target['boxes'], labels=target['labels'])

            if not crop_transforms['bboxes']:
                # Skip this instance, move to the next one
                if index + 1 < len(self):
                    index = index + 1
                else:
                    index = random.randint(0, len(self) - 1)
                return self.__getitem__(index)
            
            image =  transforms.ToTensor()(crop_transforms['image'])
            for i in range(len(crop_transforms['bboxes'])):
                crop_transforms['bboxes'][i] = box_resize(crop_transforms['bboxes'][i],crop_width,crop_height)
                
            target['boxes'] = torch.tensor(np.array(crop_transforms['bboxes']))
            target['labels'] = torch.tensor(crop_transforms['labels'])
            target['image_id'] = torch.tensor(index)
            
            return image, target
            
        elif self.data_type == 'val':
            image = transforms.ToTensor()(image)
   
            for i in range(len(target['boxes'])):
                    target['boxes'][i] = box_resize(target['boxes'][i], image.shape[2],image.shape[1])
            target['boxes'] = torch.tensor(np.array(target['boxes']))
            target['labels'] = torch.tensor(target['labels'])
            target['image_id'] = torch.tensor(index)
            
            return image, target
