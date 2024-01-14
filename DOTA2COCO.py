## This will create json ann files like coco from .txt files of DOTA. 
## We need two folders one containing path to all Images and other one with all txt files. 

## Example: Images/0001.png,0002.png Txt_files/0001.txt, 0002.txt
## This is for DOTA1.
## You need to clone this repo and install shapely library before
## !pip install shapely
## !git clone https://github.com/CAPTAIN-WHU/DOTA_devkit.git

from DOTA_devkit import dota_utils as util
import os
import cv2
import json
import numpy as np

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter','container-crane']

## imageparent -- Path to directory contianing images
## labelparent -- Path to directory containing annotation text files

## You put split_type ="train" once and split_type="val" other time. And train it will create two josn files for you by creating train-test split of 80-20
def DOTA2COCO(destfile, split_type = 'train' , split_major =0.8,imageparent,labelparent ):
    data_dict = {}
    info = {'contributor': 'captain group',
           'data_created': '2018',
           'description': 'This is 1.0 version of DOTA dataset.',
           'url': 'http://captain.whu.edu.cn/DOTAweb/',
           'version': '1.0',
           'year': 2018}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(wordname_15):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        
        filenames = util.GetFileFromThisRootDir(labelparent)
        if split_type=='train':
            filenames = sorted(filenames)[:int(split_major* len(filenames))]
        elif split_type=='val':
            filenames = sorted(filenames)[int(split_major*len(filenames)):]
            
        for file in sorted(filenames):
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.parse_dota_poly2(file)
            for obj in objects:
                
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])
                values = [xmin/width, ymin/height, xmax/width, ymax/height]
                if np.max(values)>1:
                    continue
                else:
                    single_obj = {}
                    single_obj['boxes'] = values
                    single_obj['image_id'] = image_id
                    single_obj['category_id'] = wordname_15.index(obj['name']) + 1
                    single_obj['id'] = inst_count
                    data_dict['annotations'].append(single_obj)
                
                inst_count = inst_count + 1
                
            image_id = image_id + 1
        json.dump(data_dict, f_out)
    
