import os
import torch
import torch.utils.data as data
import numpy as np
import random
from PIL import Image

random.seed(1143)

def prepare_data(hazefree_images_dir, hazeeffected_images_dir):
    data_pairs = []
    hazy_data = os.listdir(hazeeffected_images_dir)

    for h_image in hazy_data:
        hazy_path = os.path.join(hazeeffected_images_dir, h_image)
        if Image.open(hazy_path).mode != 'RGB':
            continue
        
        # Exact filename mapping (RICE1 / Haze1k Train & Val 1:1 mapping)
        h_image_name = h_image.split("/")[-1]
        id_ = h_image_name 
        
        clear_path = os.path.join(hazefree_images_dir, id_)
        
        if not os.path.exists(clear_path):
            continue
        if Image.open(clear_path).mode != 'RGB':
            continue

        data_pairs.append([clear_path, hazy_path])

    random.shuffle(data_pairs)
    return data_pairs

    
class hazy_data_loader(data.Dataset):
    def __init__(self, hazefree_images_dir, hazeeffected_images_dir, mode='train'):
        # The loader now maps 100% of the provided directory
        self.data_dict = prepare_data(hazefree_images_dir, hazeeffected_images_dir) 

        if mode == 'train':
            print("Number of Training Images:", len(self.data_dict))
        else:
            print("Number of Validation Images:", len(self.data_dict))

    def __getitem__(self, index):
        hazefree_image_path, hazy_image_path = self.data_dict[index]

        hazefree_image = Image.open(hazefree_image_path)
        hazy_image = Image.open(hazy_image_path)

        hazefree_image = hazefree_image.resize((480,640), Image.LANCZOS)
        hazy_image = hazy_image.resize((480,640), Image.LANCZOS)

        hazefree_image = (np.asarray(hazefree_image) / 255.0) 
        hazy_image = (np.asarray(hazy_image) / 255.0) 

        hazefree_image = torch.from_numpy(hazefree_image).float()
        hazy_image = torch.from_numpy(hazy_image).float()

        return hazefree_image.permute(2,0,1), hazy_image.permute(2,0,1)

    def __len__(self):
        return len(self.data_dict)