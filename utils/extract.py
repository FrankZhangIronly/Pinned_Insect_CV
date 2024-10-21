from ultralytics.utils import ops
import torch
import numpy as np
from PIL import Image
import os

def ex_label(result, name, conf=0.5, base_dir = "./models/ultralytics/runs/labels/", mode="image", frame=0):
    masks = result.masks.data
    classes = result.boxes.cls
    confs = result.boxes.conf
    shape = result.orig_shape
    image_path = result.path
    im = result.orig_img[:,:,[2,1,0]]
    
    min_conf = conf
    
    image = Image.fromarray(im).convert("RGBA")
    image_array = np.array(image)
    
    resized_masks = ops.scale_image(masks.permute(1,2,0).byte().cpu().numpy(), shape)
    resized_masks = torch.tensor(resized_masks).permute(2,0,1).cpu().numpy()
    
    for i in range(len(classes)):
        if classes[i] == 1 and confs[i]>=min_conf:
            array_2d = resized_masks[i]
            new_image_array = np.zeros((array_2d.shape[0],array_2d.shape[1], 4), dtype=np.uint8)
            new_image_array[array_2d == 1] = image_array[array_2d == 1]
    
            # getting border
            rows = np.any(array_2d, axis=1)
            cols = np.any(array_2d, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
    
            new_image_array = new_image_array[rmin:rmax+1, cmin:cmax+1]
            new_image = Image.fromarray(new_image_array)
            
            image_name = image_path.split("\\")[-1]
            os.makedirs(base_dir+name, exist_ok=True)
            
            if mode == "image":
                label_name = image_name.replace("."+image_name.split(".")[-1], "_label_{}.png".format(str(i)))
                new_image.save(base_dir+name+"/"+label_name)
            elif mode == "video":
                label_name = image_name.replace("."+image_name.split(".")[-1], "_frame_{}_label_{}.png".format(frame, str(i)))
                new_image.save(base_dir+name+"/"+label_name)

def ex_label_track(result, name, conf=0.5, base_dir = "./models/ultralytics/runs/labels/", mode="image", frame=0):
    masks = result.masks.data
    classes = result.boxes.cls
    confs = result.boxes.conf
    shape = result.orig_shape
    image_path = result.path
    id = result.boxes.id
    im = result.orig_img[:,:,[2,1,0]]
    
    min_conf = conf
    
    image = Image.fromarray(im).convert("RGBA")
    image_array = np.array(image)
    
    resized_masks = ops.scale_image(masks.permute(1,2,0).byte().cpu().numpy(), shape)
    resized_masks = torch.tensor(resized_masks).permute(2,0,1).cpu().numpy()
    
    for i in range(len(classes)):
        if classes[i] == 1 and confs[i]>=min_conf and id is not None:
            track_id = int(id[i])
            array_2d = resized_masks[i]
            new_image_array = np.zeros((array_2d.shape[0],array_2d.shape[1], 4), dtype=np.uint8)
            new_image_array[array_2d == 1] = image_array[array_2d == 1]
    
            # getting border
            rows = np.any(array_2d, axis=1)
            cols = np.any(array_2d, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
    
            new_image_array = new_image_array[rmin:rmax+1, cmin:cmax+1]
            new_image = Image.fromarray(new_image_array)
            
            image_name = image_path.split("\\")[-1]
            os.makedirs(base_dir+name+"/"+str(track_id), exist_ok=True)
            
            if mode == "image":
                label_name = image_name.replace("."+image_name.split(".")[-1], "_label_{}.png".format(str(i)))
                new_image.save(base_dir+name+"/"+label_name)
            elif mode == "video":
                label_name = image_name.replace("."+image_name.split(".")[-1], "_frame_{}_label_{}.png".format(frame, str(i)))
                new_image.save(base_dir+name+"/"+str(track_id)+"/"+label_name)