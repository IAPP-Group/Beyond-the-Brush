import numpy as np
import json
from PIL import Image
from time import time
import os
import random
import cv2
from tqdm import tqdm
import argparse

def sample_n_images(input_dir, n_images, dataset_name):
    
    json_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith("json")]
    if n_images is None or n_images > len(json_files):
        print(f"Warning: you request to sample {n_images} but there are only {len(json_files)} files in the input directory.")
        with open(f'./sampled_{len(json_files)}_images_{dataset_name}.txt', 'w') as f:
            for file in json_files:
                f.write(file+"\n")
        return json_files
    else:
        json_files_sampled = random.sample(json_files, n_images)
        with open(f'./sampled_{n_images}_images_{dataset_name}.txt', 'w') as f:
            for file in json_files_sampled:
                f.write(file+"\n")
        return json_files_sampled


def compute_mask_area(masks, box):
    counts=np.asarray([0,0,0], dtype=np.float32)
    masks = masks.astype(np.uint8)
    for chn in range(masks.shape[1]):
        counts[chn] = masks[box][chn].sum()
                    
    areas = counts / (masks.shape[2]*masks.shape[3])
    max_area = areas.max()
    chn = areas.argmax()

    return float(max_area), int(chn)

def compute_connected_components(image):
    num_labels, _, _, _= cv2.connectedComponentsWithStats(image)
    num_connected_components = num_labels - 1  
    return num_connected_components

def update_json_file(data, image_json_path):
    
    masks = np.load(image_json_path.replace(".json", ".npz"))['masks']
    
    for box_num, box in enumerate(data['boxes']):
        
        area, chn = compute_mask_area(masks, box_num)
        
        box['area'] = area
        if 0 <= area and area <= 0.15:
            dim = 'small'
        elif 0.15 < area and area <= 0.3:
            dim = 'medium'
        elif 0.3 < area and area <= 0.6:
            dim = 'large'
        else:
            dim = 'extra_large'
            
        box['area_size'] = dim
        box['channel_max_area'] = chn
        
        image_array = masks[box_num, chn, :, :].astype(np.uint8) * 255

        num_connected_components = compute_connected_components(image_array)
        box['num_connected_components'] = num_connected_components
    
    with open(image_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    

def get_top_3_masks(image_json_path, save_path):
    with open(image_json_path, 'r') as file:
        data = json.load(file)
    
    img_name = data['path'].split("/")[-1].replace(".jpg", "")
    masks = np.load(image_json_path.replace('.json', '.npz'))['masks']
    
    best_masks = {
        'small':None, 'medium':None, 'large':None
    }
    
    for box_num, box in enumerate(data['boxes'][:30]):
        if box['area_size'] != 'extra_large' and best_masks[box['area_size']] == None:
            best_masks[box['area_size']] = box_num
        
    second_medium, second_large, second_small = False, False, False
    
    smalls = [i for i, med_box in enumerate(data['boxes'][:30]) if med_box['area_size'] == 'small']
    mediums = [i for i, med_box in enumerate(data['boxes'][:30]) if med_box['area_size'] == 'medium']
    larges = [i for i, large_box in enumerate(data['boxes'][:30]) if large_box['area_size'] == 'large']
    
    if best_masks['small'] == None:
        if len(mediums) > 1:
            best_masks['small'] = mediums[1]
            second_medium = True
        elif len(larges) > 1:
            best_masks['small'] = larges[1]
            second_large = True
            
    if best_masks['medium'] == None:
        if len(larges) > 1:
            if second_large:
                best_masks['medium'] = larges[2]
            else:
                best_masks['medium'] = larges[1]
                second_large = True
        elif len(smalls) > 1:
            if best_masks['small'] != None:
                best_masks['medium'] = smalls[1]
                second_small = True
    
    if best_masks['large'] == None:
        if len(mediums) > 1:
            if not second_medium:
                best_masks['large'] = mediums[1]
                second_medium = True
            else:
                best_masks['large'] = mediums[2]
        elif len(smalls) > 1:
            if second_small:
                best_masks['large'] = smalls[2]
            else:
                best_masks['large'] = smalls[1]
    
    for key, value in best_masks.items():
        if value is not None:
            chn = data['boxes'][value]['channel_max_area']
            mask_dim = key
            image_array = masks[value, chn, :, :].astype(np.uint8) * 255
            image = Image.fromarray(image_array)
            
            # apply opening and closing operations to masks 
            img = np.array(image)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            
            cv2.imwrite(f'{save_path}/mask_{img_name}_{mask_dim}.png', closing)
            
        else:
            print(f"Missing {key} region for image {img_name}!")
            with open(os.path.join(save_path, "missed_regions.txt"), "a") as missing:
                missing.write(f"Missing {key} region for image {img_name}\n")
    return best_masks

def draw_bounding_box(image, start_point, end_point, img_name, save_path):
    image = cv2.rectangle(image, start_point, end_point, color=(0, 255, 0), thickness=3)
    cv2.imwrite(f'{save_path}/mask_{img_name}_with_box.png', image)
                 
        
def mask_with_bounding_box(image, start_point, end_point, img_name, save_path):
    black_image = np.zeros_like(image)
    
    width = end_point[0] - start_point[0]
    height = end_point[1] - start_point[1]
    
    if (width*1.1) < image.shape[0] and (height*1.1) < image.shape[1]:
        new_width = int(width * 1.1)
        new_height = int(height * 1.1)
        
        new_start_point = (start_point[0] - (new_width - width) // 2, start_point[1] - (new_height - height) // 2)
        new_end_point = (end_point[0] + (new_width - width) // 2, end_point[1] + (new_height - height) // 2)
    else: 
        new_start_point = start_point
        new_end_point = end_point
        
    image = cv2.rectangle(black_image, new_start_point, new_end_point, color=(255, 255, 255), thickness=-1)
    cv2.imwrite(f'{save_path}/mask_{img_name}_with_box.png', black_image)

def get_images_path(sampled_json_files, dataset_name):
    with open(sampled_json_files, 'r') as f:
        json_files = f.readlines()
    
    images_txt_file = open(f"./source_images_path_{dataset_name}.txt", 'w')
    
    for json_file in json_files:
        with open(json_file.strip(), 'r') as f:
            data = json.load(f)
        images_txt_file.write(data['path']+"\n")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help="Directory containing the output from the mask extraction component (.json and .npz files).")
    parser.add_argument('--num_images', type=int, default=None, help="Number of samples to construct the BtB collection for the pipeline.")
    parser.add_argument('--dataset_name', type=str, default="Flickr30k", help="Name of the source dataset used.")
    parser.add_argument('--save_dir_masks', type=str, help="Directory to save the created masks.")
    parser.add_argument('--save_dir_bb', type=str, help="Directory to save the images with the green bounding box.")
    return parser 

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    masks_save_path = args.save_dir_masks
    os.makedirs(masks_save_path, exist_ok=True)
    
    bounding_box_save_path = args.save_dir_bb
    os.makedirs(bounding_box_save_path, exist_ok=True)
    
    best_mask_labels_file = open(f'./best_mask_labels_{args.dataset_name}.txt', 'a')
    
    # 1. SAMPLE N_IMAGES FROM THE INPUT DIR
    json_files_sampled = sample_n_images(args.input_dir, args.num_images, args.dataset_name)
    
    for json_file in tqdm(json_files_sampled, desc="Processing images", bar_format=f'\033[35m{{l_bar}}{{bar}}\033[0m{{r_bar}}'):
        json_file = json_file.strip()
        
        with open(json_file, 'r') as file:
            data = json.load(file)
            
        image_name = data['path'].split("/")[-1].replace(".jpg", "")
        
        # 2. UPDATE JSON FILE 
        update_json_file(data, json_file)
        
        # 3. GET TOP 3 MASKS (SMALL, MEDIUM, AND LARGE)
        best_masks = get_top_3_masks(json_file, masks_save_path)
        
        for size in ['small', 'medium', 'large']:
            if best_masks[size] is not None:
                best_mask_labels_file.write(f"{image_name}_{size} - {data['boxes'][best_masks[size]]['label']}\n")
                start_point = (data['boxes'][best_masks[size]]['box']['xmin'], data['boxes'][best_masks[size]]['box']['ymin'])
                end_point = (data['boxes'][best_masks[size]]['box']['xmax'], data['boxes'][best_masks[size]]['box']['ymax'])
                draw_bounding_box(cv2.imread(data['path']), start_point, end_point, img_name=image_name+f"_{size}", save_path=bounding_box_save_path)

    # 4. CREATE A TXT FILE CONTAINING THE SOURCE IMAGES PATH
    get_images_path(f'./sampled_{len(json_files_sampled)}_images_{args.dataset_name}.txt', args.dataset_name)
        
    print(f"Dataset {args.dataset_name} preprocessing completed.")
    

    
    
            