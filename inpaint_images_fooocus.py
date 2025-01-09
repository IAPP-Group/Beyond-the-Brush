import requests
import json
import os
import argparse
import subprocess
from datetime import datetime
from tqdm import tqdm
from time import time

HOST = 'http://127.0.0.1:8889'

# GLOBAL PARAMETERS FOR FOOOCUS
GUIDANCE_SCALE = 4.0 
INPAINT_STRENGTH = 0.9
IMAGE_SEED = 2668419004769029052


def inpaint_outpaint(params: dict, input_image: bytes, input_mask: bytes = None) -> dict:
    response = requests.post(
        url=f"{HOST}/v1/generation/image-inpaint-outpaint",
        data=params,
        files={"input_image": input_image,
        "input_mask": input_mask})
    return response.json()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_paths', type=str, help='Txt file containing the path of source images to be inpaint.')
    parser.add_argument('--masks_path', type=str, help='Directory containing the masks of the images.')
    parser.add_argument('--prompt', type=str, help='Directory containing the prompts generated for each images.')
    parser.add_argument('--save_path', type=str, help="Directory to store the final inpainted images.")
    parser.add_argument('--fooocus_api_dir', type=str, help='Location of Fooocus-API directory.')
    return parser 

def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    
    images = os.listdir(args.prompt)
    
    start = time()
    
    for img in tqdm(images, desc="Inpaiting images", bar_format=f'\033[36m{{l_bar}}{{bar}}\033[0m{{r_bar}}'):

        image_name = img[:img.rfind("_")]
        size = img[img.rfind("_")+1:img.find(".json")]
        
        with open(args.images_paths, "r") as f:
            paths = f.readlines()
        
        images_names = [path.split("/")[-1].replace(".jpg", "").strip() for path in paths]
        
        with open(os.path.join(args.prompt, img), 'r') as file:
            prompts = json.load(file)

        for i, prompt in enumerate(prompts):
            source = open(paths[images_names.index(image_name)].strip(), "rb").read()
            mask = open(os.path.join(args.masks_path, f"mask_{image_name}_{size}.png"), "rb").read()
            
            replacement_idx = prompt.find("Replace with: ")
            if replacement_idx ==-1:
                print(f"Unable to find the replacement in the image {image_name}_{size} for prompt {i}")
            else:
                prompt = prompt[replacement_idx+len("Replace with: "):]
                    
                save_name = f"{image_name}_{size}_{i%5}_inpainted"
                
                result = inpaint_outpaint(
                    params={
                        "prompt": prompt,
                        "async_process": False, 
                        "save_name": f"{args.save_path}/{save_name}",
                        "guidance_scale": GUIDANCE_SCALE,
                        "inpaint_strength": INPAINT_STRENGTH,
                        "image_seed": IMAGE_SEED,
                    },
                    input_image=source,
                    input_mask=mask)
    print(f"Inpainting ended in {(time()-start)/60} minutes.")
    
if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
