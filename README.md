# Beyond-the-Brush
This repository containes the code of the paper: "(Beyond-the-Brush: Fully-automated Crafting of Realistic Inpainted Images)[https://ieeexplore.ieee.org/abstract/document/10810722]" in WIFS, 2024.

## Framework

The BtB framework consists of three main modules:
<ul>
    <li> <b>Mask Extraction</b>: identifies meaningful regions in an image to be inpainted using a segmentation procedure.
    <li> <b>Prompt Generation</b>: uses a Visual Language Model to determine the replacement content for the identified regions.
    <li> <b>Inpaiting</b>: inpaints the input images using the extracted masks and generated prompts.
</ul>

![BtB Framework](images/InpaintingPipeline.png)


## Usage

### Requirements
```
conda create -n btb python=3.8
conda activate btb
pip install -r requirements.txt
```

### 1. Mask Extraction

### 1.1 Postprocess Images
Once masks have been extracted, you need to postprocess the output from the mask extraction component. To this end, you can run the following command:

`python post_processing.py --input_dir --num_images --dataset_name --save_dir_masks --save_dir_bb`

where `--input_dir` is the directory containing the outputs from the mask extraction module (i.e., .json and .npz files); `--num_images` is the number of images you want to sample to construct the BtB collection for the pipeline (if None it will use all the images in the directory); `--dataset_name` is the name of the source dataset (for example Flickr30k, VISION, FloreView, etc.); `--save_dir_masks` is the directory to save the actual masks (as .png files); `--save_dir_bb` is the directory to save the images with the green bounding box.

This script will generate three txt files within the project directory:
<ul>
    <li> "best_mask_labels_<i>dataset_name</i>.txt": it contains the list of images with their labels and sizes. Specifically, each line in the file follows the format: ImageName_MaskAreaSize - Label.
    <li> "sampled_<i>num_images</i>_images_<i>dataset_name</i>.txt": it contains the list of sampled json files from the input_dir.
    <li> "source_images_path_<i>dataset_name</i>.txt": it contains the paths of the sampled source images.
</ul>

### 2. Prompt Generation
After the postprocessing operation, we use a Visual Language Model to generate the prompt for each image of the source dataset. To this end, you can run the following command:

`python generate_prompts.py --output_dir --bounding_box_dir --labels_file --num_prompt`

where `--output_dir` is the directory to store the json file containing the generated prompt for each image; `--bounding_box_dir` is the directory containing the images with the green bounding box produced in the postprocessing step; `--labels_file` is the txt file containing the labels for each image produced in the postprocessing step; `--num_prompt` (optional) is the number of prompts to generate for each image (by default is 5 prompts for each image).

### 3. Inpaiting
The final step is the actual inpaiting of the source images. We used Fooocus as image inpaiting software.

First of all, you need to install [Fooocus](https://github.com/lllyasviel/Fooocus) and [Fooocus-API](https://github.com/mrhan1993/Fooocus-API) by following the instructions in the corresponding repositories.

Once installed, start the Fooocus-API app.
Then, you can run the following command to inpaint the images:

`python inpaint_images_fooocus.py --images_path --masks_path --prompt --save_path --fooocus_api_dir`

where `--images_path` is the txt file containing the path of source images to be inpaint; `--masks_path` is the directory containing the masks of the images; `--prompt` is the directory containing the prompts generated for each images; `--save_path` is the directory to store the resulting inpainted images; `--fooocus_api_dir` is the location of Fooocus-API directory.

Please, be sure to properly set the HOST variable within the script, which depends on where the Fooocus-API app is running.

### Dataset
Link al dataset di HF

### Citation
Please, if you use this code cite out paper:
```
@inproceedings{bertazzini2024beyond,
  title={Beyond the Brush: Fully-automated Crafting of Realistic Inpainted Images},
  author={Bertazzini, Giulia and Albisani, Chiara and Baracchi, Daniele and Shullani, Dasara and Piva, Alessandro},
  booktitle={2024 IEEE International Workshop on Information Forensics and Security (WIFS)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```
