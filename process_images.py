#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForMaskGeneration, AutoModelForZeroShotObjectDetection, pipeline
from ram.models import ram
from ram import get_transform


class Tagger:
    def __init__(self, image_size, device, ram_model_path):
        self.transform = get_transform(image_size=image_size)
        self.model = ram(pretrained=ram_model_path, # './ram_swin_large_14m.pth',
                         image_size=image_size,
                         vit='swin_l')
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device

    def __call__(self, images):
        image_tensor = torch.stack(tuple(self.transform(i).to(self.device) for i in images))

        with torch.no_grad():
            tags, tags_chinese = self.model.generate_tag(image_tensor)
        return [x.split(' | ') for x in tags]


class Detector:
    def __init__(self, device):
        model_id = "IDEA-Research/grounding-dino-tiny"

        self.object_detector = pipeline(model=model_id, task="zero-shot-object-detection", device=device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def __call__(self, images, tags):
        inputs = [
            {
                "image": i,
                "candidate_labels": [t + '.' for t in ts]
            }
            for i, ts in zip(images, tags)
        ]
        return self.object_detector(inputs)


class Segmenter:
    def __init__(self, device, batch_boxes=4):
        segmenter_id = "facebook/sam-vit-base"

        self.segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
        self.processor = AutoProcessor.from_pretrained(segmenter_id)
        self.device = device
        self.batch_boxes = batch_boxes

    def __call__(self, image, boxes):
        n_batches = len(boxes) // self.batch_boxes + (1 if len(boxes) % self.batch_boxes != 0 else 0)

        all_masks = np.empty((len(boxes), 3, image.size[1], image.size[0]), dtype=bool)
        
        print(f"Running {n_batches} batches")
        for i in range(n_batches):
            input_boxes = [boxes[i * self.batch_boxes : (i + 1) * self.batch_boxes]]
            inputs = self.processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(self.device)
    
            outputs = self.segmentator(**inputs)
            masks = self.processor.post_process_masks(
                masks=outputs.pred_masks,
                original_sizes=inputs.original_sizes,
                reshaped_input_sizes=inputs.reshaped_input_sizes
            )[0]

            all_masks[i * self.batch_boxes : min((i + 1) * self.batch_boxes, len(boxes)), :, :, :] = masks.cpu().numpy()

        return all_masks


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-o', '--output-path', default='./out')
    parser.add_argument('-t', '--top-masks', type=int, default=None)
    parser.add_argument('-r', '--ram-model-path', required=True)
    parser.add_argument('input_list_path')

    return parser

class Batched:
    def __init__(self, base, batch_size):
        self.base = base
        self.batch_size = batch_size
        self.next_batch = 0

    def __len__(self):
        base_len = len(self.base)
        if base_len % self.batch_size != 0:
            return base_len // self.batch_size + 1
        else:
            return base_len // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch >= len(self):
            raise StopIteration

        items = self.base[self.next_batch * self.batch_size : (self.next_batch + 1) * self.batch_size]
        self.next_batch += 1

        return items


class BatchedZip:
    def __init__(self, base, batch_size):
        self.base = base
        self.base_len = len(self.base[0])
        for x in self.base[1:]:
            assert len(x) == self.base_len
        self.batch_size = batch_size
        self.next_batch = 0

    def __len__(self):
        if self.base_len % self.batch_size != 0:
            return self.base_len // self.batch_size + 1
        else:
            return self.base_len // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch >= len(self):
            raise StopIteration

        items = [x[self.next_batch * self.batch_size : (self.next_batch + 1) * self.batch_size] for x in self.base]
        self.next_batch += 1

        return items


def load_all_json(parent_path, concat=False):
    result = []
    for item in sorted(os.listdir(parent_path)):
        with open(os.path.join(parent_path, item), 'r') as stream:
            result.append(json.load(stream))

    if concat:
        return [item for sub in result for item in sub]
    else:
        return result


def main(args):
    tagger = Tagger(384, args.device, args.ram_model_path)
    detector = Detector(args.device)
    segmenter = Segmenter(args.device)

    with open(args.input_list_path, 'r') as stream:
        image_paths = [x.strip() for x in stream.readlines()]

    os.makedirs(os.path.join(args.output_path, 'tmp', 'tags'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'tmp', 'boxes'), exist_ok=True)
    
    for i, ps in enumerate(tqdm(Batched(image_paths, 64))):
        tags_path = os.path.join(args.output_path, 'tmp', 'tags', f'{i:07d}-tags.json')
        if not os.path.exists(tags_path):
            images = [Image.open(p).convert("RGB") for p in ps]
            tags = tagger(images)
            with open(tags_path, 'w') as stream:
                json.dump(tags, stream)

    del tagger

    all_tags = load_all_json(os.path.join(args.output_path, 'tmp', 'tags'), concat=True)
    print("Tags OK")
    
    for i, (ps, tags) in enumerate(tqdm(BatchedZip((image_paths, all_tags), 4))):
        boxes_path = os.path.join(args.output_path, 'tmp', 'boxes', f'{i:07d}-boxes.json')
        if not os.path.exists(boxes_path):
            images = [Image.open(p).convert("RGB") for p in ps]
            boxes = detector(images, tags)
            
            with open(boxes_path, 'w') as stream:
                json.dump(boxes, stream)

    del detector

    all_boxes = load_all_json(os.path.join(args.output_path, 'tmp', 'boxes'), concat=True)
    print("Boxes OK")

    for i, (p, tags, boxes) in enumerate(zip(tqdm(image_paths), all_tags, all_boxes)):
        result_path = os.path.join(args.output_path, f'{i:07d}.json')
        if not os.path.exists(result_path):
            image = Image.open(p).convert("RGB")
            if args.top_masks is not None:
                boxes = boxes[:args.top_masks]
            in_boxes = [[m['box']['xmin'], m['box']['ymin'], m['box']['xmax'], m['box']['ymax']] for m in boxes]
            segmented_masks = segmenter(image, in_boxes)
            
            result = {
                'path': p,
                'tags': tags,
                'boxes': [
                    { 'score': box['score'], 'label': box['label'][:-1], 'box': box['box'] }
                    for box in boxes
                ]
            }
            
            with open(result_path, 'w') as stream:
                json.dump(result, stream, indent=2)
            masks_path = os.path.join(args.output_path, f'{i:07d}.npz')
            np.savez_compressed(masks_path, masks=segmented_masks)
    
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
