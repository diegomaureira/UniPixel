# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause license.

import re
import uuid
from functools import partial

import gradio as gr
import imageio.v3 as iio
import spaces
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from PIL import Image

from unipixel.constants import MEM_TOKEN, SEG_TOKEN
from unipixel.dataset.utils import process_vision_info
from unipixel.model.builder import build_model
from unipixel.utils.io import load_image, load_video
from unipixel.utils.transforms import get_sam2_transform
from unipixel.utils.visualizer import draw_mask, sample_color
import yaml

MODEL = 'PolyU-ChenLab/UniPixel-3B'

model, processor = build_model(MODEL, attn_implementation='sdpa')

sam2_transform = get_sam2_transform(model.config.sam2_image_size)

device = torch.device('cuda')

colors = sample_color()
color_map = {f'Target {i + 1}': f'#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}' for i, c in enumerate(colors * 255)}
color_map_light = {
    f'Target {i + 1}': f'#{int(c[0] * 127.5 + 127.5):02x}{int(c[1] * 127.5 + 127.5):02x}{int(c[2] * 127.5 + 127.5):02x}'
    for i, c in enumerate(colors)
}


def enable_btns():
    return (gr.Button(interactive=True), ) * 4


def disable_btns():
    return (gr.Button(interactive=False), ) * 4


def reset_seg():
    return 16, gr.Button(interactive=False)


def reset_reg():
    return 1, gr.Button(interactive=False)


def update_region(blob):
    if blob['background'] is None or not blob['layers'][0].any():
        return

    region = blob['background'].copy()
    region[blob['layers'][0][:, :, -1] == 0] = [0, 0, 0, 0]

    return region


def update_video(video, prompt_idx):
    if video is None:
        return gr.ImageEditor(value=None, interactive=False)

    _, images = load_video(video, sample_frames=16)
    component = gr.ImageEditor(value=images[prompt_idx - 1], interactive=True)

    return component


@spaces.GPU
def infer_seg(media, query, sample_frames=16, media_type=None):
    global model

    if not media:
        gr.Warning('Please upload an image or a video.')
        return None, None, None

    if not query:
        gr.Warning('Please provide a text prompt.')
        return None, None, None

    if any(media.endswith(k) for k in ('jpg', 'png')):
        frames, images = load_image(media), [media]
    else:
        frames, images = load_video(media, sample_frames=sample_frames)

    messages = [{
        'role':
        'user',
        'content': [{
            'type': 'video',
            'video': images,
            'min_pixels': 128 * 28 * 28,
            'max_pixels': 256 * 28 * 28 * int(sample_frames / len(images))
        }, {
            'type': 'text',
            'text': query
        }]
    }]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)

    data = processor(text=[text], images=images, videos=videos, return_tensors='pt', **kwargs)

    data['frames'] = [sam2_transform(frames).to(model.sam2.dtype)]
    data['frame_size'] = [frames.shape[1:3]]

    model = model.to(device)

    output_ids = model.generate(
        **data.to(device),
        do_sample=False,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        max_new_tokens=512)

    assert data.input_ids.size(0) == output_ids.size(0) == 1
    output_ids = output_ids[0, data.input_ids.size(1):]

    if output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]

    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
    response = response.replace(f' {SEG_TOKEN}', SEG_TOKEN).replace(f'{SEG_TOKEN} ', SEG_TOKEN)

    entities = []
    for i, m in enumerate(re.finditer(re.escape(SEG_TOKEN), response)):
        entities.append(dict(entity=f'Target {i + 1}', start=m.start(), end=m.end()))

    answer = dict(text=response, entities=entities)

    imgs = draw_mask(frames, model.seg, colors=colors)

    path = f"/tmp/{uuid.uuid4().hex}.{'gif' if len(imgs) > 1 else 'png'}"
    iio.imwrite(path, imgs, duration=100, loop=0)

    if media_type == 'image':
        if len(model.seg) >= 1:
            masks = media, [(m[0, 0].numpy(), f'Target {i + 1}') for i, m in enumerate(model.seg)]
        else:
            masks = None
    else:
        masks = path

    return answer, masks, path


infer_seg_image = partial(infer_seg, media_type='image')
infer_seg_video = partial(infer_seg, media_type='video')


@spaces.GPU
def infer_reg(blob, query, prompt_idx=1, video=None):
    global model

    if blob['background'] is None:
        gr.Warning('Please upload an image or a video.')
        return

    if not blob['layers'][0].any():
        gr.Warning('Please provide a mask prompt.')
        return

    if not query:
        gr.Warning('Please provide a text prompt.')
        return

    if video is None:
        frames = torch.from_numpy(blob['background'][:, :, :3]).unsqueeze(0)
        images = [Image.fromarray(blob['background'], mode='RGBA')]
    else:
        frames, images = load_video(video, sample_frames=16)

    frame_size = frames.shape[1:3]

    mask = torch.from_numpy(blob['layers'][0][:, :, -1]).unsqueeze(0) > 0

    refer_mask = torch.zeros(frames.size(0), 1, *frame_size)
    refer_mask[prompt_idx - 1] = mask

    if refer_mask.size(0) % 2 != 0:
        refer_mask = torch.cat((refer_mask, refer_mask[-1, None]))
    refer_mask = refer_mask.flatten(1)
    refer_mask = F.max_pool1d(refer_mask.transpose(-1, -2), kernel_size=2, stride=2).transpose(-1, -2)
    refer_mask = refer_mask.view(-1, 1, *frame_size)

    if video is None:
        prefix = f'Here is an image with the following highlighted regions:\n[0]: <{prompt_idx}> {MEM_TOKEN}\n'
    else:
        prefix = f'Here is a video with {len(images)} frames denoted as <1> to <{len(images)}>. The highlighted regions are as follows:\n[0]: <{prompt_idx}>-<{prompt_idx + 1}> {MEM_TOKEN}\n'

    messages = [{
        'role':
        'user',
        'content': [{
            'type': 'video',
            'video': images,
            'min_pixels': 128 * 28 * 28,
            'max_pixels': 256 * 28 * 28 * int(16 / len(images))
        }, {
            'type': 'text',
            'text': prefix + query
        }]
    }]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)

    data = processor(text=[text], images=images, videos=videos, return_tensors='pt', **kwargs)

    refer_mask = T.resize(refer_mask, (data['video_grid_thw'][0][1] * 14, data['video_grid_thw'][0][2] * 14))
    refer_mask = F.max_pool2d(refer_mask, kernel_size=28, stride=28)
    refer_mask = refer_mask > 0

    data['frames'] = [sam2_transform(frames).to(model.sam2.dtype)]
    data['frame_size'] = [frames.shape[1:3]]
    data['refer_mask'] = [refer_mask]

    model = model.to(device)

    output_ids = model.generate(
        **data.to(device),
        do_sample=False,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        max_new_tokens=512)

    assert data.input_ids.size(0) == output_ids.size(0) == 1
    output_ids = output_ids[0, data.input_ids.size(1):]

    if output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]

    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
    response = response.replace(' [0]', '[0]').replace('[0] ', '[0]').replace('[0]', '<REGION>')

    entities = []
    for m in re.finditer(re.escape('<REGION>'), response):
        entities.append(dict(entity='region', start=m.start(), end=m.end(), color="#f85050"))

    answer = dict(text=response, entities=entities)

    return answer

def load_images_from_folder(folder_path):
    import os
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_formats):
            images.append(os.path.join(folder_path, filename))
    return images

import os
import shutil

def copy_file(prompt, image_path, output_dir, output_path, masks = None):
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        prompt_filename = prompt.replace(' ', '_')
        filename = f"PROMPT-{prompt_filename.lower()}_NAME-{os.path.basename(image_path)}"
        if 'mp4' in image_path:
            image_save_path = filename.replace('.mp4', '.gif')
        else:
            image_save_path = filename
        shutil.copy(output_path, os.path.join(output_dir, image_save_path))
        
        if masks:
            import pickle
            mask_output_path = os.path.join(output_dir, filename.replace('.gif', '_masks.pkl').replace('.jpg', '_masks.pkl')).replace('.png', '_masks.pkl')
            pickle.dump(masks, open(mask_output_path, 'wb'))

def run_batch_inference(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    for idx, task in enumerate(config["tasks"]):
        ttype = task["type"]

        if ttype == "image_segmentation":
            image_path = task["image_path"]
            prompt = task["prompt"]
            output_dir = task.get("output_dir", None)
            # Optionally handle other args
            result = infer_seg_image(image_path, prompt)
            answer, masks, output_path = result
            copy_file(prompt, image_path, output_dir, output_path, masks)
            
        elif ttype == "folder_segmentation":
            folder_path = task["folder_path"]
            prompt = task["prompt"]
            output_dir = task.get("output_dir", None)
            images = load_images_from_folder(folder_path)
            for img_path in images:
                result = infer_seg_image(img_path, prompt)
                answer, masks, output_path = result
                copy_file(prompt, img_path, output_dir, output_path, masks)

        elif ttype == "video_segmentation":
            video_path = task["video_path"]
            prompt = task["prompt"]
            output_dir = task.get("output_dir", None)
            sample_frames = task.get("sample_frames", 16)
            result = infer_seg_video(video_path, prompt, sample_frames)
            answer, masks, output_path = result
            copy_file(prompt, video_path, output_dir, output_path, masks)

        elif ttype == "image_regional":
            image_path = task["image_path"]
            prompt = task["prompt"]
            region_mask = task.get("region_mask", None)
            result = infer_reg(image_path, prompt, region_mask)
            print(f"Task {idx}: Image regional understanding done, result: {result}")

        else:
            print(f"Task {idx}: Unknown task type {ttype}")

if __name__ == "__main__":
    run_batch_inference("demo/infer_config.yaml")