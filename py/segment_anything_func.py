# layerstyle advance

import os
import sys
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

import copy
import torch
import numpy as np
from PIL import Image, ImageDraw
import logging
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import folder_paths
import comfy.model_management
from sam_hq.predictor import SamPredictorHQ
from sam_hq.build_sam_hq import sam_model_registry
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

import glob
import folder_paths

logger = logging.getLogger('comfyui_segment_anything')

sam_model_dir_name = "sams"
sam_model_list = {
    "sam_vit_h (2.56GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    },
    "sam_vit_l (1.25GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    },
    "sam_vit_b (375MB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    },
    "sam_hq_vit_h (2.57GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"
    },
    "sam_hq_vit_l (1.25GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"
    },
    "sam_hq_vit_b (379MB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth"
    },
    "mobile_sam(39MB)": {
        "model_url": "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt"
    },
    "sam2.1_hiera_tiny": {
        "model_url": "facebook/sam2.1-hiera-tiny"
    },
    "sam2.1_hiera_small": {
        "model_url": "facebook/sam2.1-hiera-small"
    },
    "sam2.1_hiera_base_plus": {
        "model_url": "facebook/sam2.1-hiera-base-plus"
    },
    "sam2.1_hiera_large": {
        "model_url": "facebook/sam2.1-hiera-large"
    },
    "sam2_hiera_tiny": {
        "model_url": "facebook/sam2-hiera-tiny"
    },
    "sam2_hiera_small": {
        "model_url": "facebook/sam2-hiera-small"
    },
    "sam2_hiera_base_plus": {
        "model_url": "facebook/sam2-hiera-base-plus"
    },
    "sam2_hiera_large": {
        "model_url": "facebook/sam2-hiera-large"
    },
}

groundingdino_model_dir_name = "grounding-dino"
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
    },
}

def get_bert_base_uncased_model_path():
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        print('grounding-dino is using models/bert-base-uncased')
        return comfy_bert_model_base
    return 'bert-base-uncased'

def list_sam_model():
    return list(sam_model_list.keys())

def load_sam_model(model_name):
    model_url = sam_model_list[model_name]["model_url"]

    if model_name.startswith("sam2"):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = SAM2ImagePredictor.from_pretrained(model_url, device=device)
        sam.model_name = model_name  
        return sam

    # Fallback to SAM1/HQ/mobile loading
    sam_checkpoint_path = get_local_filepath(
        sam_model_list[model_name]["model_url"], sam_model_dir_name)
    model_file_name = os.path.basename(sam_checkpoint_path)
    model_type = os.path.splitext(model_file_name)[0]

    if 'hq' not in model_type and 'mobile' not in model_type:
        model_type = '_'.join(model_type.split('_')[:-1])

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam_device = comfy.model_management.get_torch_device()
    sam.to(device=sam_device)
    sam.eval()
    sam.model_name = model_file_name
    return sam



def get_local_filepath(url, dirname, local_file_name=None):
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)

    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination:
        logger.warn(f'using extra model: {destination}')
        return destination

    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        logger.warn(f'downloading {url} to {destination}')
        download_url_to_file(url, destination)
    return destination

def load_groundingdino_model(model_name):
    from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
    from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
    from local_groundingdino.models import build_model as local_groundingdino_build_model
    dino_model_args = local_groundingdino_SLConfig.fromfile(
        get_local_filepath(
            groundingdino_model_list[model_name]["config_url"],
            groundingdino_model_dir_name
        ),
    )

    if dino_model_args.text_encoder_type == 'bert-base-uncased':
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()
    
    dino = local_groundingdino_build_model(dino_model_args)
    checkpoint = torch.load(
        get_local_filepath(
            groundingdino_model_list[model_name]["model_url"],
            groundingdino_model_dir_name,
        ),
    )
    dino.load_state_dict(local_groundingdino_clean_state_dict(
        checkpoint['model']), strict=False)
    device = comfy.model_management.get_torch_device()
    dino.to(device=device)
    dino.eval()
    return dino

def list_groundingdino_model():
    return list(groundingdino_model_list.keys())

def groundingdino_predict(
    dino_model,
    image,
    prompt,
    threshold
):
    from local_groundingdino.datasets import transforms as T
    def load_dino_image(image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def get_grounding_output(model, image, caption, box_threshold):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        return boxes_filt.cpu()

    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt = get_grounding_output(
        dino_model, dino_image, prompt, threshold
    )
    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt

def create_tensor_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_image, output_mask = split_image_mask(
            Image.fromarray(image_np_copy))
        output_masks.append(output_mask)
        output_images.append(output_image)
    return (output_images, output_masks)

def split_image_mask(image):
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,]
    if 'A' in image.getbands():
        mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image_rgb, mask)

def save_masks_and_boxes(outdir, image_rgba, boxes, masks, prefix="", scores=None):
    os.makedirs(outdir, exist_ok=True)

    # Save image with bounding boxes
    image_with_boxes = Image.fromarray(image_rgba.copy())
    draw = ImageDraw.Draw(image_with_boxes)

    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

        label = f"{i}"
        if scores is not None and i < len(scores):
            label += f": {scores[i].item():.2f}"

        draw.text((x0 + 2, y0 + 2), label, fill="white")

    image_with_boxes.save(os.path.join(outdir, f"{prefix}_image_with_boxes.png"))

    # --- Normalize all weird mask shapes ---
    masks = np.array(masks)

    if masks.ndim == 2:
        # Single mask (H, W)
        masks = masks[np.newaxis, :, :]
    elif masks.ndim == 3:
        # (N, H, W) – okay
        pass
    elif masks.ndim == 4:
        if masks.shape[0] == 1:
            masks = masks[0]
        else:
            raise ValueError(f"Don't know how to handle 4D mask with shape {masks.shape}")
    else:
        raise ValueError(f"Unsupported mask shape: {masks.shape}")

    if masks.ndim != 3:
        raise ValueError(f"Expected 3D mask after normalization, got {masks.shape}")

    # Save each mask
    for i, mask in enumerate(masks):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(os.path.join(outdir, f"{prefix}_mask_{i}.png"))

def sam_segment(sam_model, image, boxes, sam_score_threshold):
    if boxes.shape[0] == 0:
        return None

    if hasattr(sam_model, '__class__') and sam_model.__class__.__name__ == "SAM2ImagePredictor":
        image_np = np.array(image.convert("RGB"))  # Ensure RGB np array
        sam_model.set_image(image_np)
        all_masks = []
        all_scores = []
        for i, box in enumerate(boxes):
            box_arr = box.cpu().numpy()
            masks, scores, _ = sam_model.predict(
                box=box_arr,
                point_coords=None,
                point_labels=None,
                multimask_output=False
            )
            if scores[0] < sam_score_threshold:
             continue
            mask = masks[0]
            score = scores[0]
            all_scores.append(score)
            # Resize mask to match original image size (W, H)
            mask_resized = Image.fromarray((mask * 255).astype("uint8")).resize(image.size, Image.NEAREST)
            mask_resized_np = np.array(mask_resized)

            # Ensure mask is (H, W)
            if mask_resized_np.ndim == 3:
                mask_resized_np = mask_resized_np[:, :, 0]

            # Convert to boolean mask
            mask_resized_np = mask_resized_np.astype(bool)
            all_masks.append(mask_resized_np)

        stacked = np.stack(all_masks, axis=0)
        print(f"[SAM2 DEBUG] masks shape (before): {stacked.shape}")
        image_np_rgba = np.array(image.convert("RGBA"))

        if stacked.ndim == 3:
            stacked = np.expand_dims(stacked, axis=0)

        print(f"[SAM2 DEBUG] image_np shape: {image_np.shape}")
        print(f"[SAM2 DEBUG] image_rgba shape: {image_np_rgba.shape}")
        print(f"[SAM2 DEBUG] masks shape (after): {stacked.shape}")
        print(f"[SAM2 DEBUG] boxes shape: {boxes.shape}")
        print(f"[SAM2 DEBUG] scores: {all_scores}")
        save_masks_and_boxes("/tmp/sam2_output", np.array(image.convert("RGBA")), boxes.cpu().numpy(), stacked, prefix="sam2", scores=all_scores)
        return create_tensor_output(image_np_rgba, stacked, boxes)

    # Fallback
    return sam_segment_hq(sam_model, image, boxes, sam_score_threshold)


def sam_segment_hq(sam_model, image, boxes, sam_score_threshold):
    sam_is_hq = hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name
    predictor = SamPredictorHQ(sam_model, sam_is_hq)

    # 1) prepare + predict
    image_np = np.array(image)[..., :3]
    predictor.set_image(image_np)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image_np.shape[:2])
    device = comfy.model_management.get_torch_device()
    masks_t, scores_t, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False
    )

    # 2) to numpy & squeeze
    masks_np  = masks_t.squeeze(1).cpu().numpy()   # (N, H, W)
    scores_np = scores_t.squeeze(-1).cpu().numpy() # (N,)
    boxes_np  = boxes.cpu().numpy()                # (N, 4)

    # 3) filter by score
    keep = scores_np >= sam_score_threshold
    masks_np  = masks_np[keep]                     # (M, H, W)
    boxes_np  = boxes_np[keep]                     # (M, 4)
    scores_np = scores_np[keep]

    # 4) restore channel-dim for save/return
    masks_np = masks_np[:, None, :, :]             # (M, 1, H, W)

    # 5) convert only boxes back to tensor
    boxes_torch = torch.from_numpy(boxes_np)       # (M, 4)

    # 6) save & output
    image_rgba = np.array(image.convert("RGBA"))

    # pass numpy-array masks, torch-boxes:
    return create_tensor_output(image_rgba, masks_np, boxes_torch)