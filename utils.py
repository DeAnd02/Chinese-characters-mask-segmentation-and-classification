import os
from torch.utils.data import Dataset, DataLoader 
import json
from collections import Counter
import cv2
import numpy as np
import random 
import torch

TARGET_IMG_HEIGHT = 1024
TARGET_IMG_WIDTH = 1024
TARGET_IMG_SIZE = 1024 
BASE = "/srv/data/lt2326-h25/lt2326-h25/a1"
IMG_DIR = os.path.join(BASE, "images")
TRAIN_JSONL = os.path.join(BASE, "train.jsonl")

# Costants of the datasset
IMG_HEIGHT = 2048
IMG_WIDTH = 2048
IMG_CHANNELS = 3

class ClassificationDataset(Dataset):
    """
    Dataset for Bonus A: Binary classification (presence/absence of the target character).
    Reuses the logic for loading and resizing images.
    """
    def __init__(self, metadata_list, target_char, target_h=256, target_w=256):
        self.metadata = metadata_list
        self.TH = target_h # Target Height
        self.TW = target_w # Target Width
        self.IMG_CHANNELS = 3
        self.target_char = target_char # The specific character to detect

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        anno = self.metadata[idx]
        
        # IMAGE LOADING AND RESIZING
        img = cv2.imread(anno["file_path"], cv2.IMREAD_COLOR)
        if img is None:
             raise RuntimeError(f"Failed to read image: {anno['file_path']}")

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H_orig, W_orig = img.shape[:2]
        
        # Resize the image 
        if (H_orig, W_orig) != (self.TH, self.TW):
            image_resized = cv2.resize(img, (self.TW, self.TH), interpolation=cv2.INTER_AREA)
        else:
            image_resized = img

        # Convert to PyTorch tensor and normalize
        image = image_resized[:, :, :self.IMG_CHANNELS]
        image = np.transpose(image, (2, 0, 1)) 
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image)
        
         # BINARY LABEL GENERATION
        is_present = 0 # Default: 0 = Absent
        
        # Iterate through annotations to check if the target character is present
        for textline in anno.get("annotations", []):
            for inst in textline: 
                char_text = inst.get('text', '').strip()
                
                if char_text == self.target_char:
                    is_present = 1 # 1: Present
                    break 
            if is_present == 1:
                break
        
        # Create the label as a single float tensor (required by BCELoss for binary classification)
        label = torch.tensor([float(is_present)]) 
            
        return image_tensor, label

# Function for reading json files
def read_jsonl(path):
    """Read a .jsonl (JSON Lines) file and return a list of JSON objects."""
    anns = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    anns.append(json.loads(line))
    except FileNotFoundError:
        print(f"WARNING: File not found: {path}")
    return anns


# Function to create the mask from the given polygon properties
def rasterize_polygon_mask(h, w, polygons):
    # Initialize an empty mask filled with zeros (background)
    mask = np.zeros((h, w), dtype=np.uint8)
    if not polygons:
        return mask
    
    # Convert each polygon into the format required by cv2.fillPoly:
    # - Convert to float32, then round to nearest int
    # - Cast to int32
    # - Reshape into shape (N, 1, 2) as expected by OpenCV
    pts = [np.round(np.array(p, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2) for p in polygons]

    # Fill the polygons on the mask with value 1 (foreground)
    cv2.fillPoly(
        mask,
        pts=pts,
        color=1  # Set the color for the region
    )
    return mask


def load_split_metadata(jsonl_path, img_dir):
    """
    Load annotations from a JSONL file and check if the corresponding image 
    files exist (as we are using a small part of the whole dataset).
    Returns a list of metadata dictionaries instead of full NumPy arrays.
    """
       
    annos = read_jsonl(jsonl_path)
    if not os.path.isdir(img_dir):
        print(f"WARNING: Image directory not found: {img_dir}.")
        return []

    metadata_list = []
    
    # Build a metadata list with valid image paths and annotations
    for anno in annos:
        fname = anno.get("file_name")
        if fname and os.path.exists(os.path.join(img_dir, fname)):
            metadata_list.append({
                "file_path": os.path.join(img_dir, fname),
                "annotations": anno.get("annotations", []),
                "ignore": anno.get("ignore", [])
            })
            
    print(f"Loading of {os.path.basename(jsonl_path)}: Found {len(metadata_list)} valid images.")
    return metadata_list


def find_most_common_character(jsonl_path):
    """
    Reads the training JSONL file and counts the frequency of each character.
    It ignores non-Chinese characters (is_chinese = False) and special symbols.
    """
    character_counts = Counter()   # Dictionary-like object to count character frequencies
    total_characters = 0           # Total valid characters found
    total_images = 0               # Total number of images (lines) processed

    try:
        # Open the JSONL file line by line
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_images += 1
                try:
                    data = json.loads(line.strip())
                    
                    # Iterate over all annotation lists
                    for annotation_list in data.get("annotations", []):
                        # Iterate over each single annotation in the list
                        for anno in annotation_list:
                            # Check if it's a Chinese character and has a 'text' field
                            if anno.get("is_chinese", False) and "text" in anno:
                                char = anno["text"].strip()
                                
                                if char and char != ' ' and char != '·': # Filter
                                    character_counts[char] += 1
                                    total_characters += 1
                                    
                except json.JSONDecodeError as e:
                    print(f"JSON decoding error in line: {e}")
                
    except FileNotFoundError:
        # Handle missing file error
        print(f"ERROR: File not found at {jsonl_path}")
        return None

    # If no valid characters were found
    if not character_counts:
        return None, 0, total_characters
        
    most_common = character_counts.most_common(1)[0]
        
    return most_common[0], most_common[1], total_characters


# Target Images size for shrinking
TARGET_IMG_HEIGHT = 1024
TARGET_IMG_WIDTH = 1024

class ImageMaskDataset(Dataset):
    """
    Dataset with lazy loading and integrated resizing.
    Loads a high-res image (e.g., 2048x2048), rasterizes polygon labels
    at the ORIGINAL resolution, then downsamples to the target size
    (defaults: 1024 x 1024).
    Returns (image_tensor, mask_tensor).
    """
    
    def __init__(self, metadata_list, target_h=TARGET_IMG_HEIGHT, target_w=TARGET_IMG_WIDTH):
        self.metadata = metadata_list
        self.TH = target_h
        self.TW = target_w
        self.IMG_CHANNELS = 3

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # ---- LOAD IMAGE AT ORIGINAL RESOLUTION ----
        anno = self.metadata[idx]

        # Read BGR image; raise if missing or unreadable
        img = cv2.imread(anno["file_path"], cv2.IMREAD_COLOR)
        if img is None:
             raise RuntimeError(f"Failed to read image: {anno['file_path']}")

        # Convert BGR to RGB and copy to ensure a contiguous array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        chinese_polys, ignore_polys = [], []
        # Extract polygons marked as chinese (or default True)
        # Expected structure: anno["annotations"] is a list of textlines,
        # each textline is a list of instances (dicts)
        for textline in anno["annotations"]:
            for inst in textline:
                poly = inst.get("polygon")
                if poly and len(poly) >= 3 and inst.get("is_chinese", True):
                    chinese_polys.append(poly)

        # Polygons to ignore (will be subtracted from the final mask)
        for ign in anno["ignore"]:
            poly = ign.get("polygon")
            if poly and len(poly) >= 3:
                ignore_polys.append(poly)
                
        # ---- RASTERIZE MASKS AT ORIGINAL SIZE ----
        H_orig, W_orig = img.shape[:2]
        chinese_mask = rasterize_polygon_mask(H_orig, W_orig, chinese_polys)
        ignore_mask = rasterize_polygon_mask(H_orig, W_orig, ignore_polys)

        # Keep pixels that are chinese AND NOT ignored
        # final_mask is boolean; cast to uint8 {0,1}
        final_mask = (chinese_mask == 1) & (ignore_mask == 0)
        mask_orig = final_mask.astype(np.uint8) 
        
        # ---- RESIZE TO TARGET SIZE ----
        # Image: use AREA for high-quality downsampling
        if (H_orig, W_orig) != (self.TH, self.TW):
            image_resized = cv2.resize(img, (self.TW, self.TH), interpolation=cv2.INTER_AREA)
        else:
            image_resized = img

        # Mask: use NEAREST to preserve crisp class boundaries and small parts (interpolation)
        if (H_orig, W_orig) != (self.TH, self.TW):
            mask_resized = cv2.resize(mask_orig, (self.TW, self.TH), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask_orig
            
        # ---- CONVERT TO PYTORCH TENSORS ----
        # Image: HWC -> CHW, float32 in [0,1]
        image = image_resized[:, :, :self.IMG_CHANNELS]
        image = np.transpose(image, (2, 0, 1)) 
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image)
        
        # Mask: shape from (H, W) to (1, H, W), float32 in {0.0, 1.0}
        mask_tensor = torch.from_numpy(mask_resized[np.newaxis, :, :].astype(np.float32))
            
        return image_tensor, mask_tensor


def load_dataset():
    train_metadata = load_split_metadata(TRAIN_JSONL, IMG_DIR)
    train_dataset = ImageMaskDataset(train_metadata, target_h=TARGET_IMG_HEIGHT, target_w=TARGET_IMG_WIDTH)

    val_metadata = load_split_metadata(os.path.join(BASE, "val.jsonl"), IMG_DIR)
    val_dataset = ImageMaskDataset(val_metadata, target_h=TARGET_IMG_HEIGHT, target_w=TARGET_IMG_WIDTH)
    
    return train_dataset, val_dataset


def load_classification_dataset():
    MOST_COMMON_CHAR, count, total = find_most_common_character(TRAIN_JSONL) 
    val_metadata = load_split_metadata(os.path.join(BASE, "val.jsonl"), IMG_DIR)
    val_clf_dataset = ClassificationDataset(val_metadata, target_char=MOST_COMMON_CHAR,
                                        target_h=TARGET_IMG_SIZE, target_w=TARGET_IMG_SIZE)
    
    return val_metadata, val_clf_dataset, MOST_COMMON_CHAR