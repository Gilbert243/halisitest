import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from model import PetModel
import yaml
from yaml.loader import SafeLoader

# Configuration

DATA_OLD_DIR = "./wait"
DATA_NEW_DIR = "./data"
ANNOTATION_CSV = "./data/annotations.csv"

os.makedirs(DATA_NEW_DIR, exist_ok=True)

# Load config

def parseConfigurationYAML(configFile):
    with open(configFile) as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config

config = parseConfigurationYAML("./segmentation/config.yaml")

# Load model

def loadModel(model_path):
    model = PetModel(
        arch=config['HParams']['architecture'],
        encoder_name=config['HParams']['backbone'],
        in_channels=config['HParams']['nr_channels'],
        out_classes=config['HParams']['nr_identities']
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# Hair segmentation inference

def predictMask(model, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (320, 320), interpolation=cv2.INTER_AREA)

    tensor = np.moveaxis(image_resized, -1, 0)
    tensor = torch.unsqueeze(torch.tensor(tensor, dtype=torch.float32), 0)

    with torch.no_grad():
        logits = model(tensor)
        mask = logits.sigmoid()

    return mask.squeeze().cpu().numpy()

# Apply mask to image

def apply_mask(image, mask, threshold=0.5):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    binary_mask = (mask > threshold).astype(np.uint8)

    segmented = image.copy()
    segmented[binary_mask == 0] = 0
    return segmented

# Filename parser

def extract_class_name(filename):
    """
    Extracts hair type code from filename.
    Example:
        Kinky_coily_4B_14.jpg -> 4B
    """
    name = os.path.splitext(filename)[0]
    parts = name.replace("-", "_").replace("_ ", "_").split("_")

    for part in parts:
        if part.upper() in ["1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C", "4A", "4B", "4C"]:
            return part.upper()

    raise ValueError(f"No valid hair type found in filename: {filename}")


# Main processing loop

def process_dataset(model):
    annotations = []

    image_files = [
        f for f in os.listdir(DATA_OLD_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for filename in tqdm(sorted(image_files), desc="Processing images", unit="image"):
        image_path = os.path.join(DATA_OLD_DIR, filename)
        image = cv2.imread(image_path)

        if image is None:
            continue

        mask = predictMask(model, image)
        segmented_image = apply_mask(image, mask)

        class_name = extract_class_name(filename)

        output_filename = filename
        output_path = os.path.join(DATA_NEW_DIR, output_filename)

        cv2.imwrite(output_path, segmented_image)

        annotations.append({
            "image_path": f"data/{output_filename}",
            "type": class_name
        })

    return pd.DataFrame(annotations)

# Run pipeline

if __name__ == "__main__":
    model = loadModel("./models/model_mobilenetv2_150_320x320.pth")
    annotations_df = process_dataset(model)
    annotations_df.to_csv(ANNOTATION_CSV, index=False)

    print("Processing completed.")
    print(f"Segmented images saved in: {DATA_NEW_DIR}")
    print(f"Annotations saved to: {ANNOTATION_CSV}")
