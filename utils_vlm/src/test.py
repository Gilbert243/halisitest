import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DATA_OLD_DIR = "./samples"
DATA_NEW_DIR = "./samples"


def extract_class_name(filename):
    """
    Extracts hair type code from filename.
    Example:
        Kinky_coily_4B_14.jpg -> 4B
    """
    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    for part in parts:
        if part.upper() in ["1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C", "4A", "4B", "4C"]:
            return part.upper()

    raise ValueError(f"No valid hair type found in filename: {filename}")


# Main processing loop

def process_dataset():
    annotations = []

    image_files = [
        f for f in os.listdir(DATA_OLD_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for filename in tqdm(sorted(image_files), desc="Processing images", unit="image"):
        
        class_name = extract_class_name(filename)

        output_filename = filename

        annotations.append({
            "image_path": f"samples/{output_filename}",
            "type": class_name
        })

    annotations_df = pd.DataFrame(annotations)
    annotations_df.to_csv("data.csv",index=False)

    return "Done"

process_dataset()