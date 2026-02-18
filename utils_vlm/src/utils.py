import base64

def encode_image(image_path: str) -> str:
    """Encode image file to base64 string.
    
    Reads image file from disk and converts to base64-encoded string
    suitable for transmission to Vision-Language Model APIs.
    
    Args:
        image_path: Path to image file (JPEG, PNG, etc.)
        
    Returns:
        Base64-encoded string representation of image
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        IOError: On file read errors
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
