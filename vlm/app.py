import streamlit as st
import cv2
import torch
import numpy as np
import os
import base64
from PIL import Image
import io
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Force CPU usage - no GPU required
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
torch.set_default_device('cpu')

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'segmentation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from segmentation.model import PetModel
from segmentation.inference import parseConfigurationYAML, predictMask, apply_mask
from src.config import EvalConfig
from src.client import AzureLLMClient
from src.classifier import HairClassifier

# Page configuration
st.set_page_config(
    page_title="Halisi Cosmetics - Hair Analysis",
    layout="wide"
)

# Initialize session state
if 'segmentation_model' not in st.session_state:
    st.session_state.segmentation_model = None
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'config_loaded' not in st.session_state:
    st.session_state.config_loaded = False

@st.cache_resource
def load_segmentation_model():
    """Load the hair segmentation model"""
    try:
        # Force CPU usage
        torch.set_num_threads(4)  # Limit CPU threads for better performance
        
        config = parseConfigurationYAML("./segmentation/config.yaml")
        model_path = "./models/model_mobilenetv2_150_320x320.pth"
        
        model = PetModel(
            arch=config['HParams']['architecture'],
            encoder_name=config['HParams']['backbone'],
            in_channels=config['HParams']['nr_channels'],
            out_classes=config['HParams']['nr_identities']
        )
        # Load model on CPU explicitly
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        model.to('cpu')
        return model
    except Exception as e:
        st.error(f"Error loading segmentation model: {e}")
        return None

@st.cache_resource
def load_classifier():
    """Load the hair classification system"""
    try:
        # Load configuration
        config = EvalConfig("./config/eval.yaml")
        
        # Load prompt
        with open(config.paths["prompt"], "r") as f:
            prompt = f.read()
        
        # Get API credentials from environment or Streamlit secrets
        api_key = os.getenv("AZUREOPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or st.secrets.get("AZURE_OPENAI_API_KEY", None)
        endpoint = os.getenv("AZUREOPENAI_API_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT") or st.secrets.get("AZURE_OPENAI_ENDPOINT", None)
        api_version = os.getenv("AZUREOPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        if not api_key or not endpoint:
            return None
        
        # Initialize client and classifier
        client = AzureLLMClient(api_key, endpoint, api_version)
        classifier = HairClassifier(client, config, prompt)
        return classifier
    except Exception as e:
        st.warning(f"Classification disabled: {e}")
        return None

def image_to_base64(image):
    """Convert image array to base64 string"""
    # Convert from BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Convert to base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def process_image(image, seg_model, classifier=None):
    """Process image: segmentation + classification"""
    results = {}
    
    # Perform segmentation
    with st.spinner("Segmenting hair..."):
        # Ensure we're using CPU
        with torch.no_grad():
            mask = predictMask(seg_model, image)
        segmented_image = apply_mask(image, mask, threshold=0.5)
        results['mask'] = mask
        results['segmented'] = segmented_image
    
    # Perform classification if available
    if classifier is not None:
        with st.spinner("Classifying hair type..."):
            try:
                # Convert segmented image to base64
                image_b64 = image_to_base64(segmented_image)
                hair_type = classifier.predict(image_b64)
                results['classification'] = hair_type
            except Exception as e:
                st.error(f"Classification error: {e}")
                results['classification'] = "Error"
    else:
        results['classification'] = "Not available"
    
    return results

# App Header
st.title("Halisi Cosmetics - Hair Analysis")

# Load models
with st.spinner("Loading models..."):
    if st.session_state.segmentation_model is None:
        st.session_state.segmentation_model = load_segmentation_model()
    if st.session_state.classifier is None:
        st.session_state.classifier = load_classifier()

if st.session_state.segmentation_model is None:
    st.error("Failed to load segmentation model. Please check model file.")
    st.stop()

# Main interface
st.markdown("---")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Camera Capture", "Upload Image"])

with tab1:
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        # Convert to OpenCV format
        image = Image.open(camera_image)
        image_array = np.array(image)
        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Display and process
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(image_array, caption="Original Image", use_container_width=True)
        
        # Process the image
        results = process_image(image_cv, st.session_state.segmentation_model, st.session_state.classifier)
        
        with col2:
            # Display segmentation mask
            mask_display = (results['mask'] * 255).astype(np.uint8)
            mask_colored = cv2.applyColorMap(mask_display, cv2.COLORMAP_JET)
            st.image(mask_colored, caption="Segmentation Mask", use_container_width=True, channels="BGR")
        
        with col3:
            # Display segmented image
            segmented_rgb = cv2.cvtColor(results['segmented'], cv2.COLOR_BGR2RGB)
            st.image(segmented_rgb, caption="Segmented Hair", use_container_width=True)
        
        # Display classification result
        st.markdown("---")
        st.subheader("Classification Result")
        
        hair_type = results['classification']
        if hair_type and hair_type != "Not available" and hair_type != "Error":
            st.write(f"**Hair Type: {hair_type}**")
        elif hair_type == "Not available":
            st.info("Classification not available.")
        else:
            st.warning(f"Classification result: {hair_type}")

with tab2:
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        # Display and process
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(image_rgb, caption="Original Image", use_container_width=True)
        
        # Process the image
        results = process_image(image_cv, st.session_state.segmentation_model, st.session_state.classifier)
        
        with col2:
            # Display segmentation mask
            mask_display = (results['mask'] * 255).astype(np.uint8)
            mask_colored = cv2.applyColorMap(mask_display, cv2.COLORMAP_JET)
            st.image(mask_colored, caption="Segmentation Mask", use_container_width=True, channels="BGR")
        
        with col3:
            # Display segmented image
            segmented_rgb = cv2.cvtColor(results['segmented'], cv2.COLOR_BGR2RGB)
            st.image(segmented_rgb, caption="Segmented Hair", use_container_width=True)
        
        # Display classification result
        st.markdown("---")
        st.subheader("Classification Result")
        
        hair_type = results['classification']
        if hair_type and hair_type != "Not available" and hair_type != "Error":
            st.write(f"**Hair Type: {hair_type}**")
        elif hair_type == "Not available":
            st.info("Classification not available.")
        else:
            st.warning(f"Classification result: {hair_type}")
