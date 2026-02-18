# phase1_streamlit_app_pro.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import json
import base64
from datetime import datetime
from utils_vlm.utils_vlm import process_image_full_pipeline  # ta pipeline

st.set_page_config(
    page_title="Halisi Cosmetics - Smart Hair Profiler",
    page_icon="💇‍♀️",
    layout="wide"
)

st.title("💇‍♀️ Halisi Cosmetics - Smart Hair Profiler Interactive")
st.markdown("5 questions + interactive AI photo analysis")

# --- SESSION STATE INIT ---
if "profile_data" not in st.session_state:
    st.session_state.profile_data = {}
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

# --- 5 QUESTIONS ---
with st.form("profile_form"):
    st.subheader("👋 Hello! Are you shopping for yourself or a child?")
    who_for = st.radio("Select:", ["self", "child"])
    
    st.subheader("What is your age, gender, and location?")
    age_gender_location = st.text_input("Answer")
    
    st.subheader("Describe your hair type (straight, wavy, curly, coily/afro), main concerns, goals, scalp condition, and structure.")
    hair_type_desc = st.text_input("Answer")
    
    st.subheader("Tell me about your current routine: products, styling frequency, time spent, budget, and routine preference.")
    routine_info = st.text_input("Answer")
    
    st.subheader("Any allergies, medical considerations, color history, eco-preferences, lifestyle factors, past experiences, hair porosity?")
    extra_info = st.text_input("Answer")
    
    submitted = st.form_submit_button("✅ Profile Created!")
    
    if submitted:
        st.session_state.profile_data.update({
            "who_for": who_for,
            "age_gender_location": age_gender_location,
            "hair_type_desc": hair_type_desc,
            "routine_info": routine_info,
            "extra_info": extra_info,
            "image_analysis": {}
        })
        st.success("Profile Created!")

# --- IMAGE CAPTURE / UPLOAD ---
st.subheader("📸 Your Hair Photo")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
capture_btn = st.button("Use Webcam")

if uploaded_file:
    st.session_state.captured_image = Image.open(uploaded_file)
elif capture_btn:
    import cv2
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.session_state.captured_image = Image.fromarray(frame_rgb)

# --- MASK GENERATION & HAIR TYPE DETECTION ---
if st.session_state.captured_image:
    st.image(st.session_state.captured_image, caption="Your Hair Photo", use_column_width=True)
    
    # Save temporary image for processing
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        st.session_state.captured_image.save(tmp.name)
        tmp_path = tmp.name
    
    st.info("Generating mask and analyzing hair type...")
    
    try:
        result = process_image_full_pipeline(
            image_path=tmp_path,
            save_mask=False
        )
        mask = result["mask"]
        hair_analysis = result["hair_type"]
        
        # --- APPLY COLOR PALETTE ON MASK ---
        # Palette per hair type
        palette = {
            "1A": (255, 0, 0),
            "1B": (255, 128, 0),
            "2A": (255, 255, 0),
            "2B": (128, 255, 0),
            "2C": (0, 255, 0),
            "3A": (0, 255, 128),
            "3B": (0, 255, 255),
            "3C": (0, 128, 255),
            "4A": (0, 0, 255),
            "4B": (128, 0, 255),
            "4C": (255, 0, 255),
            "Unknown": (200, 200, 200)
        }
        hair_type = hair_analysis.get("hair_type", "Unknown")
        color = palette.get(hair_type, (200, 200, 200))
        
        colored_mask = mask.copy()
        colored_mask[mask != 0] = color  # apply color to mask
        
        st.subheader(f"AI Hair Analysis: {hair_type}")
        st.image(colored_mask, caption="Hair Mask Colored", use_column_width=True)
        
        # --- STORE ANALYSIS IN SESSION STATE ---
        st.session_state.profile_data["image_analysis"] = {
            "hair_type": hair_type
        }
        
        # --- CLEAN JSON FOR DOWNLOAD ---
        def generate_clean_profile():
            profile_data = st.session_state.profile_data.copy()
            # Encode image to Base64
            buffered = io.BytesIO()
            st.session_state.captured_image.save(buffered, format="JPEG")
            profile_data["hair_image_base64"] = base64.b64encode(buffered.getvalue()).decode()
            profile_data["profile_id"] = f"halisi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            profile_data["completion_date"] = datetime.now().isoformat()
            return profile_data
        
        clean_profile = generate_clean_profile()
        json_str = json.dumps(clean_profile, indent=2)
        
        st.download_button(
            "Download profile JSON",
            data=json_str,
            file_name=f"{clean_profile['profile_id']}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error during image analysis: {e}")
