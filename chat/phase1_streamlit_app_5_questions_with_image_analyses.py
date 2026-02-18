# ==================================================
# phase1_streamlit_app_5_questions_color_mask.py
# ==================================================

import streamlit as st
import json
import os
import re
import base64
import io
import sys
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import cv2
import numpy as np

# Add root path to import utils_vlm
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils_vlm.utils_vlm import process_image_full_pipeline

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Halisi Cosmetics",
    page_icon="💇‍♀️",
    layout="wide"
)

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()
AZURE_DEPLOYMENT_NAME = "gpt-4o-mini"

# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------
for key, default in [
    ("messages", []),
    ("extracted_info", {}),
    ("conversation_complete", False),
    ("question_count", 0),
    ("profile_data", {}),
    ("current_group", 0),
    ("captured_image", None),
    ("image_captured", False),
    ("image_analyzed", False),
    ("analysis_results", {})
]:
    if key not in st.session_state:
        st.session_state[key] = default

MAX_QUESTIONS = 5

# --------------------------------------------------
# QUESTION GROUPS
# --------------------------------------------------
QUESTION_GROUPS = [
    {"name": "Context", "question": "👋 Hello! Are you shopping for yourself or a child?", "topics": ["product_context"]},
    {"name": "Demographics", "question": "What is your age, gender, and location?", "topics": ["user_demographics"]},
    {"name": "Hair Portrait", "question": "Describe your hair type (straight, wavy, curly, coily/afro), main concerns, goals, scalp condition, and structure.", 
     "topics": ["hair_type_texture", "hair_scalp_concerns", "hair_goals", "scalp_condition", "hair_structure"]},
    {"name": "Routine & Products", "question": "Tell me about your current routine: products, styling frequency, time spent, budget, and routine preference.", 
     "topics": ["current_products", "hair_care_routine", "styling_habits", "time_availability", "budget_constraints", "existing_inventory", "routine_preference"]},
    {"name": "Health & Preferences", "question": "Any allergies, medical considerations, color history, eco-preferences, lifestyle factors, past experiences, hair porosity?", 
     "topics": ["sensitivities_allergies","pregnancy_status","professional_treatments","color_history","eco_preferences","lifestyle_factors","past_experiences","maintenance_commitment","hair_porosity"]}
]

# --------------------------------------------------
# NORMALIZATION HELPERS
# --------------------------------------------------
def clean_response(response):
    if not response: return ""
    response = response.lower().strip()
    response = re.sub(r'\s+', ' ', response)
    return response

def categorize_response(topic, response):
    cleaned = clean_response(response)
    if topic == "product_context":
        return "child" if any(word in cleaned for word in ["child","kid","baby","daughter","son"]) else "self"
    if topic == "user_demographics":
        extracted = {"age":"","gender":"","location":""}
        age_match = re.search(r'\b(\d{1,3})\b', cleaned)
        if age_match: extracted["age"]=age_match.group(1)
        extracted["gender"]="Female" if any(word in cleaned for word in ["female","woman","girl","she","her"]) else "Male"
        loc_match = re.search(r'from ([a-z\s,]+)', cleaned)
        extracted["location"]=loc_match.group(1).title() if loc_match else ""
        return extracted
    if topic == "hair_type_texture":
        if any(word in cleaned for word in ["straight","1a","1b","1c"]): return "Straight"
        if any(word in cleaned for word in ["wavy","2a","2b","2c"]): return "Wavy"
        if any(word in cleaned for word in ["curly","3a","3b","3c"]): return "Curly"
        if any(word in cleaned for word in ["coily","afro","kinky","4a","4b","4c"]): return "Coily/Afro-textured"
        return cleaned
    if topic == "hair_structure":
        if any(word in cleaned for word in ["braid","cornrow","plait"]): return "Braids"
        if any(word in cleaned for word in ["lock","dread","dreadlock"]): return "Locks"
        if any(word in cleaned for word in ["extension","weave","wig"]): return "Extensions"
        if "natural" in cleaned: return "Natural"
        return cleaned
    return cleaned

# --------------------------------------------------
# GENERATE NEXT QUESTION
# --------------------------------------------------
def generate_next_group_question():
    if st.session_state.current_group >= len(QUESTION_GROUPS): return None, None
    group = QUESTION_GROUPS[st.session_state.current_group]
    return group["question"], st.session_state.current_group

# --------------------------------------------------
# GENERATE PROFILE
# --------------------------------------------------
def generate_clean_profile():
    profile_data = {}
    for topic in QUESTION_GROUPS[-1]["topics"]:
        if topic in st.session_state.extracted_info:
            val = st.session_state.extracted_info[topic].get("value","")
            profile_data[topic]=st.session_state.extracted_info[topic]

    summary_parts=[]
    for topic in ["hair_type_texture","hair_scalp_concerns","hair_goals"]:
        if topic in profile_data: summary_parts.append(f"{topic.replace('_',' ').title()}: {profile_data[topic]['value']}")
    # AI image analysis
    if st.session_state.analysis_results:
        hair_type = st.session_state.analysis_results.get('hair_type',{}).get('hair_type','unknown')
        summary_parts.append(f"Image analysis: {hair_type}")
    summary="; ".join(summary_parts) if summary_parts else "Profile ready"

    image_data=None
    if st.session_state.captured_image:
        buffered=io.BytesIO()
        st.session_state.captured_image.save(buffered, format="JPEG")
        image_data=base64.b64encode(buffered.getvalue()).decode()

    return {
        "profile_id": f"halisi_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "completion_date": datetime.now().isoformat(),
        "summary": summary,
        "profile_data": profile_data,
        "hair_image_base64": image_data,
        "image_analysis": st.session_state.analysis_results
    }

# --------------------------------------------------
# INITIALIZE FIRST QUESTION
# --------------------------------------------------
if not st.session_state.messages:
    first_q,_=generate_next_group_question()
    st.session_state.messages.append({"role":"assistant","content":first_q})
    st.session_state.current_group=0

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.title("💇‍♀️ Halisi Cosmetics - Smart Hair Profiler Interactive")
st.markdown("*5 questions + interactive AI photo analysis*")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# USER INPUT LOGIC
# --------------------------------------------------
if not st.session_state.conversation_complete:
    user_input=st.chat_input("Your answer...")
    if user_input:
        st.session_state.messages.append({"role":"user","content":user_input})
        with st.chat_message("user"): st.markdown(user_input)

        group=QUESTION_GROUPS[st.session_state.current_group]
        for topic in group["topics"]:
            categorized=categorize_response(topic,user_input)
            st.session_state.extracted_info[topic]={"raw":user_input,"value":categorized}

        st.session_state.question_count+=1
        st.session_state.current_group+=1

        if st.session_state.question_count>=MAX_QUESTIONS:
            st.session_state.conversation_complete=True
            st.rerun()
        else:
            next_q,_=generate_next_group_question()
            if next_q:
                st.session_state.messages.append({"role":"assistant","content":next_q})
                with st.chat_message("assistant"): st.markdown(next_q)
                st.rerun()

# --------------------------------------------------
# PHOTO CAPTURE + MASK OVERLAY WITH COLOR PALETTE
# --------------------------------------------------
HAIR_TYPE_COLORS = {
    "Straight": [0, 255, 0],      # Green
    "Wavy": [0, 128, 255],        # Orange
    "Curly": [255, 0, 255],       # Magenta
    "Coily/Afro-textured": [255, 0, 0],  # Red
    "unknown": [128,128,128]      # Gray fallback
}

if st.session_state.conversation_complete and not st.session_state.image_analyzed:
    st.markdown("---")
    st.subheader("📸 Capture a photo of your hair")
    camera_image=st.camera_input("Take a picture", key="hair_cam")
    if camera_image is not None:
        image=Image.open(camera_image).convert("RGB")
        st.session_state.captured_image=image
        with tempfile.NamedTemporaryFile(delete=False,suffix='.jpg') as tmp:
            temp_path=tmp.name
            image.save(temp_path,format='JPEG')
        try:
            with st.spinner("Analyzing your hair..."):
                result=process_image_full_pipeline(
                    image_path=temp_path,
                    azure_api_key=os.getenv("AZUREOPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZUREOPENAI_API_ENDPOINT"),
                    save_mask=True,
                    mask_output_path=temp_path.replace(".jpg","_mask.jpg")
                )
                st.session_state.analysis_results=result
                st.session_state.image_analyzed=True

                # Overlay mask with color based on hair type
                mask=result["mask"]
                mask_rgb=cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                orig_rgb=np.array(image)

                detected_type=result['hair_type'].get('hair_type','unknown')
                color=HAIR_TYPE_COLORS.get(detected_type,[128,128,128])
                overlay_color=np.zeros_like(mask_rgb)
                overlay_color[mask_rgb>0]=color

                alpha=0.5
                overlay=(orig_rgb*(1-alpha) + overlay_color*alpha).astype(np.uint8)

                caption=f"Hair Mask Overlay - Detected Type: {detected_type}"
                st.image(overlay, caption=caption, width=350)

        except Exception as e:
            st.error(f"Image analysis failed: {e}")
            st.session_state.image_analyzed=True
        finally:
            os.unlink(temp_path)
        st.rerun()
    if st.button("Skip photo (continue without image)"):
        st.session_state.image_analyzed=True
        st.rerun()

# --------------------------------------------------
# FINAL PROFILE DISPLAY + JSON DOWNLOAD
# --------------------------------------------------
if st.session_state.conversation_complete and st.session_state.image_analyzed:
    if not st.session_state.profile_data:
        st.session_state.profile_data=generate_clean_profile()
    st.success("✅ Profile Created!")
    st.write(st.session_state.profile_data["summary"])
    if st.session_state.captured_image:
        st.image(st.session_state.captured_image, caption="Your Hair Photo", width=300)
    analysis=st.session_state.profile_data.get("image_analysis",{})
    if analysis: st.write("**AI Hair Analysis:**", analysis)
    json_str=json.dumps(st.session_state.profile_data, indent=2)
    st.download_button("Download profile JSON", data=json_str,
                       file_name=f"{st.session_state.profile_data['profile_id']}.json",
                       mime="application/json")
