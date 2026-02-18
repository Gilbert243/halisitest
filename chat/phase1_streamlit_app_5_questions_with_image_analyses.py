# ==================================================
# phase1_streamlit_app_5_questions_with_image_analyses.py
# ==================================================

import streamlit as st
import json, os, re, base64, io, sys, tempfile
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
from PIL import Image
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils_vlm.utils_vlm import process_image_full_pipeline

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Halisi Cosmetics - Smart Hair Profiler",
    page_icon="💇‍♀️",
    layout="wide"
)

# --------------------------------------------------
# LOAD ENV
# --------------------------------------------------
load_dotenv()
AZURE_DEPLOYMENT_NAME = "gpt-4o-mini"

client = AzureOpenAI(
    api_key=os.getenv("AZUREOPENAI_API_KEY"),
    api_version=os.getenv("AZUREOPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZUREOPENAI_API_ENDPOINT"),
)

# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------
if "messages" not in st.session_state: st.session_state.messages = []
if "extracted_info" not in st.session_state: st.session_state.extracted_info = {}
if "conversation_complete" not in st.session_state: st.session_state.conversation_complete = False
if "question_count" not in st.session_state: st.session_state.question_count = 0
if "profile_data" not in st.session_state: st.session_state.profile_data = {}
if "current_group" not in st.session_state: st.session_state.current_group = 0
if "user_demographics" not in st.session_state: st.session_state.user_demographics = {}
if "product_context" not in st.session_state: st.session_state.product_context = {}
if "captured_image" not in st.session_state: st.session_state.captured_image = None
if "image_captured" not in st.session_state: st.session_state.image_captured = False
if "image_analyzed" not in st.session_state: st.session_state.image_analyzed = False
if "analysis_results" not in st.session_state: st.session_state.analysis_results = {}
if "analysis_error" not in st.session_state: st.session_state.analysis_error = None

MAX_QUESTIONS = 5

# --------------------------------------------------
# TOPICS
# --------------------------------------------------
DEMOGRAPHIC_TOPICS = ["product_context","user_demographics"]
HAIR_TOPICS = ["hair_type_texture","hair_scalp_concerns","hair_goals","current_products","hair_care_routine",
               "styling_habits","scalp_condition","hair_porosity","routine_preference","time_availability",
               "budget_constraints","sensitivities_allergies","hair_structure","pregnancy_status",
               "existing_inventory","hair_length","maintenance_commitment","past_experiences",
               "professional_treatments","color_history","eco_preferences","lifestyle_factors"]
ALL_TOPICS = DEMOGRAPHIC_TOPICS + HAIR_TOPICS

# --------------------------------------------------
# QUESTION GROUPS
# --------------------------------------------------
QUESTION_GROUPS = [
    {"name":"Context","question":"👋 Hi! I'm Halisi, your hair care expert.\n\nAre you looking for products for yourself or a child?","topics":["product_context"]},
    {"name":"Demographics","question":"What is your age, gender, and location? (If you're shopping for a child, please provide the child's age and gender, and your location for climate)","topics":["user_demographics"]},
    {"name":"Hair Portrait","question":"Describe your hair: type (straight, wavy, curly, coily/afro), main concerns, goals, scalp condition, and structure (natural, braids, locks, extensions).","topics":["hair_type_texture","hair_scalp_concerns","hair_goals","scalp_condition","hair_structure"]},
    {"name":"Routine & Products","question":"Tell me about your current hair routine: products you use, how often you style, time spent per week, monthly budget, products at home, and whether you prefer a simple or detailed routine.","topics":["current_products","hair_care_routine","styling_habits","time_availability","budget_constraints","existing_inventory","routine_preference"]},
    {"name":"Health & Preferences","question":"Any allergies, sensitivities, or medical considerations (pregnancy)? Past professional treatments (keratin, relaxing)? Color history? Importance of eco‑friendly products? Lifestyle factors (stress, diet, climate)? Past product experiences? How committed are you to a routine? Hair porosity if known?","topics":["sensitivities_allergies","pregnancy_status","professional_treatments","color_history","eco_preferences","lifestyle_factors","past_experiences","maintenance_commitment","hair_porosity"]}
]

# --------------------------------------------------
# NORMALIZATION HELPERS
# --------------------------------------------------
def normalize_budget(text):
    text = text.lower()
    amount_match = re.search(r'(\d+(?:\.\d+)?)', text)
    amount = amount_match.group(1) if amount_match else None
    currency = "USD"
    if "€" in text or "eur" in text: currency="EUR"
    elif "£" in text or "gbp" in text: currency="GBP"
    elif "c$" in text or "cad" in text: currency="CAD"
    elif "$" in text or "usd" in text or "dollar" in text: currency="USD"
    if amount: return f"{amount} {currency}"
    return text

def normalize_time(text):
    text = text.lower()
    if "quarter" in text and "hour" in text: return "15 minutes"
    if "half" in text and "hour" in text: return "30 minutes"
    match = re.search(r'(\d+)\s*(?:times?|x)\s*(?:per|a)\s*week', text)
    if match: return f"{match.group(1)}x/week"
    match = re.search(r'(\d+)\s*(?:minutes?|mins?)', text)
    if match: return f"{match.group(1)} min"
    return text

def normalize_hair_length(text):
    text = text.lower()
    mapping = {"very small":"Short","short":"Short","shoulder":"Shoulder-length","medium":"Medium",
               "long":"Long","very long":"Very long"}
    for k,v in mapping.items():
        if k in text: return v
    return text

def extract_product_list(text):
    keywords = ["shampoo","conditioner","oil","serum","gel","cream","lotion","mask","treatment",
                "spray","mousse","butter","leave-in","moisturizer","detangler"]
    found = [kw.title() for kw in keywords if kw in text.lower()]
    return found if found else text

def clean_response(response):
    if not response: return ""
    response = response.lower().strip()
    response = re.sub(r'\s+', ' ', response)
    return response

# --------------------------------------------------
# CATEGORIZATION FUNCTION
# (Règles complètes pour tous les topics)
# --------------------------------------------------
def categorize_response(topic, response):
    cleaned = clean_response(response)
    # Exemple simplifié pour démonstration. Remplacer par toutes les règles que vous avez définies.
    if topic == "hair_type_texture":
        if any(x in cleaned for x in ["straight","1a","1b","1c"]): return "Straight"
        if any(x in cleaned for x in ["wavy","2a","2b","2c"]): return "Wavy"
        if any(x in cleaned for x in ["curly","3a","3b","3c"]): return "Curly"
        if any(x in cleaned for x in ["coily","afro","4a","4b","4c"]): return "Coily/Afro-textured"
    return cleaned

# --------------------------------------------------
# PROFILE GENERATION
# --------------------------------------------------
def generate_clean_profile():
    profile_data = {}
    meaningful_count = 0
    for topic in ALL_TOPICS:
        if topic in st.session_state.extracted_info:
            entry = st.session_state.extracted_info[topic]
            raw = entry.get("raw","")
            categorized = entry.get("value","")
            profile_data[topic] = {"raw": raw[:200],"value":categorized}
            if categorized and str(categorized).lower() not in ["","none","no","unspecified","not applicable","i don't know"]:
                meaningful_count += 1
    completeness = round((meaningful_count / len(ALL_TOPICS))*100,1)
    summary_parts = []
    for topic in ["hair_type_texture","hair_scalp_concerns","hair_goals","current_products"]:
        if topic in profile_data:
            val = profile_data[topic]["value"]
            if val and str(val).lower() not in ["","none","no","unspecified"]:
                summary_parts.append(f"{topic.replace('_',' ').title()}: {val}")
    if st.session_state.analysis_results:
        detected = st.session_state.analysis_results.get("hair_type","unknown")
        summary_parts.append(f"Image analysis: {detected}")
    summary = "; ".join(summary_parts) if summary_parts else "Profile ready"

    image_data = None
    if st.session_state.captured_image:
        buf = io.BytesIO()
        st.session_state.captured_image.save(buf, format="JPEG")
        image_data = base64.b64encode(buf.getvalue()).decode()

    return {
        "profile_id": f"halisi_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "completion_date": datetime.now().isoformat(),
        "stats":{"questions":st.session_state.question_count,"meaningful_data":meaningful_count,"completeness":completeness},
        "demographics": {k:v for k,v in profile_data.items() if k in DEMOGRAPHIC_TOPICS},
        "hair_profile": {k:v for k,v in profile_data.items() if k in HAIR_TOPICS},
        "summary": summary,
        "recommendation_ready": meaningful_count>=15,
        "hair_image_base64": image_data,
        "image_analysis": st.session_state.analysis_results
    }

# --------------------------------------------------
# COLOR PALETTE
# --------------------------------------------------
HAIR_TYPE_COLORS = {"Straight":[0,255,0],"Wavy":[0,128,255],"Curly":[255,0,255],
                    "Coily/Afro-textured":[255,0,0],"unknown":[128,128,128]}

# --------------------------------------------------
# INITIAL QUESTION
# --------------------------------------------------
if not st.session_state.messages:
    first_group = QUESTION_GROUPS[0]
    st.session_state.messages.append({"role":"assistant","content":first_group["question"]})
    st.session_state.current_group = 0

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.title("💇‍♀️ Halisi Cosmetics - Smart Hair Profiler")
st.markdown("*Quick, comprehensive hair profiling in just 5 questions + AI photo analysis*")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# CHAT LOGIC
# --------------------------------------------------
if not st.session_state.conversation_complete:
    user_input = st.chat_input("Your answer...")
    if user_input:
        st.session_state.messages.append({"role":"user","content":user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        group = QUESTION_GROUPS[st.session_state.current_group]
        for topic in group["topics"]:
            cleaned = clean_response(user_input)
            categorized = categorize_response(topic, cleaned)
            st.session_state.extracted_info[topic] = {"raw": user_input,"cleaned":cleaned,"value":categorized}
            if topic in DEMOGRAPHIC_TOPICS:
                st.session_state.user_demographics[topic] = categorized
        st.session_state.question_count += 1
        st.session_state.current_group += 1
        if st.session_state.question_count >= MAX_QUESTIONS or st.session_state.current_group >= len(QUESTION_GROUPS):
            demo_val = st.session_state.extracted_info.get("user_demographics", {}).get("value", {})
            if isinstance(demo_val, dict) and demo_val.get("gender") == "Male":
                if "pregnancy_status" not in st.session_state.extracted_info:
                    st.session_state.extracted_info["pregnancy_status"] = {"raw":"automatically skipped","cleaned":"not applicable","value":"Not applicable"}
            st.session_state.conversation_complete = True
            st.rerun()
        else:
            next_q = QUESTION_GROUPS[st.session_state.current_group]["question"]
            st.session_state.messages.append({"role":"assistant","content":next_q})
            with st.chat_message("assistant"):
                st.markdown(next_q)
            st.rerun()

# --------------------------------------------------
# PHOTO CAPTURE & ANALYSIS (fix NumPy included)
# --------------------------------------------------
if st.session_state.conversation_complete and not st.session_state.image_analyzed:
    st.subheader("📸 Capture a photo of your hair")
    camera_image = st.camera_input("Take a picture", key="hair_cam")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_path = tmp.name
            image.save(temp_path, format="JPEG")
        try:
            with st.spinner("Analyzing hair..."):
                result = process_image_full_pipeline(
                    image_path=temp_path,
                    azure_api_key=os.getenv("AZUREOPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZUREOPENAI_API_ENDPOINT"),
                    save_mask=True,
                    mask_output_path=temp_path.replace(".jpg","_mask.jpg")
                )
                st.session_state.analysis_results = result.get("hair_type", {"hair_type":"unknown"})
                st.session_state.captured_image = image
                st.session_state.image_captured = True
                st.session_state.image_analyzed = True

                if "mask" in result and result["mask"] is not None:
                    mask = result["mask"]
                    if len(mask.shape)==3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    orig_rgb = np.array(image)
                    if mask.shape[:2] != orig_rgb.shape[:2]: mask = cv2.resize(mask,(orig_rgb.shape[1],orig_rgb.shape[0]))
                    mask_bool = mask>0
                    overlay_color = np.zeros_like(orig_rgb)
                    detected_type = st.session_state.analysis_results.get("hair_type","unknown")
                    color = HAIR_TYPE_COLORS.get(detected_type,[128,128,128])
                    for i in range(3):
                        overlay_color[:,:,i][mask_bool] = color[i]
                    alpha = 0.5
                    overlay = (orig_rgb*(1-alpha) + overlay_color*alpha).astype(np.uint8)
                    st.image(overlay, caption=f"Hair Mask Overlay - Detected: {detected_type}", width=400)
                else:
                    st.warning("No mask available for overlay.")
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.session_state.analysis_results = {"hair_type":"unknown","error":str(e)}
        finally:
            os.unlink(temp_path)

# --------------------------------------------------
# PROFILE DISPLAY & DOWNLOAD
# --------------------------------------------------
if st.session_state.conversation_complete and st.session_state.image_analyzed:
    if not st.session_state.profile_data:
        st.session_state.profile_data = generate_clean_profile()

    st.success("✅ **Profile Successfully Created!**")
    stats = st.session_state.profile_data.get("stats",{})
    col1,col2,col3 = st.columns(3)
    with col1: st.metric("Questions", stats.get("questions",0))
    with col2: st.metric("Useful data", stats.get("meaningful_data",0))
    with col3: st.metric("Completeness", f"{stats.get('completeness',0)}%")

    with st.expander("📋 **Your Hair Profile**",expanded=True):
        st.write(st.session_state.profile_data.get("summary",""))
        if st.session_state.captured_image: st.image(st.session_state.captured_image,caption="Your hair",width=300)

    col1,col2 = st.columns(2)
    with col1:
        st.download_button(
            "💾 Download Profile",
            json.dumps(st.session_state.profile_data, indent=2),
            file_name=f"halisi_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    with col2:
        if st.button("🔄 New Profile", type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

st.markdown("---")
st.caption("Halisi Cosmetics • Smart Hair Profiler")
