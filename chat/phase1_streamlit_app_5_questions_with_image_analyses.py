# ==================================================
# phase1_streamlit_app_5_questions_with_image_analyses.py
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
from openai import AzureOpenAI
from PIL import Image

# Add root path to import utils_vlm
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
# ENV
# --------------------------------------------------
load_dotenv()
AZURE_DEPLOYMENT_NAME = "gpt-4o-mini"

client = AzureOpenAI(
    api_key=os.getenv("AZUREOPENAI_API_KEY"),
    api_version=os.getenv("AZUREOPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZUREOPENAI_API_ENDPOINT"),
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
for key, default in [
    ("messages", []),
    ("extracted_info", {}),
    ("conversation_complete", False),
    ("question_count", 0),
    ("profile_data", {}),
    ("current_group", 0),
    ("user_demographics", {}),
    ("product_context", {}),
    ("captured_image", None),
    ("image_captured", False),
    ("image_analyzed", False),
    ("analysis_results", {})
]:
    if key not in st.session_state:
        st.session_state[key] = default

MAX_QUESTIONS = 5  # Only 5 composite questions

# --------------------------------------------------
# TOPICS
# --------------------------------------------------
DEMOGRAPHIC_TOPICS = [
    "product_context",
    "user_demographics",
]

HAIR_TOPICS = [
    "hair_type_texture",
    "hair_scalp_concerns",
    "hair_goals",
    "current_products",
    "hair_care_routine",
    "styling_habits",
    "scalp_condition",
    "hair_porosity",
    "routine_preference",
    "time_availability",
    "budget_constraints",
    "sensitivities_allergies",
    "hair_structure",
    "pregnancy_status",
    "existing_inventory",
    "hair_length",
    "maintenance_commitment",
    "past_experiences",
    "professional_treatments",
    "color_history",
    "eco_preferences",
    "lifestyle_factors"
]

ALL_TOPICS = DEMOGRAPHIC_TOPICS + HAIR_TOPICS

# --------------------------------------------------
# QUESTION GROUPS
# --------------------------------------------------
QUESTION_GROUPS = [
    {
        "name": "Context",
        "question": "👋 Hi! I'm Halisi, your hair care expert.\n\nAre you looking for products for yourself or a child?",
        "topics": ["product_context"]
    },
    {
        "name": "Demographics",
        "question": "What is your age, gender, and location? (If you're shopping for a child, please provide the child's age and gender, and your location for climate)",
        "topics": ["user_demographics"]
    },
    {
        "name": "Hair Portrait",
        "question": "Describe your hair: type (straight, wavy, curly, coily/afro), main concerns, goals, scalp condition, and structure (natural, braids, locks, extensions).",
        "topics": ["hair_type_texture", "hair_scalp_concerns", "hair_goals", "scalp_condition", "hair_structure"]
    },
    {
        "name": "Routine & Products",
        "question": "Tell me about your current hair routine: products you use, how often you style, time spent per week, monthly budget, products at home, and whether you prefer a simple or detailed routine.",
        "topics": ["current_products", "hair_care_routine", "styling_habits", "time_availability", "budget_constraints", "existing_inventory", "routine_preference"]
    },
    {
        "name": "Health & Preferences",
        "question": "Any allergies, sensitivities, or medical considerations (pregnancy)? Past professional treatments (keratin, relaxing)? Color history? Importance of eco‑friendly products? Lifestyle factors (stress, diet, climate)? Past product experiences? How committed are you to a routine? Hair porosity if known?",
        "topics": ["sensitivities_allergies", "pregnancy_status", "professional_treatments", "color_history", "eco_preferences", "lifestyle_factors", "past_experiences", "maintenance_commitment", "hair_porosity"]
    }
]

# --------------------------------------------------
# NORMALIZATION HELPERS
# --------------------------------------------------
def normalize_budget(text):
    text = text.lower()
    amount_match = re.search(r'(\d+(?:\.\d+)?)', text)
    amount = amount_match.group(1) if amount_match else None
    currency = "USD"
    if "€" in text or "eur" in text:
        currency = "EUR"
    elif "£" in text or "gbp" in text:
        currency = "GBP"
    elif "c$" in text or "cad" in text:
        currency = "CAD"
    if amount:
        return f"{amount} {currency}"
    return text

def normalize_time(text):
    text = text.lower()
    match = re.search(r'(\d+)\s*(?:times?|x)\s*(?:per|a)\s*week', text)
    if match:
        return f"{match.group(1)}x/week"
    match = re.search(r'(\d+)\s*(?:minutes?|mins?)', text)
    if match:
        return f"{match.group(1)} min"
    return text

def normalize_hair_length(text):
    text = text.lower()
    mapping = {
        "very small": "Short",
        "short": "Short",
        "shoulder": "Shoulder-length",
        "medium": "Medium",
        "long": "Long",
        "very long": "Very long"
    }
    for key, val in mapping.items():
        if key in text:
            return val
    return text

def extract_product_list(text):
    keywords = ["shampoo", "conditioner", "oil", "serum", "gel", "cream", "lotion", "mask", "treatment", "spray", "mousse", "butter", "leave-in", "moisturizer", "detangler"]
    found = []
    text_lower = text.lower()
    for kw in keywords:
        if kw in text_lower:
            found.append(kw.title())
    return found if found else text

# --------------------------------------------------
# CLEAN + CATEGORIZE
# --------------------------------------------------
def clean_response(response):
    if not response:
        return ""
    response = response.lower().strip()
    response = re.sub(r'\s+', ' ', response)
    return response

def categorize_response(topic, response):
    cleaned = clean_response(response)
    if topic == "product_context":
        return "child" if any(word in cleaned for word in ["child", "kid", "baby", "daughter", "son"]) else "self"

    if topic == "user_demographics":
        extracted = {"age": "", "gender": "", "location": ""}
        age_match = re.search(r'\b(\d{1,3})\b', cleaned)
        if age_match:
            extracted["age"] = age_match.group(1)
        extracted["gender"] = "Female" if any(word in cleaned for word in ["female","woman","girl","she","her"]) else "Male"
        loc_match = re.search(r'from ([a-z\s,]+)', cleaned)
        extracted["location"] = loc_match.group(1).title() if loc_match else ""
        return extracted

    # Hair type
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

    if topic == "pregnancy_status":
        return "Not applicable" if "male" in cleaned else cleaned
    if topic == "hair_porosity":
        if "low" in cleaned: return "Low"
        if "medium" in cleaned: return "Medium"
        if "high" in cleaned: return "High"
        return cleaned
    if topic == "routine_preference":
        if any(word in cleaned for word in ["minimal","simple","quick","fast"]): return "Minimalist"
        if any(word in cleaned for word in ["multi","detailed","complex","elaborate"]): return "Multi-step"
        if any(word in cleaned for word in ["balanced","moderate"]): return "Balanced"
        return cleaned
    if topic == "scalp_condition":
        if any(word in cleaned for word in ["dry","flaky","dandruff"]): return "Dry / Flaky"
        if any(word in cleaned for word in ["oily","greasy"]): return "Oily"
        if any(word in cleaned for word in ["normal","healthy","good"]): return "Normal"
        if any(word in cleaned for word in ["itch","irritat"]): return "Itchy / Irritated"
        return cleaned
    if topic == "budget_constraints": return normalize_budget(cleaned)
    if topic == "time_availability": return normalize_time(cleaned)
    if topic == "hair_length": return normalize_hair_length(cleaned)
    if topic in ["current_products","existing_inventory"]:
        products = extract_product_list(cleaned)
        return ", ".join(products) if isinstance(products, list) else cleaned
    return cleaned

# --------------------------------------------------
# GENERATE NEXT QUESTION
# --------------------------------------------------
def generate_next_group_question():
    if st.session_state.current_group >= len(QUESTION_GROUPS): return None, None
    group = QUESTION_GROUPS[st.session_state.current_group]
    base_question = group["question"]
    if group["name"] == "Demographics" and "product_context" in st.session_state.extracted_info:
        ctx = st.session_state.extracted_info["product_context"].get("value")
        if ctx == "child":
            base_question = "What is your child's age and gender? (Your location also helps for climate recommendations)"
    return base_question, st.session_state.current_group

# --------------------------------------------------
# GENERATE CLEAN PROFILE
# --------------------------------------------------
def generate_clean_profile():
    profile_data = {}
    meaningful_count = 0
    for topic in ALL_TOPICS:
        if topic in st.session_state.extracted_info:
            entry = st.session_state.extracted_info[topic]
            raw = entry.get("raw","")
            val = entry.get("value","")
            profile_data[topic] = {"raw": raw[:200], "value": val}
            if val and str(val).lower() not in ["", "none", "no", "unspecified", "not applicable"]:
                meaningful_count += 1
    completeness = round((meaningful_count / len(ALL_TOPICS)) * 100,1)
    summary_parts = []
    for topic in ["hair_type_texture","hair_scalp_concerns","hair_goals","current_products"]:
        if topic in profile_data:
            val = profile_data[topic]["value"]
            if val: summary_parts.append(f"{topic.replace('_',' ').title()}: {val}")
    # Add image analysis
    if st.session_state.analysis_results:
        hair_type = st.session_state.analysis_results.get('hair_type', {}).get('hair_type', 'unknown')
        summary_parts.append(f"Image analysis: {hair_type}")
    summary = "; ".join(summary_parts) if summary_parts else "Profile ready"

    image_data = None
    if st.session_state.captured_image:
        buffered = io.BytesIO()
        st.session_state.captured_image.save(buffered, format="JPEG")
        image_data = base64.b64encode(buffered.getvalue()).decode()

    return {
        "profile_id": f"halisi_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "completion_date": datetime.now().isoformat(),
        "stats": {"questions": st.session_state.question_count,
                  "meaningful_data": meaningful_count,
                  "completeness": completeness},
        "demographics": {k:v for k,v in profile_data.items() if k in DEMOGRAPHIC_TOPICS},
        "hair_profile": {k:v for k,v in profile_data.items() if k in HAIR_TOPICS},
        "summary": summary,
        "recommendation_ready": meaningful_count >= 15,
        "hair_image_base64": image_data,
        "image_analysis": st.session_state.analysis_results
    }

# --------------------------------------------------
# INITIALIZE FIRST QUESTION
# --------------------------------------------------
if not st.session_state.messages:
    first_group = QUESTION_GROUPS[0]
    st.session_state.messages.append({"role":"assistant","content":first_group["question"]})
    st.session_state.current_group = 0

# --------------------------------------------------
# UI - CHAT MESSAGES
# --------------------------------------------------
st.title("💇‍♀️ Halisi Cosmetics - Smart Hair Profiler")
st.markdown("*Quick, comprehensive hair profiling in just 5 questions + AI photo analysis*")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/4A90E2/FFFFFF?text=Halisi", width=150)
    st.caption("Progress")
    progress_val = st.session_state.question_count / MAX_QUESTIONS
    st.progress(min(progress_val,1.0))
    st.caption(f"Question {st.session_state.question_count+1}/{MAX_QUESTIONS}")
    collected = len(st.session_state.extracted_info)
    st.metric("Topics covered", f"{collected}/{len(ALL_TOPICS)}")

# --------------------------------------------------
# CHAT LOGIC
# --------------------------------------------------
if not st.session_state.conversation_complete:
    user_input = st.chat_input("Your answer...")
    if user_input:
        st.session_state.messages.append({"role":"user","content":user_input})
        with st.chat_message("user"): st.markdown(user_input)

        group = QUESTION_GROUPS[st.session_state.current_group]
        for topic in group["topics"]:
            cleaned = clean_response(user_input)
            categorized = categorize_response(topic, cleaned)
            st.session_state.extracted_info[topic] = {"raw": user_input, "cleaned": cleaned, "value": categorized}

        st.session_state.question_count += 1
        st.session_state.current_group += 1

        # Auto-skip pregnancy for males
        demo_val = st.session_state.extracted_info.get("user_demographics", {}).get("value", {})
        if isinstance(demo_val, dict) and demo_val.get("gender") == "Male":
            st.session_state.extracted_info["pregnancy_status"] = {
                "raw":"automatically skipped","cleaned":"not applicable","value":"Not applicable"
            }

        if st.session_state.question_count >= MAX_QUESTIONS:
            st.session_state.conversation_complete = True
            st.rerun()
        else:
            next_q, _ = generate_next_group_question()
            if next_q:
                st.session_state.messages.append({"role":"assistant","content":next_q})
                with st.chat_message("assistant"): st.markdown(next_q)
                st.rerun()

# --------------------------------------------------
# PHOTO CAPTURE + ANALYSIS
# --------------------------------------------------
if st.session_state.conversation_complete and not st.session_state.image_analyzed:
    st.markdown("---")
    st.subheader("📸 Capture a photo of your hair")
    st.write("This helps analyze your hair for better recommendations.")
    camera_image = st.camera_input("Take a picture", key="hair_cam")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.session_state.captured_image = image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            temp_path = tmp.name
            image.save(temp_path, format='JPEG')
        try:
            with st.spinner("Analyzing your hair with AI..."):
                result = process_image_full_pipeline(
                    image_path=temp_path,
                    azure_api_key=os.getenv("AZUREOPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZUREOPENAI_API_ENDPOINT")
                )
                st.session_state.analysis_results = result
                st.session_state.image_captured = True
                st.session_state.image_analyzed = True
        except Exception as e:
            st.error(f"Image analysis failed: {e}")
            st.session_state.image_analyzed = True
        finally:
            os.unlink(temp_path)
        st.rerun()
    if st.button("Skip photo (continue without image)", use_container_width=True):
        st.session_state.image_analyzed = True
        st.rerun()

# --------------------------------------------------
# FINAL PROFILE DISPLAY
# --------------------------------------------------
if st.session_state.conversation_complete and st.session_state.image_analyzed:
    if not st.session_state.profile_data:
        st.session_state.profile_data = generate_clean_profile()

    st.success("✅ **Profile Successfully Created!**")
    stats = st.session_state.profile_data.get("stats", {})
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Questions", stats.get("questions",0))
    with col2: st.metric("Useful data", stats.get("meaningful_data",0))
    with col3: st.metric("Completeness", f"{stats.get('completeness',0)}%")

    with st.expander("📋 **Your Hair Profile**", expanded=True):
        st.write(st.session_state.profile_data.get("summary",""))
        if st.session_state.captured_image: st.image(st.session_state.captured_image, caption="Your hair", width=300)

        demo = st.session_state.profile_data.get("demographics", {})
        hair = st.session_state.profile_data.get("hair_profile", {})
        analysis = st.session_state.profile_data.get("image_analysis", {})
        st.write("**Demographics**", demo)
        st.write("**Hair Profile**", hair)
        st.write("**AI Hair Analysis**", analysis)

    # Export JSON
    json_str = json.dumps(st.session_state.profile_data, indent=2)
    st.download_button(
        "Download profile as JSON",
        data=json_str,
        file_name=f"{st.session_state.profile_data['profile_id']}.json",
        mime="application/json"
    )
