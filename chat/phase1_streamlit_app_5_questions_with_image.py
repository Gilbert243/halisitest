# ==================================================
# phase1_streamlit_app_5_questions_with_image.py
# ==================================================

import streamlit as st
import json
import os
import re
import base64
import io
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
from PIL import Image

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
if "messages" not in st.session_state:
    st.session_state.messages = []
if "extracted_info" not in st.session_state:
    st.session_state.extracted_info = {}
if "conversation_complete" not in st.session_state:
    st.session_state.conversation_complete = False
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "profile_data" not in st.session_state:
    st.session_state.profile_data = {}
if "current_group" not in st.session_state:
    st.session_state.current_group = 0
if "user_demographics" not in st.session_state:
    st.session_state.user_demographics = {}
if "product_context" not in st.session_state:
    st.session_state.product_context = {}
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None
if "image_captured" not in st.session_state:
    st.session_state.image_captured = False
if "image_analyzed" not in st.session_state:
    st.session_state.image_analyzed = False
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}

MAX_QUESTIONS = 5  # Only 5 composite questions

# --------------------------------------------------
# TOPICS (unchanged)
# --------------------------------------------------
DEMOGRAPHIC_TOPICS = [
    "product_context",       # self or child
    "user_demographics",     # age, gender, location
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
# QUESTION GROUPS (unchanged)
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
# NORMALIZATION HELPERS (unchanged)
# --------------------------------------------------
def normalize_budget(text):
    """Extract numeric amount and currency."""
    text = text.lower()
    amount_match = re.search(r'(\d+(?:\.\d+)?)', text)
    amount = amount_match.group(1) if amount_match else None
    currency = "USD"  # default
    if "€" in text or "eur" in text:
        currency = "EUR"
    elif "£" in text or "gbp" in text:
        currency = "GBP"
    elif "c$" in text or "cad" in text:
        currency = "CAD"
    elif "$" in text or "usd" in text or "dollar" in text:
        currency = "USD"
    if amount:
        return f"{amount} {currency}"
    return text

def normalize_time(text):
    """Convert time phrases to standardized format."""
    text = text.lower()
    if "quarter" in text and "hour" in text:
        return "15 minutes"
    if "half" in text and "hour" in text:
        return "30 minutes"
    match = re.search(r'(\d+)\s*(?:times?|x)\s*(?:per|a)\s*week', text)
    if match:
        return f"{match.group(1)}x/week"
    match = re.search(r'(\d+)\s*(?:minutes?|mins?)', text)
    if match:
        return f"{match.group(1)} min"
    return text

def normalize_hair_length(text):
    """Map common length descriptions."""
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
    """Extract list of product types from a string."""
    keywords = ["shampoo", "conditioner", "oil", "serum", "gel", "cream", "lotion", "mask", "treatment", "spray", "mousse", "butter", "leave-in", "moisturizer", "detangler"]
    found = []
    text_lower = text.lower()
    for kw in keywords:
        if kw in text_lower:
            found.append(kw.title())
    return found if found else text

# --------------------------------------------------
# CATEGORIZATION (unchanged)
# --------------------------------------------------
def clean_response(response):
    """Light cleaning – keep original for raw storage."""
    if not response:
        return ""
    response = response.lower().strip()
    response = re.sub(r'\s+', ' ', response)
    return response

def categorize_response(topic, response):
    """Return a clean, normalized value for the given topic."""
    cleaned = clean_response(response)

    # ----- DEMOGRAPHICS -----
    if topic == "product_context":
        if any(word in cleaned for word in ["child", "kid", "baby", "toddler", "daughter", "son"]):
            return "child"
        else:
            return "self"

    if topic == "user_demographics":
        extracted = {"age": "", "gender": "", "location": ""}
        # Age
        age_match = re.search(r'\b(\d{1,3})\s*(?:years?|yo|y\.?o\.?)\b', cleaned)
        if age_match:
            extracted["age"] = age_match.group(1)
        elif "teen" in cleaned:
            extracted["age"] = "13-19"
        elif "adult" in cleaned:
            extracted["age"] = "adult"
        elif "child" in cleaned or "kid" in cleaned:
            extracted["age"] = "child"
        # Gender
        if any(word in cleaned for word in ["female", "woman", "girl", "she", "her"]):
            extracted["gender"] = "Female"
        elif any(word in cleaned for word in ["male", "man", "boy", "he", "him"]):
            extracted["gender"] = "Male"
        elif any(word in cleaned for word in ["non-binary", "other", "prefer not"]):
            extracted["gender"] = "Other"
        # Location
        loc_patterns = [
            r'\b(?:from|in|located in|live in)\s+([a-z\s,]+?)(?:\.|$|\s+and|\s+my|\s+i\b)',
            r'\b([a-z\s,]+?)\s*$'
        ]
        for pattern in loc_patterns:
            loc_match = re.search(pattern, cleaned)
            if loc_match:
                location = loc_match.group(1).strip()
                if location and not re.match(r'^(years?|old|age|male|female|man|woman|child|kid)$', location):
                    extracted["location"] = location.title()
                    break
        return extracted

    # ----- UNSPECIFIED DETECTION -----
    unspecified_phrases = ["i don't know", "dont know", "not sure", "unsure", "not yet", "i don't have", "no idea", "unknown"]
    if any(phrase in cleaned for phrase in unspecified_phrases):
        return "unspecified"

    # ----- HAIR TYPE -----
    if topic == "hair_type_texture":
        if any(word in cleaned for word in ["straight", "1a", "1b", "1c", "type 1"]):
            return "Straight"
        if any(word in cleaned for word in ["wavy", "2a", "2b", "2c", "type 2"]):
            return "Wavy"
        if any(word in cleaned for word in ["curly", "3a", "3b", "3c", "type 3"]):
            return "Curly"
        if any(word in cleaned for word in ["coily", "afro", "kinky", "4a", "4b", "4c", "type 4"]):
            return "Coily/Afro-textured"
        return cleaned

    # ----- HAIR STRUCTURE -----
    if topic == "hair_structure":
        if any(word in cleaned for word in ["braid", "cornrow", "plait"]):
            return "Braids"
        if any(word in cleaned for word in ["lock", "dread", "dreadlock"]):
            return "Locks"
        if any(word in cleaned for word in ["extension", "weave", "wig"]):
            return "Extensions"
        if "natural" in cleaned:
            return "Natural"
        return cleaned

    # ----- PREGNANCY STATUS -----
    if topic == "pregnancy_status":
        if any(word in cleaned for word in ["pregnant", "expecting"]):
            return "Pregnant"
        if any(word in cleaned for word in ["breastfeed", "nursing", "lactating"]):
            return "Breastfeeding"
        if any(word in cleaned for word in ["no", "none", "not", "n/a"]):
            return "Not applicable"
        return cleaned

    # ----- HAIR POROSITY -----
    if topic == "hair_porosity":
        if any(word in cleaned for word in ["low", "slow", "repel", "float"]):
            return "Low"
        if any(word in cleaned for word in ["medium", "normal", "average"]):
            return "Medium"
        if any(word in cleaned for word in ["high", "fast", "absorb", "sink"]):
            return "High"
        return cleaned

    # ----- ROUTINE PREFERENCE -----
    if topic == "routine_preference":
        if any(word in cleaned for word in ["minimal", "simple", "quick", "fast"]):
            return "Minimalist"
        if any(word in cleaned for word in ["multi", "detailed", "complex", "elaborate"]):
            return "Multi-step"
        if any(word in cleaned for word in ["balanced", "moderate"]):
            return "Balanced"
        return cleaned

    # ----- SCALP CONDITION -----
    if topic == "scalp_condition":
        if any(word in cleaned for word in ["dry", "flaky", "dandruff"]):
            return "Dry / Flaky"
        if any(word in cleaned for word in ["oily", "greasy"]):
            return "Oily"
        if any(word in cleaned for word in ["normal", "healthy", "good"]):
            return "Normal"
        if any(word in cleaned for word in ["itch", "irritat"]):
            return "Itchy / Irritated"
        return cleaned

    # ----- BUDGET NORMALIZATION -----
    if topic == "budget_constraints":
        return normalize_budget(cleaned)

    # ----- TIME NORMALIZATION -----
    if topic == "time_availability":
        return normalize_time(cleaned)

    # ----- HAIR LENGTH NORMALIZATION -----
    if topic == "hair_length":
        return normalize_hair_length(cleaned)

    # ----- PRODUCT LIST EXTRACTION -----
    if topic in ["current_products", "existing_inventory"]:
        products = extract_product_list(cleaned)
        if isinstance(products, list):
            return ", ".join(products)
        return cleaned

    # ----- NEW TOPICS -----
    if topic == "professional_treatments":
        return cleaned
    if topic == "color_history":
        if any(word in cleaned for word in ["yes", "dyed", "colored", "bleach"]):
            return "Yes"
        if any(word in cleaned for word in ["no", "never", "natural"]):
            return "No"
        return cleaned
    if topic == "eco_preferences":
        if any(word in cleaned for word in ["very", "extremely", "essential"]):
            return "Very important"
        if any(word in cleaned for word in ["somewhat", "moderately", "prefer"]):
            return "Somewhat important"
        if any(word in cleaned for word in ["not", "don't", "unimportant"]):
            return "Not important"
        return cleaned
    if topic == "lifestyle_factors":
        return cleaned

    # ----- OTHER FREE‑TEXT TOPICS -----
    if topic in ["hair_goals", "hair_scalp_concerns", "hair_care_routine",
                 "styling_habits", "sensitivities_allergies", "past_experiences",
                 "maintenance_commitment"]:
        return cleaned

    return cleaned

# --------------------------------------------------
# FUNCTION TO GENERATE NEXT GROUP QUESTION (unchanged)
# --------------------------------------------------
def generate_next_group_question():
    """Return the question for the current group."""
    if st.session_state.current_group >= len(QUESTION_GROUPS):
        return None, None

    group = QUESTION_GROUPS[st.session_state.current_group]
    base_question = group["question"]

    # Adapt context for demographics
    if group["name"] == "Demographics" and "product_context" in st.session_state.extracted_info:
        ctx = st.session_state.extracted_info["product_context"].get("value")
        if ctx == "child":
            base_question = "What is your child's age and gender? (Your location also helps for climate recommendations)"

    return base_question, st.session_state.current_group

# --------------------------------------------------
# PROFILE GENERATION (with image storage, no analysis yet)
# --------------------------------------------------
def generate_clean_profile():
    """Generate a profile with correctly categorized values and captured image."""
    profile_data = {}
    meaningful_count = 0

    for topic in ALL_TOPICS:
        if topic in st.session_state.extracted_info:
            entry = st.session_state.extracted_info[topic]
            raw = entry.get("raw", "")
            categorized = entry.get("value", "")

            profile_data[topic] = {
                "raw": raw[:200],
                "value": categorized
            }

            if categorized and str(categorized).lower() not in [
                "", "none", "no", "not specified", "unsure", "i don't know", "unspecified",
                "not yet", "unknown", "not applicable", "n/a"
            ]:
                meaningful_count += 1

    completeness = round((meaningful_count / len(ALL_TOPICS)) * 100, 1)

    summary_parts = []
    for topic in ["hair_type_texture", "hair_scalp_concerns", "hair_goals", "current_products"]:
        if topic in profile_data:
            val = profile_data[topic]["value"]
            if val and str(val).lower() not in ["", "none", "no", "not specified", "unspecified"]:
                summary_parts.append(f"{topic.replace('_',' ').title()}: {val}")

    summary = "; ".join(summary_parts) if summary_parts else "Profile ready"

    # Encode captured image if any (for download)
    image_data = None
    if st.session_state.captured_image is not None:
        buffered = io.BytesIO()
        st.session_state.captured_image.save(buffered, format="JPEG")
        image_data = base64.b64encode(buffered.getvalue()).decode()

    return {
        "profile_id": f"halisi_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "completion_date": datetime.now().isoformat(),
        "stats": {
            "questions": st.session_state.question_count,
            "meaningful_data": meaningful_count,
            "completeness": completeness
        },
        "demographics": {k: v for k, v in profile_data.items() if k in DEMOGRAPHIC_TOPICS},
        "hair_profile": {k: v for k, v in profile_data.items() if k in HAIR_TOPICS},
        "summary": summary,
        "recommendation_ready": meaningful_count >= 15,
        "hair_image_base64": image_data,
        "image_analysis": st.session_state.analysis_results  # placeholder for future analysis
    }

# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------
if not st.session_state.messages:
    first_group = QUESTION_GROUPS[0]
    st.session_state.messages.append({"role": "assistant", "content": first_group["question"]})
    st.session_state.current_group = 0

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("💇‍♀️ Halisi Cosmetics - Smart Hair Profiler")
st.markdown("*Quick, comprehensive hair profiling in just 5 questions + photo capture*")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/4A90E2/FFFFFF?text=Halisi", width=150)
    st.caption("Progress")
    if not st.session_state.conversation_complete:
        progress = min(st.session_state.question_count / MAX_QUESTIONS, 1.0)
        st.progress(progress)
        st.caption(f"Question {st.session_state.question_count+1}/{MAX_QUESTIONS}")
    else:
        if st.session_state.image_captured:
            st.progress(1.0)
            st.caption("Photo captured")
        else:
            st.progress(1.0)
            st.caption("Questions completed - Take a photo")
    collected = len(st.session_state.extracted_info)
    st.metric("Topics covered", f"{collected}/{len(ALL_TOPICS)}")

    if st.session_state.extracted_info:
        st.divider()
        st.caption("**Context**")
        if "product_context" in st.session_state.extracted_info:
            ctx = st.session_state.extracted_info["product_context"].get("value", "")
            st.caption(f"• For: {ctx}")
        if "user_demographics" in st.session_state.extracted_info:
            demo = st.session_state.extracted_info["user_demographics"].get("value", {})
            if isinstance(demo, dict):
                if demo.get("age"):
                    st.caption(f"• Age: {demo['age']}")
                if demo.get("gender"):
                    st.caption(f"• Gender: {demo['gender']}")

# --------------------------------------------------
# CHAT LOGIC (questions)
# --------------------------------------------------
if not st.session_state.conversation_complete:
    user_input = st.chat_input("Your answer...")
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Extract information for all topics in the current group
        if st.session_state.current_group < len(QUESTION_GROUPS):
            group = QUESTION_GROUPS[st.session_state.current_group]
            for topic in group["topics"]:
                cleaned = clean_response(user_input)
                categorized = categorize_response(topic, cleaned)

                st.session_state.extracted_info[topic] = {
                    "raw": user_input,
                    "cleaned": cleaned,
                    "value": categorized
                }

                # Also store in demographic dicts if needed
                if topic in DEMOGRAPHIC_TOPICS:
                    st.session_state.user_demographics[topic] = categorized

        st.session_state.question_count += 1
        st.session_state.current_group += 1

        # Check if we have finished all questions
        should_end = (
            st.session_state.question_count >= MAX_QUESTIONS or
            st.session_state.current_group >= len(QUESTION_GROUPS)
        )

        if should_end:
            # Auto‑skip pregnancy for males
            demo_val = st.session_state.extracted_info.get("user_demographics", {}).get("value", {})
            if isinstance(demo_val, dict) and demo_val.get("gender") == "Male":
                if "pregnancy_status" not in st.session_state.extracted_info:
                    st.session_state.extracted_info["pregnancy_status"] = {
                        "raw": "automatically skipped",
                        "cleaned": "not applicable",
                        "value": "Not applicable"
                    }

            st.session_state.conversation_complete = True
            st.rerun()
        else:
            # Ask next group question
            next_q, _ = generate_next_group_question()
            if next_q:
                st.session_state.messages.append({"role": "assistant", "content": next_q})
                with st.chat_message("assistant"):
                    st.markdown(next_q)
                st.rerun()
            else:
                st.session_state.conversation_complete = True
                st.rerun()

# --------------------------------------------------
# PHOTO CAPTURE (after questions, no analysis yet)
# --------------------------------------------------
if st.session_state.conversation_complete and not st.session_state.image_captured:
    st.markdown("---")
    st.subheader("📸 Now, let's capture a photo of your hair")
    st.write("This will help us with future AI analysis for better recommendations.")

    col1, col2 = st.columns(2)
    with col1:
        camera_image = st.camera_input("Take a picture", key="hair_cam")
        if camera_image is not None:
            # Store the image
            image = Image.open(camera_image)
            st.session_state.captured_image = image
            st.session_state.image_captured = True
            st.success("Photo captured successfully!")
            st.rerun()

    with col2:
        if st.button("Skip photo (continue without image)", use_container_width=True):
            st.session_state.image_captured = True
            st.rerun()

# --------------------------------------------------
# FINAL PROFILE DISPLAY (after photo)
# --------------------------------------------------
if st.session_state.conversation_complete and st.session_state.image_captured:
    # Generate profile if not already done
    if not st.session_state.profile_data:
        st.session_state.profile_data = generate_clean_profile()

    st.success("✅ **Profile Successfully Created!**")
    stats = st.session_state.profile_data.get("stats", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Questions", stats.get("questions", 0))
    with col2:
        st.metric("Useful data", stats.get("meaningful_data", 0))
    with col3:
        st.metric("Completeness", f"{stats.get('completeness', 0)}%")

    with st.expander("📋 **Your Hair Profile**", expanded=True):
        st.write(st.session_state.profile_data.get("summary", ""))

        if st.session_state.captured_image is not None:
            st.image(st.session_state.captured_image, caption="Your hair", width=300)

        # Demographics
        demo = st.session_state.profile_data.get("demographics", {})
        if demo:
            st.caption("**Demographics**")
            for topic, data in demo.items():
                val = data.get("value", "")
                if val and str(val).lower() not in ["not specified", ""]:
                    if isinstance(val, dict):
                        for k, v in val.items():
                            if v:
                                st.caption(f"• {k.title()}: {v}")
                    else:
                        st.caption(f"• {topic.replace('_',' ').title()}: {val}")

        # Hair profile – show only key fields that have values
        hair = st.session_state.profile_data.get("hair_profile", {})
        if hair:
            st.caption("**Hair details**")
            key_fields = ["hair_type_texture", "hair_scalp_concerns", "hair_goals", 
                         "current_products", "hair_structure", "pregnancy_status",
                         "professional_treatments", "color_history", "eco_preferences", "lifestyle_factors"]
            for topic in key_fields:
                if topic in hair:
                    val = hair[topic].get("value", "")
                    if val and str(val).lower() not in ["", "none", "no", "not applicable", "unspecified"]:
                        st.caption(f"• {topic.replace('_',' ').title()}: {val}")

        # Placeholder for future image analysis
        if st.session_state.analysis_results:
            st.caption("**Image analysis**")
            st.caption(f"• Detected hair type: {st.session_state.analysis_results.get('classification', 'unknown')}")

    with st.expander("📄 **All Responses (Raw & Normalized)**"):
        all_data = st.session_state.profile_data.get("hair_profile", {})
        all_data.update(st.session_state.profile_data.get("demographics", {}))
        if all_data:
            for topic, data in all_data.items():
                st.markdown(f"**{topic.replace('_',' ').title()}**")
                st.caption(f"Raw: {data.get('raw', '')}")
                st.caption(f"Normalized: {data.get('value', '')}")
                st.divider()
        else:
            st.caption("No data available.")

    col1, col2 = st.columns(2)
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

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("Halisi Cosmetics")