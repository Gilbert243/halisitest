# ==================================================
# phase1_streamlit_app.py
# ==================================================

import streamlit as st
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Halisi Cosmetics - Enhanced Hair Profiler",
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
if "current_topic" not in st.session_state:
    st.session_state.current_topic = None
if "user_demographics" not in st.session_state:
    st.session_state.user_demographics = {}
if "product_context" not in st.session_state:
    st.session_state.product_context = {}

MAX_QUESTIONS = 25

# --------------------------------------------------
# TOPICS
# --------------------------------------------------
DEMOGRAPHIC_TOPICS = [
    "product_context",       # For self or child
    "user_demographics",     # Age, gender, location combined
]

HAIR_TOPICS = [
    "hair_type_texture",
    "hair_scalp_concerns",
    "hair_goals",
    "current_products",
    "hair_care_routine",
    "existing_inventory",
    "scalp_condition",
    "hair_porosity",
    "hair_length",
    "styling_habits",
    "routine_preference",
    "time_availability",
    "maintenance_commitment",
    "sensitivities_allergies",
    "past_experiences",
    "lifestyle_factors",
    "budget_constraints",
    "eco_preferences",
    "professional_treatments",
    "color_history",
    "hair_structure",       # Added: natural hair, braids, locks, extensions
    "pregnancy_status"      # Added: pregnancy/breastfeeding status
]

ALL_TOPICS = DEMOGRAPHIC_TOPICS + HAIR_TOPICS

# --------------------------------------------------
# FALLBACK QUESTIONS
# --------------------------------------------------
FALLBACK_QUESTIONS = {
    "product_context": "Are you looking for hair products for yourself, or do you have a child in mind?",
    "user_demographics": "Great! Now, could you share the age, gender, and location? This helps me personalize recommendations for {context}.",
    "hair_type_texture": "How would you describe your hair type and texture?",
    "hair_scalp_concerns": "What are your main hair and scalp concerns?",
    "hair_goals": "What are your hair goals?",
    "current_products": "What hair products and brands do you currently use?",
    "hair_care_routine": "What's your hair care routine?",
    "existing_inventory": "What hair products do you already have at home?",
    "scalp_condition": "How would you describe your scalp condition?",
    "hair_porosity": "Do you know your hair porosity?",
    "hair_length": "What is your current hair length and desired length?",
    "styling_habits": "What are your styling habits and tools usage?",
    "routine_preference": "Do you prefer minimalist or multi-step hair care routines?",
    "time_availability": "How much time can you dedicate to hair care?",
    "maintenance_commitment": "What's your maintenance commitment level?",
    "sensitivities_allergies": "Do you have any sensitivities or allergies to hair product ingredients?",
    "past_experiences": "What past hair product experiences worked or didn't work for you?",
    "lifestyle_factors": "How do lifestyle factors affect your hair?",
    "budget_constraints": "What's your budget for hair care products?",
    "eco_preferences": "How important are eco-friendly and sustainable products to you?",
    "professional_treatments": "What professional treatments do you get and how often?",
    "color_history": "What's your color treatment history?",
    "hair_structure": "Is your hair natural, or do you wear braids, locks, extensions, or other styles?",
    "pregnancy_status": "Are you currently pregnant or breastfeeding? This is important for product safety."
}

# --------------------------------------------------
# ENHANCED CATEGORIZATION
# --------------------------------------------------
def clean_response(response):
    """Clean and normalize response"""
    if not response:
        return response
    
    response_lower = response.lower()
    
    corrections = {
        "shampp": "shampoo",
        "conditionning": "conditioning",
        "augmantation": "augmentation",
        "porositie": "porosity",
        "minimaliste": "minimalist",
        "colour": "color",
        "cheveux": "hair",
        "cheveu": "hair",
        "dandruf": "dandruff",
        "psoriasys": "psoriasis",
        "eczma": "eczema"
    }
    
    for wrong, correct in corrections.items():
        if wrong in response_lower:
            response_lower = response_lower.replace(wrong, correct)
    
    response = re.sub(r'\s+', ' ', response_lower).strip()
    
    return response

def categorize_response(topic, response):
    """Categorize responses with enhanced logic"""
    cleaned = clean_response(response)
    
    if topic == "product_context":
        if any(word in cleaned for word in ["child", "kid", "baby", "toddler", "son", "daughter"]):
            return {"context": "child", "details": cleaned}
        elif any(word in cleaned for word in ["myself", "self", "me", "i am"]):
            return {"context": "self", "details": cleaned}
        else:
            return {"context": "self", "details": cleaned}
    
    elif topic == "user_demographics":
        extracted = {"age": "not specified", "gender": "not specified", "location": "not specified"}
        
        # Age extraction
        age_patterns = [
            (r'\b(\d{1,3})\s*years?\s*old\b', "age"),
            (r'\b(\d{1,3})\s*yo\b', "age"),
            (r'\bage\s*(\d{1,3})\b', "age"),
            (r'\bteen\b', "13-19"),
            (r'\bteenager\b', "13-19"),
            (r'\badult\b', "adult"),
            (r'\bmiddle\s*aged\b', "40-60"),
            (r'\bsenior\b', "60+"),
            (r'\belderly\b', "65+")
        ]
        
        for pattern, label in age_patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                extracted["age"] = match.group(1) if label == "age" else label
                break
        
        # Gender extraction
        if any(word in cleaned for word in ["female", "woman", "girl", "she", "her", "women"]):
            extracted["gender"] = "Female"
        elif any(word in cleaned for word in ["male", "man", "boy", "he", "him", "men"]):
            extracted["gender"] = "Male"
        elif any(word in cleaned for word in ["non-binary", "other", "prefer not", "nonbinary"]):
            extracted["gender"] = "Other/Prefer not to say"
        
        # Location extraction - enhanced
        location_patterns = [
            (r'\bfrom\s+([a-zA-Z\s,]+)(?:\s|$)', 1),
            (r'\bin\s+([a-zA-Z\s,]+)(?:\s|$)', 1),
            (r'\blocated\s+in\s+([a-zA-Z\s,]+)(?:\s|$)', 1),
            (r'\b([a-zA-Z\s]+)(?:,\s*[a-zA-Z]+)?\s*$', 0)  # Last word as location
        ]
        
        for pattern, group in location_patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                location = match.group(group).strip()
                if location and len(location) > 1 and not location.isdigit():
                    # Filter out common non-location words
                    non_locations = ["years", "old", "year", "yo", "age", "male", "female", "man", "woman"]
                    if location.lower() not in non_locations:
                        extracted["location"] = location.title()
                        break
        
        return extracted
    
    elif topic == "hair_type_texture":
        if any(word in cleaned for word in ["straight", "1a", "1b", "1c", "type 1"]):
            return "Straight"
        elif any(word in cleaned for word in ["wavy", "2a", "2b", "2c", "type 2"]):
            return "Wavy"
        elif any(word in cleaned for word in ["curly", "3a", "3b", "3c", "type 3"]):
            return "Curly"
        elif any(word in cleaned for word in ["coily", "4a", "4b", "4c", "afro", "kinky", "type 4"]):
            return "Coily/Afro-textured"
        elif "afro" in cleaned:
            return "Coily/Afro-textured"
    
    elif topic == "hair_structure":
        if any(word in cleaned for word in ["braid", "braids", "cornrow", "plaits"]):
            return "Braids"
        elif any(word in cleaned for word in ["lock", "locks", "dread", "dreadlock", "dreads"]):
            return "Locks/Dreadlocks"
        elif any(word in cleaned for word in ["extension", "weave", "wig", "hairpiece"]):
            return "Extensions/Weave"
        elif any(word in cleaned for word in ["natural", "virgin", "no extensions"]):
            return "Natural"
    
    elif topic == "pregnancy_status":
        if any(word in cleaned for word in ["pregnant", "expecting", "pregnancy", "expectant"]):
            return "Pregnant"
        elif any(word in cleaned for word in ["breastfeed", "nursing", "lactating", "breastfeeding"]):
            return "Breastfeeding"
        elif any(word in cleaned for word in ["none", "no", "not applicable", "n/a"]):
            return "Not pregnant/breastfeeding"
    
    elif topic == "hair_porosity":
        if any(word in cleaned for word in ["low", "slow", "repels", "floats", "takes time"]):
            return "Low porosity"
        elif any(word in cleaned for word in ["medium", "normal", "balanced", "average"]):
            return "Medium porosity"
        elif any(word in cleaned for word in ["high", "fast", "absorbs", "sinks", "quickly"]):
            return "High porosity"
    
    elif topic == "routine_preference":
        if any(word in cleaned for word in ["minimalist", "simple", "quick", "basic", "fast"]):
            return "Minimalist"
        elif any(word in cleaned for word in ["multi-step", "detailed", "complex", "elaborate", "comprehensive"]):
            return "Multi-step"
        elif any(word in cleaned for word in ["balanced", "moderate", "middle", "regular"]):
            return "Balanced"
    
    return cleaned

# --------------------------------------------------
# CONTEXT-AWARE QUESTION GENERATION
# --------------------------------------------------
def generate_contextual_question():
    """Generate context-aware questions based on user responses"""
    
    # Prepare conversation context
    conversation_context = ""
    for msg in st.session_state.messages[-6:]:
        role = "User" if msg["role"] == "user" else "Halisi"
        conversation_context += f"{role}: {msg['content']}\n"
    
    # Determine next topic
    topics_asked = list(st.session_state.extracted_info.keys())
    next_topics = [t for t in ALL_TOPICS if t not in topics_asked]
    
    if not next_topics:
        return None, None
    
    next_topic = next_topics[0]
    
    # Get context for personalization
    context_info = {}
    
    if "product_context" in st.session_state.extracted_info:
        product_ctx = st.session_state.extracted_info["product_context"].get("categorized", {})
        if isinstance(product_ctx, dict):
            context_info["product_for"] = product_ctx.get("context", "self")
    
    if "user_demographics" in st.session_state.extracted_info:
        user_info = st.session_state.extracted_info["user_demographics"].get("categorized", {})
        if isinstance(user_info, dict):
            context_info.update(user_info)
    
    # Prepare context string
    context_str = ""
    pronoun = "your"
    possessive = "your"
    
    if context_info.get("product_for") == "child":
        pronoun = "your child's"
        possessive = "your child's"
        if context_info.get("age"):
            context_str += f"For a {context_info['age']} year old child. "
        if context_info.get("gender"):
            context_str += f"Gender: {context_info['gender']}. "
    else:
        if context_info.get("age"):
            context_str += f"Age: {context_info['age']}. "
        if context_info.get("gender"):
            context_str += f"Gender: {context_info['gender']}. "
        if context_info.get("location"):
            context_str += f"Location: {context_info['location']}. "
    
    # Prepare prompt
    prompt = f"""You are Halisi, an expert hair care consultant.

Context: {context_str}

Previous conversation:
{conversation_context}

Your task: Ask about {next_topic.replace('_', ' ')}
Base guidance: "{FALLBACK_QUESTIONS.get(next_topic, '')}"

Generate ONE natural, conversational question that:
1. Adapts to the context provided above
2. Uses appropriate pronouns: {pronoun} for questions about hair/scalp
3. Flows naturally from the previous conversation
4. Is clear, friendly, and professional
5. May reference something mentioned earlier
6. Keeps the user engaged

Keep it to 1-2 sentences maximum. Be natural and warm."""

    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        
        question = response.choices[0].message.content.strip()
        return question, next_topic
        
    except Exception as e:
        # Fallback with context
        fallback = FALLBACK_QUESTIONS.get(next_topic, "Tell me more about your hair.")
        
        # Add context to fallback question
        if next_topic == "user_demographics" and "product_context" in st.session_state.extracted_info:
            product_ctx = st.session_state.extracted_info["product_context"].get("categorized", {})
            context = "your child" if isinstance(product_ctx, dict) and product_ctx.get("context") == "child" else "you"
            fallback = fallback.format(context=context)
        
        return fallback, next_topic

# --------------------------------------------------
# ENHANCED PROFILE GENERATION
# --------------------------------------------------
def generate_enhanced_profile():
    """Generate enhanced profile with all data"""
    
    profile_data = {}
    meaningful_count = 0
    
    for topic in ALL_TOPICS:
        if topic in st.session_state.extracted_info:
            data = st.session_state.extracted_info[topic]
            raw_value = data.get("raw", "")
            cleaned_value = data.get("cleaned", "")
            categorized_value = data.get("categorized", "")
            
            final_value = categorized_value if categorized_value and categorized_value not in ["", "not specified"] else cleaned_value
            
            profile_data[topic] = {
                "raw_response": raw_value,
                "cleaned_value": cleaned_value,
                "categorized_value": categorized_value,
                "final_value": final_value
            }
            
            if final_value and str(final_value).lower() not in [
                "not specified", "none", "no", "yes", 
                "i don't know", "unsure", "not sure", "", "n/a", "skip"
            ]:
                meaningful_count += 1
    
    completeness = round((meaningful_count / len(ALL_TOPICS)) * 100, 1)
    
    summary_parts = []
    
    if "user_demographics" in profile_data:
        user_info = profile_data["user_demographics"].get("final_value", {})
        if isinstance(user_info, dict):
            if user_info.get("age"):
                summary_parts.append(f"Age: {user_info['age']}")
            if user_info.get("gender") and user_info["gender"] != "not specified":
                summary_parts.append(f"Gender: {user_info['gender']}")
            if user_info.get("location") and user_info["location"] != "not specified":
                summary_parts.append(f"Location: {user_info['location']}")
    
    if "product_context" in profile_data:
        product_ctx = profile_data["product_context"].get("final_value", {})
        if isinstance(product_ctx, dict):
            ctx = product_ctx.get("context", "self")
            summary_parts.append(f"Product for: {ctx}")
    
    hair_key_topics = ["hair_type_texture", "hair_scalp_concerns", "hair_goals", 
                      "current_products", "hair_porosity", "hair_structure"]
    
    for topic in hair_key_topics:
        if topic in profile_data:
            value = profile_data[topic].get("final_value", "")
            if value:
                display_name = topic.replace('_', ' ').title()
                summary_parts.append(f"{display_name}: {value}")
    
    summary = "; ".join(summary_parts) if summary_parts else "Complete hair profile"
    
    recommendation_ready = (
        meaningful_count >= 15 and
        "hair_type_texture" in profile_data and
        "hair_scalp_concerns" in profile_data and
        "current_products" in profile_data
    )
    
    enhanced_profile = {
        "profile_id": f"halisi_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "completion_date": datetime.now().isoformat(),
        "profile_statistics": {
            "total_questions": st.session_state.question_count,
            "demographic_topics": len([t for t in DEMOGRAPHIC_TOPICS if t in profile_data]),
            "hair_topics": len([t for t in HAIR_TOPICS if t in profile_data]),
            "meaningful_data_points": meaningful_count,
            "completeness_score": completeness
        },
        "demographic_profile": {
            k: v for k, v in profile_data.items() 
            if k in DEMOGRAPHIC_TOPICS
        },
        "hair_profile": {
            k: v for k, v in profile_data.items() 
            if k in HAIR_TOPICS
        },
        "profile_summary": summary,
        "recommendation_ready": recommendation_ready,
        "special_notes": {
            "has_child_context": "product_context" in profile_data and 
                               isinstance(profile_data["product_context"].get("final_value"), dict) and
                               profile_data["product_context"]["final_value"].get("context") == "child",
            "has_pregnancy_info": "pregnancy_status" in profile_data,
            "has_hair_structure_info": "hair_structure" in profile_data
        }
    }
    
    return enhanced_profile

# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------
if not st.session_state.messages:
    initial_message = """👋 Hello! I'm Halisi, your expert hair care consultant.

I'll guide you through some questions to build your complete hair profile.
This helps us find the perfect products and routine for you.

Are you looking for hair products for yourself, or do you have a child in mind?"""
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": initial_message
    })
    st.session_state.current_topic = "product_context"

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("💇‍♀️ Halisi Cosmetics - Enhanced Hair Profiler")
st.markdown("*Personalized hair analysis with demographic context*")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/4A90E2/FFFFFF?text=Halisi", width=150)
    st.title("Progress Dashboard")
    
    current_q = min(st.session_state.question_count + 1, MAX_QUESTIONS)
    st.subheader(f"Question {current_q} of {MAX_QUESTIONS}")
    progress = min(st.session_state.question_count / MAX_QUESTIONS, 1.0)
    st.progress(progress)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", f"{st.session_state.question_count}/{MAX_QUESTIONS}")
    with col2:
        collected = len(st.session_state.extracted_info)
        st.metric("Topics", f"{collected}/{len(ALL_TOPICS)}")
    
    if any(t in st.session_state.extracted_info for t in DEMOGRAPHIC_TOPICS):
        st.subheader("User Context")
        
        if "product_context" in st.session_state.extracted_info:
            ctx_data = st.session_state.extracted_info["product_context"].get("categorized", {})
            if isinstance(ctx_data, dict):
                ctx = ctx_data.get("context", "self")
                st.caption(f"Products for: {ctx}")
        
        if "user_demographics" in st.session_state.extracted_info:
            user_info = st.session_state.extracted_info["user_demographics"].get("categorized", {})
            if isinstance(user_info, dict):
                if user_info.get("age"):
                    st.caption(f"Age: {user_info['age']}")
                if user_info.get("gender") and user_info["gender"] != "not specified":
                    st.caption(f"Gender: {user_info['gender']}")
                if user_info.get("location") and user_info["location"] != "not specified":
                    st.caption(f"Location: {user_info['location']}")
    
    if st.session_state.extracted_info:
        st.subheader("Progress Overview")
        
        demo_completed = sum(1 for t in DEMOGRAPHIC_TOPICS if t in st.session_state.extracted_info)
        demo_total = len(DEMOGRAPHIC_TOPICS)
        demo_status = "✅" if demo_completed == demo_total else "🔄"
        st.caption(f"{demo_status} Demographics: {demo_completed}/{demo_total}")
        
        hair_groups = {
            "Basic Info": ["hair_type_texture", "hair_scalp_concerns", "hair_goals", "hair_structure"],
            "Products & Routine": ["current_products", "hair_care_routine", "existing_inventory"],
            "Hair Characteristics": ["scalp_condition", "hair_porosity", "hair_length", "color_history"],
            "Habits & Lifestyle": ["styling_habits", "routine_preference", "time_availability", 
                                 "maintenance_commitment", "lifestyle_factors"],
            "Health & Preferences": ["sensitivities_allergies", "past_experiences", 
                                   "budget_constraints", "eco_preferences", "pregnancy_status"],
            "Treatments": ["professional_treatments"]
        }
        
        for group_name, group_topics in hair_groups.items():
            completed = sum(1 for t in group_topics if t in st.session_state.extracted_info)
            total = len(group_topics)
            if total > 0:
                status = "✅" if completed == total else "🔄" if completed > 0 else "⏳"
                st.caption(f"{status} {group_name}: {completed}/{total}")

# --------------------------------------------------
# CHAT LOGIC
# --------------------------------------------------
if not st.session_state.conversation_complete:
    user_input = st.chat_input("Your answer...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        if st.session_state.current_topic:
            cleaned = clean_response(user_input)
            categorized = categorize_response(st.session_state.current_topic, cleaned)
            
            if categorized and categorized != cleaned:
                final_value = categorized
            else:
                final_value = cleaned
            
            st.session_state.extracted_info[st.session_state.current_topic] = {
                "raw": user_input,
                "cleaned": cleaned,
                "categorized": categorized,
                "value": final_value,
                "timestamp": datetime.now().isoformat()
            }
            
            if st.session_state.current_topic in DEMOGRAPHIC_TOPICS:
                st.session_state.user_demographics[st.session_state.current_topic] = final_value
        
        st.session_state.question_count += 1
        
        should_end = (
            st.session_state.question_count >= MAX_QUESTIONS or
            len(st.session_state.extracted_info) >= len(ALL_TOPICS)
        )
        
        if should_end:
            st.session_state.profile_data = generate_enhanced_profile()
            st.session_state.conversation_complete = True
            
            profile_summary = st.session_state.profile_data.get("profile_summary", "")
            ready_for_recs = st.session_state.profile_data.get("recommendation_ready", False)
            
            complete_msg = f"""✅ **Profile Complete!**

Thank you for completing {st.session_state.question_count} questions!

**Profile Summary:**
{profile_summary}

{"✅ Ready for personalized recommendations!" if ready_for_recs else "📝 Additional details would help optimize recommendations."}

Your enhanced profile is ready below."""
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": complete_msg
            })
            
            st.rerun()
        else:
            next_question, next_topic = generate_contextual_question()
            
            if next_question:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": next_question
                })
                st.session_state.current_topic = next_topic
                
                with st.chat_message("assistant"):
                    st.markdown(next_question)
                
                st.rerun()
            else:
                st.session_state.profile_data = generate_enhanced_profile()
                st.session_state.conversation_complete = True
                st.rerun()

else:
    st.success("🎉 **Enhanced Profile Successfully Completed!**")
    
    stats = st.session_state.profile_data.get("profile_statistics", {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Questions", stats.get("total_questions", 0))
    with col2:
        st.metric("Useful Data", stats.get("meaningful_data_points", 0))
    with col3:
        st.metric("Completeness", f"{stats.get('completeness_score', 0)}%")
    with col4:
        demo_topics = stats.get("demographic_topics", 0)
        hair_topics = stats.get("hair_topics", 0)
        st.metric("Topics", f"{demo_topics}+{hair_topics}")
    
    special_notes = st.session_state.profile_data.get("special_notes", {})
    
    if special_notes.get("has_child_context"):
        st.info("👶 **Child profile** - Recommendations will be age-appropriate")
    
    if special_notes.get("has_pregnancy_info"):
        st.warning("🤰 **Pregnancy/breastfeeding noted** - Safety-focused recommendations")
    
    st.subheader("📋 Enhanced Hair & Demographic Profile")
    
    with st.expander("📊 Profile Summary", expanded=True):
        st.write(st.session_state.profile_data.get("profile_summary", ""))
        
        st.caption("**Key Information:**")
        
        demo_data = st.session_state.profile_data.get("demographic_profile", {})
        if demo_data:
            st.markdown("**Demographics:**")
            for topic, data in demo_data.items():
                value = data.get("final_value", "")
                if value and str(value).lower() not in ["not specified", "none"]:
                    display_name = topic.replace('_', ' ').title()
                    if isinstance(value, dict):
                        for k, v in value.items():
                            if v and v != "not specified":
                                st.caption(f"• {k.title()}: {v}")
                    else:
                        st.caption(f"• {display_name}: {value}")
        
        hair_data = st.session_state.profile_data.get("hair_profile", {})
        key_hair_topics = ["hair_type_texture", "hair_scalp_concerns", "hair_goals", 
                          "current_products", "hair_porosity", "routine_preference", "hair_structure"]
        
        st.markdown("**Hair Profile:**")
        for topic in key_hair_topics:
            if topic in hair_data:
                value = hair_data[topic].get("final_value", "")
                if value:
                    display_name = topic.replace('_', ' ').title()
                    st.caption(f"• {display_name}: {value}")
    
    with st.expander("📄 Complete Profile Data (JSON)"):
        st.json(st.session_state.profile_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "💾 Download Enhanced Profile",
            json.dumps(st.session_state.profile_data, indent=2),
            file_name=f"halisi_enhanced_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
    
    with col2:
        if st.button("🔄 Start New Profile", type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("Halisi Cosmetics • Enhanced Profiler • Context-aware analysis")