import streamlit as st
import pandas as pd
import json
import os

from utils import split_into_sentences, extract_aspects, compute_insights, ASPECT_META
from model import SentimentModel
from evaluation import evaluate_models

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Review Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS overrides for Amazon-inspired UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
    background-color: #0e1117;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #161b22;
}

/* Custom Text colors */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

header[data-testid="stHeader"] {
    background-color: transparent !important;
}
.block-container {
    padding-top: 2rem !important;
}

.az-title {
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0px;
    color: #ffffff !important;
}
.az-subtitle {
    font-size: 1rem;
    color: #8b949e;
    margin-bottom: 24px;
}

/* Amazon star rating */
label, label p, label span, details summary, details summary * {
    color: #e5e7eb !important;
}

.feature-rating span {
    color: #e5e7eb !important;
}

.az-star-rating {
    font-size: 3rem;
    font-weight: 700;
    color: #f59e0b;
    margin: 0px;
    line-height: 1.1;
}
.az-star-text {
    font-size: 1.1rem;
    color: #e5e7eb;
    margin-bottom: 20px;
}
.star-icon {
    color: #f59e0b;
    font-size: 1.4rem;
}

/* Review horizontal bars */
.rating-bar-row {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
    font-size: 0.95rem;
}
.rating-bar-label {
    width: 80px;
    color: #4fb2e2;
    cursor: pointer;
}
.rating-bar-label:hover {
    color: #e56012;
    text-decoration: underline;
}
.rating-bar-bg {
    flex-grow: 1;
    height: 18px;
    background: #21262d;
    border-radius: 4px;
    margin: 0 12px;
    border: 1px solid #30363d;
    overflow: hidden;
}
.rating-bar-fill {
    height: 100%;
    background: #f59e0b;
    border-radius: 3px;
}
.rating-bar-pct {
    width: 40px;
    text-align: right;
    color: #8b949e;
}

/* By Feature ratings */
.feature-rating {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    font-size: 1rem;
}
.feature-stars {
    color: #f59e0b;
    letter-spacing: 2px;
}

/* Pills for mentions */
.mentions-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 24px;
}
.mention-pill {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 0.9rem;
    color: #e5e7eb;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    cursor: default;
    transition: background 0.1s;
}
.mention-pill:hover {
    background: #374151;
}

/* Review Cards */
.review-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.review-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.review-badge-pos {
    color: #10b981;
    font-weight: 700;
}
.review-badge-neg {
    color: #ef4444;
    font-weight: 700;
}

.custom-divider {
    border-top: 1px solid #30363d;
    margin: 30px 0;
}

/* Hide default streamlit metrics background */
div[data-testid="stMetric"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Persisting
# ─────────────────────────────────────────────────────────────────────────────
REVIEWS_PATH = os.path.join("data", "product_reviews.json")

def load_data():
    if not os.path.exists(REVIEWS_PATH):
        return {"_meta": {}}, {}
    with open(REVIEWS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        meta = data.pop("_meta", {})
        return meta, data

def save_data(meta: dict, reviews_dict: dict):
    out = {"_meta": meta, **reviews_dict}
    with open(REVIEWS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

if "reviews_data" not in st.session_state:
    m, r = load_data()
    st.session_state.product_meta = m
    st.session_state.reviews_data = r

def get_product_label(name):
    meta = st.session_state.product_meta.get(name, {})
    emoji = meta.get("emoji", "📦")
    cat = f" {meta.get('category')}" if meta.get("category") else ""
    return f"{emoji} {name} {cat}"

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Dashboard Controls")
    
    product_list = list(st.session_state.reviews_data.keys())
    if not product_list:
        st.warning("No products found! Add one below.")
        selected_product_name = None
    else:
        product_options = {get_product_label(k): k for k in product_list}
        labels = list(product_options.keys())
        
        default_index = 0
        if st.session_state.get("target_product") in product_list:
            target_val = st.session_state["target_product"]
            for i, lbl in enumerate(labels):
                if product_options[lbl] == target_val:
                    default_index = i
                    break
        
        selected_label = st.selectbox("📦 Select Product", labels, index=default_index)
        selected_product_name = product_options[selected_label]
        st.session_state["target_product"] = selected_product_name

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    st.markdown("### 🤖 Model Settings")
    model_choice = st.selectbox("Classifier", ["Logistic Regression", "Naive Bayes"])
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05, 
                            help="Predictions below this confidence trigger the rule-based fallback.")
    
    @st.cache_resource
    def get_model():
        m = SentimentModel()
        m.load_models()
        return m
    model_mgr = get_model()
    model_mgr.confidence_threshold = conf_thresh

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    with st.expander("➕ Submit a Review"):
        action_type = st.radio("Choose Action", ["Add review to existing product", "Create a new product"])
        
        if action_type == "Add review to existing product":
            with st.form("add_review_existing_form", clear_on_submit=True):
                # Ensure we have products before allowing selection
                if product_list:
                    existing_prod = st.selectbox("Select Product", list(product_options.keys()))
                    review_text = st.text_area("Review Details")
                    if st.form_submit_button("Submit Review", use_container_width=True):
                        if review_text.strip():
                            target_name = product_options[existing_prod]
                            st.session_state.reviews_data[target_name].append(review_text)
                            save_data(st.session_state.product_meta, st.session_state.reviews_data)
                            st.session_state["target_product"] = target_name
                            st.rerun()
                else:
                    st.info("No products exist yet. Please create a new product first.")
                    st.form_submit_button("Submit Review", disabled=True, use_container_width=True)
                    
        else:
            with st.form("add_review_new_form", clear_on_submit=True):
                new_prod_name = st.text_input("New Product Name")
                new_prod_emoji = st.text_input("Emoji (e.g. 📸)", "📦")
                new_prod_review = st.text_area("Initial Review")
                if st.form_submit_button("Create Product", use_container_width=True):
                    if new_prod_name and new_prod_name not in st.session_state.reviews_data:
                        st.session_state.reviews_data[new_prod_name] = [new_prod_review] if new_prod_review.strip() else []
                        st.session_state.product_meta[new_prod_name] = {"emoji": new_prod_emoji, "category": "New"}
                        save_data(st.session_state.product_meta, st.session_state.reviews_data)
                        st.session_state["target_product"] = new_prod_name
                        st.rerun()

    if selected_product_name:
        with st.expander("🛠️ Manage Current Product"):
            if st.button("🗑️ Delete Product", use_container_width=True):
                st.session_state.reviews_data.pop(selected_product_name)
                st.session_state.product_meta.pop(selected_product_name, None)
                save_data(st.session_state.product_meta, st.session_state.reviews_data)
                st.session_state["target_product"] = None
                st.rerun()

if not selected_product_name:
    st.title("Welcome! Please add a product.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Engine (Cached per threshold & reviews state)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def analyze_reviews(reviews_list, model_name, threshold):
    """
    Cached analysis. 
    Notice: We strictly use standard standard Python dictionaries 
    instead of defaultdict(lambda) to guarantee it successfully pickles/caches!
    """
    aspect_sentiments = {}
    all_sentiments = []
    aspect_sentences = {}
    
    flat_sentences = []
    for r in reviews_list:
        flat_sentences.extend(split_into_sentences(r))
        
    if not flat_sentences:
        return aspect_sentiments, all_sentiments, aspect_sentences, {}
        
    predictions = model_mgr.predict_batch(flat_sentences, model_choice=model_name)
    
    for sent, (pred_label, proba_dict, fallback) in zip(flat_sentences, predictions):
        aspects = extract_aspects(sent)
        conf = max(proba_dict.values()) if proba_dict else 0.0
        
        all_sentiments.append({"text": sent, "label": pred_label, "conf": conf, "fallback": fallback})
        
        for asp in aspects:
            # Initialize explicitly to avoid pickle errors
            if asp not in aspect_sentiments:
                aspect_sentiments[asp] = {"Positive": 0, "Negative": 0, "Neutral": 0, "total": 0}
            if asp not in aspect_sentences:
                aspect_sentences[asp] = []
                
            aspect_sentiments[asp][pred_label] += 1
            aspect_sentiments[asp]["total"] += 1
            
            aspect_sentences[asp].append({
                "text": sent, 
                "label": pred_label, 
                "conf": conf,
                "fallback": fallback
            })
            
    # compute_insights uses the structure we just built safely
    insights = compute_insights(aspect_sentiments)
            
    return aspect_sentiments, all_sentiments, aspect_sentences, insights

with st.spinner("Analyzing sentiments..."):
    reviews = st.session_state.reviews_data[selected_product_name]
    asp_sent, all_sent_dicts, asp_sents, insights = analyze_reviews(tuple(reviews), model_choice, conf_thresh)
    
    # Extract just list of strings for overall sentiments logic
    all_sent = [x["label"] for x in all_sent_dicts]


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
product_meta = st.session_state.product_meta.get(selected_product_name, {})
st.markdown(f'<p class="az-title">Customer Reviews: {selected_product_name}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="az-subtitle">Analytics based on {len(all_sent)} customer sentences.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions for Fake Star Ratings
# ─────────────────────────────────────────────────────────────────────────────
def sentiment_to_stars_overall(sents_list):
    if not sents_list: return 0.0
    # Pos=5, Neu=3, Neg=1
    score = sum(5 if s == "Positive" else 3 if s == "Neutral" else 1 for s in sents_list)
    return round(score / len(sents_list), 1)

def rating_to_star_html(rating_val):
    full_stars = int(rating_val)
    half_stars = 1 if (rating_val - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_stars
    
    html = '<span class="feature-stars">'
    html += '★' * full_stars
    html += '½' * half_stars
    html += '☆' * empty_stars
    html += f' <span style="font-size:0.85rem;color:#8b949e">({rating_val:.1f})</span></span>'
    return html

# ─────────────────────────────────────────────────────────────────────────────
# Amazon-Style Layout: Top Section
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

# ------ LEFT COLUMN (Metrics) ------
with col_left:
    overall_rating = sentiment_to_stars_overall(all_sent)
    
    # Big star widget
    font_stars = '★' * int(overall_rating) + ('½' if overall_rating % 1 >= 0.5 else '')
    font_stars += '☆' * (5 - len(font_stars.replace('½', '')))
    
    st.markdown(f'''
        <div class="az-star-rating">{overall_rating} out of 5</div>
        <div class="az-star-text"><span class="star-icon">{font_stars}</span> Global Sentiment Rating</div>
    ''', unsafe_allow_html=True)
    
    # Sentiment distribution bars
    tot_sents = len(all_sent) or 1
    pos_pct = round(all_sent.count('Positive') / tot_sents * 100)
    neu_pct = round(all_sent.count('Neutral') / tot_sents * 100)
    neg_pct = round(all_sent.count('Negative') / tot_sents * 100)
    
    st.markdown(f'''
    <div class="rating-bar-row">
        <span class="rating-bar-label">Positive</span>
        <div class="rating-bar-bg"><div class="rating-bar-fill" style="width:{pos_pct}%; background:#10b981"></div></div>
        <span class="rating-bar-pct">{pos_pct}%</span>
    </div>
    <div class="rating-bar-row">
        <span class="rating-bar-label">Neutral</span>
        <div class="rating-bar-bg"><div class="rating-bar-fill" style="width:{neu_pct}%; background:#f59e0b"></div></div>
        <span class="rating-bar-pct">{neu_pct}%</span>
    </div>
    <div class="rating-bar-row">
        <span class="rating-bar-label">Critical</span>
        <div class="rating-bar-bg"><div class="rating-bar-fill" style="width:{neg_pct}%; background:#ef4444"></div></div>
        <span class="rating-bar-pct">{neg_pct}%</span>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # By feature
    st.markdown("<h4 style='font-size:1.1rem;margin-bottom:16px'>By feature</h4>", unsafe_allow_html=True)
    
    if asp_sent:
        for aspect, counts in sorted(asp_sent.items(), key=lambda x: x[1]['total'], reverse=True)[:5]:
            # Fake star out of 5 for this aspect
            # 100% positive = 5.0. 0% positive = 1.0 (assuming the rest are negative)
            # A simple math: 1 + 4 * (Positive / Total)
            total = counts["total"]
            asp_rate = 1 + 4 * (counts["Positive"] / total) if total else 0
            asp_rate = min(max(asp_rate, 1.0), 5.0)
            
            st.markdown(f'''
            <div class="feature-rating">
                <span>{aspect}</span>
                {rating_to_star_html(asp_rate)}
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.write("Not enough feature data.")

# ------ RIGHT COLUMN (Feed and Pills) ------
with col_right:
    
    if insights:
        st.markdown(f"<div style='margin-bottom:20px; font-style:italic; color:#e5e7eb'>💡 AI Summary: {insights['sentiment_narrative']}</div>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='font-size:1.0rem'>Read reviews that mention</h4>", unsafe_allow_html=True)
    
    if asp_sent:
        pills_html = '<div class="mentions-container">'
        for aspect, _ in sorted(asp_sent.items(), key=lambda x: x[1]['total'], reverse=True):
            pills_html += f'<div class="mention-pill">{aspect}</div>'
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)
    
    # Find Best Positive and Worst CRITICAL sentences
    top_pos_sentence = None
    top_neg_sentence = None
    
    # Sort all_sent_dicts by confidence descending
    sorted_sents = sorted(all_sent_dicts, key=lambda x: x["conf"], reverse=True)
    
    for s in sorted_sents:
        if s["label"] == "Positive" and not top_pos_sentence:
            top_pos_sentence = s
        elif s["label"] == "Negative" and not top_neg_sentence:
            top_neg_sentence = s
            
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top positive review")
        if top_pos_sentence:
            st.markdown(f'''
            <div class="review-card">
                <div class="review-header">
                    <span class="star-icon">★★★★★</span>
                    <span class="review-badge-pos">Positive</span>
                </div>
                <div style="font-size:1.05rem;font-weight:600;margin-bottom:6px">"Highly recommended"</div>
                <div style="color:#e5e7eb">{top_pos_sentence['text']}</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.info("No positive reviews found.")

    with c2:
        st.markdown("#### Top critical review")
        if top_neg_sentence:
            st.markdown(f'''
            <div class="review-card">
                <div class="review-header">
                    <span class="star-icon">★☆☆☆☆</span>
                    <span class="review-badge-neg">Critical</span>
                </div>
                <div style="font-size:1.05rem;font-weight:600;margin-bottom:6px">"Major flaws"</div>
                <div style="color:#e5e7eb">{top_neg_sentence['text']}</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.info("No critical reviews found.")


# ─────────────────────────────────────────────────────────────────────────────
# Standard Filtered Feed
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
st.markdown("### 🔎 All Mentions Filter")

f_col1, f_col2 = st.columns([1, 2])
with f_col1:
    aspect_filter = st.selectbox("Topic", ["All"] + list(asp_sents.keys()))
with f_col2:
    sentiment_filter = st.radio("Rating", ["All", "Positive", "Negative", "Neutral"], horizontal=True)

count_shown = 0
aspects_to_show = asp_sents.keys() if aspect_filter == "All" else [aspect_filter]

for aspect in aspects_to_show:
    sentences = asp_sents[aspect]
    
    if sentiment_filter != "All":
        sentences = [s for s in sentences if s["label"] == sentiment_filter]
        
    if not sentences: continue
    
    with st.expander(f"{ASPECT_META.get(aspect, {}).get('emoji','📌')} {aspect} ({len(sentences)} results)", expanded=True):
        for s in sentences:
            lab = s["label"]
            icon = "👍" if lab == "Positive" else "👎" if lab == "Negative" else "😐"
            color = "#10b981" if lab == "Positive" else "#ef4444" if lab == "Negative" else "#f59e0b"
            
            st.markdown(f'''
            <div class="review-card" style="padding: 10px 16px; margin-bottom: 8px;">
                <span style="color:{color}; font-weight:600">{icon} {lab}</span>
                <span style="color:#6b7280; font-size:0.85rem; margin-left:8px">(Conf: {s['conf']:.0%})</span><br>
                <div style="margin-top:6px; color:#d1d5db">{s['text']}</div>
            </div>
            ''', unsafe_allow_html=True)
        count_shown += 1

if count_shown == 0:
    st.info("No reviews match your filters.")
