"""
ReviewLens - Premium Dashboard (Brown Theme)
"""

import streamlit as st
import pandas as pd
import json
import os
from nlp_engine import generate_review_intelligence, USING_CUSTOM_MODEL
from batch_engine import batch_analyze
from product_identifier import identify_product


st.set_page_config(
    page_title="ReviewLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* ===== GLOBAL ===== */
.main {
    background: linear-gradient(160deg, #faf6f1 0%, #f0e8dd 100%);
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #3E2723 0%, #4E342E 50%, #5D4037 100%);
}
[data-testid="stSidebar"] * { color: #d7ccc8 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 15px; padding: 12px 16px; border-radius: 12px;
    margin: 4px 0; transition: all 0.3s ease;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.08); }

/* ===== HEADER ===== */
.main-header {
    background: linear-gradient(135deg, #8D6E63 0%, #6D4C41 50%, #5D4037 100%);
    padding: 40px; border-radius: 20px; margin-bottom: 30px;
    box-shadow: 0 10px 40px rgba(93, 64, 55, 0.25); color: white;
}
.main-header h1 { margin: 0; font-size: 2.5em; font-weight: 800; letter-spacing: -0.5px; }
.main-header p { margin: 8px 0 0 0; font-size: 1.1em; opacity: 0.9; font-weight: 300; }

/* ===== GLASS CARDS ===== */
.glass-card {
    background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(20px);
    border-radius: 20px; padding: 28px;
    box-shadow: 0 8px 32px rgba(93, 64, 55, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.8);
    margin-bottom: 20px; transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(93, 64, 55, 0.12); }

/* ===== SENTIMENT CARDS ===== */
.sentiment-card {
    border-radius: 20px; padding: 28px; text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    border: 2px solid transparent; transition: all 0.3s ease;
}
.sentiment-card:hover { transform: translateY(-4px); box-shadow: 0 16px 48px rgba(0, 0, 0, 0.15); }
.card-positive { background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); border-color: #81C784; }
.card-negative { background: linear-gradient(135deg, #fbe9e7 0%, #ffccbc 100%); border-color: #E57373; }
.card-neutral { background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%); border-color: #FFB74D; }

.text-positive { color: #2E7D32; font-weight: 700; }
.text-negative { color: #C62828; font-weight: 700; }
.text-neutral { color: #E65100; font-weight: 700; }

/* ===== PROGRESS BARS ===== */
.premium-bar {
    height: 10px; border-radius: 10px;
    background: linear-gradient(90deg, #e0e0e0 0%, #f5f5f5 100%);
    overflow: hidden; margin: 8px 0;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.08);
}
.premium-fill { height: 100%; border-radius: 10px; transition: width 1s ease; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
.fill-positive { background: linear-gradient(90deg, #81C784, #66BB6A); }
.fill-negative { background: linear-gradient(90deg, #E57373, #EF5350); }
.fill-neutral { background: linear-gradient(90deg, #FFB74D, #FFA726); }
.fill-blue { background: linear-gradient(90deg, #A1887F, #8D6E63); }

/* ===== VERDICT ===== */
.verdict-box {
    padding: 40px; border-radius: 24px; text-align: center;
    font-size: 1.8em; font-weight: 800; margin: 25px 0;
    letter-spacing: 0.5px; box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}
.verdict-green { background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%); color: #1B5E20; border: 2px solid #66BB6A; }
.verdict-yellow { background: linear-gradient(135deg, #ffe0b2 0%, #ffcc80 100%); color: #E65100; border: 2px solid #FFA726; }
.verdict-orange { background: linear-gradient(135deg, #ffccbc 0%, #ffab91 100%); color: #BF360C; border: 2px solid #FF8A65; }
.verdict-red { background: linear-gradient(135deg, #ffcdd2 0%, #ef9a9a 100%); color: #B71C1C; border: 2px solid #EF5350; }

/* ===== PROS CONS ===== */
.pros-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
    border-left: 5px solid #81C784; border-radius: 0 16px 16px 0; padding: 24px;
}
.cons-box {
    background: linear-gradient(135deg, #fbe9e7 0%, #fce4ec 100%);
    border-left: 5px solid #E57373; border-radius: 0 16px 16px 0; padding: 24px;
}

/* ===== WARNING ===== */
.warning-box {
    background: linear-gradient(135deg, #fff8e1 0%, #fff3e0 100%);
    border-left: 5px solid #FFB74D; padding: 16px 24px;
    margin: 10px 0; border-radius: 0 12px 12px 0; font-weight: 500;
}

/* ===== ASPECT CARD ===== */
.aspect-card {
    background: white; border-radius: 16px; padding: 24px;
    box-shadow: 0 4px 20px rgba(93, 64, 55, 0.06);
    border: 1px solid #efebe9; margin-bottom: 12px; transition: all 0.2s ease;
}
.aspect-card:hover { box-shadow: 0 8px 30px rgba(93, 64, 55, 0.1); border-color: #d7ccc8; }

/* ===== STAT BOX ===== */
.stat-box {
    background: white; border-radius: 16px; padding: 20px; text-align: center;
    box-shadow: 0 4px 20px rgba(93, 64, 55, 0.06); border: 1px solid #efebe9;
}
.stat-number {
    font-size: 2.2em; font-weight: 800;
    background: linear-gradient(135deg, #8D6E63 0%, #5D4037 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.stat-label { font-size: 0.85em; color: #8D6E63; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* ===== PRODUCT CARD ===== */
.product-card {
    background: linear-gradient(135deg, #3E2723 0%, #5D4037 100%);
    color: white; border-radius: 20px; padding: 30px;
    box-shadow: 0 10px 40px rgba(62, 39, 35, 0.3);
}

/* ===== INPUT ===== */
.stTextArea > div > div > textarea {
    border-radius: 16px !important; border: 2px solid #d7ccc8 !important;
    font-size: 15px !important; padding: 20px !important; transition: border-color 0.3s ease !important;
}
.stTextArea > div > div > textarea:focus {
    border-color: #8D6E63 !important; box-shadow: 0 0 0 4px rgba(141, 110, 99, 0.1) !important;
}

/* ===== BUTTONS ===== */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #8D6E63 0%, #5D4037 100%) !important;
    border: none !important; border-radius: 16px !important;
    padding: 16px 32px !important; font-size: 16px !important;
    font-weight: 600 !important; color: white !important;
    box-shadow: 0 8px 24px rgba(93, 64, 55, 0.3) !important;
    transition: all 0.3s ease !important; width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(93, 64, 55, 0.4) !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: white; padding: 8px;
    border-radius: 16px; box-shadow: 0 4px 20px rgba(93, 64, 55, 0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px !important; padding: 12px 28px !important;
    font-weight: 600 !important; font-size: 14px !important;
    color: #8D6E63 !important; background: transparent !important;
    transition: all 0.3s ease !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #8D6E63 0%, #5D4037 100%) !important;
    color: white !important; box-shadow: 0 4px 16px rgba(93, 64, 55, 0.3) !important;
}

/* ===== METRIC ===== */
[data-testid="stMetric"] {
    background: white; border-radius: 16px; padding: 20px !important;
    box-shadow: 0 4px 20px rgba(93, 64, 55, 0.06); border: 1px solid #efebe9;
}

/* ===== BADGE ===== */
.badge { display: inline-flex; align-items: center; gap: 8px; padding: 8px 18px; border-radius: 50px; font-size: 13px; font-weight: 600; }
.badge-success { background: linear-gradient(135deg, #c8e6c9, #a5d6a7); color: #1B5E20; }
.badge-warning { background: linear-gradient(135deg, #ffe0b2, #ffcc80); color: #E65100; }

/* ===== SECTION TITLE ===== */
.section-title { font-size: 1.4em; font-weight: 700; color: #3E2723; margin-bottom: 20px; display: flex; align-items: center; gap: 12px; }
.section-icon {
    width: 40px; height: 40px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center; font-size: 20px;
    background: linear-gradient(135deg, #8D6E63 0%, #5D4037 100%);
    box-shadow: 0 4px 12px rgba(93, 64, 55, 0.25);
}

.fancy-divider { height: 2px; background: linear-gradient(90deg, transparent, #8D6E63, transparent); margin: 30px 0; border: none; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ==================== SIDEBAR ====================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 3em;">🔍</div>
        <div style="font-size: 1.4em; font-weight: 800; color: white; margin-top: 8px;">ReviewLens</div>
        <div style="font-size: 0.85em; color: #a1887f; margin-top: 4px;">NLP Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    if USING_CUSTOM_MODEL:
        st.markdown('<div class="badge badge-success" style="margin: 10px 0;">✅ Custom Model Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge badge-warning" style="margin: 10px 0;">⚠️ Default Model</div>', unsafe_allow_html=True)

    if os.path.exists("./trained_sentiment_model/training_info.json"):
        with open("./trained_sentiment_model/training_info.json") as f:
            info = json.load(f)
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); border-radius: 16px; padding: 20px; margin-top: 20px;">
            <div style="font-weight: 700; margin-bottom: 12px; color: white;">📊 Model Stats</div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1: st.metric("Accuracy", f"{info.get('accuracy', 0):.2%}")
        with col2: st.metric("F1 Score", f"{info.get('f1', 0):.2%}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="position: fixed; bottom: 20px; left: 20px; right: 20px; text-align: center;">
        <div style="color: #a1887f; font-size: 12px;">Trained on 21K+ Amazon Reviews</div>
    </div>
    """, unsafe_allow_html=True)


# ==================== HEADER ====================

st.markdown("""
<div class="main-header">
    <h1>🔍 ReviewLens</h1>
    <p>NLP-Powered Review Intelligence • Aspect Extraction • Confidence Scoring</p>
</div>
""", unsafe_allow_html=True)

mode = st.radio("Navigation", ["📝 Single Review", "📊 Batch Analysis"], label_visibility="collapsed", horizontal=True)
st.markdown("<hr style='margin: 10px 0 25px 0; border: none;'>", unsafe_allow_html=True)

# ============================================================
# SINGLE REVIEW
# ============================================================

if "Single" in mode:

    SAMPLES = [
        "I've been using these headphones for a month. The sound quality is fantastic with deep bass. Battery life is impressive - about 20 hours. However, the microphone is mediocre and people complain they can't hear me on calls. Build quality feels premium but ear cushions wear out quickly.",
        "This laptop is a mixed bag. The display is stunning with vibrant colors. Performance is snappy. But battery life is terrible, barely 3-4 hours. Keyboard feels nice but trackpad is unresponsive sometimes.",
        "These shoes look great and are very comfortable. Material quality seems good. But after two weeks, the sole started separating. Grip excellent on dry surfaces but slips on wet ones.",
        "Absolutely love this product! Best purchase ever. Everything works perfectly and quality exceeded expectations.",
        "Worst product ever. Broke after one week. Customer service was unhelpful. Complete waste of money."
    ]

    with st.expander("💡 Quick Samples"):
        cols = st.columns(5)
        for i, s in enumerate(SAMPLES):
            with cols[i]:
                if st.button(f"Sample {i+1}", key=f"s_{i}", use_container_width=True):
                    st.session_state["single_review"] = s
                    st.rerun()

    st.markdown('<div class="section-title"><div class="section-icon">📝</div> Your Review</div>', unsafe_allow_html=True)

    review_text = st.text_area("", value=st.session_state.get("single_review", ""), height=150, placeholder="Paste a product review here...", label_visibility="collapsed")

    if st.button(" Analyze Review", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("Please enter a review first.")
        else:
            with st.spinner("Analyzing with AI..."):
                result = generate_review_intelligence(review_text)

            # st.markdown('<div class="section-title"><div class="section-icon">📄</div> Original Review</div>', unsafe_allow_html=True)
                        # Product Identification
            product_info = identify_product([review_text])
            prod = product_info
            
            st.markdown('<div class="section-title"><div class="section-icon">📦</div> Product Identified</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown(f'''<div class="product-card">
                    <div style="font-size: 0.8em; color: #a1887f; text-transform: uppercase; letter-spacing: 1px;">Product</div>
                    <div style="font-size: 1.3em; font-weight: 700; margin-top: 6px;">{prod['name'] or 'Unknown'}</div>
                    <div style="font-size: 0.85em; color: #a1887f; margin-top: 4px;">{prod['detection_method']}</div>
                </div>''', unsafe_allow_html=True)
            
            with c2:
                st.markdown(f'''<div class="product-card">
                    <div style="font-size: 0.8em; color: #a1887f; text-transform: uppercase; letter-spacing: 1px;">Category</div>
                    <div style="font-size: 1.3em; font-weight: 700; margin-top: 6px;">{prod['category']}</div>
                </div>''', unsafe_allow_html=True)
            
            with c3:
                brands = ', '.join([b[0] for b in prod.get('brands', [])[:3]]) or 'N/A'
                models = ', '.join([m[0] for m in prod.get('models', [])[:3]]) or 'N/A'
                st.markdown(f'''<div class="product-card">
                    <div style="font-size: 0.8em; color: #a1887f; text-transform: uppercase; letter-spacing: 1px;">Detected</div>
                    <div style="font-size: 0.95em; margin-top: 6px;">🏷️ {brands}</div>
                    <div style="font-size: 0.95em; margin-top: 4px;">📋 {models}</div>
                </div>''', unsafe_allow_html=True)
            st.markdown(f'<div class="glass-card" style="font-size: 15px; line-height: 1.7; color: #5D4037;">{review_text}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title"><div class="section-icon">📊</div> Sentiment Analysis</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)

            for col, title, data, icon in [
                (c1, "VADER", result["overall_sentiment"]["vader"], "⚡"),
                (c2, "Fine-tuned Model", result["overall_sentiment"]["custom_model"], "🧠"),
                (c3, "Combined", {"sentiment": result["overall_sentiment"]["combined"], "confidence": result["overall_sentiment"]["confidence"]}, "🎯")
            ]:
                with col:
                    sent = data["sentiment"]
                    conf = data["confidence"]
                    st.markdown(f'''
                    <div class="sentiment-card card-{sent}">
                        <div style="font-size: 2em; margin-bottom: 8px;">{icon}</div>
                        <div style="font-size: 0.85em; color: #8D6E63; font-weight: 500;">{title}</div>
                        <div class="text-{sent}" style="font-size: 1.5em; margin: 8px 0;">{sent.upper()}</div>
                        <div style="font-size: 0.85em; color: #8D6E63;">Confidence</div>
                        <div style="font-size: 1.3em; font-weight: 700; color: #3E2723;">{conf:.0%}</div>
                        <div class="premium-bar"><div class="premium-fill fill-{sent}" style="width:{conf*100}%"></div></div>
                    </div>
                    ''', unsafe_allow_html=True)

            st.markdown('<div class="section-title"><div class="section-icon">🎯</div> Key Insights</div>', unsafe_allow_html=True)
            p1, p2 = st.columns(2)

            with p1:
                pros = result["pros"]
                if pros:
                    h = "".join([f'<div style="padding: 10px 0; border-bottom: 1px solid #c8e6c9;"><span style="color: #66BB6A; font-weight: 700;">✅</span> <span style="font-weight: 500; color: #2E7D32;">{p.title()}</span></div>' for p in pros])
                    st.markdown(f'<div class="pros-box"><div style="font-weight: 700; color: #2E7D32; margin-bottom: 12px;">👍 Pros ({len(pros)})</div>{h}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="glass-card" style="text-align: center; color: #a1887f;">No positive aspects detected</div>', unsafe_allow_html=True)

            with p2:
                cons = result["cons"]
                if cons:
                    h = "".join([f'<div style="padding: 10px 0; border-bottom: 1px solid #ffcdd2;"><span style="color: #E57373; font-weight: 700;">❌</span> <span style="font-weight: 500; color: #C62828;">{c.title()}</span></div>' for c in cons])
                    st.markdown(f'<div class="cons-box"><div style="font-weight: 700; color: #C62828; margin-bottom: 12px;">👎 Cons ({len(cons)})</div>{h}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="glass-card" style="text-align: center; color: #a1887f;">No negative aspects detected</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title"><div class="section-icon">🔍</div> Aspect Analysis</div>', unsafe_allow_html=True)
            sorted_aspects = sorted(result["aspects"], key=lambda x: x["aspect_confidence"], reverse=True)
            a1, a2 = st.columns(2)

            for i, asp in enumerate(sorted_aspects[:8]):
                with a1 if i % 2 == 0 else a2:
                    sent = asp["sentiment"]
                    ac = asp["aspect_confidence"]
                    sc = asp["sentiment_confidence"]
                    bg = "#e8f5e9" if sent == "positive" else "#fbe9e7" if sent == "negative" else "#FFF8E1"
                    st.markdown(f'''
                    <div class="aspect-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="font-size: 1.1em; font-weight: 700; color: #3E2723;">{asp["aspect"].title()}</div>
                            <span class="text-{sent}" style="font-size: 0.85em; background: {bg}; padding: 4px 12px; border-radius: 20px;">{sent.upper()}</span>
                        </div>
                        <div style="margin-top: 16px;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: #a1887f; margin-bottom: 4px;">
                                <span>Aspect Detection</span><span style="font-weight: 600; color: #3E2723;">{ac:.0%}</span>
                            </div>
                            <div class="premium-bar"><div class="premium-fill fill-blue" style="width:{ac*100}%"></div></div>
                        </div>
                        <div style="margin-top: 12px;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: #a1887f; margin-bottom: 4px;">
                                <span>Sentiment Confidence</span><span style="font-weight: 600; color: #3E2723;">{sc:.0%}</span>
                            </div>
                            <div class="premium-bar"><div class="premium-fill fill-{sent}" style="width:{sc*100}%"></div></div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

            with st.expander("📎 Raw JSON Data"):
                st.json(result)


# ============================================================
# BATCH ANALYSIS
# ============================================================

else:

    reviews = []
    product_name = ""

    input_tab1, input_tab2, input_tab3 = st.tabs(["📝 Paste Reviews", "📁 Upload CSV", "📋 Samples"])

    with input_tab1:
        product_name = st.text_input("Product Name (optional):", placeholder="e.g., Sony WH-1000XM5", key="bn1")
        reviews_text = st.text_area("Paste reviews (one per line):", height=180, placeholder="Great battery life\nTerrible microphone\nBest headphones...", key="bt1")
        if reviews_text:
            reviews = [r.strip() for r in reviews_text.split("\n") if r.strip()]
            st.info(f"📊 Detected **{len(reviews)}** reviews")

    with input_tab2:
        uploaded = st.file_uploader("Upload CSV with 'Review Text' column", type=['csv'], key="bu")
        product_name = st.text_input("Product Name (optional):", placeholder="e.g., Sony WH-1000XM5", key="bn2")
        if uploaded:
            try:
                df = pd.read_csv(uploaded, engine='python', on_bad_lines='skip')
                if 'Review Text' in df.columns:
                    reviews = df['Review Text'].dropna().astype(str).tolist()
                    reviews = [r for r in reviews if len(r.strip()) > 10]
                    st.success(f"✅ Loaded **{len(reviews)}** reviews")
                else:
                    st.error(f"❌ 'Review Text' not found. Columns: `{list(df.columns)}`")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    with input_tab3:
        if st.button("🎧 Load 10 Headphone Samples", use_container_width=True):
            reviews = [
                "Best headphones ever. Noise cancellation incredible, battery lasts forever.",
                "Good sound but microphone is terrible. People can't hear me on calls.",
                "Battery life amazing - 30 hours. Sound clear with great bass.",
                "Build quality feels cheap. Plastic creaks when adjusting.",
                "Comfortable for long sessions but ears get sweaty after 2 hours.",
                "Bluetooth keeps disconnecting. Very frustrating.",
                "The app is garbage - crashes constantly.",
                "Worth every penny. ANC blocks everything on my commute.",
                "Microphone quality unacceptable for premium product.",
                "Sound fantastic, battery great, but build quality and mic are dealbreakers."
            ]
            st.session_state["batch_reviews"] = reviews
            st.rerun()

    if 'batch_reviews' in st.session_state and not reviews:
        reviews = st.session_state["batch_reviews"]

    if st.button("✨ Analyze All Reviews", type="primary", use_container_width=True):
        if not reviews: st.warning("Please enter reviews first.")
        elif len(reviews) < 2: st.warning("Need at least 2 reviews.")
        else:
            with st.spinner(f"Analyzing {len(reviews)} reviews..."):
                result = batch_analyze(reviews, product_name)

            if "error" in result: st.error(result["error"])
            else:

                st.markdown('<div class="section-title"><div class="section-icon">📦</div> Product Identified</div>', unsafe_allow_html=True)
                prod = result["product"]
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown(f'''<div class="product-card">
                        <div style="font-size: 0.8em; color: #a1887f; text-transform: uppercase; letter-spacing: 1px;">Product</div>
                        <div style="font-size: 1.3em; font-weight: 700; margin-top: 6px;">{prod['name'] or 'Unknown'}</div>
                        <div style="font-size: 0.85em; color: #a1887f; margin-top: 4px;">{prod['detection_method']}</div>
                    </div>''', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'''<div class="product-card">
                        <div style="font-size: 0.8em; color: #a1887f; text-transform: uppercase; letter-spacing: 1px;">Category</div>
                        <div style="font-size: 1.3em; font-weight: 700; margin-top: 6px;">{prod['category']}</div>
                    </div>''', unsafe_allow_html=True)
                with c3:
                    brands = ', '.join([b[0] for b in prod.get('brands', [])[:3]]) or 'N/A'
                    models = ', '.join([m[0] for m in prod.get('models', [])[:3]]) or 'N/A'
                    st.markdown(f'''<div class="product-card">
                        <div style="font-size: 0.8em; color: #a1887f; text-transform: uppercase; letter-spacing: 1px;">Detected</div>
                        <div style="font-size: 0.95em; margin-top: 6px;">🏷️ {brands}</div>
                        <div style="font-size: 0.95em; margin-top: 4px;">📋 {models}</div>
                    </div>''', unsafe_allow_html=True)

                verdict = result["verdict"]
                vclass = {"Strongly Recommended": "verdict-green", "Recommended with Reservations": "verdict-yellow", "Mixed Reviews": "verdict-orange", "Not Recommended": "verdict-red", "Avoid": "verdict-red"}.get(verdict["text"], "verdict-orange")
                st.markdown(f'<div class="verdict-box {vclass}">{verdict["emoji"]} {verdict["text"]}</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-title"><div class="section-icon">📈</div> Overview</div>', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.markdown(f'<div class="stat-box"><div class="stat-number">{result["analysis_summary"]["total_reviews"]}</div><div class="stat-label">Reviews</div></div>', unsafe_allow_html=True)
                with c2: st.markdown(f'<div class="stat-box"><div class="stat-number" style="-webkit-text-fill-color: #2E7D32;">{verdict["positive_pct"]:.0f}%</div><div class="stat-label">Positive</div></div>', unsafe_allow_html=True)
                with c3: st.markdown(f'<div class="stat-box"><div class="stat-number" style="-webkit-text-fill-color: #E65100;">{result["sentiment_distribution"]["custom"]["neutral_pct"]:.0f}%</div><div class="stat-label">Neutral</div></div>', unsafe_allow_html=True)
                with c4: st.markdown(f'<div class="stat-box"><div class="stat-number" style="-webkit-text-fill-color: #C62828;">{verdict["negative_pct"]:.0f}%</div><div class="stat-label">Negative</div></div>', unsafe_allow_html=True)

                st.markdown('<div class="section-title"><div class="section-icon">📊</div> Sentiment Distribution</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)

                for col, title, data in [(c1, "⚡ VADER", result["sentiment_distribution"]["vader"]), (c2, "🧠 Fine-tuned Model", result["sentiment_distribution"]["custom"])]:
                    with col:
                        st.markdown(f'''
                        <div class="glass-card">
                            <div style="font-weight: 700; font-size: 1.1em; margin-bottom: 20px;">{title}</div>
                            <div style="margin-bottom: 16px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                                    <span style="color: #2E7D32; font-weight: 600;">✅ Positive</span>
                                    <span style="font-weight: 700; color: #3E2723;">{data['positive']} ({data['positive_pct']}%)</span>
                                </div>
                                <div class="premium-bar"><div class="premium-fill fill-positive" style="width:{data['positive_pct']}%"></div></div>
                            </div>
                            <div style="margin-bottom: 16px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                                    <span style="color: #E65100; font-weight: 600;">⚠️ Neutral</span>
                                    <span style="font-weight: 700; color: #3E2723;">{data['neutral']} ({data['neutral_pct']}%)</span>
                                </div>
                                <div class="premium-bar"><div class="premium-fill fill-neutral" style="width:{data['neutral_pct']}%"></div></div>
                            </div>
                            <div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                                    <span style="color: #C62828; font-weight: 600;">❌ Negative</span>
                                    <span style="font-weight: 700; color: #3E2723;">{data['negative']} ({data['negative_pct']}%)</span>
                                </div>
                                <div class="premium-bar"><div class="premium-fill fill-negative" style="width:{data['negative_pct']}%"></div></div>
                            </div>
                        </div>''', unsafe_allow_html=True)

                st.markdown('<div class="section-title"><div class="section-icon">🎯</div> Key Insights</div>', unsafe_allow_html=True)
                p1, p2 = st.columns(2)
                with p1:
                    pros = result["pros"]
                    if pros:
                        h = "".join([f'<div style="padding: 8px 0; border-bottom: 1px solid #c8e6c9;"><span style="color: #66BB6A; font-weight: 700;">✅</span> <span style="font-weight: 500; color: #2E7D32;">{p.title()}</span></div>' for p in pros])
                        st.markdown(f'<div class="pros-box"><div style="font-weight: 700; color: #2E7D32; margin-bottom: 12px;">👍 Pros ({len(pros)})</div>{h}</div>', unsafe_allow_html=True)
                with p2:
                    cons = result["cons"]
                    if cons:
                        h = "".join([f'<div style="padding: 8px 0; border-bottom: 1px solid #ffcdd2;"><span style="color: #E57373; font-weight: 700;">❌</span> <span style="font-weight: 500; color: #C62828;">{c.title()}</span></div>' for c in cons])
                        st.markdown(f'<div class="cons-box"><div style="font-weight: 700; color: #C62828; margin-bottom: 12px;">👎 Cons ({len(cons)})</div>{h}</div>', unsafe_allow_html=True)

                if result["warnings"]:
                    st.markdown('<div class="section-title"><div class="section-icon">⚠️</div> Warnings</div>', unsafe_allow_html=True)
                    for w in result["warnings"]:
                        st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-title"><div class="section-icon">🔍</div> Aspect Breakdown</div>', unsafe_allow_html=True)
                total_reviews = result["analysis_summary"]["total_reviews"]

                for asp in result["aspects"]:
                    sent = asp["overall"]
                    bg = "#e8f5e9" if sent == "positive" else "#fbe9e7" if sent == "negative" else "#FFF8E1"
                    tc = "#2E7D32" if sent == "positive" else "#C62828" if sent == "negative" else "#E65100"
                    st.markdown(f'''
                    <div class="aspect-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                            <div style="font-size: 1.15em; font-weight: 700; color: #3E2723;">{asp["aspect"].title()}</div>
                            <div style="display: flex; align-items: center; gap: 12px;">
                                <span style="color: #a1887f; font-size: 0.85em;">{asp["mentions"]}/{total_reviews} ({asp["mention_pct"]}%)</span>
                                <span style="background: {bg}; color: {tc}; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85em;">{sent.upper()}</span>
                            </div>
                        </div>
                        <div style="display: flex; gap: 16px;">
                            <div style="flex: 1;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: #8D6E63; margin-bottom: 4px;">
                                    <span style="color: #2E7D32; font-weight: 600;">✅ {asp["positive_pct"]}%</span><span>{asp["positive"]}</span>
                                </div>
                                <div class="premium-bar" style="height: 8px;"><div class="premium-fill fill-positive" style="width:{asp["positive_pct"]}%"></div></div>
                            </div>
                            <div style="flex: 1;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: #8D6E63; margin-bottom: 4px;">
                                    <span style="color: #E65100; font-weight: 600;">⚠️ {asp["neutral_pct"]}%</span><span>{asp["neutral"]}</span>
                                </div>
                                <div class="premium-bar" style="height: 8px;"><div class="premium-fill fill-neutral" style="width:{asp["neutral_pct"]}%"></div></div>
                            </div>
                            <div style="flex: 1;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: #8D6E63; margin-bottom: 4px;">
                                    <span style="color: #C62828; font-weight: 600;">❌ {asp["negative_pct"]}%</span><span>{asp["negative"]}</span>
                                </div>
                                <div class="premium-bar" style="height: 8px;"><div class="premium-fill fill-negative" style="width:{asp["negative_pct"]}%"></div></div>
                            </div>
                        </div>
                    </div>''', unsafe_allow_html=True)

                with st.expander("📎 Raw JSON Data"):
                    clean_result = {k: v for k, v in result.items() if k != "individual_results"}
                    clean_result["individual_results_count"] = len(result.get("individual_results", []))
                    st.json(clean_result)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #a1887f; font-size: 13px;">ReviewLens • Fine-tuned DistilBERT + VADER + BART • Trained on 21,000+ Amazon Reviews</div>', unsafe_allow_html=True)