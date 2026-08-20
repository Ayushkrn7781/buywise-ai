"""
Streamlit app for batch review analysis.
Run: streamlit run batch_app.py
"""

import streamlit as st
import pandas as pd
import json
from batch_engine import batch_analyze

st.set_page_config(
    page_title="ReviewLens - Batch Analysis",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f5f7fa; }
.card {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
.sentiment-positive { color: #4CAF50; font-weight: bold; }
.sentiment-negative { color: #f44336; font-weight: bold; }
.sentiment-neutral { color: #FF9800; font-weight: bold; }
.bar-positive { background-color: #4CAF50; }
.bar-negative { background-color: #f44336; }
.bar-neutral { background-color: #FF9800; }
.progress-bar {
    height: 24px;
    border-radius: 12px;
    background-color: #e0e0e0;
    overflow: hidden;
    margin: 5px 0;
}
.progress-fill {
    height: 100%;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 12px;
    font-weight: bold;
}
.verdict-box {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin: 20px 0;
}
.verdict-green { background-color: #E8F5E9; color: #2E7D32; border: 2px solid #4CAF50; }
.verdict-yellow { background-color: #FFF8E1; color: #F57F17; border: 2px solid #FFC107; }
.verdict-orange { background-color: #FFF3E0; color: #E65100; border: 2px solid #FF9800; }
.verdict-red { background-color: #FFEBEE; color: #C62828; border: 2px solid #f44336; }
.warning-box {
    background-color: #FFF3E0;
    border-left: 4px solid #FF9800;
    padding: 12px 20px;
    margin: 8px 0;
    border-radius: 0 8px 8px 0;
}
.aspect-row {
    display: flex;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #f0f0f0;
}
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================

st.title("📊 ReviewLens - Batch Analysis")
st.markdown("Analyze multiple reviews for a product → Get aggregate insights")

st.markdown("---")

# ==================== INPUT SECTION ====================

tab1, tab2, tab3 = st.tabs(["📝 Paste Reviews", "📁 Upload CSV", "📋 Sample Data"])

with tab1:
    product_name = st.text_input("Product Name (optional - will auto-detect if empty):")
    reviews_text = st.text_area(
        "Paste reviews (one per line):",
        height=250,
        placeholder="""Great battery life and sound quality
Terrible microphone, can't use for calls
Best headphones I've ever owned
Build quality is cheap, plastic feels flimsy
..."""
    )
    
    if reviews_text:
        reviews = [r.strip() for r in reviews_text.split("\n") if r.strip()]
        st.info(f"Detected {len(reviews)} reviews")

with tab2:
    uploaded = st.file_uploader("Upload CSV with 'Review Text' column", type=['csv'])
    product_name_csv = st.text_input("Product Name (optional):", key="csv_product")
    
    if uploaded:
        try:
            df = pd.read_csv(uploaded, engine='python', on_bad_lines='skip')
            if 'Review Text' in df.columns:
                reviews = df['Review Text'].dropna().astype(str).tolist()
                reviews = [r for r in reviews if len(r.strip()) > 10]
                st.success(f"Loaded {len(reviews)} reviews from CSV")
            else:
                st.error(f"Column 'Review Text' not found. Available: {list(df.columns)}")
                reviews = []
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            reviews = []
    else:
        reviews = []

with tab3:
    st.markdown("Use sample headphone reviews for testing:")
    if st.button("Load 10 Sample Reviews", use_container_width=True):
        reviews = [
            "Best headphones I've ever owned. Noise cancellation is incredible and battery lasts forever.",
            "Good sound quality but the microphone is terrible. People can't hear me on calls.",
            "Battery life is amazing - easily 30 hours. Sound is clear with great bass.",
            "Build quality feels cheap for the price. Plastic creaks when I adjust them.",
            "Comfortable for long sessions but my ears get sweaty after 2 hours.",
            "Bluetooth keeps disconnecting from my phone. Very frustrating.",
            "The app is garbage - crashes constantly and settings don't save.",
            "Worth every penny. ANC blocks out everything on my commute.",
            "Microphone quality is unacceptable for a premium product.",
            "Sound is fantastic, battery is great, but the build quality and mic are dealbreakers."
        ]
        st.session_state["batch_reviews"] = reviews
        st.rerun()

# Use session state for sample reviews
if 'batch_reviews' in st.session_state and not reviews:
    reviews = st.session_state["batch_reviews"]

# ==================== ANALYZE BUTTON ====================

if st.button("🔍 Analyze All Reviews", type="primary", use_container_width=True):
    if not reviews or len(reviews) == 0:
        st.warning("Please enter or upload reviews first.")
    elif len(reviews) < 2:
        st.warning("Please enter at least 2 reviews for batch analysis.")
    else:
        with st.spinner(f"Analyzing {len(reviews)} reviews..."):
            result = batch_analyze(reviews, product_name if 'product_name' in dir() else None)
        
        if "error" in result:
            st.error(result["error"])
        else:
            # ==================== PRODUCT INFO ====================
            st.subheader("📦 Product Identified")
            
            prod = result["product"]
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown(f'<div class="card">', unsafe_allow_html=True)
                st.markdown(f"**Product:** {prod['name']}")
                st.markdown(f"**Detection:** {prod['detection_method']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with c2:
                st.markdown(f'<div class="card">', unsafe_allow_html=True)
                st.markdown(f"**Category:** {prod['category']}")
                if prod.get('category_score'):
                    st.markdown(f"**Category Score:** {prod['category_score']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with c3:
                st.markdown(f'<div class="card">', unsafe_allow_html=True)
                if prod.get('brands'):
                    st.markdown(f"**Brands Found:** {', '.join([b[0] for b in prod['brands'][:3]])}")
                if prod.get('models'):
                    st.markdown(f"**Models Found:** {', '.join([m[0] for m in prod['models'][:3]])}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ==================== VERDICT ====================
            verdict = result["verdict"]
            verdict_class = {
                "Strongly Recommended": "verdict-green",
                "Recommended with Reservations": "verdict-yellow",
                "Mixed Reviews": "verdict-orange",
                "Not Recommended": "verdict-red",
                "Avoid": "verdict-red"
            }.get(verdict["text"], "verdict-orange")
            
            st.markdown(f'<div class="verdict-box {verdict_class}">{verdict["emoji"]} {verdict["text"]}</div>', unsafe_allow_html=True)
            
            # ==================== SUMMARY STATS ====================
            st.subheader("📈 Analysis Summary")
            summary = result["analysis_summary"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Reviews", summary["total_reviews"])
            c2.metric("Skipped", summary["skipped_reviews"])
            c3.metric("Positive", f"{verdict['positive_pct']}%")
            c4.metric("Negative", f"{verdict['negative_pct']}%")
            
            # ==================== SENTIMENT DISTRIBUTION ====================
            st.subheader("📊 Sentiment Distribution")
            
            c1, c2 = st.columns(2)
            
            for col, title, data in [
                (c1, "VADER (Rule-based)", result["sentiment_distribution"]["vader"]),
                (c2, "Custom Model (Fine-tuned)", result["sentiment_distribution"]["custom"])
            ]:
                with col:
                    st.markdown(f'<div class="card">', unsafe_allow_html=True)
                    st.markdown(f"**{title}**")
                    
                    total = data["positive"] + data["negative"] + data["neutral"]
                    
                    # Positive bar
                    st.markdown(f"✅ Positive: {data['positive']} ({data['positive_pct']}%)")
                    st.markdown(f'''<div class="progress-bar">
                        <div class="progress-fill bar-positive" style="width:{data['positive_pct']}%">{data['positive_pct']}%</div>
                    </div>''', unsafe_allow_html=True)
                    
                    # Neutral bar
                    st.markdown(f"⚠️ Neutral: {data['neutral']} ({data['neutral_pct']}%)")
                    st.markdown(f'''<div class="progress-bar">
                        <div class="progress-fill bar-neutral" style="width:{data['neutral_pct']}%">{data['neutral_pct']}%</div>
                    </div>''', unsafe_allow_html=True)
                    
                    # Negative bar
                    st.markdown(f"❌ Negative: {data['negative']} ({data['negative_pct']}%)")
                    st.markdown(f'''<div class="progress-bar">
                        <div class="progress-fill bar-negative" style="width:{data['negative_pct']}%">{data['negative_pct']}%</div>
                    </div>''', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # ==================== PROS & CONS ====================
            st.subheader("🎯 Key Insights")
            
            c1, c2 = st.columns(2)
            
            with c1:
                pros = result["pros"]
                pros_html = "<br>".join([f"✅ {p.title()}" for p in pros]) if pros else "None"
                st.markdown(f'<div class="card"><b>👍 Pros ({len(pros)})</b><br><br>{pros_html}</div>', unsafe_allow_html=True)
            
            with c2:
                cons = result["cons"]
                cons_html = "<br>".join([f"❌ {c.title()}" for c in cons]) if cons else "None"
                st.markdown(f'<div class="card"><b>👎 Cons ({len(cons)})</b><br><br>{cons_html}</div>', unsafe_allow_html=True)
            
            # ==================== WARNINGS ====================
            if result["warnings"]:
                st.subheader("⚠️ Warnings")
                for w in result["warnings"]:
                    st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)
            
            # ==================== ASPECT BREAKDOWN ====================
            st.subheader("🔍 Aspect Breakdown")
            
            for asp in result["aspects"]:
                color = "#4CAF50" if asp["overall"] == "positive" else "#f44336" if asp["overall"] == "negative" else "#FF9800"
                
                st.markdown(f'''
                <div class="card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <b>{asp["aspect"].title()}</b>
                            <span class="sentiment-{asp['overall']}"> ({asp["overall"].upper()})</span>
                        </div>
                        <div style="text-align: right; color: #666; font-size: 14px;">
                            Mentioned in {asp["mention_pct"]}% of reviews ({asp["mentions"]}/{result["analysis_summary"]["total_reviews"]})
                        </div>
                    </div>
                    
                    <div style="display: flex; gap: 20px; margin-top: 10px;">
                        <div style="flex: 1;">
                            <span style="color: #4CAF50;">✅ {asp["positive_pct"]}%</span>
                            <div class="progress-bar" style="height: 12px;">
                                <div class="progress-fill bar-positive" style="width:{asp["positive_pct"]}%; font-size: 10px;">{asp["positive"]}</div>
                            </div>
                        </div>
                        <div style="flex: 1;">
                            <span style="color: #FF9800;">⚠️ {asp["neutral_pct"]}%</span>
                            <div class="progress-bar" style="height: 12px;">
                                <div class="progress-fill bar-neutral" style="width:{asp["neutral_pct"]}%; font-size: 10px;">{asp["neutral"]}</div>
                            </div>
                        </div>
                        <div style="flex: 1;">
                            <span style="color: #f44336;">❌ {asp["negative_pct"]}%</span>
                            <div class="progress-bar" style="height: 12px;">
                                <div class="progress-fill bar-negative" style="width:{asp["negative_pct"]}%; font-size: 10px;">{asp["negative"]}</div>
                            </div>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            # ==================== RAW DATA ====================
            with st.expander("📎 Raw JSON Data"):
                # Remove individual results to keep JSON clean
                clean_result = {k: v for k, v in result.items() if k != "individual_results"}
                clean_result["individual_results_count"] = len(result.get("individual_results", []))
                st.json(clean_result)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
**How to use:**
1. Paste reviews (one per line) OR upload a CSV with "Review Text" column
2. Optionally enter product name (will auto-detect if empty)
3. Click "Analyze All Reviews"
4. View product identification, sentiment distribution, and aspect breakdown
""")
st.caption("ReviewLens Batch Analysis | Fine-tuned DistilBERT + VADER | Trained on Amazon Reviews")