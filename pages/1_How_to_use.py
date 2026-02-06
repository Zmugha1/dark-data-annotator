"""
How to use the Dark Data Annotator — feature guide
"""

import streamlit as st

st.set_page_config(
    page_title="How to use | Dark Data Annotator",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Match main app light theme
st.markdown("""
<style>
    .stApp { background-color: #f1f5f9; }
    h1, h2, h3 { color: #0f172a !important; }
    p, li { color: #334155; }
    .feature-block {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
    }
    .feature-block h3 { color: #0d9488 !important; margin-top: 0; }
    code { background: #e2e8f0; color: #0d9488; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

if st.button("← Back to main app", type="primary", use_container_width=True):
    st.switch_page("app.py")

st.title("📖 How to use this app")
st.markdown("""
**Dark Data Annotator** is a step-by-step pipeline that turns raw documents into ML-ready data.  
Use this guide to understand each feature and how to run the pipeline.
""")

st.markdown("---")

# Overview
st.header("🎯 Overview")
st.markdown("""
The app walks you through **6 steps** in order. You click a button after each step to move to the next.  
You can go **Back** at any time to re-run a step with different settings. At the end you get **ML-ready JSON** and a **metrics comparison**.
""")

st.markdown("---")

# Sidebar features
st.header("⚙️ Sidebar: Pipeline configuration")
st.markdown("These settings apply when you **generate data** and run the pipeline.")

st.markdown("""
<div class="feature-block">
<h3>📊 Data Type</h3>
<p><strong>What it does:</strong> Chooses the kind of synthetic data to generate.</p>
<ul>
<li><strong>Customer Support Tickets</strong> — Fake tickets with customer PII (name, email, phone), issue category, priority, description, and agent metadata.</li>
<li><strong>Product Reviews</strong> — Fake reviews with product name, rating, sentiment, and customer identifier.</li>
</ul>
<p><strong>How to use:</strong> Pick one before clicking "Generate Sample Data". You can only change it before starting a new pipeline (step 0).</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-block">
<h3>📏 Number of Records</h3>
<p><strong>What it does:</strong> Sets how many synthetic documents (tickets or reviews) are generated.</p>
<p><strong>How to use:</strong> Use the slider (3–20). More records = more chunks and a longer pipeline. Start with 5 to see the flow quickly.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-block">
<h3>🔧 Processing options (toggles)</h3>
<ul>
<li><strong>Enable Anonymization</strong> — When ON, step 3 hashes/redacts PII (names, emails, phones, agent names). When OFF, you can skip anonymization and go straight to chunking.</li>
<li><strong>Enable Chunking</strong> — When ON, step 4 splits documents into small segments (e.g. by sentence or word groups). When OFF, each document is treated as one chunk.</li>
<li><strong>Enable Flattening (RAG Optimization)</strong> — When ON, step 5 turns JSON into natural-language text before embedding (better for ML). When OFF, raw JSON is used (worse metrics but still runs).</li>
</ul>
<p><strong>How to use:</strong> Leave all ON for the full, recommended pipeline. Turn OFF to compare behavior or skip a step.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Pipeline steps
st.header("📋 Pipeline steps (what each one does)")

st.markdown("""
<div class="feature-block">
<h3>Step 1 — Data Intake</h3>
<p><strong>What happens:</strong> You click <strong>Generate Sample Data</strong>. The app creates fake support tickets or product reviews with realistic PII and text.</p>
<p><strong>How to use:</strong> Choose Data Type and Number of Records in the sidebar, then click the button. You’ll see a preview of the raw JSON. Click <strong>Cleanse Data</strong> to continue.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-block">
<h3>Step 2 — Data Cleansing</h3>
<p><strong>What happens:</strong> Text is normalized (extra spaces removed, punctuation standardized), timestamps converted to ISO format, and categories (e.g. billing, technical) mapped to a fixed set of labels.</p>
<p><strong>How to use:</strong> Click <strong>Cleanse Data</strong>. Expand "View Output" to see cleaned records. Then click <strong>Anonymize Data</strong> (or <strong>Skip to Chunking</strong> if anonymization is off).</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-block">
<h3>Step 3 — Anonymization</h3>
<p><strong>What happens:</strong> PII is removed or replaced: names → [REDACTED], emails → anonymized IDs, phones → [REDACTED], agent names → agent IDs. Account IDs become short hashes.</p>
<p><strong>How to use:</strong> Click <strong>Anonymize Data</strong>. Check the output to confirm no real PII remains. Then click <strong>Chunk Documents</strong> (or <strong>Skip to Flattening</strong> if chunking is off).</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-block">
<h3>Step 4 — Semantic Chunking</h3>
<p><strong>What happens:</strong> Each document is split into smaller pieces (e.g. 2–3 words per chunk for tickets, or one sentence per chunk for reviews). Each chunk keeps a link to its parent and metadata (category, sentiment, etc.).</p>
<p><strong>How to use:</strong> Click <strong>Chunk Documents</strong>. You’ll see many chunk objects. Then click <strong>Flatten for Embedding</strong> (or <strong>Continue with Raw JSON</strong> if flattening is off).</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-block">
<h3>Step 5 — Flattening</h3>
<p><strong>What happens:</strong> Each chunk’s structured JSON is turned into a single natural-language string (e.g. "Support ticket TKT-10001 chunk 1 of 3. Category: billing. Priority: high. Content: ..."). This is the <strong>key optimization</strong> for embedding models.</p>
<p><strong>How to use:</strong> Click <strong>Flatten for Embedding</strong>. Compare "Raw JSON (Before)" and "Flattened (After)" to see the difference. Then click <strong>Generate Embeddings</strong>.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-block">
<h3>Step 6 — Embedding generation</h3>
<p><strong>What happens:</strong> Each flattened text is turned into a 384-dimensional vector (simulated in this app). The result is ML-ready: chunk_id, text, embedding, and metadata.</p>
<p><strong>How to use:</strong> Click <strong>Generate Embeddings</strong>. You’ll see the final JSON, a <strong>Download ML-Ready Dataset (JSON)</strong> button, and a <strong>metrics comparison</strong> (with vs without flattening). Use <strong>Start New Pipeline</strong> to run again from step 0.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Metrics and download
st.header("📈 Metrics and download")
st.markdown("""
<div class="feature-block">
<h3>ML Performance comparison</h3>
<p>At the end of the pipeline you see two panels:</p>
<ul>
<li><strong>WITHOUT Flattening (Raw JSON)</strong> — Simulated accuracy, precision, recall, F1, retrieval precision@10, MRR when embedding raw JSON.</li>
<li><strong>WITH Flattening (Optimized)</strong> — Same metrics when embedding the flattened natural-language text. You should see improvements (e.g. +20% precision, +19% recall, +15% token efficiency).</li>
</ul>
<p><strong>How to use:</strong> Compare the numbers to see why flattening is recommended. The "Token Efficiency" section explains token and cost savings.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-block">
<h3>Download ML-Ready Dataset (JSON)</h3>
<p><strong>What it does:</strong> Downloads a JSON file containing all chunk IDs, flattened text, embedding vectors (384 dims), and metadata.</p>
<p><strong>How to use:</strong> Click the button after step 6. Use the file for training, RAG, or vector search in your own tools.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Back and reset
st.header("⬅️ Back and Reset")
st.markdown("""
<ul>
<li><strong>Back</strong> — On steps 1–5, the <strong>Back</strong> button returns you one step and keeps your data. Change sidebar settings or re-run the current step after going back.</li>
<li><strong>Start New Pipeline</strong> — On step 6, this clears all data and returns you to step 0. Use it to run a new pipeline with different Data Type or options.</li>
</ul>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("Return to the **main app** using the button below or the sidebar.")
if st.button("← Back to main app", key="back_bottom", type="primary", use_container_width=True):
    st.switch_page("app.py")
