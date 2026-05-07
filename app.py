import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import os
import shutil
import matplotlib.pyplot as plt
from zipfile import ZipFile
from collections import defaultdict
import random
import time

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="ForestSense AI | Tree Detection & Carbon MRV",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ForestSense AI — A Climate Initiative by Mayank, Tanmay & Yash"}
)

# ===== SPLASH SCREEN =====
if "splash_done" not in st.session_state:
    st.session_state.splash_done = False

if not st.session_state.splash_done:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    [data-testid="stSidebar"] { display: none; }
    .main .block-container { padding: 0 !important; max-width: 100% !important; }
    .splash-wrap {
        position: fixed; inset: 0;
        background: linear-gradient(135deg, #071a0e 0%, #0d2b1a 40%, #1a4a2e 100%);
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        z-index: 9999; min-height: 100vh;
    }
    .splash-icon { font-size: 5rem; margin-bottom: 1.2rem; animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(1.1);opacity:0.8} }
    .splash-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3.2rem; color: white; text-align: center;
        margin-bottom: 0.6rem; letter-spacing: -0.01em;
    }
    .splash-sub {
        font-size: 1rem; color: rgba(255,255,255,0.55);
        text-align: center; letter-spacing: 0.12em;
        text-transform: uppercase; margin-bottom: 3rem;
    }
    .splash-bar-wrap {
        width: 280px; height: 3px;
        background: rgba(255,255,255,0.1); border-radius: 10px; overflow: hidden;
    }
    .splash-bar {
        height: 100%; width: 0%;
        background: linear-gradient(90deg, #4caf50, #a5d6a7);
        border-radius: 10px;
        animation: loadbar 2.2s ease-in-out forwards;
    }
    @keyframes loadbar { 0%{width:0%} 60%{width:70%} 100%{width:100%} }
    .splash-tagline {
        margin-top: 2rem; font-size: 0.8rem;
        color: rgba(255,255,255,0.3); letter-spacing: 0.06em;
    }
    </style>
    <div class="splash-wrap">
        <div class="splash-icon">🌳</div>
        <div class="splash-title">ForestSense AI</div>
        <div class="splash-sub">Tree Detection · Carbon MRV · Climate Action</div>
        <div class="splash-bar-wrap"><div class="splash-bar"></div></div>
        <div class="splash-tagline">Initialising satellite intelligence...</div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(2.5)
    st.session_state.splash_done = True
    st.rerun()

# ===== GLOBAL CSS =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071a0e 0%, #0d2b1a 50%, #071a0e 100%);
    border-right: 1px solid #1a3d28;
}
[data-testid="stSidebar"] * { color: #e8f5e9 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 10px 16px !important; border-radius: 8px !important;
    transition: background 0.2s; cursor: pointer; display: block;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.08) !important; }

.main .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1100px; }

.hero-card {
    background: linear-gradient(135deg, #071a0e 0%, #1b5e35 55%, #2e7d52 100%);
    border-radius: 20px; padding: 60px 50px; color: white;
    position: relative; overflow: hidden; margin-bottom: 2rem;
}
.hero-card::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:260px; height:260px; background:rgba(255,255,255,0.04); border-radius:50%;
}
.hero-card::after {
    content:''; position:absolute; bottom:-80px; left:-40px;
    width:320px; height:320px; background:rgba(255,255,255,0.03); border-radius:50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif; font-size: 3rem;
    font-weight: 400; line-height: 1.15; margin-bottom: 1rem; color: white;
}
.hero-subtitle {
    font-size: 1.05rem; font-weight: 300; color: rgba(255,255,255,0.8);
    max-width: 600px; line-height: 1.7; margin-bottom: 2rem;
}
.hero-badge {
    display: inline-block; background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2); border-radius: 50px;
    padding: 6px 18px; font-size: 0.78rem; font-weight: 500;
    letter-spacing: 0.05em; text-transform: uppercase;
    color: #a5d6a7; margin-bottom: 1.2rem;
}
.stat-row { display: flex; gap: 16px; margin-bottom: 2rem; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 160px; background: white;
    border: 1px solid #e8f5e9; border-radius: 14px; padding: 22px 20px;
    box-shadow: 0 2px 12px rgba(46,125,82,0.08); text-align: center;
}
.stat-number { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #1b5e35; }
.stat-label { font-size: 0.75rem; color: #666; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 4px; }

.section-header { font-family: 'DM Serif Display', serif; font-size: 1.9rem; color: #071a0e; font-weight: 400; margin-bottom: 0.4rem; }
.section-subheader { font-size: 0.95rem; color: #666; margin-bottom: 1.8rem; }

.info-card {
    background: white; border-radius: 14px; padding: 28px 26px;
    border: 1px solid #e0ede6; box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    height: 100%; margin-bottom: 1rem;
}
.info-card-icon { font-size: 1.8rem; margin-bottom: 10px; }
.info-card-title { font-family: 'DM Serif Display', serif; font-size: 1.15rem; color: #071a0e; margin-bottom: 8px; }
.info-card-text { font-size: 0.88rem; color: #555; line-height: 1.7; }

.step-card {
    background: white; border-radius: 14px; padding: 24px 22px;
    border-left: 4px solid #2e7d52; box-shadow: 0 2px 10px rgba(0,0,0,0.04); margin-bottom: 1rem;
}
.step-number { font-family: 'DM Serif Display', serif; font-size: 2.2rem; color: #c8e6c9; line-height: 1; }
.step-title { font-size: 1rem; font-weight: 600; color: #1b5e35; margin-bottom: 6px; }
.step-text { font-size: 0.85rem; color: #555; line-height: 1.6; }

.tag-pill {
    display: inline-block; background: #e8f5e9; color: #1b5e35;
    border-radius: 50px; padding: 4px 14px; font-size: 0.78rem;
    font-weight: 500; margin: 3px; border: 1px solid #c8e6c9;
}

.timeline-item { display: flex; gap: 16px; margin-bottom: 1.2rem; align-items: flex-start; }
.timeline-dot { width:12px; height:12px; background:#2e7d52; border-radius:50%; margin-top:5px; flex-shrink:0; }
.timeline-content { font-size: 0.88rem; color: #444; line-height: 1.6; }
.timeline-year { font-weight: 600; color: #1b5e35; }

.result-highlight {
    background: linear-gradient(135deg, #e8f5e9, #f1f8f4);
    border-radius: 14px; padding: 24px; border: 1px solid #c8e6c9;
    text-align: center; margin-bottom: 1rem;
}
.result-big { font-family: 'DM Serif Display', serif; font-size: 2.5rem; color: #1b5e35; }
.result-label { font-size: 0.85rem; color: #555; font-weight: 500; margin-top: 4px; }

.green-divider {
    height: 3px; background: linear-gradient(90deg, #2e7d52, transparent);
    border: none; border-radius: 2px; margin: 2rem 0;
}

/* Before/After comparison */
.compare-wrap { display: flex; gap: 0; border-radius: 16px; overflow: hidden; margin-bottom: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
.compare-before {
    flex: 1; background: #fff5f5; padding: 28px 24px;
    border-right: 2px solid #f0f0f0;
}
.compare-after { flex: 1; background: #f0faf4; padding: 28px 24px; }
.compare-label-bad { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: #e53935; font-weight: 700; margin-bottom: 12px; }
.compare-label-good { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: #2e7d52; font-weight: 700; margin-bottom: 12px; }
.compare-title { font-family: 'DM Serif Display', serif; font-size: 1.2rem; color: #333; margin-bottom: 14px; }
.compare-row { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; font-size: 0.86rem; color: #444; line-height: 1.5; }

/* Initiative page */
.initiative-hero {
    background: linear-gradient(135deg, #071a0e, #0d3320, #1b5e35);
    border-radius: 20px; padding: 70px 50px; text-align: center;
    position: relative; overflow: hidden; margin-bottom: 2rem;
}
.initiative-hero::before {
    content: ''; position: absolute; top: -100px; left: 50%;
    transform: translateX(-50%);
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(76,175,80,0.12) 0%, transparent 70%);
}
.initiative-quote {
    font-family: 'DM Serif Display', serif; font-size: 2.4rem;
    color: white; line-height: 1.3; margin-bottom: 1.2rem;
    position: relative; z-index: 1;
}
.initiative-quote em { color: #a5d6a7; font-style: italic; }
.initiative-sub { font-size: 1rem; color: rgba(255,255,255,0.65); max-width: 600px; margin: 0 auto; line-height: 1.7; position: relative; z-index: 1; }

/* Context card */
.context-card {
    background: linear-gradient(135deg, #071a0e, #1b5e35);
    border-radius: 14px; padding: 28px; color: white; margin-top: 1.5rem;
}
.context-card-title { font-family: 'DM Serif Display', serif; font-size: 1.2rem; color: #a5d6a7; margin-bottom: 14px; }
.context-row { display: flex; align-items: center; gap: 14px; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08); }
.context-row:last-child { border-bottom: none; }
.context-icon { font-size: 1.4rem; flex-shrink: 0; }
.context-text { font-size: 0.88rem; color: rgba(255,255,255,0.8); line-height: 1.5; }
.context-text strong { color: #a5d6a7; }

/* Sidebar brand */
.sidebar-brand { text-align: center; padding: 20px 0 28px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px; }
.sidebar-brand-title { font-family: 'DM Serif Display', serif; font-size: 1.35rem; color: #a5d6a7 !important; }
.sidebar-brand-sub { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(255,255,255,0.35) !important; margin-top: 4px; }

[data-testid="stFileUploader"] { background: #f8fdf9; border: 2px dashed #a5d6a7; border-radius: 14px; padding: 10px; }
[data-testid="stMetricValue"] { font-family: 'DM Serif Display', serif !important; color: #1b5e35 !important; }

.stButton > button {
    background: linear-gradient(135deg, #1b5e35, #2e7d52) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 10px 28px !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 0.9rem !important;
    letter-spacing: 0.02em !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
</style>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size:2.2rem; margin-bottom:6px;">🌳</div>
        <div class="sidebar-brand-title">ForestSense AI</div>
        <div class="sidebar-brand-sub">Tree Detection & Carbon MRV</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "🏠  Home",
            "🌍  The Initiative",
            "📘  MRV & Carbon Market",
            "🔬  How It Works",
            "📊  Sample Output",
            "🌳  Launch Tool",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:14px 16px; background:rgba(255,255,255,0.05); border-radius:10px; font-size:0.78rem; color:rgba(255,255,255,0.45); line-height:1.7;">
        <strong style="color:#a5d6a7;">Tech Stack</strong><br>
        YOLOv8 · Streamlit · OpenCV<br>
        Pandas · Matplotlib · PIL
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:12px 16px; font-size:0.74rem; color:rgba(255,255,255,0.25); border-top:1px solid rgba(255,255,255,0.07); line-height:1.7;">
        A climate initiative by<br>
        <strong style="color:#a5d6a7;">Mayank Kumar Sharma</strong><br>
        B.Tech AI & Data Science · CTAE Udaipur
    </div>
    """, unsafe_allow_html=True)


# ===================================================================
# PAGE: HOME
# ===================================================================
if page == "🏠  Home":

    st.markdown("""
    <div class="hero-card">
        <div class="hero-badge">🛰️ Satellite AI · Carbon MRV · Forest Intelligence</div>
        <div class="hero-title">Detecting Trees.<br><em>Quantifying Carbon.</em></div>
        <div class="hero-subtitle">
            ForestSense AI uses deep learning on satellite imagery to detect individual trees,
            classify their size &amp; maturity, and estimate CO₂ sequestration — enabling
            scalable, digital MRV for nature-based carbon projects.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── FACT TICKER — HTML + JS in ONE block ──
    st.markdown("""
    <div id="fact-ticker-wrap" style="
        background: linear-gradient(90deg, #071a0e, #1b5e35, #071a0e);
        border-radius: 12px; padding: 14px 24px; margin-bottom: 1.6rem;
        display: flex; align-items: center; gap: 14px; overflow: hidden;
        border: 1px solid #2d5a3d;">
        <span style="
            background: #a5d6a7; color: #071a0e; font-size: 0.68rem;
            font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
            padding: 3px 10px; border-radius: 50px; white-space: nowrap;">
            🌿 DID YOU KNOW
        </span>
        <span id="ticker-text" style="
            font-size: 0.9rem; color: rgba(255,255,255,0.88);
            font-weight: 400; transition: opacity 0.6s ease;">
            🌳 The world has lost 46% of its trees since the dawn of human civilisation.
        </span>
    </div>
    <script>
    (function() {
        var facts = [
            "🌳 The world has lost 46% of its trees since the dawn of human civilisation.",
            "🛰️ Satellite-based dMRV can monitor forests 365 days a year at near-zero cost.",
            "💨 Forests absorb roughly 2.6 trillion tons of CO₂ every single year.",
            "🌍 Deforestation causes 23% of all global greenhouse gas emissions.",
            "💰 The voluntary carbon market is projected to reach $50 billion by 2030.",
            "🌱 Nature-based solutions can deliver 30% of climate targets needed by 2030.",
            "🤖 YOLOv8 can detect hundreds of trees in a satellite image in under 3 seconds.",
            "📉 Every minute, the world loses forest area equivalent to 40 football fields."
        ];
        var idx = 1;
        function showFact() {
            var el = document.getElementById('ticker-text');
            if (!el) return;
            el.style.opacity = 0;
            setTimeout(function() {
                el.textContent = facts[idx % facts.length];
                el.style.opacity = 1;
                idx++;
            }, 600);
        }
        setInterval(showFact, 4000);
    })();
    </script>
    """, unsafe_allow_html=True)

    # ── ANIMATED COUNTERS — HTML + JS in ONE block ──
    st.markdown("""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-number" id="cnt-trees">0B</div>
            <div class="stat-label">Trees lost per year globally</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" id="cnt-co2">0T</div>
            <div class="stat-label">Tons CO₂ absorbed by forests annually</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" id="cnt-market">$0B+</div>
            <div class="stat-label">Nature-based carbon market size</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" id="cnt-climate">0%</div>
            <div class="stat-label">Climate targets met via forests</div>
        </div>
    </div>

    <script>
    (function() {
        function animateCounter(id, start, end, duration, prefix, suffix, decimals) {
            var el = document.getElementById(id);
            if (!el) return;
            var startTime = null;
            function step(timestamp) {
                if (!startTime) startTime = timestamp;
                var progress = Math.min((timestamp - startTime) / duration, 1);
                var eased = 1 - Math.pow(1 - progress, 3);
                var current = start + (end - start) * eased;
                el.textContent = prefix + current.toFixed(decimals) + suffix;
                if (progress < 1) requestAnimationFrame(step);
            }
            requestAnimationFrame(step);
        }
        animateCounter('cnt-trees',   0, 15,  1800, '',  'B', 0);
        animateCounter('cnt-co2',     0, 2.6, 2000, '',  'T', 1);
        animateCounter('cnt-market',  0, 2,   1600, '$', 'B+', 0);
        animateCounter('cnt-climate', 0, 30,  1500, '',  '%', 0);
    })();
    </script>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # Problem
    st.markdown('<div class="section-header">The Problem</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Why traditional forest monitoring falls short</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="info-card">
            <div class="info-card-icon">🚶</div>
            <div class="info-card-title">Manual Field Surveys</div>
            <div class="info-card-text">Traditional tree counting requires teams on the ground — slow, expensive, and impossible at scale across thousands of hectares.</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card">
            <div class="info-card-icon">📉</div>
            <div class="info-card-title">Inaccurate CO₂ Estimates</div>
            <div class="info-card-text">Without individual tree-level data, carbon estimates rely on broad averages — leading to inflated credits that hurt carbon market credibility.</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="info-card">
            <div class="info-card-icon">⏳</div>
            <div class="info-card-title">No Real-Time Monitoring</div>
            <div class="info-card-text">Carbon project monitoring is annual at best. Deforestation goes undetected for months, invalidating credit claims retroactively.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # Before / After comparison
    st.markdown('<div class="section-header">The Gap We\'re Filling</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Manual MRV vs ForestSense AI — side by side</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="compare-wrap">
        <div class="compare-before">
            <div class="compare-label-bad">❌ Without ForestSense AI</div>
            <div class="compare-title">Traditional Manual MRV</div>
            <div class="compare-row"><span>🚶</span><span>Field teams survey forests on foot — weeks of work per site</span></div>
            <div class="compare-row"><span>📅</span><span>Monitoring happens once per year, missing seasonal changes</span></div>
            <div class="compare-row"><span>💸</span><span>Cost: ₹5–15 lakh per survey for a medium-sized project</span></div>
            <div class="compare-row"><span>📊</span><span>Tree counts are estimates — sampled, not exhaustive</span></div>
            <div class="compare-row"><span>🌫️</span><span>CO₂ estimates based on forest-type averages, not individual trees</span></div>
            <div class="compare-row"><span>📁</span><span>Manual data entry, spreadsheets, high error rate</span></div>
        </div>
        <div class="compare-after">
            <div class="compare-label-good">✅ With ForestSense AI</div>
            <div class="compare-title">AI-Powered Digital MRV</div>
            <div class="compare-row"><span>🛰️</span><span>Satellite image analysed in seconds — no field team needed</span></div>
            <div class="compare-row"><span>🔄</span><span>Can be run weekly or monthly for continuous monitoring</span></div>
            <div class="compare-row"><span>💡</span><span>Near-zero marginal cost per additional image processed</span></div>
            <div class="compare-row"><span>🎯</span><span>Every individual tree detected and counted by YOLOv8</span></div>
            <div class="compare-row"><span>🌱</span><span>Per-tree CO₂ estimate based on actual canopy size &amp; maturity</span></div>
            <div class="compare-row"><span>📥</span><span>Automated CSV + cropped image report, download in one click</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # Solution
    st.markdown('<div class="section-header">The Solution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""<div class="info-card">
            <div class="info-card-text" style="font-size:0.93rem;">
                <strong style="color:#1b5e35;">ForestSense AI</strong> replaces manual surveys with an automated pipeline —
                upload a satellite image, and the system detects every individual tree, classifies it by
                canopy size, estimates its maturity stage, and calculates annual CO₂ sequestration potential.<br><br>
                This is <strong>digital MRV (dMRV)</strong> in action — the same approach being adopted by
                leading carbon registries like <strong>Verra</strong> and <strong>Gold Standard</strong> to improve
                transparency and accuracy in nature-based carbon credit issuance.
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card" style="background: linear-gradient(135deg, #e8f5e9, #f1f8f4);">
            <div class="info-card-icon">✅</div>
            <div class="info-card-title">What This Tool Does</div>
            <div class="info-card-text">
                🛰️ Accepts satellite images<br><br>
                🤖 Detects trees via YOLOv8<br><br>
                📐 Classifies S / M / L by canopy ratio<br><br>
                🌱 Estimates CO₂ sequestration<br><br>
                📥 Exports CSV + cropped images
            </div>
        </div>""", unsafe_allow_html=True)


# ===================================================================
# PAGE: THE INITIATIVE
# ===================================================================
elif page == "🌍  The Initiative":

    st.markdown("""
    <div class="initiative-hero">
        <div class="initiative-quote">
            This is not just a tool.<br>
            It is an <em>initiative</em> to make<br>
            the planet cooler again.
        </div>
        <div class="initiative-sub">
            ForestSense AI was built with one belief — that technology, when pointed at the right problem,
            can heal the world. We chose forests. We chose carbon. We chose to act.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Alarming stats
    st.markdown('<div class="section-header">The Scale of the Crisis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Why this work is urgent — the numbers don\'t lie</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-number">46%</div>
            <div class="stat-label">of Earth's trees lost since humans appeared</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">10M</div>
            <div class="stat-label">hectares of forest destroyed every year</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">1.5°C</div>
            <div class="stat-label">global warming threshold we're racing against</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">23%</div>
            <div class="stat-label">of global CO₂ emissions from deforestation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # Mission / Vision / Values
    st.markdown('<div class="section-header">Our Mission & Vision</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="info-card" style="border-left: 4px solid #2e7d52;">
            <div class="info-card-icon">🎯</div>
            <div class="info-card-title">Mission</div>
            <div class="info-card-text">
                To make forest carbon monitoring <strong>accessible, accurate, and automated</strong> —
                so that every tree in every nature-based carbon project can be counted,
                tracked, and credited with the precision it deserves.<br><br>
                We are democratising dMRV — making tools that were previously available only to
                well-funded organisations accessible to small project developers and forest communities.
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card" style="border-left: 4px solid #a5d6a7;">
            <div class="info-card-icon">🔭</div>
            <div class="info-card-title">Vision</div>
            <div class="info-card-text">
                A world where <strong>every carbon credit is backed by verifiable, AI-powered data</strong> —
                where no credit is issued without proof, and no forest goes unmonitored.<br><br>
                We envision ForestSense AI evolving into a full-stack dMRV platform —
                covering not just tree detection but species classification, biomass estimation,
                biodiversity indices, and real-time deforestation alerts.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # Why it matters
    st.markdown('<div class="section-header">Why It Matters</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">The real-world impact of better forest monitoring</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="info-card">
            <div class="info-card-icon">💰</div>
            <div class="info-card-title">Carbon Market Integrity</div>
            <div class="info-card-text">
                Poor MRV is the #1 reason carbon credits get cancelled or lose buyer trust.
                Better tree-level data means credits that are <strong>credible, tradeable, and impactful</strong>.
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card">
            <div class="info-card-icon">🌾</div>
            <div class="info-card-title">Community & Livelihood</div>
            <div class="info-card-text">
                Forest communities earn from carbon credits. Accurate monitoring ensures
                they are <strong>fairly compensated</strong> for protecting forests that the whole world benefits from.
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="info-card">
            <div class="info-card-icon">🌡️</div>
            <div class="info-card-title">Climate Targets</div>
            <div class="info-card-text">
                Nature-based solutions can deliver <strong>up to 30% of the emission reductions</strong>
                needed by 2030. But only if we can measure them accurately — that's exactly what we're building.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # Roadmap
    st.markdown('<div class="section-header">Where We\'re Headed</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">The ForestSense AI roadmap</div>', unsafe_allow_html=True)

    roadmap = [
        ("✅ v1.0 — Now", "YOLOv8 tree detection, size classification, CO₂ estimation, CSV + ZIP report generation."),
        ("🔜 v2.0 — Next", "Tree species classification using multi-spectral satellite bands. Above-ground biomass (AGB) estimation using allometric equations."),
        ("🔭 v3.0 — Future", "Change detection — compare two time-period images to detect deforestation or new growth. Deforestation alert system."),
        ("🚀 v4.0 — Vision", "Full dMRV platform with GIS integration, project boundary mapping, automated VVB-ready reports compatible with Verra VCS methodology."),
    ]
    for stage, desc in roadmap:
        st.markdown(f"""
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content"><span class="timeline-year">{stage}:</span> {desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # Closing quote
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e8f5e9, #f1f8f4); border-radius: 16px; padding: 40px; text-align: center; border: 1px solid #c8e6c9;">
        <div style="font-family: 'DM Serif Display', serif; font-size: 1.6rem; color: #071a0e; margin-bottom: 12px; line-height: 1.4;">
            "The best time to plant a tree was 20 years ago.<br>The second best time is <em style='color:#2e7d52;'>now.</em>"
        </div>
        <div style="font-size: 0.85rem; color: #888;">— Chinese Proverb · the reason we built ForestSense AI</div>
    </div>
    """, unsafe_allow_html=True)


# ===================================================================
# PAGE: MRV & CARBON MARKET
# ===================================================================
elif page == "📘  MRV & Carbon Market":

    st.markdown('<div class="section-header">MRV, dMRV & the Carbon Market</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">The science and economics behind carbon credit verification</div>', unsafe_allow_html=True)
    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="info-card">
            <div class="info-card-icon">📏</div>
            <div class="info-card-title">What is MRV?</div>
            <div class="info-card-text">
                <strong>MRV — Monitoring, Reporting &amp; Verification</strong> — is the framework used to
                measure how much CO₂ a carbon project actually removes or avoids.<br><br>
                <strong>Monitoring:</strong> Continuously tracking emissions, biomass, or sequestration data.<br><br>
                <strong>Reporting:</strong> Documenting findings in standardised formats accepted by registries
                like Verra (VCS), Gold Standard, or ICR.<br><br>
                <strong>Verification:</strong> Independent third-party auditors (VVBs) confirm the data
                before carbon credits are issued.<br><br>
                MRV is the backbone of carbon market integrity — without it, carbon credits are just promises.
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card">
            <div class="info-card-icon">🛰️</div>
            <div class="info-card-title">What is dMRV?</div>
            <div class="info-card-text">
                <strong>dMRV — Digital MRV</strong> — replaces manual field surveys with remote sensing,
                AI, and satellite data for continuous, scalable monitoring.<br><br>
                Instead of sending surveyors into forests once a year, dMRV uses:<br><br>
                🛰️ <strong>Satellite imagery</strong> (Sentinel, Landsat, Planet)<br>
                🤖 <strong>Deep learning models</strong> (YOLO, CNNs, transformers)<br>
                📡 <strong>IoT sensors</strong> for soil, weather, and biomass<br>
                🗺️ <strong>GIS platforms</strong> for spatial analysis<br><br>
                ForestSense AI is a dMRV tool — automating tree-level monitoring using
                computer vision on satellite images.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">The Carbon Market</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">How carbon credits are created, verified, and traded</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="info-card">
            <div class="info-card-icon">🌍</div>
            <div class="info-card-title">Voluntary Carbon Market</div>
            <div class="info-card-text">
                Companies voluntarily offset emissions by purchasing carbon credits.
                Each credit = 1 tonne of CO₂ removed or avoided.<br><br>
                Key registries: <strong>Verra, Gold Standard, ACR, ICR</strong>.
                Market projected to reach <strong>$50B by 2030</strong>.
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card">
            <div class="info-card-icon">🌿</div>
            <div class="info-card-title">Nature-Based Solutions</div>
            <div class="info-card-text">
                Forests, mangroves, and soil absorb CO₂ naturally. Projects that
                protect or restore these generate <strong>nature-based carbon credits</strong>.<br><br>
                Types: REDD+, ARR (Afforestation), IFM (Forest Management),
                Biochar, Soil Carbon.
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="info-card">
            <div class="info-card-icon">🔄</div>
            <div class="info-card-title">Credit Lifecycle</div>
            <div class="info-card-text">
                <strong>1.</strong> Project Developer designs the project<br>
                <strong>2.</strong> Methodology defines measurement rules<br>
                <strong>3.</strong> MRV collects &amp; verifies data<br>
                <strong>4.</strong> VVB conducts third-party audit<br>
                <strong>5.</strong> Registry issues the credits<br>
                <strong>6.</strong> Buyer purchases &amp; retires credits
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Future of dMRV</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Where forest monitoring is headed in the next decade</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("""<div class="info-card" style="background: linear-gradient(135deg, #071a0e, #1a4a2e); color: white;">
            <div class="info-card-icon">🚀</div>
            <div class="info-card-title" style="color: #a5d6a7;">Key Trends</div>
            <div class="info-card-text" style="color: rgba(255,255,255,0.75);">
                🌐 <strong style="color:#a5d6a7;">Hyperspectral satellites</strong> — tree species ID from orbit<br><br>
                🤖 <strong style="color:#a5d6a7;">Foundation models</strong> — one model for all forest types<br><br>
                ⛓️ <strong style="color:#a5d6a7;">Blockchain registries</strong> — tamper-proof credit issuance<br><br>
                📡 <strong style="color:#a5d6a7;">Continuous monitoring</strong> — daily updates, not annual<br><br>
                🔗 <strong style="color:#a5d6a7;">IoT + satellite fusion</strong> — ground truth at scale
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card">
            <div class="info-card-icon">📅</div>
            <div class="info-card-title">Timeline of dMRV Evolution</div>
        </div>""", unsafe_allow_html=True)
        timeline_items = [
            ("Pre-2015", "Manual field surveys dominate. Expensive, slow, infrequent."),
            ("2015–2018", "Landsat & Sentinel satellites enable country-level forest cover mapping."),
            ("2019–2021", "Deep learning applied to satellite images. YOLO models used for individual tree detection."),
            ("2022–2023", "Gold Standard & Verra begin accepting dMRV-based submissions. Pachama, SilviaTerra emerge."),
            ("2024–2025", "Foundation models (SAM, TreeSAT) enable zero-shot detection. LiDAR + AI for 3D biomass."),
            ("2026+", "Real-time continuous dMRV becomes the standard. Blockchain-backed credit issuance."),
        ]
        for year, text in timeline_items:
            st.markdown(f"""
            <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-content"><span class="timeline-year">{year}:</span> {text}</div>
            </div>""", unsafe_allow_html=True)


# ===================================================================
# PAGE: HOW IT WORKS
# ===================================================================
elif page == "🔬  How It Works":

    st.markdown('<div class="section-header">How ForestSense AI Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">From satellite image to carbon report in seconds</div>', unsafe_allow_html=True)
    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    steps = [
        ("01", "Upload Satellite Image", "User uploads a high-resolution satellite or aerial image (JPG/PNG). The system supports images up to 10,000×10,000 px. Images are temporarily stored for inference and deleted after processing."),
        ("02", "YOLOv8 Tree Detection", "A custom-trained YOLOv8 model runs inference on the image. It detects individual tree canopies and outputs bounding box coordinates (x1, y1, x2, y2) for each detected tree. Bounding boxes are drawn on the image for visual confirmation."),
        ("03", "Size Classification via Area Ratio", "Each bounding box area is computed as a ratio of the total image area — making classification device-independent and scale-invariant. S < 1% | M = 1–2% | L > 2% image area."),
        ("04", "Maturity & CO₂ Estimation", "Size class maps to estimated maturity (Young / Semi-Mature / Mature) and annual CO₂ sequestration potential (10 / 20 / 30 kg CO₂/year). These are conservative baseline estimates for tropical/subtropical tree species."),
        ("05", "Report Generation & Export", "A structured CSV report is generated with tree number, size, maturity, CO₂ estimate, and canopy area. Cropped images of each tree are saved. Everything is bundled into a single downloadable ZIP file."),
    ]
    for num, title, text in steps:
        st.markdown(f"""<div class="step-card">
            <div class="step-number">{num}</div>
            <div class="step-title">{title}</div>
            <div class="step-text">{text}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Classification & CO₂ Mapping</div>', unsafe_allow_html=True)

    table_data = {
        "Size Class": ["S — Small", "M — Medium", "L — Large"],
        "Area Ratio": ["< 1% of image", "1% – 2% of image", "> 2% of image"],
        "Estimated Maturity": ["Likely Young", "Semi-Mature", "Mature"],
        "CO₂ Estimate": ["10 kg/year", "20 kg/year", "30 kg/year"],
        "Canopy Description": ["Sapling / early growth", "Established tree", "Full canopy / dominant"],
    }
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Tech Stack</div>', unsafe_allow_html=True)
    techs = ["YOLOv8 (Ultralytics)", "Streamlit", "OpenCV", "PIL / Pillow", "Pandas", "NumPy", "Matplotlib", "Python 3.10+"]
    pills_html = "".join([f'<span class="tag-pill">{t}</span>' for t in techs])
    st.markdown(f"<div style='margin-top:10px'>{pills_html}</div>", unsafe_allow_html=True)


# ===================================================================
# PAGE: SAMPLE OUTPUT
# ===================================================================
elif page == "📊  Sample Output":

    st.markdown('<div class="section-header">Sample Output</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">What ForestSense AI generates from a satellite image</div>', unsafe_allow_html=True)
    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    st.info("💡 This is a **demo output** using simulated data — representative of real results from a forested satellite image.")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [("34", "Trees Detected"), ("740", "Total CO₂ (kg/year)"), ("0.014", "Avg. Canopy Ratio"), ("21.8", "Avg CO₂/Tree (kg/yr)")]
    for col, (num, label) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""<div class="result-highlight">
                <div class="result-big">{num}</div>
                <div class="result-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🥧 Size Distribution**")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='none')
        wedges, texts, autotexts = ax.pie(
            [14, 12, 8], labels=['Small (S)', 'Medium (M)', 'Large (L)'],
            autopct='%1.1f%%', startangle=90,
            colors=['#a5d6a7', '#388e3c', '#1b5e35'],
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        for t in texts: t.set_fontsize(10)
        for at in autotexts: at.set_fontsize(9); at.set_color('white')
        ax.axis('equal'); fig.patch.set_alpha(0)
        st.pyplot(fig, transparent=True)

    with col2:
        st.markdown("**📊 CO₂ by Size Class**")
        fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor='none')
        bars = ax2.bar(['Small (S)', 'Medium (M)', 'Large (L)'], [140, 240, 360],
                       color=['#a5d6a7', '#388e3c', '#1b5e35'], edgecolor='white', linewidth=1.5, width=0.5)
        ax2.set_ylabel('CO₂ (kg/year)', fontsize=9)
        ax2.set_facecolor('none'); fig2.patch.set_alpha(0)
        for sp in ['top', 'right']: ax2.spines[sp].set_visible(False)
        for sp in ['left', 'bottom']: ax2.spines[sp].set_color('#ccc')
        ax2.tick_params(colors='#555'); ax2.yaxis.label.set_color('#555')
        for bar, val in zip(bars, [140, 240, 360]):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                     f'{val}', ha='center', va='bottom', fontsize=9, color='#1b5e35', fontweight='bold')
        st.pyplot(fig2, transparent=True)

    # Real-world CO2 context card
    st.markdown("""
    <div class="context-card">
        <div class="context-card-title">🌍 What does 740 kg CO₂/year actually mean?</div>
        <div class="context-row">
            <div class="context-icon">🚗</div>
            <div class="context-text">Equivalent to the CO₂ emitted from driving <strong>~2,900 km</strong> in a petrol car — roughly Delhi to Bangalore and back</div>
        </div>
        <div class="context-row">
            <div class="context-icon">✈️</div>
            <div class="context-text">Offsets <strong>~1 economy flight</strong> from Mumbai to Dubai (approx 600–700 kg CO₂ per passenger)</div>
        </div>
        <div class="context-row">
            <div class="context-icon">💡</div>
            <div class="context-text">Powers an average Indian household for <strong>~8 months</strong> worth of electricity emissions</div>
        </div>
        <div class="context-row">
            <div class="context-icon">🏭</div>
            <div class="context-text">This is from just <strong>34 trees</strong> in one satellite image — scale to 10,000 trees across a project and the impact becomes enormous</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
    st.markdown("**📋 Sample Tree Report (first 10 trees)**")

    random.seed(42)
    sample_data = []
    for i in range(1, 11):
        size = random.choice(["S", "S", "M", "M", "M", "L"])
        sample_data.append({
            "Tree #": i, "Size": size,
            "Maturity": {"S": "Likely Young", "M": "Semi-Mature", "L": "Mature"}[size],
            "CO₂ (kg/year)": {"S": 10, "M": 20, "L": 30}[size],
            "Canopy Area (px²)": random.randint(800, 15000)
        })
    st.dataframe(pd.DataFrame(sample_data), use_container_width=True, hide_index=True)


# ===================================================================
# PAGE: LAUNCH TOOL
# ===================================================================
elif page == "🌳  Launch Tool":

    st.markdown('<div class="section-header">Tree Detection Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Upload a satellite image to begin detection and CO₂ estimation</div>', unsafe_allow_html=True)
    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    @st.cache_resource
    def load_model():
        return YOLO("deetection.pt")

    model = load_model()

    uploaded_image = st.file_uploader(
        "Upload a Satellite Image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Best results with high-resolution satellite or aerial imagery"
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        if max(image.size) > 10000:
            image.thumbnail((10000, 10000), Image.Resampling.LANCZOS)

        image_np = np.array(image)
        image_path = "uploaded_image.jpg"
        image.save(image_path)

        with st.spinner("🤖 Running YOLOv8 detection..."):
            results = model(image_path)[0]

        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (46, 125, 82), 2)

        image_with_boxes = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(image_with_boxes, caption="Detected Trees", use_container_width=True)

        output_data = []
        canopy_areas = []
        co2_total = 0
        class_counts = defaultdict(int)
        co2_map = {"S": 10, "M": 20, "L": 30}
        maturity_map = {"S": "Likely Young", "M": "Semi-Mature", "L": "Mature"}
        image_area = image_bgr.shape[0] * image_bgr.shape[1]
        crop_dir = "tree_crops"
        os.makedirs(crop_dir, exist_ok=True)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            crop = image_bgr[y1:y2, x1:x2]
            bbox_area = (x2 - x1) * (y2 - y1)
            bbox_ratio = bbox_area / image_area
            size_class = "S" if bbox_ratio < 0.01 else ("M" if bbox_ratio < 0.02 else "L")
            co2 = co2_map[size_class]
            maturity = maturity_map[size_class]
            cv2.imwrite(os.path.join(crop_dir, f"tree_{i+1}_{size_class}.jpg"), crop)
            co2_total += co2
            class_counts[size_class] += 1
            canopy_areas.append(bbox_area)
            output_data.append({"Tree #": i+1, "Size": size_class, "Maturity": maturity,
                                 "CO₂ (kg/year)": co2, "Canopy Area (px²)": bbox_area})

        with col2:
            st.markdown(f"""
            <div class="result-highlight" style="margin-bottom:12px;">
                <div class="result-big">{len(boxes)}</div>
                <div class="result-label">Trees Detected</div>
            </div>
            <div class="result-highlight" style="margin-bottom:12px;">
                <div class="result-big">{co2_total:.0f}</div>
                <div class="result-label">Total CO₂ (kg/year)</div>
            </div>
            <div class="result-highlight">
                <div class="result-big">{np.mean(canopy_areas):.0f}</div>
                <div class="result-label">Avg Canopy Area (px²)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Size Distribution**")
            if class_counts:
                fig, ax = plt.subplots(figsize=(4, 3.5), facecolor='none')
                colors_used = ['#a5d6a7', '#388e3c', '#1b5e35'][:len(class_counts)]
                wedges, texts, autotexts = ax.pie(
                    class_counts.values(), labels=class_counts.keys(),
                    autopct='%1.1f%%', startangle=90, colors=colors_used,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
                )
                for at in autotexts: at.set_color('white'); at.set_fontsize(9)
                ax.axis('equal'); fig.patch.set_alpha(0)
                st.pyplot(fig, transparent=True)
        with col2:
            st.markdown("**Detailed Report**")
            df = pd.DataFrame(output_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Real-world context after real results
        equiv_km = round((co2_total / 0.255), 0)
        st.markdown(f"""
        <div class="context-card" style="margin-top:1rem;">
            <div class="context-card-title">🌍 What does your {co2_total:.0f} kg CO₂/year result mean in real life?</div>
            <div class="context-row">
                <div class="context-icon">🚗</div>
                <div class="context-text">Equivalent to emissions from driving <strong>~{equiv_km:,.0f} km</strong> in a petrol car</div>
            </div>
            <div class="context-row">
                <div class="context-icon">🌳</div>
                <div class="context-text">Sequestered by <strong>{len(boxes)} trees</strong> — imagine scaling this to an entire forest project</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Download
        csv_path = "tree_report.csv"
        df.to_csv(csv_path, index=False)
        zip_path = "tree_report_package.zip"
        with ZipFile(zip_path, 'w') as zipf:
            zipf.write(csv_path)
            for file_name in os.listdir(crop_dir):
                zipf.write(os.path.join(crop_dir, file_name), arcname=os.path.join("tree_crops", file_name))
        with open(zip_path, "rb") as f:
            st.download_button("📥 Download Full Report (CSV + Cropped Trees)", f,
                               file_name="forestsense_report.zip", use_container_width=True)

        shutil.rmtree(crop_dir)
        os.remove(csv_path); os.remove(zip_path); os.remove(image_path)

    else:
        st.markdown("""
        <div class="info-card" style="text-align:center; padding:50px; border:2px dashed #c8e6c9;">
            <div style="font-size:3rem; margin-bottom:16px;">🛰️</div>
            <div class="info-card-title" style="font-size:1.2rem;">Upload a satellite image to begin</div>
            <div class="info-card-text">Supported formats: JPG, JPEG, PNG · Max recommended: 10,000 × 10,000 px</div>
        </div>
        """, unsafe_allow_html=True)
