import streamlit as stimport cv2import numpy as npimport pandas as pdfrom ultralytics import YOLOfrom PIL import Imageimport osimport shutilimport matplotlib.pyplot as pltimport matplotlib.patches as mpatchesfrom zipfile import ZipFilefrom collections import defaultdict

===== PAGE CONFIG =====

st.set_page_config(page_title="ForestSense AI | Tree Detection & Carbon MRV",layout="wide",initial_sidebar_state="expanded",menu_items={"About": "ForestSense AI — Built by Mayank Kumar Sharma"})

===== GLOBAL CSS =====

st.markdown("""

""", unsafe_allow_html=True)

===== SIDEBAR =====

with st.sidebar:st.markdown("""🌳ForestSense AITree Detection & Carbon MRV""", unsafe_allow_html=True)

page = st.radio(
    "Navigate",
    ["🏠  Home", "📘  MRV & Carbon Market", "🔬  How It Works", "📊  Sample Output", "🌳  Launch Tool"],
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="padding: 14px 16px; background: rgba(255,255,255,0.06); border-radius: 10px; font-size: 0.78rem; color: rgba(255,255,255,0.5); line-height:1.6;">
    <strong style="color:#a5d6a7;">Tech Stack</strong><br>
    YOLOv8 · Streamlit · OpenCV<br>
    Pandas · Matplotlib · PIL
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="padding: 12px 16px; font-size:0.75rem; color:rgba(255,255,255,0.3); border-top: 1px solid rgba(255,255,255,0.08); line-height:1.6;">
    Made with ❤️ by<br>
    <strong style="color:#a5d6a7;">Mayank Kumar Sharma</strong><br>
    B.Tech AI & Data Science · CTAE Udaipur
</div>
""", unsafe_allow_html=True)

===================================================================

PAGE: HOME

===================================================================

if page == "🏠  Home":

st.markdown("""
<div class="hero-card">
    <div class="hero-badge">🛰️ Satellite AI · Carbon MRV · Forest Intelligence</div>
    <div class="hero-title">Detecting Trees.<br><em>Quantifying Carbon.</em></div>
    <div class="hero-subtitle">
        ForestSense AI uses deep learning on satellite imagery to detect individual trees, 
        classify their size & maturity, and estimate CO₂ sequestration — enabling 
        scalable, digital MRV for nature-based carbon projects.
    </div>
</div>
""", unsafe_allow_html=True)

# Stats row
st.markdown("""
<div class="stat-row">
    <div class="stat-card">
        <div class="stat-number">15B</div>
        <div class="stat-label">Trees lost per year globally</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">2.6T</div>
        <div class="stat-label">Tons CO₂ absorbed by forests annually</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">$2B+</div>
        <div class="stat-label">Nature-based carbon market size</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">30%</div>
        <div class="stat-label">Climate targets met via forests</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

# Problem statement
st.markdown('<div class="section-header">The Problem</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subheader">Why traditional forest monitoring falls short</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-icon">🚶</div>
        <div class="info-card-title">Manual Field Surveys</div>
        <div class="info-card-text">Traditional tree counting requires teams on the ground — slow, expensive, and impossible at scale across thousands of hectares.</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-icon">📉</div>
        <div class="info-card-title">Inaccurate CO₂ Estimates</div>
        <div class="info-card-text">Without individual tree-level data, carbon estimates rely on broad averages — leading to inflated or under-reported credits that hurt market credibility.</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-icon">⏳</div>
        <div class="info-card-title">No Real-Time Monitoring</div>
        <div class="info-card-text">Carbon project monitoring is annual at best. Deforestation or tree loss goes undetected for months, invalidating credit claims retroactively.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

# Solution
st.markdown('<div class="section-header">The Solution</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subheader">AI-powered digital MRV at satellite scale</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-text" style="font-size:0.93rem;">
            <strong style="color:#1b5e35;">ForestSense AI</strong> replaces manual surveys with an automated pipeline — 
            upload a satellite image, and the system detects every individual tree, classifies it by 
            canopy size, estimates its maturity stage, and calculates annual CO₂ sequestration potential.<br><br>
            This is <strong>digital MRV (dMRV)</strong> in action — the same approach being adopted by 
            leading carbon registries like Verra and Gold Standard to improve transparency 
            and accuracy in nature-based carbon credit issuance.
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="info-card" style="background: linear-gradient(135deg, #e8f5e9, #f1f8f4);">
        <div class="info-card-icon">✅</div>
        <div class="info-card-title">What This Tool Does</div>
        <div class="info-card-text">
            🛰️ Accepts satellite images<br><br>
            🤖 Detects trees via YOLOv8<br><br>
            📐 Classifies S / M / L by canopy ratio<br><br>
            🌱 Estimates CO₂ sequestration<br><br>
            📥 Exports CSV + cropped images
        </div>
    </div>
    """, unsafe_allow_html=True)

===================================================================

PAGE: MRV & CARBON MARKET

===================================================================

elif page == "📘  MRV & Carbon Market":

st.markdown('<div class="section-header">MRV, dMRV & the Carbon Market</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subheader">The science and economics behind carbon credit verification</div>', unsafe_allow_html=True)

st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

# MRV
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-icon">📏</div>
        <div class="info-card-title">What is MRV?</div>
        <div class="info-card-text">
            <strong>MRV — Monitoring, Reporting & Verification</strong> — is the framework used to 
            measure how much CO₂ a carbon project actually removes or avoids.<br><br>
            <strong>Monitoring:</strong> Continuously tracking emissions, biomass, or sequestration data.<br><br>
            <strong>Reporting:</strong> Documenting findings in standardised formats accepted by registries 
            like Verra (VCS), Gold Standard, or ICR.<br><br>
            <strong>Verification:</strong> Independent third-party auditors (VVBs) confirm the data before 
            carbon credits are issued.<br><br>
            MRV is the backbone of carbon market integrity — without it, carbon credits are just promises.
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="info-card">
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
            ForestSense AI is a dMRV tool — it automates tree-level monitoring 
            using computer vision on satellite images.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">The Carbon Market</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subheader">How carbon credits are created, verified, and traded</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-icon">🌍</div>
        <div class="info-card-title">Voluntary Carbon Market (VCM)</div>
        <div class="info-card-text">
            Companies and individuals voluntarily offset their emissions by purchasing carbon credits. 
            Each credit = 1 tonne of CO₂ removed or avoided.<br><br>
            Key players: <strong>Verra, Gold Standard, ACR, ICR</strong>. 
            Market size projected to reach <strong>$50B by 2030</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-icon">🌿</div>
        <div class="info-card-title">Nature-Based Solutions (NbS)</div>
        <div class="info-card-text">
            Forests, mangroves, wetlands, and soil absorb CO₂ naturally. Projects that protect or 
            restore these ecosystems generate <strong>nature-based carbon credits</strong>.<br><br>
            Types: REDD+, ARR (Afforestation), IFM (Improved Forest Management), 
            Biochar, Soil Carbon.
        </div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-icon">🔄</div>
        <div class="info-card-title">Credit Lifecycle</div>
        <div class="info-card-text">
            <strong>1. Project Developer</strong> designs the project<br>
            <strong>2. Methodology</strong> defines measurement rules<br>
            <strong>3. MRV</strong> collects & verifies data<br>
            <strong>4. VVB</strong> conducts third-party audit<br>
            <strong>5. Registry</strong> issues the credits<br>
            <strong>6. Buyer</strong> purchases & retires credits
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Future of dMRV</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subheader">Where forest monitoring is headed in the next decade</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 3])
with col1:
    st.markdown("""
    <div class="info-card" style="background: linear-gradient(135deg, #0d2b1a, #1a4a2e); color: white;">
        <div class="info-card-icon">🚀</div>
        <div class="info-card-title" style="color: #a5d6a7;">Key Trends</div>
        <div class="info-card-text" style="color: rgba(255,255,255,0.75);">
            🌐 <strong style="color:#a5d6a7;">Hyperspectral satellites</strong> — tree species ID from orbit<br><br>
            🤖 <strong style="color:#a5d6a7;">Foundation models</strong> — one model for all forest types<br><br>
            ⛓️ <strong style="color:#a5d6a7;">Blockchain registries</strong> — tamper-proof credit issuance<br><br>
            📡 <strong style="color:#a5d6a7;">Continuous monitoring</strong> — daily updates, not annual<br><br>
            🔗 <strong style="color:#a5d6a7;">IoT + satellite fusion</strong> — ground truth at scale
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-icon">📅</div>
        <div class="info-card-title">Timeline of dMRV Evolution</div>
    </div>
    """, unsafe_allow_html=True)

    timeline_items = [
        ("Pre-2015", "Manual field surveys dominate. Expensive, slow, infrequent."),
        ("2015–2018", "Landsat & Sentinel satellites enable country-level forest cover mapping."),
        ("2019–2021", "Deep learning (CNNs) applied to satellite images for tree detection. YOLO models used for individual tree segmentation."),
        ("2022–2023", "Gold Standard & Verra begin accepting dMRV-based project submissions. Companies like Pachama, SilviaTerra emerge."),
        ("2024–2025", "Foundation models (SAM, TreeSAT) enable zero-shot tree detection. LiDAR + AI for 3D biomass estimation."),
        ("2026+", "Real-time continuous dMRV becomes the standard. Blockchain-backed credit issuance. IoT sensor fusion with satellite data."),
    ]
    for year, text in timeline_items:
        st.markdown(f"""
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content"><span class="timeline-year">{year}:</span> {text}</div>
        </div>
        """, unsafe_allow_html=True)

===================================================================

PAGE: HOW IT WORKS

===================================================================

elif page == "🔬  How It Works":

st.markdown('<div class="section-header">How ForestSense AI Works</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subheader">From satellite image to carbon report in seconds</div>', unsafe_allow_html=True)

st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

# Steps
steps = [
    ("01", "Upload Satellite Image", "User uploads a high-resolution satellite or aerial image (JPG/PNG). The system supports images up to 10,000×10,000 px. Images are temporarily stored for inference and deleted after processing."),
    ("02", "YOLOv8 Tree Detection", "A custom-trained YOLOv8 model runs inference on the image. It detects individual tree canopies and outputs bounding box coordinates (x1, y1, x2, y2) for each detected tree. Bounding boxes are drawn on the image for visual confirmation."),
    ("03", "Size Classification via Area Ratio", "Each bounding box area is computed as a ratio of the total image area — making classification device-independent and scale-invariant. S < 1% | M = 1–2% | L > 2% image area."),
    ("04", "Maturity & CO₂ Estimation", "Size class maps to estimated maturity (Young / Semi-Mature / Mature) and annual CO₂ sequestration potential (10 / 20 / 30 kg CO₂/year). These are conservative baseline estimates for tropical/subtropical tree species."),
    ("05", "Report Generation & Export", "A structured CSV report is generated with tree number, size, maturity, CO₂ estimate, and canopy area. Cropped images of each tree are saved. Everything is bundled into a single downloadable ZIP file."),
]

for num, title, text in steps:
    st.markdown(f"""
    <div class="step-card">
        <div class="step-number">{num}</div>
        <div class="step-title">{title}</div>
        <div class="step-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Classification & CO₂ Mapping</div>', unsafe_allow_html=True)

table_data = {
    "Size Class": ["S — Small", "M — Medium", "L — Large"],
    "Area Ratio": ["< 1% of image", "1% – 2% of image", "> 2% of image"],
    "Estimated Maturity": ["Likely Young", "Semi-Mature", "Mature"],
    "CO₂ Estimate": ["10 kg/year", "20 kg/year", "30 kg/year"],
    "Canopy Description": ["Sapling / early growth", "Established tree", "Full canopy / dominant"],
}
df_table = pd.DataFrame(table_data)
st.dataframe(df_table, use_container_width=True, hide_index=True)

st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Tech Stack</div>', unsafe_allow_html=True)

techs = ["YOLOv8 (Ultralytics)", "Streamlit", "OpenCV", "PIL / Pillow", "Pandas", "NumPy", "Matplotlib", "ZipFile", "Python 3.10+"]
pills_html = "".join([f'<span class="tag-pill">{t}</span>' for t in techs])
st.markdown(f"<div style='margin-top:10px'>{pills_html}</div>", unsafe_allow_html=True)

===================================================================

PAGE: SAMPLE OUTPUT

===================================================================

elif page == "📊  Sample Output":

st.markdown('<div class="section-header">Sample Output</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subheader">What ForestSense AI generates from a satellite image</div>', unsafe_allow_html=True)
st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

st.info("💡 This is a **demo output** using simulated data — representative of real results from a forested satellite image.")

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="result-highlight">
        <div class="result-big">34</div>
        <div class="result-label">Trees Detected</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="result-highlight">
        <div class="result-big">740</div>
        <div class="result-label">Total CO₂ (kg/year)</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="result-highlight">
        <div class="result-big">0.014</div>
        <div class="result-label">Avg. Canopy Ratio</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="result-highlight">
        <div class="result-big">21.8</div>
        <div class="result-label">Avg CO₂/Tree (kg/yr)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**🥧 Size Distribution**")
    fig, ax = plt.subplots(figsize=(5, 4), facecolor='none')
    sizes = [14, 12, 8]
    labels = ['Small (S)', 'Medium (M)', 'Large (L)']
    colors = ['#a5d6a7', '#388e3c', '#1b5e35']
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=90, colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for t in texts: t.set_fontsize(10)
    for at in autotexts: at.set_fontsize(9); at.set_color('white')
    ax.axis('equal')
    fig.patch.set_alpha(0)
    st.pyplot(fig, transparent=True)

with col2:
    st.markdown("**📊 CO₂ by Size Class**")
    fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor='none')
    classes = ['Small (S)', 'Medium (M)', 'Large (L)']
    co2_vals = [140, 240, 360]
    bars = ax2.bar(classes, co2_vals, color=['#a5d6a7', '#388e3c', '#1b5e35'],
                   edgecolor='white', linewidth=1.5, width=0.5)
    ax2.set_ylabel('CO₂ (kg/year)', fontsize=9)
    ax2.set_facecolor('none')
    fig2.patch.set_alpha(0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#ccc')
    ax2.spines['bottom'].set_color('#ccc')
    ax2.tick_params(colors='#555')
    ax2.yaxis.label.set_color('#555')
    for bar, val in zip(bars, co2_vals):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                 f'{val}', ha='center', va='bottom', fontsize=9, color='#1b5e35', fontweight='bold')
    st.pyplot(fig2, transparent=True)

st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
st.markdown("**📋 Sample Tree Report (first 10 trees)**")

sample_data = []
import random
random.seed(42)
for i in range(1, 11):
    size = random.choice(["S", "S", "M", "M", "M", "L"])
    co2 = {"S": 10, "M": 20, "L": 30}[size]
    mat = {"S": "Likely Young", "M": "Semi-Mature", "L": "Mature"}[size]
    area = random.randint(800, 15000)
    sample_data.append({"Tree #": i, "Size": size, "Maturity": mat,
                         "CO₂ (kg/year)": co2, "Canopy Area (px²)": area})
df_sample = pd.DataFrame(sample_data)
st.dataframe(df_sample, use_container_width=True, hide_index=True)

===================================================================

PAGE: LAUNCH TOOL

===================================================================

elif page == "🌳  Launch Tool":

st.markdown('<div class="section-header">Tree Detection Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subheader">Upload a satellite image to begin detection and CO₂ estimation</div>', unsafe_allow_html=True)
st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

# Load model
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

    max_dim = 10000
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

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

        if bbox_ratio < 0.01:
            size_class = "S"
        elif bbox_ratio < 0.02:
            size_class = "M"
        else:
            size_class = "L"

        co2 = co2_map[size_class]
        maturity = maturity_map[size_class]
        crop_path = os.path.join(crop_dir, f"tree_{i+1}_{size_class}.jpg")
        cv2.imwrite(crop_path, crop)
        co2_total += co2
        class_counts[size_class] += 1
        canopy_areas.append(bbox_area)
        output_data.append({
            "Tree #": i+1,
            "Size": size_class,
            "Maturity": maturity,
            "CO₂ (kg/year)": co2,
            "Canopy Area (px²)": bbox_area
        })

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
            colors = ['#a5d6a7', '#388e3c', '#1b5e35'][:len(class_counts)]
            wedges, texts, autotexts = ax.pie(
                class_counts.values(), labels=class_counts.keys(),
                autopct='%1.1f%%', startangle=90, colors=colors,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2}
            )
            for at in autotexts: at.set_color('white'); at.set_fontsize(9)
            ax.axis('equal')
            fig.patch.set_alpha(0)
            st.pyplot(fig, transparent=True)

    with col2:
        st.markdown("**Detailed Report**")
        df = pd.DataFrame(output_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Download
    csv_path = "tree_report.csv"
    df.to_csv(csv_path, index=False)
    zip_path = "tree_report_package.zip"
    with ZipFile(zip_path, 'w') as zipf:
        zipf.write(csv_path)
        for file_name in os.listdir(crop_dir):
            zipf.write(os.path.join(crop_dir, file_name), arcname=os.path.join("tree_crops", file_name))

    with open(zip_path, "rb") as f:
        st.download_button(
            "📥 Download Full Report (CSV + Cropped Trees)",
            f, file_name="forestsense_report.zip",
            use_container_width=True
        )

    shutil.rmtree(crop_dir)
    os.remove(csv_path)
    os.remove(zip_path)
    os.remove(image_path)

else:
    st.markdown("""
    <div class="info-card" style="text-align:center; padding: 50px; border: 2px dashed #c8e6c9;">
        <div style="font-size:3rem; margin-bottom:16px;">🛰️</div>
        <div class="info-card-title" style="font-size:1.2rem;">Upload a satellite image to begin</div>
        <div class="info-card-text">Supported formats: JPG, JPEG, PNG · Max recommended: 10,000 × 10,000 px</div>
    </div>
    """, unsafe_allow_html=True)
