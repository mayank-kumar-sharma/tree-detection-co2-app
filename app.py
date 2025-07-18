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

# ===== Streamlit App Config =====
st.set_page_config(page_title="Tree Detection & CO₂ Estimation", layout="centered")
st.title("🌳 Tree Detection and CO₂ Estimation")
st.markdown("""
Upload a **satellite image** to detect trees, classify them by size (S/M/L), estimate maturity and CO₂ sequestration, and get a CSV report with cropped images.

⚠️ **Note:** Works only with satellite imagery.
---
""")

# ===== Load YOLOv8 Detection Model =====
model = YOLO("deetection.pt")  # Replace with your trained model name

# ===== Upload Image =====
uploaded_image = st.file_uploader("Upload a Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")

    # Resize only if image is too large
    max_dim = 10000
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    image_np = np.array(image)
    image_path = "uploaded_image.jpg"
    image.save(image_path)

    # ===== Inference =====
    results = model(image_path)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # ===== Draw Bounding Boxes =====
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Convert back to RGB for display
    image_with_boxes = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(image_with_boxes, caption="🖼️ Image with Detected Trees", use_container_width=True)

    # ===== Size + CO2 Estimation =====
    output_data = []
    canopy_areas = []
    co2_total = 0
    class_counts = defaultdict(int)

    # New logic based on area ratio
    co2_map = {"S": 10, "M": 20, "L": 30}
    maturity_map = {"S": "likely young", "M": "semi-mature", "L": "mature"}
    image_area = image_bgr.shape[0] * image_bgr.shape[1]

    crop_dir = "tree_crops"
    os.makedirs(crop_dir, exist_ok=True)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        crop = image_bgr[y1:y2, x1:x2]
        bbox_area = (x2 - x1) * (y2 - y1)
        bbox_ratio = bbox_area / image_area

        # Device-independent size classification
        if bbox_ratio < 0.01:
            size_class = "S"
        elif bbox_ratio < 0.03:
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

    # ===== Display Summary =====
    st.success(f"Total Trees Detected: {len(boxes)}")
    st.info(f"Total Estimated CO₂ Sequestration: {co2_total:.2f} kg/year")
    st.write(f"Average Canopy Area: {np.mean(canopy_areas):.2f} px²")

    # ===== Pie Chart =====
    fig, ax = plt.subplots()
    ax.pie(class_counts.values(), labels=class_counts.keys(), autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    # ===== DataFrame & CSV =====
    df = pd.DataFrame(output_data)
    st.dataframe(df)

    csv_path = "tree_report.csv"
    df.to_csv(csv_path, index=False)

    # ===== Download ZIP =====
    zip_path = "tree_report_package.zip"
    with ZipFile(zip_path, 'w') as zipf:
        zipf.write(csv_path)
        for file_name in os.listdir(crop_dir):
            zipf.write(os.path.join(crop_dir, file_name), arcname=os.path.join("tree_crops", file_name))

    with open(zip_path, "rb") as f:
        st.download_button("📥 Download ZIP Report (CSV + Crops)", f, file_name="tree_report_package.zip")

    # ===== Clean up (optional) =====
    shutil.rmtree(crop_dir)
    os.remove(csv_path)
    os.remove(zip_path)
    os.remove(image_path)

# ===== Footer =====
st.markdown("---")
st.markdown("<center>Made with ❤️ by Mayank Kumar Sharma</center>", unsafe_allow_html=True)



