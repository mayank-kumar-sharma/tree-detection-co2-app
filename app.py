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
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Point, Polygon

# === Streamlit App Config ===
st.set_page_config(page_title="Tree Detection & CO₂ Estimation", layout="centered")
st.title("🌳 Tree Detection and CO₂ Estimation")
st.markdown("""
Upload a **satellite image**, draw a polygon to select your region of interest, and get:
- Detected trees inside that region
- Size classification (S / M / L)
- Estimated maturity
- Yearly CO₂ sequestration per tree
- Canopy area per tree
- Downloadable CSV + Cropped images ZIP

⚠️ **Note:** Use satellite images. Trees outside the polygon will be ignored.
---
""")

# === Load Model ===
model = YOLO("deetection.pt")  # make sure it's in root dir

# === Upload Image ===
uploaded_image = st.file_uploader("Upload a Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)
    image_path = "uploaded_image.jpg"
    image.save(image_path)

    st.image(image, caption="📷 Uploaded Satellite Image", use_container_width=True)

    # === Polygon Drawing Canvas ===
    st.subheader("✏️ Draw a Polygon to Select ROI (Region of Interest)")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        stroke_color="red",
        background_image=uploaded_image,
        height=image.height,
        width=image.width,
        drawing_mode="polygon",
        key="canvas"
    )

    if canvas_result.json_data and "objects" in canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        if objects:
            polygon_points = objects[0]["path"]
            polygon_coords = [(pt[1], pt[2]) for pt in polygon_points if pt[0] == "L"]
            if polygon_coords[0] != polygon_coords[-1]:
                polygon_coords.append(polygon_coords[0])  # close polygon
            polygon = Polygon(polygon_coords)

            # === Inference ===
            results = model(image_path)[0]
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # === Processing ===
            output_data = []
            canopy_areas = []
            co2_total = 0
            class_counts = defaultdict(int)

            size_map = {"S": (0, 400000), "M": (400000, 800000), "L": (800001, float("inf"))}
            co2_map = {"S": 10, "M": 20, "L": 30}
            maturity_map = {"S": "likely young", "M": "semi-mature", "L": "mature"}

            crop_dir = "tree_crops"
            os.makedirs(crop_dir, exist_ok=True)

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                if not polygon.contains(Point(center_x, center_y)):
                    continue

                crop = image_bgr[y1:y2, x1:x2]
                bbox_area = (x2 - x1) * (y2 - y1)

                size_class = "L" if bbox_area > 800000 else "M" if bbox_area > 400000 else "S"
                co2 = co2_map[size_class]
                maturity = maturity_map[size_class]

                crop_path = os.path.join(crop_dir, f"tree_{i+1}_{size_class}.jpg")
                cv2.imwrite(crop_path, crop)

                co2_total += co2
                class_counts[size_class] += 1
                canopy_areas.append(bbox_area)

                output_data.append({
                    "Tree #": i + 1,
                    "Size": size_class,
                    "Maturity": maturity,
                    "CO₂ (kg/year)": co2,
                    "Canopy Area (px²)": bbox_area
                })

            # === Results ===
            if output_data:
                st.success(f"✅ Total Trees Detected in Polygon: {len(output_data)}")
                st.info(f"🌱 Total Estimated CO₂: {co2_total:.2f} kg/year")
                st.write(f"🟩 Average Canopy Area: {round(np.mean(canopy_areas), 2)} px²")

                # Pie chart
                fig, ax = plt.subplots()
                ax.pie(class_counts.values(), labels=class_counts.keys(), autopct="%1.1f%%", startangle=90)
                ax.axis("equal")
                st.pyplot(fig)

                # CSV
                df = pd.DataFrame(output_data)
                st.dataframe(df)

                csv_path = "tree_report.csv"
                df.to_csv(csv_path, index=False)

                # ZIP
                zip_path = "tree_report_package.zip"
                with ZipFile(zip_path, 'w') as zipf:
                    zipf.write(csv_path)
                    for file_name in os.listdir(crop_dir):
                        zipf.write(os.path.join(crop_dir, file_name), arcname=os.path.join("tree_crops", file_name))

                with open(zip_path, "rb") as f:
                    st.download_button("📥 Download ZIP Report", f, file_name="tree_report_package.zip")

                shutil.rmtree(crop_dir)
                os.remove(csv_path)
                os.remove(zip_path)
                os.remove(image_path)
            else:
                st.warning("⚠️ No trees found inside the selected region.")
        else:
            st.warning("⚠️ Please draw a polygon on the image.")
    else:
        st.warning("🛑 Waiting for polygon to be drawn.")

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ by Mayank Kumar Sharma</center>", unsafe_allow_html=True)


