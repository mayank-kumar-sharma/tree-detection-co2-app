import streamlit as st
from streamlit_drawable_canvas import st_canvas
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
from shapely.geometry import Point, Polygon
import base64
from io import BytesIO

st.set_page_config(page_title="Tree Detection & CO₂ Estimation", layout="centered")
st.title("🌳 Tree Detection and CO₂ Estimation")

st.markdown("""
Upload a **satellite image**, draw a polygon to select Region of Interest (ROI), and get tree detection, size (S/M/L), maturity, CO₂ sequestration, and a downloadable report.

---

**🔺 Only trees inside the polygon will be processed.**
""")

# Load YOLO model
model = YOLO("deetection.pt")  # Your model file

# Helper: Convert image to base64 URL for canvas
def image_to_data_url(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

uploaded_image = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_path = "uploaded_image.jpg"
    image.save(image_path)
    img_url = image_to_data_url(image)

    # Draw Polygon
    st.subheader("✏️ Draw Polygon ROI")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        stroke_color="red",
        background_image=img_url,
        height=image.height,
        width=image.width,
        drawing_mode="polygon",
        key="canvas",
    )

    if canvas_result.json_data and "objects" in canvas_result.json_data:
        polygons = canvas_result.json_data["objects"]
        if polygons:
            polygon_points = polygons[0]["path"]
            polygon_xy = [(pt[1], pt[2]) for pt in polygon_points if len(pt) == 3]
            st.success(f"Polygon with {len(polygon_xy)} points received!")

            # Convert to Shapely polygon
            roi_polygon = Polygon(polygon_xy)

            # Run detection
            results = model(image_path)[0]
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)

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
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_point = Point(center_x, center_y)

                if not roi_polygon.contains(center_point):
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
                    "Tree #": len(output_data)+1,
                    "Size": size_class,
                    "Maturity": maturity,
                    "CO₂ (kg/year)": co2,
                    "Canopy Area (px²)": bbox_area
                })

            # Show results
            st.success(f"✅ Trees inside polygon: {len(output_data)}")
            st.info(f"Estimated Total CO₂: {co2_total:.2f} kg/year")
            st.write(f"Average Canopy Area: {np.mean(canopy_areas) if canopy_areas else 0:.2f} px²")

            # Pie Chart
            fig, ax = plt.subplots()
            ax.pie(class_counts.values(), labels=class_counts.keys(), autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

            # DataFrame
            df = pd.DataFrame(output_data)
            st.dataframe(df)

            # CSV + ZIP
            csv_path = "tree_report.csv"
            df.to_csv(csv_path, index=False)

            zip_path = "tree_report_package.zip"
            with ZipFile(zip_path, 'w') as zipf:
                zipf.write(csv_path)
                for file in os.listdir(crop_dir):
                    zipf.write(os.path.join(crop_dir, file), arcname=os.path.join("tree_crops", file))

            with open(zip_path, "rb") as f:
                st.download_button("📥 Download ZIP Report", f, file_name="tree_report_package.zip")

            # Cleanup
            shutil.rmtree(crop_dir)
            os.remove(csv_path)
            os.remove(zip_path)
            os.remove(image_path)

        else:
            st.warning("⚠️ Please draw a polygon.")
    else:
        st.warning("⚠️ Use the tool above to draw a region before continuing.")

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ by Mayank Kumar Sharma</center>", unsafe_allow_html=True)

