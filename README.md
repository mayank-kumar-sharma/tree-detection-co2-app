# 🌳 Tree Detection and CO₂ Estimation App

This Streamlit-powered application enables users to upload **satellite imagery** and automatically:

- Detect trees
- Classify each tree by **size** (Small / Medium / Large)
- Estimate tree **maturity**
- Calculate **CO₂ sequestration**
- Generate a **CSV report** and provide cropped tree images in a downloadable ZIP

---

## 🚀 Features

✅ **Tree Detection**  
Uses a YOLOv8 model trained on high-resolution satellite images to accurately detect individual tree canopies.

📏 **Size Classification**  
Each detected tree is classified into one of three categories based on its canopy (bounding box) area:

| Size | Canopy Area (px²)     | CO₂ Estimate (kg/year) | Maturity       |
|------|------------------------|------------------------|----------------|
| S    | 0 – 400,000            | 10                     | Likely Young   |
| M    | 400,001 – 800,000      | 20                     | Semi-Mature    |
| L    | 800,001 and above      | 30                     | Mature         |

📊 **Summary Outputs**

- 🌲 **Total Trees Detected**
- 🌱 **Average Canopy Area**
- 🌍 **Total Estimated CO₂ Sequestration**
- 📈 **Pie Chart** for tree size distribution

📁 **ZIP Download**  
Includes:
- `tree_report.csv` — details of each tree (size, maturity, CO₂, canopy area)
- A folder of cropped images for each detected tree

---

## 🔗 Live App

👉 [Click here to launch the app](https://tree-detection-co2-app-bgu6yqhyf3hb7rqvm4uncw.streamlit.app/)

---

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **Model**: YOLOv8 (Ultralytics)
- **Image Processing**: OpenCV, Pillow
- **Data Analysis**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Packaging**: zipfile, shutil
- **Deployment**: Streamlit Cloud

---

## 📸 Sample Output

Example summary after image upload:

- ✅ Trees Detected: 42  
- 📐 Average Canopy Area: 611,000 px²  
- 🌿 Estimated Total CO₂ Sequestration: 860 kg/year  
- 📊 Size Distribution Pie Chart  
- 📥 ZIP file with `CSV + Cropped Images`

---

💬 About the App
This project aims to demonstrate how AI + satellite imagery can help in environmental monitoring, urban forestry, and carbon sequestration estimation. The app is lightweight, fast, and optimized for practical use cases like:

Estimating green cover in urban areas

Supporting reforestation monitoring

Educational and research-based analysis

🤝 Acknowledgements
Ultralytics for YOLOv8

Roboflow for dataset preparation

Streamlit for the UI framework

OpenCV, NumPy, Matplotlib, and the Python open-source ecosystem

❤️ Made with Love
<p align="center">
  <b>Made with ❤️ by Mayank Kumar Sharma</b><br>
  🌱 Empowering environmental insights through AI & satellite vision.
</p>

