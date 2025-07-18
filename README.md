# 🌳 Tree Detection & CO₂ Estimation App

This Streamlit-based application enables users to upload **satellite images** and automatically:

- Detect trees using a YOLOv8 model
- Classify them into Small (S), Medium (M), or Large (L) sizes using **device-independent area ratio**
- Estimate **tree maturity** and **CO₂ sequestration**
- Generate a downloadable report with cropped tree images and structured CSV data

---

## 🔗 Live App

👉 [Launch Now](https://tree-detection-co2-app-bgu6yqhyf3hb7rqvm4uncw.streamlit.app/)

---

## 🧠 How It Works

1. **Upload Satellite Image** (JPG, PNG)
2. App uses a **YOLOv8 model** (`deetection.pt`) to detect trees
3. Each bounding box is classified by **relative area ratio** to the entire image:
   - `S` (Small): Area ratio `< 1%`
   - `M` (Medium): Area ratio `1–2%`
   - `L` (Large): Area ratio `> 2%`
4. App estimates:
   - **Maturity**
   - **CO₂ Sequestration Potential**
5. App generates:
   - A **summary table**
   - A **pie chart** of size distribution
   - A **ZIP file** containing:
     - `tree_report.csv`
     - Cropped tree images sorted by class

---

## 📏 Size Classification & CO₂ Mapping

| Size Class | Area Ratio      | CO₂ Estimate (kg/year) | Maturity       |
|------------|------------------|------------------------|----------------|
| S          | `< 1%`           | 10                     | Likely Young   |
| M          | `1–2%`           | 20                     | Semi-Mature    |
| L          | `> 2%`           | 30                     | Mature         |

This dynamic method ensures consistent results **across all devices** regardless of screen resolution or image size.

---

## 📁 Download Package Includes

- **`tree_report.csv`** — Detailed table with tree number, size, maturity, CO₂ estimate, and canopy area.
- **Cropped Tree Images** — Each saved as `tree_#_size.jpg` inside `tree_crops/`.

All files are bundled into a single **ZIP file**, downloadable directly from the app.

---

## 🛠 Tech Stack

- **Model**: YOLOv8 via Ultralytics
- **Frontend**: Streamlit
- **Image Processing**: OpenCV, PIL
- **Data Handling**: Pandas, NumPy
- **Charting**: Matplotlib
- **Packaging**: ZipFile, Shutil

---

## 🌍 Use Cases

- Estimating carbon capture from urban tree canopies
- Environmental monitoring via satellite imagery
- Eco-conscious urban planning
- Research on vegetation distribution

---

## 🎯 Sample Output

- ✅ Total Trees Detected: 34  
- 🌱 Total CO₂ Sequestration: 740 kg/year  
- 📐 Average Canopy Area: 0.014 image ratio  
- 📊 Size Distribution: Pie chart (S/M/L)  
- 📥 ZIP file: Includes CSV + cropped images

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [PIL (Pillow)](https://python-pillow.org/)

---

### ❤️ Made with Love  
<p align="center"><b>Made with ❤️ by Mayank Kumar Sharma</b><br>🌱 Empowering environmental insights through AI & satellite vision.</p>


