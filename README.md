# 📖 Smart Book Reader

> An intelligent, visual document exploration and question-answering system designed for complex textbooks, scientific papers, and scanned documents.

---

## 🌟 Overview

**Smart Book Reader** bridges the gap between text extraction and visual comprehension. Traditional PDF readers only parse plain text, missing diagrams, apparatus setups, and scientific figures. Smart Book Reader uses a **Hybrid Multi-Lane Pipeline** combining:

- **PyMuPDF**: High-speed digital text & vector graphics extraction.
- **Tesseract OCR**: Full-text and coordinate layout parsing for scanned pages.
- **YOLOv8 Deep Learning**: Visual object detection for apparatus, charts, diagrams, and figures.
- **OpenCV**: Pixel-level morphological boundary snapping and text masking.
- **Gemini LLM**: Strict, grounded question-answering with conceptual reasoning and source page citations.

---

## 🚀 Key Features

- **Automatic Two-Lane Routing**: Dynamically inspects each page's embedded character density to route digital pages through native text extraction (sub-second speed) and scanned image pages through high-resolution OCR.
- **YOLOv8 Visual Region Detection**: Uses deep learning to localize apparatus, experiments, and visual figures without needing hardcoded coordinate heuristics.
- **Multi-Signal Figure Retrieval**: Ranks visual diagrams using a 5-component weighted scoring formula (internal diagram labels, caption matches, citation proximity, surrounding context, and conceptual intent).
- **LLM-Assisted Visual Reranker**: Understands semantic intent to surface the exact relevant diagram(s) for a query (e.g., matching *"burning magnesium ribbon"* to *Activity 1.1 / Figure 1.1*).
- **Conceptual Deduction & Citations**: Answers chapter exercises and reasoning questions by connecting concepts to book context while always citing source page numbers (e.g., `[Page 1]`, `[Page 6]`).
- **Offline Extractive Fallback**: Operates fully offline without an API key by generating structured direct evidence snippets from matched pages.

---

## 🏗️ Architecture & Pipeline Flow

```
                               ┌─────────────────────────┐
                               │     Uploaded PDF        │
                               └────────────┬────────────┘
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │ Page Classification & Inspection│
                           │   (Character Density Analysis)  │
                           └────────┬───────────────┬────────┘
                                    │               │
                 [>= 40 chars]      │               │ [< 40 chars]
                 Digital Page       │               │ Scanned Page
                                    ▼               ▼
                      ┌──────────────────┐    ┌──────────────────┐
                      │ Digital Pipeline │    │ Scanned Pipeline │
                      │ - Vector Paths   │    │ - 300 DPI Render │
                      │ - Font Streams   │    │ - Tesseract OCR  │
                      │ - PyMuPDF Images │    │ - OpenCV Contours│
                      └────────┬─────────┘    └────────┬─────────┘
                               │                       │
                               └───────────┬───────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────┐
                            │  YOLOv8 Visual Detection    │
                            │  (Bounding Box Predictions) │
                            └──────────────┬──────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────┐
                            │  Structured PageResult JSON │
                            └──────────────┬──────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        │                                     │
                        ▼                                     ▼
           ┌─────────────────────────┐           ┌─────────────────────────┐
           │ 6-Layer Text Retrieval  │           │ 5-Signal Figure Scoring │
           │ - Exact & Stem Matching │           │ - Semantic Labels (40%) │
           │ - Substring & Prefix    │           │ - Caption Match (25%)   │
           │ - Bigram Co-occurrence  │           │ - Citation Window (20%) │
           └────────────┬────────────┘           └────────────┬────────────┘
                        │                                     │
                        └──────────────────┬──────────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────┐
                            │   LLM Visual Selection &    │
                            │ Grounded Answer Generation  │
                            └──────────────┬──────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────┐
                            │ Interactive Streamlit View  │
                            └─────────────────────────────┘
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites

- **Python**: Version 3.10, 3.11, 3.12, or 3.13
- **Tesseract OCR**:
  - **Windows**: Download installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Add to PATH or standard install location (`C:\Program Files\Tesseract-OCR\tesseract.exe`).
  - **Linux (Ubuntu/Debian)**: `sudo apt-get install tesseract-ocr`
  - **macOS**: `brew install tesseract`

---

### 2. Clone the Repository

```bash
git clone https://github.com/pushkaranand07/Smart-Book-Reader.git
cd Smart-Book-Reader
```

---

### 3. Create & Activate Virtual Environment

```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

*(YOLOv8 model weights `yolov8n.pt` will automatically download on the first run).*

---

### 5. (Optional) Set Google Gemini API Key

For natural language synthesis and LLM-assisted figure ranking:

```powershell
# Windows (PowerShell)
$env:GEMINI_API_KEY="your-api-key-here"

# Linux / macOS
export GEMINI_API_KEY="your-api-key-here"
```

> **Note**: You can also enter your API key directly inside the Streamlit sidebar at runtime.

---

## 🖥️ Running the Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`.

---

## 📁 Project Structure

```
Smart-Book-Reader/
├── app.py                      # Main Streamlit user interface
├── requirements.txt            # Locked project dependencies
├── README.md                   # Documentation & setup guide
├── .gitignore                  # Git ignore rules
│
├── src/                        # Core application modules
│   ├── __init__.py
│   ├── pdf_processor.py        # Pipeline orchestrator & Two-Lane classifier
│   ├── digital_pipeline.py     # Native PyMuPDF text & vector drawing engine
│   ├── scanned_pipeline.py     # Tesseract OCR & OpenCV contour extractor
│   ├── yolo_detector.py        # YOLOv8 visual diagram & layout analyzer
│   ├── search.py               # 6-layer text relevance & snippet ranking
│   ├── qa_engine.py            # Evidence assembly, LLM visual selector & QA
│   ├── ocr_config.py           # Cross-platform Tesseract auto-discovery
│   └── storage.py              # File caching & directory management
│
└── data/                       # Local data directories
    ├── uploads/                # Uploaded PDF documents
    ├── extracted/              # Processed PageResult JSON cache
    ├── images/                 # Cropped figures & diagram PNGs
    └── pages/                  # Page preview thumbnails
```

---

## 🔬 Multi-Signal Figure Ranking Formula

Each extracted figure is evaluated using a composite multi-signal formula scaled from 0 to 100:

$$\text{Score} = \left( 0.40 \cdot S_{\text{semantic}} + 0.25 \cdot S_{\text{caption}} + 0.20 \cdot S_{\text{citation}} + 0.10 \cdot S_{\text{context}} + 0.05 \cdot S_{\text{type}} \right) \times 100$$

- **$S_{\text{semantic}}$ (40%)**: Matches query keywords against internal diagram text labels.
- **$S_{\text{caption}}$ (25%)**: Evaluates hits in the formal figure caption (`Figure 1.1`).
- **$S_{\text{citation}}$ (20%)**: Keyword proximity in sentences where the text cites the figure.
- **$S_{\text{context}}$ (10%)**: Paragraph context directly above or below the figure bounding box.
- **$S_{\text{type}}$ (5%)**: Intent alignment (process/mechanism vs. structure/apparatus).

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
