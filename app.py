import json
import os
import sys
from pathlib import Path
import streamlit as st
from PIL import Image

# Venv execution guard — fails loudly with a clear message if run outside virtualenv
_venv_marker = os.path.join(os.path.dirname(sys.executable), "pyvenv.cfg")
_venv_marker_parent = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "pyvenv.cfg")
if not (os.path.exists(_venv_marker) or os.path.exists(_venv_marker_parent) or hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
    st.error(
        "⚠️ **Python Environment Warning**: App may not be running inside the project virtual environment (`venv`).\n\n"
        "To ensure all native dependencies (`fitz`, `cv2`, `tesseract`) load properly, start the app with:\n"
        "```powershell\n"
        ".\\venv\\Scripts\\python.exe -m streamlit run app.py\n"
        "```"
    )

from src.ocr_config import configure_tesseract
from src.yolo_detector import get_yolo_detector
from src.pdf_processor import PDFProcessor, process_book
from src.qa_engine import answer_question
from src.storage import (
    clear_cache,
    ensure_directories,
    load_extraction_results,
    save_uploaded_file,
    PIPELINE_VERSION,
)

# Page configuration
st.set_page_config(
    page_title="Smart Book Reader",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure data folders
ensure_directories()

# Custom modern CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        color: #6B7280;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .answer-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .badge-src {
        background: #4F46E5;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        display: inline-block;
    }
    .badge-digital {
        background-color: #10B981;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
    }
    .badge-scanned {
        background-color: #F59E0B;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/book-stack.png", width=60)
    st.title("Smart Book Reader")

    tess_ok, tess_msg = configure_tesseract()
    if tess_ok:
        st.success("🟢 **Tesseract OCR Ready**")
    else:
        st.warning(f"⚠️ **OCR Limited**\n\n{tess_msg}")

    yolo_ok, _, yolo_msg = get_yolo_detector()
    if yolo_ok:
        st.success("🎯 **YOLOv8 Visual Engine Ready**")
    else:
        st.info("ℹ️ YOLO Visual Engine (Heuristic Fallback)")

    st.markdown("---")
    st.subheader("⚙️ Settings & Configuration")
    gemini_api_key = st.text_input(
        "API Key (Optional)",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Provide a free API key for conversational answers. Without it, direct evidence is extracted from the book.",
    )

    st.markdown("---")
    st.subheader("⚙️ Processing Settings")
    min_char_threshold = st.slider(
        "Digital vs. Scanned Threshold",
        min_value=10,
        max_value=200,
        value=40,
        help="Pages with fewer characters than this are rasterized and processed with OCR.",
    )
    ocr_dpi = st.select_slider(
        "OCR Image DPI",
        options=[150, 200, 300, 400],
        value=300,
    )

    st.markdown("---")
    st.subheader("🧹 Cache Management")
    st.caption(f"Pipeline Engine: **{PIPELINE_VERSION}**")
    if st.button("🔄 Re-process / Clear Cache", use_container_width=True):
        clear_cache()
        st.session_state.clear()
        st.toast("Cache purged! Re-processing book...", icon="🧹")
        st.rerun()

    st.markdown("---")
    st.caption("Smart Book Reader • Visual Document Explorer")

# Main Header
st.markdown('<div class="main-header">📖 Smart Book Reader</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload any book or PDF document to search, explore diagrams, and answer questions.</div>',
    unsafe_allow_html=True,
)

# Optional LLM Key Quick Setting in Main Header if not provided in sidebar
if not gemini_api_key:
    with st.expander("🔑 Optional: Add Gemini API Key for conversational answers (Click to expand)"):
        main_key = st.text_input(
            "Enter Gemini API Key (starts with AIzaSy...):",
            type="password",
            key="main_gemini_key_input",
            help="Get your free key at aistudio.google.com",
        )
        if main_key:
            gemini_api_key = main_key

# Upload Component
uploaded_file = st.file_uploader(
    "Choose a PDF file to analyze",
    type=["pdf"],
    help="Upload your document or book to begin.",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    saved_pdf_path = save_uploaded_file(file_bytes, uploaded_file.name)

    # Process Book Flow with Pipeline Version Hash
    cache_key = f"book_{saved_pdf_path.name}_{min_char_threshold}_{ocr_dpi}_{PIPELINE_VERSION}"

    if cache_key not in st.session_state:
        saved_data = load_extraction_results(uploaded_file.name, min_char_threshold=min_char_threshold)
        if saved_data:
            st.session_state[cache_key] = saved_data
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(current: int, total: int, msg: str):
                progress = min(1.0, current / max(1, total))
                progress_bar.progress(progress)
                status_text.text(f"⏳ {msg} ({int(progress * 100)}%)")

            book_data = process_book(
                pdf_path=saved_pdf_path,
                min_char_threshold=min_char_threshold,
                ocr_dpi=ocr_dpi,
                progress_callback=update_progress,
            )
            progress_bar.empty()
            status_text.empty()
            st.session_state[cache_key] = book_data
            st.toast("PDF Processing complete!", icon="✅")

    book_data = st.session_state[cache_key]
    pages_list = book_data.get("pages", [])

    # Overview Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pages", book_data.get("total_pages", len(pages_list)))
    with col2:
        st.metric("Digital Pages", book_data.get("digital_pages", 0))
    with col3:
        st.metric("Scanned Pages", book_data.get("scanned_pages", 0))
    with col4:
        total_figs = sum(len(p.get("figures", [])) for p in pages_list)
        st.metric("Extracted Figures", total_figs)

    st.markdown("---")

    # Q&A Interactive Section
    st.subheader("💬 Ask Questions About This Book")
    q_col1, q_col2 = st.columns([4, 1])
    with q_col1:
        user_question = st.text_input(
            "Enter your question:",
            placeholder="e.g., Oxygen enters the blood in the lungs, Movement of water during transpiration...",
            label_visibility="collapsed",
        )
    with q_col2:
        ask_clicked = st.button("🔎 Search Book", use_container_width=True, type="primary")

    if (user_question.strip() and ask_clicked) or (user_question.strip() and "last_q" in st.session_state and st.session_state["last_q"] == user_question):
        st.session_state["last_q"] = user_question

        with st.spinner("Retrieving evidence, ranking figures, and generating answer..."):
            qa_result = answer_question(
                query=user_question,
                book_data=book_data,
                api_key=gemini_api_key,
                top_k=5,
            )

        # Answer Display
        with st.container():
            st.markdown('<div class="answer-card">', unsafe_allow_html=True)
            st.markdown("### 💡 Answer")
            st.markdown(qa_result["answer"])

            # Source Pages
            if qa_result["source_pages"]:
                st.markdown("#### 📚 Source Citations")
                source_badges = " ".join(
                    [f'<span class="badge-src">📖 Page {p}</span>' for p in qa_result["source_pages"]]
                )
                st.markdown(source_badges, unsafe_allow_html=True)

            # Relevant Images & Visual Figures
            if qa_result.get("figures") or qa_result.get("images"):
                st.markdown("#### 🖼️ Associated Figures & Diagrams")
                figures = qa_result.get("figures", [])
                
                if figures:
                    cols = st.columns(min(len(figures), 3))
                    for idx, fig in enumerate(figures[:3]):
                        img_path = fig.get("image_path")
                        if img_path and os.path.exists(img_path):
                            label = fig.get("figure_label", "")
                            cap = fig.get("caption", "")
                            cap_display = f"**{label}**: {cap}" if label and cap else (cap or label)
                            with cols[idx % len(cols)]:
                                st.image(img_path, caption=cap_display, use_container_width=True)
                elif qa_result.get("images"):
                    cols = st.columns(min(len(qa_result["images"]), 3))
                    for idx, img_p in enumerate(qa_result["images"][:3]):
                        if os.path.exists(img_p):
                            with cols[idx % len(cols)]:
                                st.image(img_p, caption=Path(img_p).name, use_container_width=True)

            # Visual Evidence / Original Page Inspection
            if qa_result["evidence"]:
                with st.expander("🔍 Inspect Source Page Excerpts"):
                    for ev in qa_result["evidence"]:
                        st.markdown(f"**Page {ev['page_number']} ({ev.get('page_type', 'Digital')})**")
                        st.info(ev.get("text", ""))

            st.markdown('</div>', unsafe_allow_html=True)

    # Document Exploration Tabs
    st.markdown("---")
    tab_inspect, tab_text, tab_meta = st.tabs(["📄 Browse Pages", "📝 Full Text", "📊 Raw JSON"])

    with tab_inspect:
        selected_page_idx = st.selectbox(
            "Select Page",
            options=list(range(len(pages_list))),
            format_func=lambda idx: f"Page {idx + 1} ({pages_list[idx]['page_type']})",
        )
        p_data = pages_list[selected_page_idx]
        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            st.markdown(f"### Page {p_data['page_number']}")
            is_dig = p_data['page_type'] == 'Digital'
            badge_html = (
                '<span class="badge-digital">🟢 Digital Pipeline (Direct Text & Vectors)</span>'
                if is_dig else
                '<span class="badge-scanned">🔍 Scanned Pipeline (OCR & OpenCV)</span>'
            )
            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown(f"**Words**: {p_data.get('word_count', 0)} | **Chars**: {p_data.get('char_count', 0)}")
            p_imgs = p_data.get("images", [])
            if p_imgs:
                for img_p in p_imgs:
                    if os.path.exists(img_p):
                        st.image(img_p, caption=Path(img_p).name, use_container_width=True)
        with col_p2:
            st.text_area(
                "Page Text",
                value=p_data.get("text", ""),
                height=300,
                key=f"browse_page_{p_data['page_number']}",
            )

    with tab_text:
        all_text = "\n\n".join([f"--- PAGE {p['page_number']} ({p['page_type']}) ---\n{p['text']}" for p in pages_list])
        st.download_button(
            "📥 Download Full Text (.txt)",
            data=all_text,
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_full.txt",
            mime="text/plain",
        )
        st.text_area("Full Document Text", value=all_text, height=400)

    with tab_meta:
        json_str = json.dumps(book_data, indent=2, ensure_ascii=False)
        st.download_button(
            "📥 Download JSON Data",
            data=json_str,
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_data.json",
            mime="application/json",
        )
        st.json(book_data)

else:
    st.info("👆 Please upload a PDF file to begin.")
