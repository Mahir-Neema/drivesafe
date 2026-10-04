import asyncio
import base64
import html
import json
import os
import secrets
from dataclasses import asdict
from pathlib import Path

import streamlit as st
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from drivesafe.activities import (
    analyze_window_activity,
    export_clip_activity,
    extract_windows_activity,
    save_session_activity,
)
from drivesafe.config import (
    DATA_ROOT, MAX_UPLOAD_BYTES, MAX_VIDEO_SECONDS, MODEL_NAME,
    ensure_data_root, ENGINE, GOOGLE_CLOUD_MODEL, GOOGLE_CLOUD_PROJECT,
    GEMINI_API_KEY, WINDOW_SECONDS, SAMPLES_PER_WINDOW, CONTEXT_SECONDS,
)
from drivesafe.gemma import check_model_ready, describe_window
from drivesafe.models import AnalysisSession, WindowResult
from drivesafe.storage import save_session, delete_session, session_dir
from drivesafe.temporal_client import TASK_QUEUE
from drivesafe.video import VideoError, get_video_info, make_windows, export_clip, ffmpeg_path
from drivesafe.workflows import AnalysisWorkflowInput, VideoAnalysisWorkflow

LOGO_PATH = Path(__file__).resolve().parent / "DriveSafe_logo.png"
if not LOGO_PATH.is_file():
    LOGO_PATH = Path("DriveSafe_logo.png")

page_icon_val = str(LOGO_PATH.resolve()) if LOGO_PATH.is_file() else "🚗"

# ---------------------------------------------------------------------------
# Streamlit Page Config (Tab Favicon & Title)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DriveSafe — Durable Dashcam Review with Gemma 4 & Temporal",
    page_icon=page_icon_val,
    layout="wide",
    initial_sidebar_state="collapsed",
)

LOGO_B64 = ""
if LOGO_PATH.is_file():
    try:
        LOGO_B64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    except Exception:
        LOGO_B64 = ""

APP_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap");

:root {
  --canvas: #FBFAF8; --surface: #FFFFFF; --tint: #F4F2EF;
  --inverse: #0E1014; --inverse-2: #16181D; --inverse-rule: #23262E;
  --ink: #16181D; --body: #4B525C; --faint: #8B929C;
  --rule: #E6E3DE; --rule-strong: #D6D2CB;
  --volt: #2558E6; --volt-soft: #EAEFFD; --volt-lite: #7CA0FF;
  --sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, monospace;
}

/* Global Reset */
*, *::before, *::after { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  background: var(--canvas) !important;
  color: var(--ink);
  font-family: var(--sans);
  overflow-x: hidden !important;
}

/* Hide Streamlit chrome */
[data-testid="stHeader"], #MainMenu, footer, .stDeployButton { display: none !important; }

/* Constrain and Center Entire App Container */
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"],
[data-testid="block-container"] {
  max-width: 1080px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-top: 40px !important;
  padding-bottom: 60px !important;
  padding-left: 24px !important;
  padding-right: 24px !important;
}
@media (max-width: 640px) {
  [data-testid="block-container"] {
    padding-left: 16px !important;
    padding-right: 16px !important;
    padding-top: 20px !important;
  }
}

section[data-testid="stSidebar"] { display: none !important; }

/* Hero Section */
.ds-hero {
  padding: 10px 0 28px;
}
.ds-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--volt-soft); color: var(--volt);
  font-family: var(--mono); font-size: 11px; font-weight: 600;
  letter-spacing: .08em; text-transform: uppercase;
  padding: 6px 14px; border-radius: 999px; margin-bottom: 18px;
}
.ds-badge-dot { width: 6px; height: 6px; border-radius: 999px; background: var(--volt); }
.ds-hero h1 {
  margin: 0 0 16px;
  font-size: clamp(32px, 4.8vw, 54px);
  font-weight: 700;
  line-height: 1.12;
  letter-spacing: -0.03em;
  color: var(--ink);
  word-break: normal;
  overflow-wrap: break-word;
}
.ds-lede {
  margin: 0;
  font-size: 17px;
  line-height: 1.6;
  color: var(--body);
  max-width: 65ch;
}

/* Radio Widget Container */
div[data-testid="stRadio"] > div {
  gap: 8px !important;
  background: var(--tint);
  padding: 6px;
  border-radius: 10px;
  border: 1px solid var(--rule);
  flex-wrap: wrap;
}
div[data-testid="stRadio"] label {
  font-size: 13px !important;
  font-weight: 500 !important;
}

/* Buttons */
div.stButton > button[kind="primary"] {
  background: var(--volt) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 10px 20px !important;
  box-shadow: 0 4px 12px rgba(37, 88, 230, 0.2) !important;
  transition: all .15s ease !important;
}
div.stButton > button[kind="primary"]:hover {
  background: #1e46c7 !important;
  box-shadow: 0 6px 16px rgba(37, 88, 230, 0.3) !important;
}
div.stButton > button {
  background: var(--surface) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 10px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--ink) !important;
  transition: all .15s ease !important;
}
div.stButton > button:hover {
  border-color: var(--rule-strong) !important;
  background: var(--tint) !important;
}

/* Spec Card */
.ds-spec {
  background: var(--surface); border: 1px solid var(--rule);
  border-radius: 12px; padding: 20px;
  box-shadow: 0 2px 8px rgba(22,24,29,.03);
}
.ds-spec-title {
  font: 600 11px var(--mono); letter-spacing: 0.1em;
  color: var(--faint); text-transform: uppercase; margin-bottom: 14px;
}
.ds-spec-row {
  display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;
  font: 13px var(--sans);
}
.ds-spec-row span:first-child { color: var(--faint); }
.ds-spec-row span:last-child { font-weight: 500; color: var(--ink); }

/* Inspector */
.ds-inspector {
  background: var(--surface); border: 1px solid var(--rule-strong);
  border-radius: 16px; overflow: hidden; margin-top: 32px; margin-bottom: 24px;
  box-shadow: 0 8px 24px rgba(22,24,29,.05);
}
.ds-insp-bar {
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;
  padding: 12px 18px; border-bottom: 1px solid var(--rule); background: var(--tint);
}
.ds-file-name { font: 600 13px var(--mono); color: var(--ink); word-break: break-all; }
.ds-chip {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--surface); border: 1px solid var(--rule);
  padding: 4px 10px; border-radius: 999px;
  font: 500 11px var(--mono); color: var(--ink);
}
.ds-chip--dim { color: var(--body); background: transparent; border-color: transparent; padding: 4px 0; }
.ds-live { width: 6px; height: 6px; border-radius: 50%; background: var(--volt); display: inline-block; }
.ds-insp-stage { padding: 28px 20px 36px; background: var(--surface); }

/* Timeline track */
.ds-tl-wrap { position: relative; margin: 0 auto; width: 100%; max-width: 860px; }
.ds-tl-track {
  position: relative; height: 10px; border-radius: 5px;
  background: var(--tint); border: 1px solid var(--rule); margin-bottom: 10px;
}
.ds-tl-fill { position: absolute; left: 0; top: 0; bottom: 0; width: 100%; background: var(--rule); border-radius: 5px; opacity: 0.3; }
.ds-tl-pin {
  position: absolute; top: -5px; width: 14px; height: 18px;
  margin-left: -7px; display: flex; flex-direction: column;
  align-items: center; z-index: 10; cursor: pointer;
}
.ds-tl-pin-dot {
  width: 14px; height: 14px; border-radius: 50%;
  border: 2px solid var(--surface); box-shadow: 0 1px 4px rgba(0,0,0,.25);
}
.ds-tl-pin-label {
  position: absolute; top: -20px;
  font: 600 10px var(--mono); color: var(--ink);
  white-space: nowrap; background: var(--surface); padding: 2px 4px; border-radius: 4px;
  border: 1px solid var(--rule); box-shadow: 0 1px 3px rgba(0,0,0,.08);
}
.ds-tl-tick {
  position: absolute; top: 0;
  font: 500 10px var(--mono); color: var(--faint);
  transform: translateX(-50%);
}
.ds-tl-tick::before {
  content: ""; position: absolute; top: -8px; left: 50%;
  width: 1px; height: 4px; background: var(--rule-strong);
}

.ds-insp-foot {
  display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
  padding: 10px 18px; border-top: 1px solid var(--rule); background: var(--tint);
  font: 500 11px var(--mono); text-transform: uppercase;
  letter-spacing: .06em; color: var(--faint);
}
.ds-insp-foot .ds-ok { color: var(--volt); font-weight: 600; }

/* Streamlit Expander styling */
div[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 12px !important;
  margin-bottom: 12px !important;
  box-shadow: 0 1px 4px rgba(22,24,29,.02) !important;
}
div[data-testid="stExpander"] summary {
  padding: 12px 18px !important;
  font-family: var(--sans) !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  color: var(--ink) !important;
}
div[data-testid="stExpanderDetails"] {
  padding: 16px 18px 20px !important;
  border-top: 1px solid var(--rule) !important;
}

/* Landing Page Sections */
.section { padding: 56px 0; border-top: 1px solid var(--rule); }
.section-head { max-width: 640px; margin-bottom: 36px; }
.section-label { font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--faint); margin-bottom: 10px; }
.section-head h2 { font-size: clamp(24px, 3.5vw, 34px); font-weight: 700; line-height: 1.15; letter-spacing: -0.02em; margin: 0 0 12px; }
.section-head p { color: var(--body); font-size: 15px; line-height: 1.55; margin: 0; }

.steps-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px;
}
.step-card {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 14px; padding: 20px;
  display: flex; flex-direction: column; justify-content: space-between;
}
.step-card-shot {
  aspect-ratio: 22/14; border: 1px solid var(--rule); border-radius: 8px; background: var(--tint);
  overflow: hidden; display: flex; align-items: center; justify-content: center; margin-bottom: 18px;
}
.step-card-shot svg { width: 100%; height: 100%; }
.step-num { font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .12em; color: var(--faint); margin-bottom: 6px; }
.step-card h3 { font-size: 17px; font-weight: 600; margin: 0 0 6px; color: var(--ink); }
.step-card p { font-size: 13px; line-height: 1.55; color: var(--body); margin: 0; }

.chrono {
  border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
  padding: 14px 0; background: var(--tint); text-align: center;
  font-family: var(--mono); font-size: 12px; color: var(--faint); letter-spacing: .08em; text-transform: uppercase;
  margin: 40px 0; border-radius: 8px;
}
.chrono-tag { color: var(--volt); font-weight: 600; }

/* Privacy Section */
.privacy-wrap {
  background: var(--inverse); border-radius: 16px; padding: 40px 32px; color: #fff;
  display: grid; grid-template-columns: 1fr 1fr; gap: 36px; align-items: center;
}
@media (max-width: 768px) {
  .privacy-wrap { grid-template-columns: 1fr; padding: 28px 18px; }
}
.privacy-wrap h2 { font-size: clamp(22px, 3.2vw, 30px); font-weight: 700; line-height: 1.2; margin: 0 0 14px; color: #fff; }
.privacy-wrap p { color: #9AA1AB; font-size: 14px; line-height: 1.6; margin: 0 0 20px; }
.privacy-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.privacy-chip {
  display: inline-flex; align-items: center; gap: 6px;
  border: 1px solid var(--inverse-rule); border-radius: 999px; padding: 5px 12px;
  font-family: var(--mono); font-size: 11px; color: #fff; background: var(--inverse-2);
}
.terminal {
  background: var(--inverse-2); border: 1px solid var(--inverse-rule); border-radius: 10px;
  padding: 18px; font-family: var(--mono); font-size: 12px; line-height: 1.8; overflow-x: auto;
}
.terminal .prompt { color: var(--volt-lite); }
.terminal .out { color: #D7DBE0; }
.terminal .dim { color: #6B727C; }
.terminal .zero { color: #4ADE80; }

/* Footer */
.ds-footer {
  border-top: 1px solid var(--rule); padding: 48px 0 24px; margin-top: 48px;
}
.ds-footer-grid {
  display: grid; grid-template-columns: 2fr 1fr 1fr 1.5fr; gap: 32px; margin-bottom: 36px;
}
@media (max-width: 860px) {
  .ds-footer-grid { grid-template-columns: 1fr 1fr; gap: 24px; }
}
@media (max-width: 540px) {
  .ds-footer-grid { grid-template-columns: 1fr; gap: 20px; }
}
.ds-footer-brand h3 { font-family: var(--mono); font-size: 15px; font-weight: 700; letter-spacing: .1em; color: var(--ink); margin: 0 0 10px; }
.ds-footer-brand p { font-size: 13px; line-height: 1.55; color: var(--body); margin: 0; }
.ds-footer-col h4 { font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--faint); margin: 0 0 12px; }
.ds-footer-col ul { list-style: none; padding: 0; margin: 0; }
.ds-footer-col li { margin-bottom: 8px; font-size: 13px; color: var(--body); }
.ds-footer-tags { display: flex; flex-wrap: wrap; gap: 6px; }
.ds-footer-tag { font-family: var(--mono); font-size: 10px; background: var(--tint); border: 1px solid var(--rule); padding: 3px 8px; border-radius: 4px; color: var(--body); }
.ds-footer-bottom {
  border-top: 1px solid var(--rule); padding-top: 18px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
  font-family: var(--mono); font-size: 12px; color: var(--faint);
}

/* Mobile Stacking for Columns */
@media (max-width: 768px) {
  [data-testid="column"] {
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 100% !important;
    margin-bottom: 16px;
  }
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_time(seconds: float) -> str:
    value = max(0, int(seconds))
    return f"{value // 60:02d}:{value % 60:02d}"


def human_label(value: str) -> str:
    return {
        "possible_close_encounter": "Close encounter",
        "pedestrian_or_cyclist": "Pedestrian / cyclist",
        "sudden_visual_change": "Sudden visual change",
        "unclear": "Needs a closer look",
    }.get(value, value.replace("_", " ").capitalize())


def pin_color(event_type: str) -> str:
    return {
        "possible_close_encounter": "#ef4444",
        "pedestrian_or_cyclist": "#f59e0b",
        "sudden_visual_change": "#2558E6",
        "unclear": "#8B929C",
    }.get(event_type, "#8B929C")


def clear_active_session() -> None:
    session_id = st.session_state.get("session_id")
    if session_id and not session_id.startswith("sample"):
        delete_session(session_id)
    for key in ("session_id", "source_path", "analysis", "upload_token", "selected_event"):
        st.session_state.pop(key, None)


async def run_temporal_analysis(
    input_data: AnalysisWorkflowInput,
    status_el,
    progress_el,
) -> dict:
    """Executes the Temporal VideoAnalysisWorkflow with live progress reporting."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[VideoAnalysisWorkflow],
            activities=[
                extract_windows_activity,
                analyze_window_activity,
                export_clip_activity,
                save_session_activity,
            ],
        ):
            handle = await env.client.start_workflow(
                VideoAnalysisWorkflow.run,
                input_data,
                id=f"workflow-{input_data.session_id}",
                task_queue=TASK_QUEUE,
            )

            while True:
                prog = await handle.query(VideoAnalysisWorkflow.get_progress)
                percent = prog.get("percent", 0) / 100.0
                status_msg = prog.get("status_message", "Processing video...")
                status_el.info(f"Temporal Workflow: {status_msg}")
                progress_el.progress(min(1.0, max(0.0, percent)))

                if prog.get("is_completed") or prog.get("error"):
                    break
                await asyncio.sleep(0.3)

            return await handle.result()


# ---------------------------------------------------------------------------
# API Key Modal Dialog
# ---------------------------------------------------------------------------
@st.dialog("🔑 Google Cloud & Gemini API Settings")
def api_key_modal():
    current_key = st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY", "")
    
    st.markdown(
        """
        DriveSafe runs **100% locally and privately** by default using Ollama & MongoDB.
        
        To unlock Google's massive **Gemma 4 26B** cloud vision model, paste your free Gemini API key below:
        
        🔗 **[Get your free API key at Google AI Studio &rarr;](https://aistudio.google.com/app/apikey)**
        """
    )
    
    user_key = st.text_input(
        "Gemini API Key",
        value=current_key,
        type="password",
        placeholder="Paste your AI Studio API key here...",
        help="Used to authenticate Gemma 4 26B cloud vision requests.",
    )
    
    c_save, c_clear = st.columns(2)
    with c_save:
        if st.button("Save & Enable Cloud Model", type="primary", use_container_width=True):
            clean = user_key.strip()
            if clean:
                st.session_state["gemini_api_key"] = clean
                os.environ["GEMINI_API_KEY"] = clean
                st.success("API key saved! Cloud Gemma 4 26B is now enabled.")
            else:
                st.session_state.pop("gemini_api_key", None)
                os.environ.pop("GEMINI_API_KEY", None)
            st.rerun()
            
    with c_clear:
        if st.button("Clear (Use 100% Local)", use_container_width=True):
            st.session_state.pop("gemini_api_key", None)
            os.environ.pop("GEMINI_API_KEY", None)
            st.info("Reverted to local edge mode.")
            st.rerun()


# ---------------------------------------------------------------------------
# Top Navbar & Hero Header
# ---------------------------------------------------------------------------
logo_img_tag = (
    f'<img src="data:image/png;base64,{LOGO_B64}" alt="DriveSafe Logo" style="height: 38px; width: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">'
    if LOGO_B64
    else ""
)

nav_col1, nav_col2 = st.columns([2.6, 1.4])
with nav_col1:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
          {logo_img_tag}
          <div>
            <div style="font-family: var(--mono); font-size: 18px; font-weight: 700; letter-spacing: 0.12em; color: var(--ink); line-height: 1;">DRIVESAFE</div>
            <div style="font-size: 10px; color: var(--faint); font-family: var(--mono); font-weight: 500; letter-spacing: 0.08em; margin-top: 3px;">DURABLE DASHCAM VISION &amp; VECTOR SEARCH</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col2:
    active_key = st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY", "")
    btn_label = "☁️ Cloud Key: Active" if active_key else "🔑 Add Cloud API Key"
    if st.button(btn_label, use_container_width=True, help="Configure your free Google AI Studio API key to use Gemma 4 26B cloud model"):
        api_key_modal()

st.markdown(
    f"""
    <div class="ds-hero" style="padding-top: 14px;">
      <span class="ds-badge"><span class="ds-badge-dot"></span>Powered by Temporal, Gemma 4 &amp; MongoDB Atlas</span>
      <h1>Understand what happened on the road.</h1>
      <p class="ds-lede">DriveSafe reviews dashcam footage using durable Temporal workflows with Google's Gemma 4 vision models (Cloud 26B or Local Edge E2B via Ollama). Events are saved to MongoDB Atlas for long-term vector search.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Upload & Controls Section
# ---------------------------------------------------------------------------
left, right = st.columns([1.3, 0.7], gap="large")

with left:
    st.markdown("**Inference Engine & Model**")
    engine_choice = st.radio(
        "Inference Engine",
        options=[
            "☁️ Cloud — Gemma 4 26B (Gemini API)",
            "💻 Local Edge — Gemma 4 E2B (Ollama)",
        ],
        index=0 if bool(GEMINI_API_KEY) else 1,
        horizontal=True,
        label_visibility="collapsed",
    )

    if "Cloud" in engine_choice:
        selected_engine = "gemini_api"
        selected_model = GOOGLE_CLOUD_MODEL
        endpoint_text = "Gemini API (Cloud)"
    else:
        selected_engine = "ollama"
        selected_model = MODEL_NAME
        endpoint_text = "127.0.0.1:11434 (Local)"

    ready, model_status = check_model_ready(engine=selected_engine, model_name=selected_model)
    ffmpeg_ready = ffmpeg_path() is not None

    st.markdown("**Upload a recording**")
    uploaded = st.file_uploader(
        "Choose a dashcam video",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed",
    )
    upload_token = f"{uploaded.name}:{uploaded.size}:{selected_engine}" if uploaded else None
    if upload_token and st.session_state.get("upload_token") not in (None, upload_token):
        clear_active_session()
    if upload_token:
        st.session_state["upload_token"] = upload_token
        size_mb = uploaded.size / (1024 * 1024)
        st.caption(f"**{uploaded.name}** &middot; {size_mb:.1f} MB")

    max_minutes = 60 if selected_engine == "ollama" else 5
    max_duration_seconds = max_minutes * 60
    limit_note = "Up to 60 min (Local Edge)" if selected_engine == "ollama" else "Up to 5 min (Cloud API)"
    st.caption(f"MP4, MOV, AVI or MKV &middot; {limit_note} &middot; Orchestrated by Temporal &amp; saved to MongoDB.")

    analyze_clicked = st.button(
        f"Analyze this clip ({'Cloud 26B' if selected_engine == 'gemini_api' else 'Local E2B'})",
        type="primary",
        disabled=not bool(uploaded) or not ready or not ffmpeg_ready,
        use_container_width=True,
    )

    st.markdown("**Or try a sample video:**")
    scol1, scol2 = st.columns(2)
    sample_to_use = None
    with scol1:
        if st.button("Traffic / Near Miss (30s)", use_container_width=True, disabled=not ready):
            sample_to_use = "person-bicycle-car-detection.mp4"
    with scol2:
        if st.button("Dashcam Accident (4m)", use_container_width=True, disabled=not ready):
            sample_to_use = "accident_sample.mp4"

    if sample_to_use:
        sample_path = Path(sample_to_use)
        if sample_path.is_file():
            class _MockUpload:
                def __init__(self, p):
                    self.name = p.name
                    self.size = p.stat().st_size
                    self._path = p
                    self._is_sample = True
                def getvalue(self):
                    return self._path.read_bytes()
                def __bool__(self):
                    return True
            uploaded = _MockUpload(sample_path)
            analyze_clicked = True

    # --- Run Analysis ---
    if analyze_clicked and uploaded:
        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error("This file exceeds the size limit. Choose a shorter or smaller clip.")
        else:
            is_sample = getattr(uploaded, "_is_sample", False)
            if is_sample:
                session_id = "sample" + "".join(c for c in Path(uploaded.name).stem if c.isalnum())
            else:
                session_id = secrets.token_hex(8)

            suffix = Path(uploaded.name).suffix.lower()
            suffix = suffix if suffix in {".mp4", ".mov", ".avi", ".mkv"} else ".mp4"
            folder = session_dir(session_id)
            session_json_path = folder / "session.json"
            source_path = folder / f"source{suffix}"

            if is_sample and session_json_path.exists():
                with open(session_json_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                st.session_state["analysis"] = cached_data
                st.session_state["session_id"] = session_id
                st.session_state["source_path"] = str(source_path)
            else:
                ensure_data_root()
                incoming_path = DATA_ROOT / f".incoming-{session_id}{suffix}"
                incoming_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    incoming_path.write_bytes(uploaded.getvalue())
                    duration, _, _ = get_video_info(incoming_path)
                    if duration > max_duration_seconds:
                        raise VideoError(
                            f"Videos up to {max_minutes} minutes are supported for this engine. "
                            f"This video is {duration / 60:.1f} minutes ({duration:.0f}s)."
                        )
                    folder.mkdir(parents=True, exist_ok=True)
                    incoming_path.replace(source_path)

                    st.session_state["session_id"] = session_id
                    st.session_state["source_path"] = str(source_path)

                    status_el = st.empty()
                    progress_el = st.progress(0)

                    status_el.info(f"Initializing durable Temporal workflow ({selected_model})...")
                    input_data = AnalysisWorkflowInput(
                        session_id=session_id,
                        source_name=uploaded.name,
                        source_path_str=str(source_path.resolve()),
                        duration_seconds=duration,
                        model_name=selected_model,
                        engine=selected_engine,
                        window_seconds=WINDOW_SECONDS,
                        samples_per_window=2 if selected_engine == "ollama" else SAMPLES_PER_WINDOW,
                        context_seconds=CONTEXT_SECONDS,
                    )

                    final_analysis_dict = asyncio.run(
                        run_temporal_analysis(input_data, status_el, progress_el)
                    )

                    st.session_state["analysis"] = final_analysis_dict
                    status_el.empty()
                    progress_el.empty()
                    st.success(f"Review ready — {len(final_analysis_dict.get('events', []))} event(s) found (Temporal Durable Execution)")
                except (VideoError, OSError, ValueError, Exception) as exc:
                    incoming_path.unlink(missing_ok=True)
                    if st.session_state.get("session_id") == session_id:
                        clear_active_session()
                    st.error(str(exc))

with right:
    st.markdown(
        f"""
        <div class="ds-spec">
          <div class="ds-spec-title">Runtime Spec</div>
          <div class="ds-spec-row"><span>Status</span><span>{"ready" if ready and ffmpeg_ready else "offline"}</span></div>
          <div class="ds-spec-row"><span>Orchestrator</span><span>Temporal (Durable)</span></div>
          <div class="ds-spec-row"><span>Engine</span><span>{endpoint_text}</span></div>
          <div class="ds-spec-row"><span>Model</span><span>{selected_model}</span></div>
          <div class="ds-spec-row"><span>FFmpeg</span><span>{"installed" if ffmpeg_ready else "missing"}</span></div>
          <div class="ds-spec-row" style="margin-bottom:0"><span>Storage</span><span>Atlas Vector Search</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not ready or not ffmpeg_ready:
        st.warning(model_status if not ready else "FFmpeg is required. Install it and restart.")


# ---------------------------------------------------------------------------
# Results: Timeline Inspector & Flagged Moments
# ---------------------------------------------------------------------------
analysis_data = st.session_state.get("analysis")
session_id = st.session_state.get("session_id")

if analysis_data and session_id:
    analysis = AnalysisSession.from_dict(analysis_data)
    duration = max(analysis.duration_seconds, 1)

    pins_html = ""
    for ev in analysis.events:
        mid = (ev.start_seconds + ev.end_seconds) / 2
        pct = min(max((mid / duration) * 100, 2), 98)
        bg = pin_color(ev.event_type)
        pins_html += (
            f'<div class="ds-tl-pin" style="left:{pct:.1f}%" title="{human_label(ev.event_type)}">'
            f'  <div class="ds-tl-pin-dot" style="background:{bg}"></div>'
            f'  <div class="ds-tl-pin-label">{format_time(ev.start_seconds)}</div>'
            f'</div>'
        )

    tick_interval = 10 if duration <= 60 else 30
    ticks_html = ""
    t = 0.0
    while t <= duration:
        pct = (t / duration) * 100
        ticks_html += f'<div class="ds-tl-tick" style="left:{pct:.1f}%">{format_time(t)}</div>'
        t += tick_interval

    st.markdown(
        f"""
        <div class="ds-inspector">
          <div class="ds-insp-bar">
            <span class="ds-file-name">{html.escape(analysis.source_name)}</span>
            <span class="ds-chip ds-chip--dim">{format_time(duration)}</span>
            <span class="ds-chip"><span class="ds-live"></span>{len(analysis.events)} events</span>
          </div>
          <div class="ds-insp-stage">
            <div class="ds-tl-wrap">
              <div class="ds-tl-track"><div class="ds-tl-fill"></div>{pins_html}</div>
              <div style="position:relative;height:24px;">{ticks_html}</div>
            </div>
          </div>
          <div class="ds-insp-foot">
            <span>endpoint {endpoint_text}</span>
            <span>model {analysis.model_name or selected_model}</span>
            <span class="ds-ok">temporal ready</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if analysis.warnings:
        for w in analysis.warnings:
            st.warning(w)

    # Locate Full Source Video
    source_video_path = st.session_state.get("source_path")
    if not source_video_path or not Path(source_video_path).is_file():
        folder = session_dir(session_id)
        for ext in [".mp4", ".mov", ".avi", ".mkv"]:
            cand = folder / f"source{ext}"
            if cand.is_file():
                source_video_path = str(cand)
                break
        if not source_video_path or not Path(source_video_path).is_file():
            cand = Path(analysis.source_name)
            if cand.is_file():
                source_video_path = str(cand)

    if source_video_path and Path(source_video_path).is_file():
        with st.expander(f"🎬 Watch Full Recording ({format_time(duration)})", expanded=False):
            st.video(str(source_video_path))

    if not analysis.events:
        st.info("No critical road events were flagged in this clip.")
    else:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin: 24px 0 16px;">
              <span style="font-size: 18px; font-weight: 700; color: var(--ink);">Flagged Moments</span>
              <span style="font-family: var(--mono); font-size: 12px; color: var(--faint); text-transform: uppercase; letter-spacing: .08em;">{len(analysis.events)} events detected</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for index, event in enumerate(analysis.events):
            t0 = format_time(event.start_seconds)
            t1 = format_time(event.end_seconds)
            label = human_label(event.event_type)
            with st.expander(f"{t0} – {t1}  |  {label}", expanded=(index == 0)):
                st.markdown(f"**Summary:** {event.summary}")
                if event.observed_cues:
                    st.caption("Observed cues: " + ", ".join(event.observed_cues))
                clip_path = session_dir(session_id) / "exports" / (event.clip_name or "")
                if clip_path.is_file():
                    st.video(str(clip_path))
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button(
                            "Download Clip", data=clip_path.read_bytes(),
                            file_name=clip_path.name, mime="video/mp4",
                            key=f"dl_clip_{index}", use_container_width=True,
                        )
                    with c2:
                        st.download_button(
                            "Download JSON",
                            data=json.dumps(asdict(event), indent=2).encode("utf-8"),
                            file_name=f"event-{index+1:02d}.json", mime="application/json",
                            key=f"dl_json_{index}", use_container_width=True,
                        )
                elif event.warning:
                    st.info(event.warning)

    st.markdown('<div style="margin-top: 24px;">', unsafe_allow_html=True)
    if st.button("Delete this session and start over", use_container_width=True):
        clear_active_session()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Processed Videos Library & Historical Sessions
# ---------------------------------------------------------------------------
from drivesafe.storage import get_all_sessions, get_session, search_historical_events

all_sessions = get_all_sessions()

st.markdown(
    """
    <div style="margin: 48px 0 16px; padding-top: 32px; border-top: 1px solid var(--rule);">
      <div style="font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--volt); margin-bottom: 6px;">
        📂 Video Memory &amp; Historical Library
      </div>
      <h3 style="font-size: 22px; font-weight: 700; margin: 0 0 6px; color: var(--ink);">Processed Drives in MongoDB Storage</h3>
      <p style="color: var(--body); font-size: 14px; margin: 0 0 16px;">All dashcam videos processed by DriveSafe. Review previous drive timelines, inspect flagged safety events, or reload any recording into the inspector.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not all_sessions:
    st.info("No processed videos stored yet. Upload a video above or run one of the sample drives to build your library!")
else:
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.caption(f"**{len(all_sessions)}** processed drive(s) found in MongoDB Atlas / local storage:")
    with col_hdr2:
        show_all = st.checkbox("Show all drives", value=False, key="toggle_show_all_sessions") if len(all_sessions) > 5 else True

    visible_sessions = all_sessions if show_all else all_sessions[:5]
    for idx, s_data in enumerate(visible_sessions):
        s_id = s_data.get("session_id", "")
        s_name = s_data.get("source_name", "Dashcam Video")
        s_dur = float(s_data.get("duration_seconds", 0))
        s_events = s_data.get("events", [])
        is_active = (st.session_state.get("session_id") == s_id)

        # Count event types
        type_counts = {}
        for ev in s_events:
            t = ev.get("event_type", "unclear")
            type_counts[t] = type_counts.get(t, 0) + 1
        
        type_str = " · ".join([f"{human_label(k)}: {v}" for k, v in type_counts.items()]) if type_counts else "No events flagged"
        status_badge = "🟢 ACTIVE" if is_active else f"ID: {s_id[:10]}"
        
        with st.expander(f"📹 **{s_name}** ({format_time(s_dur)}) — {len(s_events)} event(s) — {status_badge}", expanded=is_active):
            c_info, c_action = st.columns([3, 1])
            with c_info:
                st.markdown(f"**Source:** `{s_name}` &middot; **Duration:** {format_time(s_dur)} ({s_dur:.0f}s)")
                st.markdown(f"**Events Breakdown:** {type_str}")
                if s_events:
                    st.caption("Event previews:")
                    for e_i, ev in enumerate(s_events[:3]):
                        st.caption(f"• **{format_time(ev.get('start_seconds', 0))} – {format_time(ev.get('end_seconds', 0))}** ({human_label(ev.get('event_type', ''))}): {ev.get('summary', '')[:110]}...")
                    if len(s_events) > 3:
                        st.caption(f"*(and {len(s_events) - 3} more event(s)...)*")
            
            with c_action:
                if not is_active:
                    if st.button("📂 Load in Inspector", key=f"load_sess_{s_id}_{idx}", use_container_width=True, type="primary"):
                        st.session_state["analysis"] = s_data
                        st.session_state["session_id"] = s_id
                        
                        # Find source file if present
                        folder = session_dir(s_id)
                        found_src = None
                        for cand in folder.glob("source.*"):
                            if cand.is_file():
                                found_src = str(cand)
                                break
                        st.session_state["source_path"] = found_src
                        st.session_state.pop("selected_event", None)
                        st.rerun()
                else:
                    st.success("Currently Active")
                
                if st.button("🗑️ Delete Drive", key=f"del_sess_{s_id}_{idx}", use_container_width=True):
                    delete_session(s_id)
                    if st.session_state.get("session_id") == s_id:
                        clear_active_session()
                    st.rerun()

# ---------------------------------------------------------------------------
# Semantic Vector Search Across All Historical Drives
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin: 48px 0 16px; padding-top: 32px; border-top: 1px solid var(--rule);">
      <div style="font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--volt); margin-bottom: 6px;">
        🔍 AI Semantic Vector Search
      </div>
      <h3 style="font-size: 22px; font-weight: 700; margin: 0 0 6px; color: var(--ink);">Search Past Processed Drives</h3>
      <p style="color: var(--body); font-size: 14px; margin: 0 0 16px;">Query historical driving events using natural language. Powered by local vector embeddings and MongoDB storage.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

search_col1, search_col2 = st.columns([3, 1])
with search_col1:
    search_query = st.text_input(
        "Search past moments",
        placeholder="e.g. 'cyclist on roadway', 'close car encounter', 'sudden stop'",
        label_visibility="collapsed",
    )
with search_col2:
    search_btn = st.button("Search Drives 🔍", use_container_width=True)

# Preset Search Suggestions
chip1, chip2, chip3, chip4 = st.columns(4)
if chip1.button("🚴 Cyclists & Bikes", use_container_width=True):
    search_query = "cyclists and bicycles on roadway"
    search_btn = True
if chip2.button("🚗 Close Encounters", use_container_width=True):
    search_query = "close vehicle encounter or sudden stop"
    search_btn = True
if chip3.button("🚶 Pedestrians", use_container_width=True):
    search_query = "pedestrians crossing street"
    search_btn = True
if chip4.button("🚛 Trucks & Buses", use_container_width=True):
    search_query = "large truck or bus near miss"
    search_btn = True

if (search_btn or search_query) and search_query.strip():
    with st.spinner("Searching vector database..."):
        search_results = search_historical_events(search_query.strip(), top_k=4)
        if not search_results:
            st.info(f"No matching moments found for '{search_query}'. Try analyzing more clips or using different keywords.")
        else:
            st.markdown(f"**Found {len(search_results)} matching event(s) for:** *\"{html.escape(search_query)}\"*")
            for s_idx, match in enumerate(search_results):
                pct = match["score"] * 100
                m_label = human_label(match["event_type"])
                time_range = f"{format_time(match['start_seconds'])} – {format_time(match['end_seconds'])}"
                
                with st.expander(f"⭐ **{pct:.1f}% Match** | {time_range} | {m_label} ({match['source_name']})", expanded=(s_idx == 0)):
                    st.markdown(f"**Summary:** {match['summary']}")
                    if match.get("observed_cues"):
                        st.caption("Observed cues: " + ", ".join(match["observed_cues"]))
                    if match.get("clip_path") and Path(match["clip_path"]).is_file():
                        st.video(match["clip_path"])
                        st.download_button(
                            "Download Event Clip",
                            data=Path(match["clip_path"]).read_bytes(),
                            file_name=f"search-match-{s_idx+1}.mp4",
                            mime="video/mp4",
                            key=f"search_dl_{s_idx}",
                            use_container_width=True,
                        )

# ---------------------------------------------------------------------------
# Native Landing Page Sections (How it works, Privacy, Footer)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <!-- HOW IT WORKS -->
    <section class="section" id="how">
      <div class="section-head">
        <div class="section-label">How it works</div>
        <h2>Three steps. One machine &mdash; yours.</h2>
        <p>A clip goes in, a contact sheet comes out, and the moments worth a second look are marked. Everything runs durably via Temporal workflows.</p>
      </div>
      <div class="steps-grid">
        <article class="step-card">
          <div class="step-card-shot">
            <svg viewBox="0 0 220 140"><rect width="220" height="140" fill="#F4F2EF"/><rect x="30" y="86" width="160" height="38" rx="8" fill="none" stroke="#D6D2CB" stroke-width="1.5" stroke-dasharray="5 5"/><g><rect x="82" y="30" width="56" height="36" rx="4" fill="#FFFFFF" stroke="#D6D2CB" stroke-width="1.5"/><rect x="82" y="30" width="56" height="9" rx="4" fill="#E6E3DE"/><rect x="90" y="47" width="26" height="3" rx="1.5" fill="#D6D2CB"/><rect x="90" y="55" width="16" height="3" rx="1.5" fill="#E6E3DE"/></g><path d="M110 72 L110 84" stroke="#B9B4AC" stroke-width="1.5" stroke-dasharray="3 3"/><path d="M106 80 L110 85 L114 80" stroke="#B9B4AC" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/><rect x="74" y="96" width="72" height="18" rx="9" fill="#FFFFFF" stroke="#D6D2CB" stroke-width="1"/><text x="110" y="108" text-anchor="middle" font-family="ui-monospace,monospace" font-size="8" letter-spacing="1" fill="#8B929C">dashcam_0214.mp4</text></svg>
          </div>
          <div>
            <div class="step-num">01</div>
            <h3>Upload Recording</h3>
            <p>Select a dashcam clip up to 5 minutes. The file is prepared and registered with the Temporal workflow.</p>
          </div>
        </article>
        <article class="step-card">
          <div class="step-card-shot">
            <svg viewBox="0 0 220 140"><rect width="220" height="140" fill="#F4F2EF"/><g fill="#E9E6E1"><rect x="16" y="16" width="42" height="32"/><rect x="64" y="16" width="42" height="32"/><rect x="112" y="16" width="42" height="32"/><rect x="160" y="16" width="42" height="32"/><rect x="16" y="54" width="42" height="32"/><rect x="64" y="54" width="42" height="32"/><rect x="112" y="54" width="42" height="32"/><rect x="160" y="54" width="42" height="32"/><rect x="16" y="92" width="42" height="32"/><rect x="64" y="92" width="42" height="32"/><rect x="112" y="92" width="42" height="32"/><rect x="160" y="92" width="42" height="32"/></g><g fill="#DAD5CC"><rect x="22" y="34" width="30" height="10"/><rect x="70" y="34" width="30" height="10"/><rect x="118" y="34" width="30" height="10"/><rect x="166" y="34" width="30" height="10"/><rect x="22" y="72" width="30" height="10"/><rect x="70" y="72" width="30" height="10"/><rect x="118" y="72" width="30" height="10"/><rect x="166" y="72" width="30" height="10"/><rect x="22" y="110" width="30" height="10"/><rect x="70" y="110" width="30" height="10"/><rect x="118" y="110" width="30" height="10"/><rect x="166" y="110" width="30" height="10"/></g><g fill="#B7B1A8"><rect x="30" y="28" width="12" height="7" rx="1"/><rect x="78" y="28" width="12" height="7" rx="1"/><rect x="126" y="28" width="12" height="7" rx="1"/><rect x="174" y="28" width="12" height="7" rx="1"/><rect x="30" y="66" width="12" height="7" rx="1"/><rect x="78" y="66" width="12" height="7" rx="1"/><rect x="126" y="66" width="12" height="7" rx="1"/><rect x="174" y="66" width="12" height="7" rx="1"/><rect x="30" y="104" width="12" height="7" rx="1"/><rect x="78" y="104" width="12" height="7" rx="1"/><rect x="126" y="104" width="12" height="7" rx="1"/><rect x="174" y="104" width="12" height="7" rx="1"/></g><rect x="111" y="53" width="44" height="34" fill="none" stroke="#2558E6" stroke-width="2"/><text x="16" y="138" font-family="ui-monospace,monospace" font-size="7" letter-spacing="1" fill="#8B929C">CONTACT SHEET</text></svg>
          </div>
          <div>
            <div class="step-num">02</div>
            <h3>Gemma 4 Vision &amp; Temporal</h3>
            <p>Keyframes are sampled into contact sheets and analyzed with auto-retrying Temporal activities.</p>
          </div>
        </article>
        <article class="step-card">
          <div class="step-card-shot">
            <svg viewBox="0 0 220 140"><rect width="220" height="140" fill="#F4F2EF"/><rect x="16" y="20" width="188" height="26" rx="4" fill="#FFFFFF" stroke="#E6E3DE"/><rect x="16" y="54" width="188" height="26" rx="4" fill="#16181D"/><rect x="16" y="88" width="188" height="26" rx="4" fill="#FFFFFF" stroke="#E6E3DE"/><g font-family="ui-monospace,monospace" font-size="8" letter-spacing=".8"><rect x="26" y="29" width="6" height="6" fill="#D6D2CB"/><text x="40" y="37" fill="#8B929C">00:07  PEDESTRIAN</text><rect x="26" y="63" width="6" height="6" fill="#2558E6"/><text x="40" y="71" fill="#F4F2EF">00:14  CLOSE ENCOUNTER</text><rect x="26" y="97" width="6" height="6" fill="#D6D2CB"/><text x="40" y="105" fill="#8B929C">00:22  SUDDEN CHANGE</text></g></svg>
          </div>
          <div>
            <div class="step-num">03</div>
            <h3>Review &amp; Vector Search</h3>
            <p>Inspect flagged moments, play exported event clips, and sync events to MongoDB Atlas.</p>
          </div>
        </article>
      </div>
    </section>

    <!-- CHRONOLINE -->
    <div class="chrono">
      <span>Frames &rarr; <span class="chrono-tag">Temporal Workflow</span> &rarr; Gemma 4 (Cloud 26B / Local E2B) &rarr; <span class="chrono-tag">Atlas</span></span>
    </div>

    <!-- PRIVACY -->
    <section class="section" id="privacy">
      <div class="privacy-wrap">
        <div>
          <h2>Privacy &amp; Local Edge Control</h2>
          <p>DriveSafe gives you complete control over your footage. Choose Local Edge mode to run Gemma 4 E2B directly via Ollama on your machine, or Cloud mode for deep Gemma 4 26B analysis.</p>
          <div class="privacy-chips">
            <span class="privacy-chip">Temporal Workflows</span>
            <span class="privacy-chip">Exponential Backoff</span>
            <span class="privacy-chip">Local Frame Sampling</span>
            <span class="privacy-chip">Gemma 4 Vision</span>
            <span class="privacy-chip">Atlas Vector Search</span>
          </div>
        </div>
        <div>
          <div class="terminal">
            <span class="prompt">$</span> <span class="out">temporal workflow start --type VideoAnalysisWorkflow</span><br>
            <span class="dim">&rarr;</span> <span class="out">task-queue drivesafe-analysis-queue</span><br>
            <span class="dim">&rarr;</span> <span class="out">activities extract &bull; vision &bull; ffmpeg &bull; atlas</span><br>
            <span class="dim">&rarr;</span> <span class="out">auto-retry exponential backoff</span><br>
            <span class="zero">&check; workflow status: COMPLETED</span>
          </div>
        </div>
      </div>
    </section>

    <!-- FOOTER -->
    <footer class="ds-footer">
      <div class="ds-footer-grid">
        <div class="ds-footer-brand">
          <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
            {f'<img src="data:image/png;base64,{LOGO_B64}" alt="DriveSafe Logo" style="height: 28px; width: auto; border-radius: 6px;">' if LOGO_B64 else ''}
            <h3 style="margin: 0; font-family: var(--mono); font-size: 15px; font-weight: 700; letter-spacing: .1em; color: var(--ink);">DRIVESAFE</h3>
          </div>
          <p>Intelligent, durable dashcam footage review powered by Temporal Workflows, Google Gemma 4, and MongoDB Atlas.</p>
        </div>
        <div class="ds-footer-col">
          <h4>Architecture</h4>
          <ul>
            <li>Frame Sampling</li>
            <li>Temporal Activities</li>
            <li>Vision Detection</li>
            <li>Privacy Engine</li>
          </ul>
        </div>
        <div class="ds-footer-col">
          <h4>Models &amp; Cloud</h4>
          <ul>
            <li>Temporal IO</li>
            <li>Gemini API (Cloud 26B)</li>
            <li>Ollama (Edge E2B)</li>
            <li>MongoDB Atlas</li>
          </ul>
        </div>
        <div class="ds-footer-col">
          <h4>Tech Stack</h4>
          <div class="ds-footer-tags">
            <span class="ds-footer-tag">Temporal</span>
            <span class="ds-footer-tag">Python 3.10</span>
            <span class="ds-footer-tag">Streamlit</span>
            <span class="ds-footer-tag">OpenCV</span>
            <span class="ds-footer-tag">FFmpeg</span>
            <span class="ds-footer-tag">Gemma 4 26B</span>
            <span class="ds-footer-tag">Gemma 4 E2B</span>
            <span class="ds-footer-tag">Ollama</span>
            <span class="ds-footer-tag">MongoDB Atlas</span>
          </div>
        </div>
      </div>
      <div class="ds-footer-bottom">
        <div>&copy; 2026 DriveSafe &middot; Built for Hacktober Weekend Hackathon</div>
        <div style="color: var(--volt); font-weight: 600;">&bull; Temporal Ready</div>
      </div>
    </footer>
    """,
    unsafe_allow_html=True,
)
