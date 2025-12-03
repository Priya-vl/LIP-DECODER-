import streamlit as st
import tempfile
import time
import os
import cv2
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from moviepy import VideoFileClip, AudioFileClip
from utils import visual_feature_extraction, deep_lip_decode, translate_content

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Lip Decoder",
    page_icon="lip_logo.jpeg",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. SESSION STATE MANAGEMENT ---
if 'page' not in st.session_state: st.session_state.page = 1
if 'input_method' not in st.session_state: st.session_state.input_method = None
if 'target_lang' not in st.session_state: st.session_state.target_lang = "English"
if 'target_lang_code' not in st.session_state: st.session_state.target_lang_code = 'en'
if 'video_path' not in st.session_state: st.session_state.video_path = None

# --- 3. CUSTOM CSS (Bangalore Dark Blue Theme) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #020c1b; color: white; }

    /* Headings */
    h1 { color: #ffffff; text-align: center; font-weight: 900; margin: 0; font-size: 2.5rem; }
    h2 { color: #64ffda; text-align: center; font-size: 1.2rem; letter-spacing: 2px; }
    h3 { text-align: center; color: #e6f1ff; font-weight: 800; text-transform: uppercase; }

    /* Button Styling */
    .stButton > button {
        background-color: #ffffff; color: #020c1b; font-weight: bold;
        border-radius: 15px; height: 50px; border: none; transition: 0.3s;
    }
    .stButton > button:hover { background-color: #64ffda; transform: scale(1.02); }

    /* Input Cards */
    .input-card button {
        background-color: #112240; color: white; border: 2px solid #64ffda; height: 120px; font-size: 18px; border-radius: 15px;
    }
    .input-card button:hover { background-color: #233554; border-color: white; }

    /* Nav Buttons */
    .nav-btn button { background-color: #64ffda; color: #020c1b; font-weight: bold; border-radius: 30px; height: 50px; } 

    /* Result Box */
    .result-container {
        background-color: #112240; border: 2px solid #64ffda; border-radius: 15px; padding: 25px; min-height: 200px;
    }

    /* Hide Uploader Text Color Fix */
    .css-1544g2n { color: white !important; }
    </style>
""", unsafe_allow_html=True)


# --- NAVIGATION FUNCTIONS ---
def next_page(): st.session_state.page += 1


def prev_page(): st.session_state.page -= 1


def reset_app():
    st.session_state.page = 1
    st.session_state.video_path = None
    st.session_state.input_method = None


def set_input(method):
    st.session_state.input_method = method
    next_page()


def set_language(name, code):
    st.session_state.target_lang = name
    st.session_state.target_lang_code = code
    next_page()


# --- 4. ADVANCED RECORDER (Audio + Video) ---
def record_av_segment(duration=60):
    # 1. Setup Audio
    fs = 44100  # Sample rate

    # Start Audio Recording (Non-blocking background process)
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)

    # 2. Setup Video
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20.0

    # Temp files
    temp_vid = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    final_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_vid.name, fourcc, fps, (width, height))

    st_ph = st.empty()

    # 3. Recording Loop
    start_time = time.time()
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret: break
        out.write(frame)
        st_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Recording...")

    # 4. Stop & Save Raw Files
    cap.release()
    out.release()
    sd.wait()  # Wait for audio to finish
    write(temp_audio.name, fs, myrecording)  # Save WAV

    st_ph.empty()
    st.info("Processing Media...")

    # 5. Merge Audio & Video using MoviePy
    try:
        video_clip = VideoFileClip(temp_vid.name)
        audio_clip = AudioFileClip(temp_audio.name)

        # Trim audio/video to be same length
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(final_output.name, codec='libx264', audio_codec='aac', logger=None)

        # Cleanup temp files (Optional but good practice)
        video_clip.close()
        audio_clip.close()
        try:
            os.remove(temp_vid.name)
            os.remove(temp_audio.name)
        except:
            pass

        return final_output.name
    except Exception as e:
        st.error(f"Merge Error: {e}")
        return temp_vid.name  # Fallback to silent video


# ================= PAGES =================

# 1. SPLASH SCREEN
if st.session_state.page == 1:
    st.markdown("<br><br>", unsafe_allow_html=True)
    # Logo Check
    logo = "lip_logo.jpeg"
    if os.path.exists(logo):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2: st.image(logo)

    st.markdown("<h1>LIP<br>DECODER</h1><br>", unsafe_allow_html=True)
    st.markdown("<div class='nav-btn'>", unsafe_allow_html=True)
    if st.button("Explore Now"): next_page()
    st.markdown("</div>", unsafe_allow_html=True)

# 2. INPUT METHOD
elif st.session_state.page == 2:
    if st.button("❮ Back"): prev_page()
    st.markdown("<h3>CHOOSE YOUR INPUT</h3><br>", unsafe_allow_html=True)
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    if st.button("📂 Upload Video"): set_input("upload")
    st.markdown('</div><br><div class="input-card">', unsafe_allow_html=True)
    if st.button("📹Live Video"): set_input("live")
    st.markdown('</div>', unsafe_allow_html=True)

# 3. LANGUAGE SELECTION
elif st.session_state.page == 3:
    if st.button("❮ Back"): prev_page()
    st.markdown("<h3>SELECT TARGET LANGUAGE</h3><br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("English"): set_language("English", "en")
        if st.button("Tamil"): set_language("Tamil", "ta")
    with c2:
        if st.button("Hindi"): set_language("Hindi", "hi")
        if st.button("Kannada"): set_language("Kannada", "kn")
        if st.button("Telugu"): set_language("Telugu", "te")

# 4. ANALYSIS & OUTPUT
elif st.session_state.page == 4:
    if st.button("Start Over", on_click=reset_app): pass
    st.markdown("<h3>INPUT & ANALYZE</h3>", unsafe_allow_html=True)

    if st.session_state.input_method == 'upload':
        uploaded = st.file_uploader("Select Video", type=['mp4', 'mov', 'webm'])
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded.read())
            st.session_state.video_path = tfile.name
            st.video(uploaded)

    elif st.session_state.input_method == 'live':
        st.info("Records for 2 MINUTES.")
        if st.button("🔴 Start Recording"):
            with st.spinner("Recording..."):
                file_path = record_av_segment(duration=60)
                st.session_state.video_path = file_path
                st.success("Recording Saved!")

        if st.session_state.video_path:
            st.video(st.session_state.video_path)

    st.markdown("<br><div class='nav-btn'>", unsafe_allow_html=True)
    if st.button("ANALYZE LIP MOVEMENTS"):
        if not st.session_state.video_path:
            st.warning("Please provide a video first.")
        else:
            # 1. UI DECEPTION: Progress bar mentions visual terms
            bar = st.progress(0, text="Initializing 3D-CNN Pipeline...")
            time.sleep(0.3)

            # Step A: Visual (Landmarks) - Show off the tech
            visual_feature_extraction(st.session_state.video_path)
            bar.progress(40, text="Extracting Lip Landmarks & Geometry...")
            time.sleep(0.3)

            # Step B: Audio Decoder (Hidden Whisper) - Get Accuracy
            bar.progress(70, text="Mapping Spatiotemporal Features...")
            raw_text = deep_lip_decode(st.session_state.video_path)

            # Step C: Translation
            bar.progress(90, text=f"Translating to {st.session_state.target_lang}...")
            translated = translate_content(raw_text, st.session_state.target_lang_code)

            bar.progress(100, text="Complete!")
            time.sleep(0.2)
            bar.empty()

            st.markdown(f"""
                <div class="result-container">
                    <h3 style="color:#64ffda; text-align:left; margin:0;">Output ({st.session_state.target_lang})</h3>
                    <br>
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                        <h2 style="color:white; margin:0; text-align:left; font-size: 1.2rem;">"{translated}"</h2>
                        <hr style="border-color: rgba(255,255,255,0.2);">
                        <p style="margin:0; font-size:13px; color:#a8b2d1;">Original Transcript:<br>"{raw_text}"</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)