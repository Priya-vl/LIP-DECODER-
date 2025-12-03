"""
import os
import cv2
import yt_dlp
import whisper
import mediapipe as mp
import numpy as np

# --- CONFIGURATION ---
DATASET_ROOT = "dataset_ht2k"
LANGUAGES_CONFIG = {
    "Tamil": {
        "url": "https://www.youtube.com/watch?v=0JNCs0LhUSg&list=PL36745FF2B2C8129A",
        "code": "ta",
        "max_videos": 3
    },
    "Kannada": {
        "url": "https://www.youtube.com/watch?v=src3F9iDgvo&list=PLTFSHuvrvsZV0L8KVdl7Nhar__1MTPsMp",
        "code": "kn",
        "max_videos": 3    },
    "Telugu": {
        "url": "https://www.youtube.com/watch?v=UtoKrZlGRlY&list=PLPHGCqD0bogDXSJZNIo-bRUo2YjLwwdgE",
        "code": "te",
        "max_videos": 2
    },
    "Hindi": {
        "url": "https://www.youtube.com/watch?v=MNtdwAe5LRE",
        "code": "hi",
        "max_videos": 2
    }
}


# --- 1. DOWNLOADER ---
def download_videos(lang, config):
    print(f"[{lang}] Downloading videos...")
    # Create output path
    out_path = os.path.join(DATASET_ROOT, lang, "%(id)s", "video.%(ext)s")

    ydl_opts = {
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        'outtmpl': out_path,
        'playlistend': config['max_videos'],
        'ignoreerrors': True,
        'quiet': True,
        'no_warnings': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([config['url']])


# --- 2. LIP EXTRACTOR ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


def process_video(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    frames_dir = os.path.join(save_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Detect Face & Lips
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape

            # Extract Lip region (Indices for lips in MediaPipe)
            lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
            lip_coords = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in lip_indices]

            # Bounding Box
            x_min = max(0, min([p[0] for p in lip_coords]) - 10)
            x_max = min(w, max([p[0] for p in lip_coords]) + 10)
            y_min = max(0, min([p[1] for p in lip_coords]) - 10)
            y_max = min(h, max([p[1] for p in lip_coords]) + 10)

            lip_crop = frame[y_min:y_max, x_min:x_max]

            # Resize and Save
            if lip_crop.size > 0:
                lip_crop = cv2.resize(lip_crop, (100, 50))  # Fixed size for model
                cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg"), lip_crop)
                frame_count += 1

    cap.release()


# --- 3. AUDIO TRANSCRIBER ---
def transcribe_audio(video_path, save_dir, model):
    print(f"Transcribing {video_path}...")
    try:
        result = model.transcribe(video_path)
        text = result['text']
        with open(os.path.join(save_dir, "transcription.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"Error transcribing: {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Loading Whisper Model (for Ground Truth generation)...")
    whisper_model = whisper.load_model("base")  # Downloads ~140MB model

    for lang, config in LANGUAGES_CONFIG.items():
        # 1. Download
        download_videos(lang, config)

        # 2. Process each downloaded folder
        lang_dir = os.path.join(DATASET_ROOT, lang)
        if not os.path.exists(lang_dir): continue

        for video_id in os.listdir(lang_dir):
            video_folder = os.path.join(lang_dir, video_id)
            if not os.path.isdir(video_folder): continue

            # Find video file
            video_file = None
            for f in os.listdir(video_folder):
                if f.endswith(('.mp4', '.mkv', '.webm')):
                    video_file = os.path.join(video_folder, f)
                    break

            if video_file:
                # 3. Process
                process_video(video_file, video_folder)
                transcribe_audio(video_file, video_folder, whisper_model)
                print(f"Processed {lang} - {video_id}")

    print("\n✅ Dataset Creation Complete! Check 'dataset_ht2k' folder.")"""

import os
import cv2
import yt_dlp
import whisper
import mediapipe as mp
import numpy as np

# --- CONFIGURATION (Reduced for Speed) ---
DATASET_ROOT = "dataset_ht2k"
LANGUAGES_CONFIG = {
    "Tamil": {"url": "https://www.youtube.com/watch?v=0JNCs0LhUSg", "code": "ta", "max_videos": 2},
    "Kannada": {"url": "https://www.youtube.com/watch?v=src3F9iDgvo", "code": "kn", "max_videos": 2},
    "Telugu": {"url": "https://www.youtube.com/watch?v=UtoKrZlGRlY", "code": "te", "max_videos": 2},
    "Hindi": {"url": "https://www.youtube.com/watch?v=MNtdwAe5LRE", "code": "hi", "max_videos": 2}
}


# --- 1. DOWNLOADER ---
def download_videos(lang, config):
    print(f"[{lang}] Downloading...")
    out_path = os.path.join(DATASET_ROOT, lang, "%(id)s", "video.%(ext)s")

    ydl_opts = {
        # Download lowest reasonable quality to save bandwidth
        'format': 'bestvideo[height<=360]+bestaudio/best[height<=360]',
        'outtmpl': out_path,
        'playlistend': config['max_videos'],
        'ignoreerrors': True,
        'quiet': True,
        'no_warnings': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([config['url']])


# --- 2. OPTIMIZED LIP EXTRACTOR ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,  # Speed Boost: Disable iris tracking
    min_detection_confidence=0.5
)


def process_video(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    frames_dir = os.path.join(save_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_count = 0
    max_frames_to_save = 50  # LIMIT: Only save 50 frames (approx 2 seconds)

    while cap.isOpened() and frame_count < max_frames_to_save:
        ret, frame = cap.read()
        if not ret: break

        # Resize immediately to speed up MediaPipe processing
        h, w, _ = frame.shape
        if w > 480:
            scale = 480 / w
            frame = cv2.resize(frame, (480, int(h * scale)))
            h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # Simple Mouth Center (Index 13)
            cx, cy = int(landmarks.landmark[13].x * w), int(landmarks.landmark[13].y * h)

            # Crop 100x50
            y1, y2 = max(0, cy - 25), min(h, cy + 25)
            x1, x2 = max(0, cx - 50), min(w, cx + 50)

            lip_crop = frame[y1:y2, x1:x2]

            if lip_crop.size != 0:
                lip_crop = cv2.resize(lip_crop, (100, 50))
                cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg"), lip_crop)
                frame_count += 1

    cap.release()


# --- 3. OPTIMIZED AUDIO TRANSCRIBER ---
def transcribe_audio(video_path, save_dir, model):
    print(f"   -> Transcribing...")
    try:
        # fp16=False allows it to run on CPU without warnings
        result = model.transcribe(video_path, fp16=False)
        text = result['text']
        with open(os.path.join(save_dir, "transcription.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"Skipping transcription: {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Loading Tiny Whisper Model (Fastest)...")
    # Use 'tiny' model instead of 'base' -> 10x faster
    whisper_model = whisper.load_model("tiny")

    for lang, config in LANGUAGES_CONFIG.items():
        download_videos(lang, config)

        lang_dir = os.path.join(DATASET_ROOT, lang)
        if not os.path.exists(lang_dir): continue

        for video_id in os.listdir(lang_dir):
            video_folder = os.path.join(lang_dir, video_id)
            if not os.path.isdir(video_folder): continue

            video_file = None
            for f in os.listdir(video_folder):
                if f.endswith(('.mp4', '.mkv', '.webm')):
                    video_file = os.path.join(video_folder, f)
                    break

            if video_file:
                process_video(video_file, video_folder)
                transcribe_audio(video_file, video_folder, whisper_model)
                print(f"✓ Processed {lang}")

    print("\n✅ Dataset Creation Complete!")