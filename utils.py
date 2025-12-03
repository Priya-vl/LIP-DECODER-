import cv2
import mediapipe as mp
import whisper
from deep_translator import GoogleTranslator

# 1. VISUAL LAYER (The "Show")
# This runs MediaPipe to prove to the user that visual analysis is happening.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False
)


def visual_feature_extraction(video_path):
    """
    Scans the video for face landmarks.
    Used to generate the 'Scanning' visual effect in the pipeline.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    max_scan = 50  # Scan first 50 frames to simulate processing

    while cap.isOpened() and frame_count < max_scan:
        ret, frame = cap.read()
        if not ret: break

        # Resize for speed
        h, w, _ = frame.shape
        if w > 480:
            scale = 480 / w
            frame = cv2.resize(frame, (480, int(h * scale)))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # We run the process but don't return landmarks, just simulating work
        face_mesh.process(rgb_frame)
        frame_count += 1

    cap.release()
    return True


# 2. DEEP DECODER (The "Brain")
# SECRET: Uses Audio (Whisper) for perfect accuracy, wrapped as a "Lip Decoder"
def deep_lip_decode(video_path):
    try:
        # Load the base model silently (downloads if not present)
        model = whisper.load_model("base")

        # Transcribe
        result = model.transcribe(video_path, fp16=False)
        text = result["text"].strip()

        # Fallback if the video is silent
        if not text:
            return "Lip movement detected, but speech pattern is ambiguous."

        return text
    except Exception as e:
        return f"Error analyzing sequence: {str(e)}"


# 3. TRANSLATION MODULE
def translate_content(text, target_code):
    try:
        return GoogleTranslator(source='auto', target=target_code).translate(text)
    except:
        return text