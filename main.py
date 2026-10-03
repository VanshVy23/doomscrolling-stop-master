import cv2
import mediapipe as mp
import numpy as np
import math
import subprocess
import threading
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

VIDEO_PATH = '/Users/jayeshvishwakarma/Documents/Documents/Stuffs/doomscroller_ctrl/1777266732970374.MP4'

QT_OPEN_SCRIPT = f'''
tell application "QuickTime Player"
    activate
    set theDoc to open POSIX file "{VIDEO_PATH}"
    tell theDoc to play
end tell
'''

QT_CLOSE_SCRIPT = '''
tell application "QuickTime Player"
    if (count of documents) > 0 then
        close every document
    end if
end tell
'''

def start_video():
    subprocess.Popen(['osascript', '-e', QT_CLOSE_SCRIPT],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait()
    time.sleep(0.1)
    subprocess.Popen(['osascript', '-e', QT_OPEN_SCRIPT],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def stop_video():
    subprocess.Popen(['osascript', '-e', QT_CLOSE_SCRIPT],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def dist3d(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0,  -150.0, -125.0)
], dtype=np.float64)
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

def get_head_pitch(landmarks, img_w, img_h):
    image_points = np.array([
        (landmarks[i].x * img_w, landmarks[i].y * img_h)
        for i in LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0

    rot_mat, _ = cv2.Rodrigues(rotation_vec)
    pitch = math.degrees(math.asin(-rot_mat[2][1]))
    return pitch

def eye_looking_down(landmarks):
    try:
        results = []
        for top_idx, bottom_idx, iris_idx in [(159, 145, 468), (386, 374, 473)]:
            top    = landmarks[top_idx]
            bottom = landmarks[bottom_idx]
            iris   = landmarks[iris_idx]
            eye_h  = dist3d(top, bottom)
            if eye_h < 0.005:
                return None
            iris_rel = (iris.y - top.y) / (bottom.y - top.y + 1e-6)
            results.append(iris_rel)

        avg = sum(results) / len(results)
        return avg > 0.52, avg
    except:
        return None

def draw_eye_boxes(image, landmarks, img_w, img_h):
    for ids in [(33, 133, 159, 145), (263, 362, 386, 374)]:
        xs = [int(landmarks[i].x * img_w) for i in ids]
        ys = [int(landmarks[i].y * img_h) for i in ids]
        pad = 8
        cv2.rectangle(image,
                      (min(xs)-pad, min(ys)-pad),
                      (max(xs)+pad, max(ys)+pad),
                      (0, 255, 0), 2)

cap = cv2.VideoCapture(1)
cv2.namedWindow('Doomscroller Ctrl', cv2.WINDOW_NORMAL)

FRAMES_TO_PLAY = 2
FRAMES_TO_STOP = 35

at_screen_frames = 0
away_frames      = 0
no_face_frames   = 0
video_playing    = False
last_timestamp_ms = 0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    img_h, img_w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb.copy())

    current_timestamp_ms = int(time.time() * 1000)
    if current_timestamp_ms <= last_timestamp_ms:
        current_timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = current_timestamp_ms

    results = face_landmarker.detect_for_video(mp_image, current_timestamp_ms)

    display = frame.copy()
    looking_at_screen = not video_playing

    if results.face_landmarks:
        no_face_frames = 0
        lms = results.face_landmarks[0]

        draw_eye_boxes(display, lms, img_w, img_h)

        pitch = get_head_pitch(lms, img_w, img_h)
        eye_result = eye_looking_down(lms)

        if eye_result is None:
            looking_at_screen = not video_playing
        else:
            _, iris_val = eye_result
            head_straight = -10 < pitch < 20
            eyes_centered = 0.20 < iris_val < 0.68
            looking_at_screen = head_straight and eyes_centered

            color_p = (0, 200, 0) if head_straight else (0, 0, 255)
            color_e = (0, 200, 0) if eyes_centered else (0, 0, 255)
            cv2.putText(display, f'Pitch: {pitch:+.1f}', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_p, 2)
            cv2.putText(display, f'Eye: {iris_val:.2f}', (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_e, 2)

        state_label = 'AT SCREEN' if looking_at_screen else 'AWAY'
        label_color = (0, 200, 0) if looking_at_screen else (0, 0, 255)
        cv2.putText(display, state_label, (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

    else:
        no_face_frames += 1
        looking_at_screen = True if no_face_frames >= 30 else False
        cv2.putText(display, 'No face', (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

    if looking_at_screen:
        at_screen_frames += 1
        away_frames = 0
    else:
        away_frames += 1
        at_screen_frames = 0

    if not video_playing and away_frames >= FRAMES_TO_PLAY:
        video_playing = True
        away_frames = 0
        threading.Thread(target=start_video, daemon=True).start()

    elif video_playing and at_screen_frames >= FRAMES_TO_STOP:
        video_playing = False
        at_screen_frames = 0
        threading.Thread(target=stop_video, daemon=True).start()

    status_color = (0, 0, 255) if video_playing else (0, 200, 0)
    cv2.circle(display, (img_w - 25, 25), 12, status_color, -1)
    cv2.putText(display, 'REC' if video_playing else 'IDLE',
                (img_w - 70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if video_playing:
        label = 'DOOMSCROLLING ALARM'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, th + 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)
        tx = (img_w - tw) // 2
        cv2.putText(display, label, (tx, th + 8),
                    font, font_scale, (0, 255, 80), thickness, cv2.LINE_AA)

    cv2.imshow('Doomscroller Ctrl', display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

stop_video()
face_landmarker.close()
cap.release()
cv2.destroyAllWindows()
