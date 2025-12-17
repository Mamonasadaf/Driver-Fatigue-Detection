import cv2
import time
import math
import numpy as np
from threading import Thread

# Try to import PyTorch (for CNN)
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ========================= CONFIG =========================
EAR_THRESH = 0.25
EYE_WAIT_TIME = 0.2   # seconds

USE_CNN = True
CNN_MODEL_PATH = "/home/nvidia/Downloads/eye_cnn_nano.pth"
CNN_PROB_THRESH = 0.8

SHOW_FPS = True
SELFIE_FLIP = True

CAM_WIDTH = 1920   # lower resolution for speed
CAM_HEIGHT = 1200
CAM_FPS = 15
FRAME_SKIP = 1    # skip every other frame

# Haar cascade paths
FACE_CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH  = "/usr/share/opencv4/haarcascades/haarcascade_eye.xml"

# =========================================================
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ======================= CNN MODEL =======================
class NanoEyeCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class EyeStateClassifier:
    def __init__(self, model_path=None, device=None, input_size=64):
        self.enabled = TORCH_AVAILABLE and model_path is not None
        self.device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.model = None
        self.input_size = input_size

        if self.enabled:
            try:
                self.model = NanoEyeCNN(1, 2)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device).eval()

                self.transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((input_size, input_size)),
                    T.Grayscale(),
                    T.ToTensor(),
                    T.Normalize((0.5,), (0.5,))
                ])
                print("[INFO] CNN model loaded")
            except Exception as e:
                print("[ERROR] CNN load failed:", e)
                self.enabled = False

    def predict_closed_prob(self, eye_roi_bgr):
        if not self.enabled or eye_roi_bgr is None or eye_roi_bgr.size == 0:
            return None
        eye_rgb = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            x = self.transform(eye_rgb).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        return float(probs[1])  # closed probability

# ======================= DROWSINESS DETECTOR =======================
class DrowsinessDetector:
    def __init__(self):
        # Haar cascades for fallback detection
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.eye_cascade  = cv2.CascadeClassifier(EYE_CASCADE_PATH)

        self.eye_closed_start = None
        self.drowsy_alarm = False
        self.eye_cnn = EyeStateClassifier(CNN_MODEL_PATH if USE_CNN else None)

        self.last_fps_time = time.time()
        self.frame_counter = 0
        self.current_fps = 0

    def process_frame(self, frame, now):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye_closed = False

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
            for (ex, ey, ew, eh) in eyes:
                eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
                p_closed = self.eye_cnn.predict_closed_prob(eye_roi)
                if p_closed and p_closed > CNN_PROB_THRESH:
                    eye_closed = True

        if eye_closed:
            if self.eye_closed_start is None:
                self.eye_closed_start = now
            if now - self.eye_closed_start > EYE_WAIT_TIME:
                self.drowsy_alarm = True
        else:
            self.eye_closed_start = None
            self.drowsy_alarm = False

        color = (0,0,255) if self.drowsy_alarm else (0,255,0)
        cv2.putText(frame, "DROWSY!" if self.drowsy_alarm else "AWAKE",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        return frame

    def update_fps(self, now):
        self.frame_counter += 1
        if now - self.last_fps_time >= 1:
            self.current_fps = self.frame_counter / (now - self.last_fps_time)
            self.frame_counter = 0
            self.last_fps_time = now

# ======================= ASYNC CAMERA =======================
class VideoCaptureAsync:
    def __init__(self, src=0):  # USB webcam at /dev/video0
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        self.ret = False
        self.frame = None
        self.running = True

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

# ======================= MAIN =======================
def main():
    cap = VideoCaptureAsync(src=0).start()  # USB webcam at /dev/video0
    detector = DrowsinessDetector()
    frame_skip_counter = 0
    print("✅ USB Webcam Drowsiness Detection Started (press q)")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        if SELFIE_FLIP:
            frame = cv2.flip(frame, 1)

        frame_skip_counter += 1
        if frame_skip_counter % (FRAME_SKIP + 1) != 0:
            continue  # skip frames to reduce load

        now = time.time()
        detector.update_fps(now)
        frame = detector.process_frame(frame, now)

        if SHOW_FPS:
            cv2.putText(frame, f"FPS: {detector.current_fps:.1f}", (20,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()