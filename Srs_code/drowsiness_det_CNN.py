import cv2
import time
import math
import mediapipe as mp
import numpy as np

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
EYE_WAIT_TIME = 0.5   # seconds

MAR_THRESH = 0.5
YAWN_WAIT_TIME = 1.0  # seconds

USE_CNN = True
CNN_MODEL_PATH = "D:\\7th Semester\\Computer Vision\\Project\\Drowsiness_Detection\\eye_cnn_nano.pth"   # <-- use the file from Colab
CNN_PROB_THRESH = 0.8

SHOW_FPS = True
SELFIE_FLIP = True

CAM_WIDTH = 640
CAM_HEIGHT = 480

# =========================================================

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ======================= CNN MODEL =======================

class NanoEyeCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
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
        x = self.features(x)
        x = self.classifier(x)
        return x

class EyeStateClassifier:
    """
    Loads NanoEyeCNN and returns p_closed for a given eye ROI (BGR patch).
    """
    def __init__(self, model_path=None, device=None, input_size=64):
        self.enabled = TORCH_AVAILABLE and model_path is not None
        self.model = None
        self.device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        if self.enabled:
            try:
                self.model = NanoEyeCNN(in_channels=1, num_classes=2)
                state = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state)
                self.model.to(self.device)
                self.model.eval()

                self.transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((input_size, input_size)),
                    T.Grayscale(),
                    T.ToTensor(),
                    T.Normalize((0.5,), (0.5,))
                ])

                print(f"[EyeStateClassifier] Loaded CNN model from {model_path} on {self.device}")
            except Exception as e:
                print(f"[EyeStateClassifier] Could not load model: {e}")
                self.model = None
                self.enabled = False
        else:
            print("[EyeStateClassifier] CNN disabled (PyTorch not available or no model path).")

    def predict_closed_prob(self, eye_roi_bgr):
        if not self.enabled or self.model is None:
            return None
        if eye_roi_bgr is None or eye_roi_bgr.size == 0:
            return None

        eye_rgb = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            x = self.transform(eye_rgb)        # 1 x H x W
            x = x.unsqueeze(0).to(self.device) # 1 x 1 x H x W
            logits = self.model(x)             # 1 x 2
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            p_open  = float(probs[0])  # class 0 = open
            p_closed = float(probs[1]) # class 1 = closed
        return p_closed

# ======================= MAIN DETECTOR =======================

class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.eye_idxs = {
            "left":  [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }
        self.mouth_idxs = [61, 81, 13, 311, 402, 14, 178, 308]

        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED   = (0,   0, 255)
        self.COLOR_BLUE  = (255, 0, 0)
        self.COLOR_YELLOW= (0, 255, 255)
        self.COLOR_WHITE = (255, 255, 255)

        self.eye_closed_start = None
        self.yawn_start = None
        self.drowsy_alarm = False
        self.yawn_alarm = False

        self.last_fps_time = time.time()
        self.frame_counter = 0
        self.current_fps = 0.0

        self.eye_cnn = EyeStateClassifier(
            model_path=CNN_MODEL_PATH if USE_CNN else None
        )

    def _landmarks_to_pixels(self, landmarks, w, h):
        pts = []
        for lm in landmarks:
            pts.append((int(lm.x * w), int(lm.y * h)))
        return pts

    def _get_eyeEAR(self, all_landmarks, w, h):
        def ear_from_indices(ids):
            pts = [all_landmarks[i] for i in ids]
            pts = self._landmarks_to_pixels(pts, w, h)
            p1, p2, p3, p4, p5, p6 = pts
            dist_26 = euclidean_distance(p2, p6)
            dist_35 = euclidean_distance(p3, p5)
            dist_14 = euclidean_distance(p1, p4)
            if dist_14 == 0:
                return 0.0, pts
            ear_val = (dist_26 + dist_35) / (2.0 * dist_14)
            return ear_val, pts

        left_ear, left_pts = ear_from_indices(self.eye_idxs["left"])
        right_ear, right_pts = ear_from_indices(self.eye_idxs["right"])
        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear, left_pts, right_pts

    def _get_mouthMAR(self, all_landmarks, w, h):
        pts = [all_landmarks[i] for i in self.mouth_idxs]
        pts = self._landmarks_to_pixels(pts, w, h)

        p61, p81, p13, p311, p402, p14, p178, p308 = pts

        v1 = euclidean_distance(p61, p81)
        v2 = euclidean_distance(p13, p14)
        v3 = euclidean_distance(p311, p402)
        h1 = euclidean_distance(p61, p311) + 1e-6

        mar_val = (v1 + v2 + v3) / (3.0 * h1)
        return mar_val, pts

    def _get_eye_roi(self, all_landmarks, w, h, padding=10):
        idxs = self.eye_idxs["left"] + self.eye_idxs["right"]
        pts = [all_landmarks[i] for i in idxs]
        pts = self._landmarks_to_pixels(pts, w, h)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min = max(0, min(xs) - padding)
        x_max = min(w, max(xs) + padding)
        y_min = max(0, min(ys) - padding)
        y_max = min(h, max(ys) + padding)
        return x_min, y_min, x_max, y_max

    def process_frame(self, frame, now):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        ear = None
        mar = None
        p_closed = None

        eye_closed_flag = False
        yawn_flag = False

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            ear, left_pts, right_pts = self._get_eyeEAR(face_landmarks, w, h)
            mar, mouth_pts = self._get_mouthMAR(face_landmarks, w, h)
            x1, y1, x2, y2 = self._get_eye_roi(face_landmarks, w, h)

            for p in left_pts + right_pts:
                cv2.circle(frame, p, 2, self.COLOR_GREEN, -1)
            for p in mouth_pts:
                cv2.circle(frame, p, 2, self.COLOR_BLUE, -1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_YELLOW, 1)

            eye_roi = frame[y1:y2, x1:x2].copy()
            p_closed = self.eye_cnn.predict_closed_prob(eye_roi)

            if p_closed is not None:
                if p_closed > CNN_PROB_THRESH:
                    eye_closed_flag = True
            elif ear is not None and ear < EAR_THRESH:
                eye_closed_flag = True

            if mar is not None and mar > MAR_THRESH:
                yawn_flag = True

        # temporal eye logic
        if eye_closed_flag:
            if self.eye_closed_start is None:
                self.eye_closed_start = now
            closed_duration = now - self.eye_closed_start
            if closed_duration >= EYE_WAIT_TIME:
                self.drowsy_alarm = True
        else:
            self.eye_closed_start = None
            self.drowsy_alarm = False

        # temporal yawn logic
        if yawn_flag:
            if self.yawn_start is None:
                self.yawn_start = now
            yawn_duration = now - self.yawn_start
            if yawn_duration >= YAWN_WAIT_TIME:
                self.yawn_alarm = True
        else:
            self.yawn_start = None
            self.yawn_alarm = False

        base_color = self.COLOR_GREEN
        if self.drowsy_alarm or self.yawn_alarm:
            base_color = self.COLOR_RED

        y_offset = 30
        if ear is not None:
            cv2.putText(frame, f"EAR: {ear:.3f}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, base_color, 2)
            y_offset += 30

        if mar is not None:
            cv2.putText(frame, f"MAR: {mar:.3f}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, self.COLOR_BLUE, 2)
            y_offset += 30

        if p_closed is not None:
            cv2.putText(frame, f"p_closed (CNN): {p_closed:.2f}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, self.COLOR_WHITE, 2)
            y_offset += 30

        if self.eye_closed_start is not None and not self.drowsy_alarm:
            remaining = max(0.0, EYE_WAIT_TIME - (now - self.eye_closed_start))
            cv2.putText(frame, f"Eyes closed... alarm in {remaining:.1f}s",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, self.COLOR_YELLOW, 2)
            y_offset += 30

        if self.yawn_start is not None and not self.yawn_alarm:
            remaining = max(0.0, YAWN_WAIT_TIME - (now - self.yawn_start))
            cv2.putText(frame, f"Yawning... alarm in {remaining:.1f}s",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, self.COLOR_BLUE, 2)
            y_offset += 30

        if self.drowsy_alarm:
            cv2.putText(frame, "DROWSINESS ALERT!",
                        (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, self.COLOR_RED, 3)

        if self.yawn_alarm:
            cv2.putText(frame, "YAWN / FATIGUE ALERT!",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, self.COLOR_BLUE, 3)

        if SHOW_FPS:
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}",
                        (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, self.COLOR_WHITE, 2)

        return frame

    def update_fps(self, now):
        self.frame_counter += 1
        if now - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_counter / (now - self.last_fps_time)
            self.frame_counter = 0
            self.last_fps_time = now

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    detector = DrowsinessDetector()

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        if SELFIE_FLIP:
            frame = cv2.flip(frame, 1)

        now = time.time()
        detector.update_fps(now)

        frame_out = detector.process_frame(frame, now)

        cv2.imshow("Jetson Drowsiness Detection (CNN + EAR + MAR)", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
