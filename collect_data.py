import cv2
import mediapipe as mp
import csv
import os

# ===============================
# CHANGE THIS LABEL EACH TIME
# hello / yes / no / help / thanks / okay
# ===============================
LABEL = "Victory"

CSV_FILE = "gestures.csv"

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Create CSV header if file doesn't exist
if not os.path.exists(CSV_FILE):
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")

    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# Open webcam
cap = cv2.VideoCapture(0)

sample_count = 0

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to access webcam")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    row = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

    # Display instructions
    cv2.putText(frame, f"Label: {LABEL}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Samples Saved: {sample_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, "Press 'S' to Save | Press 'Q' to Quit", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Collect Gesture Data", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and row is not None:
        row.append(LABEL)
        with open(CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        sample_count += 1
        print(f"Saved sample {sample_count} for label: {LABEL}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()