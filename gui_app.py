import cv2
import mediapipe as mp
import pickle
import numpy as np
import tkinter as tk
from tkinter import Label, Button, StringVar
from PIL import Image, ImageTk

# ===============================
# LOAD MODEL
# ===============================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===============================
# TKINTER APP
# ===============================
root = tk.Tk()
root.title("Sign Language Sentence Builder")
root.geometry("1100x750")
root.configure(bg="#f2f2f2")

# Webcam
cap = cv2.VideoCapture(0)

# Variables
current_sign_var = StringVar()
current_sign_var.set("Current Sign: None")

confidence_var = StringVar()
confidence_var.set("Confidence: 0%")

sentence_var = StringVar()
sentence_var.set("Sentence: ")

sentence_words = []
last_added_sign = ""
stable_sign = ""
stable_count = 0

# Settings
ADD_THRESHOLD = 8       # lower = faster word add
CONFIDENCE_THRESHOLD = 70

# ===============================
# UI ELEMENTS
# ===============================
title_label = Label(
    root,
    text="Real-Time Sign Language Sentence Builder",
    font=("Arial", 24, "bold"),
    bg="#f2f2f2",
    fg="#222"
)
title_label.pack(pady=10)

video_label = Label(root, bg="#f2f2f2")
video_label.pack()

current_sign_label = Label(
    root,
    textvariable=current_sign_var,
    font=("Arial", 18, "bold"),
    bg="#f2f2f2",
    fg="green"
)
current_sign_label.pack(pady=10)

confidence_label = Label(
    root,
    textvariable=confidence_var,
    font=("Arial", 16),
    bg="#f2f2f2",
    fg="blue"
)
confidence_label.pack()

sentence_title = Label(
    root,
    text="Generated Sentence:",
    font=("Arial", 18, "bold"),
    bg="#f2f2f2",
    fg="#111"
)
sentence_title.pack(pady=(20, 5))

sentence_label = Label(
    root,
    textvariable=sentence_var,
    font=("Arial", 20, "bold"),
    bg="white",
    fg="black",
    wraplength=950,
    width=50,
    height=3,
    relief="solid",
    bd=2
)
sentence_label.pack(pady=10)

# ===============================
# FUNCTIONS
# ===============================
def clear_sentence():
    global sentence_words, last_added_sign
    sentence_words = []
    last_added_sign = ""
    sentence_var.set("")

def remove_last_word():
    global sentence_words
    if sentence_words:
        sentence_words.pop()
        sentence_var.set(" ".join(sentence_words))

def add_space():
    global sentence_words
    sentence_words.append("|")
    sentence_var.set(" ".join(sentence_words))

def exit_app():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# ===============================
# CAMERA UPDATE FUNCTION
# ===============================
def update_frame():
    global stable_sign, stable_count, last_added_sign

    success, frame = cap.read()
    if not success:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    predicted_label = "No Hand"
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            if len(row) == 63:
                X_input = np.array(row).reshape(1, -1)
                predicted_label = model.predict(X_input)[0]

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_input)[0]
                    confidence = np.max(probs) * 100

    # Normalize label display
    predicted_label = str(predicted_label).lower()

    # If confidence too low, ignore prediction
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_label = "Unknown"

    # Sentence builder logic
    if predicted_label not in ["No Hand", "Unknown"]:
        if predicted_label == stable_sign:
            stable_count += 1
        else:
            stable_sign = predicted_label
            stable_count = 1

        if stable_count >= ADD_THRESHOLD and predicted_label != last_added_sign:
            sentence_words.append(predicted_label.upper())
            sentence_var.set(" ".join(sentence_words))
            last_added_sign = predicted_label
            stable_count = 0
    else:
        stable_sign = ""
        stable_count = 0

    current_sign_var.set(f"Current Sign: {predicted_label.upper()}")
    confidence_var.set(f"Confidence: {confidence:.2f}%")

    # Show on camera frame
    cv2.putText(frame, f"Sign: {predicted_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, "Hold gesture steady to add word", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Convert for Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# ===============================
# BUTTONS
# ===============================
button_frame = tk.Frame(root, bg="#f2f2f2")
button_frame.pack(pady=20)

clear_btn = Button(
    button_frame,
    text="Clear Sentence",
    font=("Arial", 14),
    command=clear_sentence,
    width=15,
    bg="#ffcc00"
)
clear_btn.grid(row=0, column=0, padx=10)

remove_btn = Button(
    button_frame,
    text="Remove Last",
    font=("Arial", 14),
    command=remove_last_word,
    width=15,
    bg="#ff9999"
)
remove_btn.grid(row=0, column=1, padx=10)

space_btn = Button(
    button_frame,
    text="Add Space",
    font=("Arial", 14),
    command=add_space,
    width=15,
    bg="#99ccff"
)
space_btn.grid(row=0, column=2, padx=10)

exit_btn = Button(
    button_frame,
    text="Exit",
    font=("Arial", 14),
    command=exit_app,
    width=15,
    bg="#ff6666"
)
exit_btn.grid(row=0, column=3, padx=10)

# Start updating frames
update_frame()

# Run app
root.mainloop()