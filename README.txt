🚀 Overview

SignSpeak AI is a real-time sign language translation system that uses a webcam to detect hand gestures and convert them into text.
It combines Computer Vision (MediaPipe + OpenCV) with Machine Learning (Random Forest) to recognize gestures and generate meaningful sentences through an interactive GUI.

💡 This project demonstrates how AI can be used to improve accessibility and communication for sign language users.

✨ Key Features
📷 Live Gesture Detection using webcam
✋ Hand Tracking with 21 landmark points
🤖 ML-Based Classification (Random Forest)
🧠 Confidence Filtering for accurate predictions
📝 Real-Time Sentence Builder
🎨 Modern GUI Interface (Tkinter)
🔁 Edit Controls (Clear / Remove / Space)
⚡ Fast & Lightweight Implementation
🧠 How It Works
Webcam Input
     ↓
Hand Detection (MediaPipe)
     ↓
Landmark Extraction (21 points)
     ↓
Feature Vector (63 values)
     ↓
Random Forest Model
     ↓
Gesture Prediction
     ↓
Sentence Builder GUI
🛠 Tech Stack
Category	Technology Used
Programming	Python
Computer Vision	OpenCV
Hand Tracking	MediaPipe
Machine Learning	Scikit-learn
Data Handling	Pandas, NumPy
GUI	Tkinter
📊 Model Details
Algorithm: Random Forest Classifier
Learning Type: Supervised Learning
Input Features: 63 (21 landmarks × 3 coordinates)
Output Classes:
hello, yes, no, thanks, help, okay
🎯 Accuracy Achieved: ~94.49%
📂 Project Structure
ML_MINI_PROJECT/
│
├── collect_data.py        # Collect gesture data
├── train_model.py         # Train ML model
├── predict_live.py        # Live prediction (CLI)
├── gui_app.py             # Final GUI Application
│
├── gestures.csv           # Dataset
├── model.pkl              # Trained Model
│
└── README.md              # Documentation

🧪 Usage Guide
Start the application
Show a hand gesture in front of the webcam
Hold the gesture steady for 1–2 seconds
The system detects and adds the word
Build a full sentence using gestures
📸 Screenshots (Add Your Own)

Add screenshots here for better presentation:

Example:
- GUI Interface
- Live Detection
- Sentence Output
🎥 Demo (Optional)

You can add a demo video link here (Google Drive / YouTube)

⚠️ Limitations
Limited gesture vocabulary
Sensitive to lighting and background
Similar gestures may be misclassified
Supports only single-hand gestures
🔮 Future Scope
🔊 Text-to-Speech output
📱 Mobile application version
🧠 Deep Learning (CNN / LSTM) integration
✌️ Two-hand gesture recognition
🌐 Multi-language support
📚 Full sentence grammar translation
🎓 Learning Outcomes
Understanding of Computer Vision & ML integration
Real-time data processing using webcam
Feature extraction using landmarks
Training and evaluation of ML models
GUI development using Tkinter
🏁 Conclusion

SignSpeak AI shows how machine learning and computer vision can be combined to create a practical real-time sign language translator.
It highlights the potential of AI in building accessible and inclusive communication systems.

👨‍💻 Author

Sahil Kumar

⭐ Support

If you like this project:

⭐ Star the repository
🍴 Fork it
🧠 Improve it
