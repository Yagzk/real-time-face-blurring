# Real-Time Face Blurring with Python

This project performs **real-time face detection and face blurring** using a webcam.
Detected faces are automatically blurred to demonstrate **privacy protection and censorship**
techniques in image processing.

The application works in real time and remains stable even when the face moves or turns slightly.

---

## 🎯 Project Purpose

The aim of this project is to:
- Detect human faces from a live camera stream
- Automatically blur detected face regions
- Demonstrate real-time image processing for privacy and anonymization purposes

This project is suitable for educational use in computer vision and image processing courses.

---

## 🛠 Technologies Used

| Component | Description |
|---------|-------------|
| Python | Programming language |
| OpenCV | Video capture, image processing, blurring |
| MediaPipe | Face detection model |
| OpenCV Tracker (KCF) | Face tracking for real-time performance |

---

## ⚙️ How It Works

1. The webcam provides a live video stream.
2. Face detection is performed **periodically** using MediaPipe.
3. Once a face is detected, an OpenCV **tracking algorithm** follows the face in subsequent frames.
4. The detected/tracked face region is blurred using **Gaussian Blur**.
5. If tracking fails, face detection is triggered again.
6. The processed video is displayed to the user in real time.

This hybrid approach (detection + tracking) improves performance and prevents blur interruptions.

---

## 🚀 Features

- Real-time face detection
- Automatic face blurring
- Stable blur even when the face moves
- Adjustable blur intensity
- High-resolution camera support (1920×1080)
- Keyboard controls for live tuning

---

## ⌨️ Keyboard Controls

| Key | Action |
|----|-------|
| `q` | Quit application |
| `+` / `-` | Increase / decrease blur intensity |
| `[` / `]` | Decrease / increase detection frequency |
| `p` / `o` | Increase / decrease face padding |

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/face-blurring.git
cd face-blurring
