# Hand Gesture Tree Visualizer

## About This Project
Hi! I'm Aaryan Verma, and this is my personal project: **Hand Gesture Tree Visualizer**. I created this tool to explore the intersection of computer vision, interactive art, and creative coding. With your webcam and real-time hand tracking, you can reveal, rotate, and grow a glowing 3D tree using simple hand gestures. The visualization is optimized for smooth performance and features beautiful bloom and particle effects for an immersive experience.

## Features

- Real-time hand gesture recognition in the browser using your webcam
- 3D tree generation and rendering with OpenCV and NumPy
- Interactive controls for growing, rotating, and exploding the tree
- Visual effects: bloom, glow, and particles
- Flask + Flask-SocketIO backend for real-time communication
- Deployable on Render or any cloud platform

---

## Try It Out!
**[🌳 Render Demo (Click Here)](https://hand-gesture-tree-visualizer.onrender.com)**

---

## How to Run Locally

1. **Clone the repository:**
   ```sh
   git clone https://github.com/AaryanVerma17/Hand-Gesture-Tree-Visualizer.git
   cd Advanced-Hand-Gesture-3D-Visualizer
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the server:**
   ```sh
   python Run.py
   ```

4. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

---

## Requirements

- Python 3.8+
- Flask
- Flask-SocketIO
- OpenCV (opencv-python-headless)
- MediaPipe
- NumPy
- eventlet
- gunicorn (for production)

All dependencies are listed in [`requirements.txt`](requirements.txt).

---

## Project Structure

```
index.html
README.md
requirements.txt
Run.py
.vscode/
    settings.json
```

---

## Relevant Wikipedia Links

- [Hand tracking](https://en.wikipedia.org/wiki/Hand_tracking)
- [Computer vision](https://en.wikipedia.org/wiki/Computer_vision)
- [OpenCV](https://en.wikipedia.org/wiki/OpenCV)
- [NumPy](https://en.wikipedia.org/wiki/NumPy)
- [MediaPipe](https://en.wikipedia.org/wiki/MediaPipe)
- [3D computer graphics](https://en.wikipedia.org/wiki/3D_computer_graphics)
- [GitHub Forking](https://en.wikipedia.org/wiki/Fork_(software_development))

---