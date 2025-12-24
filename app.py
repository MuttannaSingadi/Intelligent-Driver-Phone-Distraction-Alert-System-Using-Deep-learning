from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
from werkzeug.utils import secure_filename
from flask_material import Material

import os
import time
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    IST = None

import numpy as np
import pandas as pd
import cv2
import pygame
from threading import Thread
import subprocess

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO

# -------------------- App & UI --------------------
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app = Flask(__name__, static_url_path='/static')
Material(app)
app.secret_key = 'secret_key_here'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure upload dir exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------- Models --------------------
# Load the trained models
model = load_model(r'vggmodel.h5')     # VGG-based classifier
model1 = YOLO(r"best.pt")              # YOLO classifier (expects .probs)

# Define class labels
class_labels = [
    'safe driving',                    # c0
    'texting - right',                 # c1
    'talking on the phone - right',    # c2
    'texting - left',                  # c3
    'talking on the phone - left',     # c4
    'operating the radio',             # c5
    'drinking',                        # c6
    'reaching behind',                 # c7
    'hair and makeup',                 # c8
    'talking to passenger'             # c9
]

# -------------------- Audio Alert --------------------
AUDIO_PATH = os.path.abspath("alarm.mp3")
frame_count = {label: 0 for label in class_labels}
last_detected = None

pygame.mixer.init()

def play_audio():
    try:
        pygame.mixer.music.load(AUDIO_PATH)
        pygame.mixer.music.play()
    except Exception as e:
        print("[ERROR] pygame audio playback failed:", e)

# -------------------- Email Alerts --------------------
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))  # TLS
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "muttufs565@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "ubqc yays ejxr gmac")  # Use App Password for Gmail
ALERT_FROM   = os.getenv("ALERT_FROM", SMTP_USERNAME)
ALERT_TO     = os.getenv("ALERT_TO", "recipient@example.com")
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", 60))  # throttle per activity label

last_email_time = {}  # {activity_label: epoch_secs}

def now_ist_str():
    if IST:
        return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S %Z")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def format_video_time(frame_idx, fps):
    fps = max(int(fps), 1)
    total_seconds = int(frame_idx // fps)
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def send_alert_email(activity: str, confidence: float, context: str):
    """Send throttled email alert for a given activity (non-safe)."""
    now_ts = time.time()
    last_ts = last_email_time.get(activity, 0)
    if now_ts - last_ts < ALERT_COOLDOWN_SECONDS:
        return  # throttle

    last_email_time[activity] = now_ts
    ts = now_ist_str()

    subject = f"[DriverAlert] ⚠ {activity} detected"
    body = (
        f"An unsafe activity was detected.\n\n"
        f"Time: {ts}\n"
        f"Activity: {activity}\n"
        f"Confidence: {confidence:.2f}\n"
        f"Details: {context}\n"
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = ALERT_FROM
    msg["To"] = ALERT_TO
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls(context=ssl.create_default_context())
            if SMTP_USERNAME and SMTP_PASSWORD:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"[EMAIL] Alert sent → {ALERT_TO}: {subject}")
    except Exception as e:
        print(f"[EMAIL-ERROR] {e}")

# -------------------- Helpers --------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------- Routes --------------------
@app.route('/', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        if username == "admin" and password == "admin":
            return render_template('index.html')
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# ---------- File Video Inference (VGG) ----------
@app.route('/upload_video', methods=["POST"])
def upload_video():
    if 'file' not in request.files:
        flash('⚠ No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('⚠ No video selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        cap = cv2.VideoCapture(save_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25

        # Output video path (in uploads dir)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        global last_detected, frame_count
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            # Preprocess for VGG model
            resized = cv2.resize(frame, (256, 256))
            img_array = img_to_array(resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            predictions = model.predict(img_array, verbose=0)
            predicted_class = int(np.argmax(predictions, axis=1)[0])
            predicted_label = class_labels[predicted_class]
            confidence = float(np.max(predictions))

            # Logic with 20-frame threshold
            if predicted_label != 'safe driving':
                if predicted_label == last_detected:
                    frame_count[predicted_label] += 1
                else:
                    frame_count = {label: 0 for label in class_labels}
                    frame_count[predicted_label] = 1
                    last_detected = predicted_label

                if frame_count[predicted_label] == 20:
                    Thread(target=play_audio).start()
                    context = f"Source: file '{filename}', position {format_video_time(frame_index, fps)} (frame {frame_index})"
                    send_alert_email(predicted_label, confidence, context)
                    frame_count[predicted_label] = 0
            else:
                frame_count = {label: 0 for label in class_labels}
                last_detected = predicted_label
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()

            # Annotate and write
            cv2.putText(frame, f"Prediction: {predicted_label} ({confidence:.2f})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            out.write(frame)
            cv2.imshow('Driver Behavior Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Convert to H.264 (requires ffmpeg in PATH)
        converted_filename = "converted.mp4"
        converted_path = os.path.join(app.config['UPLOAD_FOLDER'], converted_filename)
        os.system(f'ffmpeg -y -i "{output_path}" -vcodec libx264 -acodec aac "{converted_path}"')

        video_url = f"/static/uploads/{converted_filename}"
        result = "Processed successfully"
        return render_template('contact.html', aclass=result, res=1, filename=converted_filename, video_url=video_url)

    return "⚠ Invalid file type", 400

@app.route('/static/uploads/<filename>')
def serve_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(video_path):
        return "⚠ Video not found", 404
    return Response(open(video_path, "rb"), mimetype="video/mp4")

# ---------- Webcam Inference (VGG) ----------
@app.route('/upload_image1', methods=["POST"])
def upload_image1():
    global last_detected, frame_count

    cv2.namedWindow('your_face', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('your_face', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (256, 256))
        frame_array = img_to_array(frame_resized) / 255.0
        frame_array = np.expand_dims(frame_array, axis=0)

        predictions = model.predict(frame_array, verbose=0)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        predicted_label = class_labels[predicted_class]
        confidence = float(np.max(predictions))

        if predicted_label != 'safe driving':
            if predicted_label == last_detected:
                frame_count[predicted_label] += 1
            else:
                frame_count = {label: 0 for label in class_labels}
                frame_count[predicted_label] = 1
                last_detected = predicted_label

            if frame_count[predicted_label] == 20:
                Thread(target=play_audio).start()
                context = "Source: webcam (VGG)"
                send_alert_email(predicted_label, confidence, context)
                frame_count[predicted_label] = 0
        else:
            frame_count = {label: 0 for label in class_labels}
            last_detected = predicted_label
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()

        cv2.putText(frame, f"Prediction: {predicted_label} ({confidence:.2f})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow('your_face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render_template('index.html')

# ---------- Webcam Inference (YOLO classifier) ----------
@app.route('/upload_video1', methods=["POST"])
def upload_video1():
    cv2.namedWindow('Driver Behavior Detection', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Driver Behavior Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0)
    global last_detected, frame_count

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO classification; expects a classification model producing .probs
        results = model1.predict(source=frame, save=False, verbose=False)[0]
        class_index = int(results.probs.top1)
        class_name = class_labels[class_index]
        confidence = float(results.probs.top1conf)

        if class_name != 'safe driving':
            if class_name == last_detected:
                frame_count[class_name] += 1
            else:
                frame_count = {label: 0 for label in class_labels}
                frame_count[class_name] = 1
                last_detected = class_name

            if frame_count[class_name] == 20:
                Thread(target=play_audio).start()
                context = "Source: webcam (YOLO)"
                send_alert_email(class_name, confidence, context)
                frame_count[class_name] = 0
        else:
            frame_count = {label: 0 for label in class_labels}
            last_detected = class_name
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()

        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow('Driver Behavior Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render_template('index.html')

# ---------- Drowsiness demo ----------
@app.route('/drowsiness')
def drowsiness():
    return render_template('drowsiness.html')

@app.route('/start-detection', methods=['POST'])
def start_detection():
    subprocess.Popen(f'python "drowsiness detection.py"', shell=True)
    return "Drowsiness Detection Started!", 200

# -------------------- Main --------------------
if __name__ == '__main__':
    app.run(debug=True)
