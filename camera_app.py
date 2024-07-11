import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import onnxruntime as ort
from tkinter import messagebox
import datetime
import os
import pygame  # For playing the siren sound

# Initialize pygame for sound playback
pygame.mixer.init()

# Load the siren sound
siren_sound = "Smoking detector siren.wav"
pygame.mixer.music.load(siren_sound)

# Global variables for recording and detection
is_recording = False
output = None
output_file = None
objects_detected = False
siren_playing = False

# Create detections folder if it doesn't exist
output_folder = "detections"
os.makedirs(output_folder, exist_ok=True)

# Load the YOLOv5 ONNX model
model_path = 'C:/Users/Hp/yolov5/runs/train/exp7/weights/best.onnx'
ort_session = ort.InferenceSession(model_path)

# Function to preprocess the image
def preprocess(img, input_shape):
    img = cv2.resize(img, input_shape)
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img[np.newaxis, ...]  # add batch dimension
    img = img.astype(np.float32) / 255.0  # normalize
    return img

# Function for Non-Maximum Suppression (NMS)
def non_max_suppression(boxes, scores, score_threshold, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, iou_threshold)
    return indices.flatten()

# Function to start recording
def start_recording():
    global output, output_file, output_folder
    
    # Generate filename with current date and time including seconds
    current_datetime = datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S%p")
    output_file = os.path.join(output_folder, f"recording_{current_datetime}.avi")
    
    # Initialize video writer
    output = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
    print(f"Recording started: {output_file}")

# Function to stop recording
def stop_recording():
    global output, output_file
    
    if output is not None:
        output.release()
        print(f"Recording stopped: {output_file}")

# Function to run inference and draw bounding boxes
def run_inference_and_draw_boxes(frame, ort_session):
    global is_recording, objects_detected, siren_playing
    
    input_shape = (640, 640)
    input_tensor = preprocess(frame, input_shape)
    outputs = ort_session.run(None, {'images': input_tensor})

    detections = outputs[0][0]  # shape is (1, 25200, 6)
    
    # Filter detections by confidence threshold and apply NMS
    boxes = []
    scores = []
    for detection in detections:
        x1, y1, x2, y2, score, class_id = detection
        if score > 0.3:  # confidence threshold
            # Calculate center and dimensions of the box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            width = int(x2 - x1)
            height = int(y2 - y1)
            
            # Adjust coordinates to form a tight bounding box around the object
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
    
    if boxes:
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Apply Non-Maximum Suppression (NMS)
        indices = non_max_suppression(boxes, scores, score_threshold=0.3, iou_threshold=0.5)
        
        # Draw filtered boxes
        for idx in indices:
            x1, y1, x2, y2 = boxes[idx]
            score = scores[idx]  # score of the detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'Smoking detected: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Start recording if not already recording
            if not is_recording:
                start_recording()
                is_recording = True
                objects_detected = True
                
                # Play the siren sound if not already playing
                if not siren_playing:
                    pygame.mixer.music.play()
                    siren_playing = True
    
    else:
        # No objects detected
        if is_recording and objects_detected:
            stop_recording()
            is_recording = False
            objects_detected = False
            
            # Stop the siren sound
            if siren_playing:
                pygame.mixer.music.stop()
                siren_playing = False
    
    return frame

# Function to open the webcam
def open_webcam():
    global output, is_recording
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference and draw bounding boxes
        frame = run_inference_and_draw_boxes(frame, ort_session)
        
        # Display the frame
        cv2.imshow('YOLOv5 Object Detection', frame)
        
        # Write frame to video file if recording
        if is_recording:
            output.write(frame)
        
        # Exit on pressing 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Release resources
    stop_recording()
    cap.release()
    cv2.destroyAllWindows()

# Function to open detections folder
def open_detections_folder():
    os.startfile(output_folder)

# Create the main window
root = tk.Tk()
root.title("Gesture Smoke Guard")

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the window size to full screen
root.geometry('800x600')

# Load and set the background image
background_image = Image.open("background_image.jpg")  # Replace with your background image path
background_image = background_image.resize((screen_width, screen_height), Image.LANCZOS)  # Resize to fit window
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Define button styles
button_open_webcam_normal_bg = "white"  # Green background
button_open_webcam_hover_bg = "#2ecc71"   # Darker green on hover
button_open_webcam_normal_fg = "#2ecc71"    # White text
button_open_webcam_hover_fg = "white"     # White text on hover

button_open_folder_normal_bg = "white"  # Red background
button_open_folder_hover_bg = "#e74c3c"   # Darker red on hover
button_open_folder_normal_fg = "#e74c3c"    # White text
button_open_folder_hover_fg = "white"     # White text on hover

# Function to apply hover effect
def on_enter_open_webcam(event):
    event.widget.config(bg=button_open_webcam_hover_bg, fg=button_open_webcam_hover_fg)

def on_leave_open_webcam(event):
    event.widget.config(bg=button_open_webcam_normal_bg, fg=button_open_webcam_normal_fg)

def on_enter_open_folder(event):
    event.widget.config(bg=button_open_folder_hover_bg, fg=button_open_folder_hover_fg)

def on_leave_open_folder(event):
    event.widget.config(bg=button_open_folder_normal_bg, fg=button_open_folder_normal_fg)

# Create a frame to hold the buttons
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, pady=20)

# Create and place the buttons inside the frame
btn_open_webcam = tk.Button(button_frame, text="Open Webcam", command=open_webcam, bg=button_open_webcam_normal_bg, fg=button_open_webcam_normal_fg, font=('Helvetica', 12, 'bold'))
btn_open_webcam.pack(side=tk.LEFT)
btn_open_webcam.bind("<Enter>", on_enter_open_webcam)
btn_open_webcam.bind("<Leave>", on_leave_open_webcam)

btn_open_folder = tk.Button(button_frame, text="Open Detections Folder", command=open_detections_folder, bg=button_open_folder_normal_bg, fg=button_open_folder_normal_fg, font=('Helvetica', 12, 'bold'))
btn_open_folder.pack(side=tk.LEFT)
btn_open_folder.bind("<Enter>", on_enter_open_folder)
btn_open_folder.bind("<Leave>", on_leave_open_folder)

# Start the Tkinter main loop
root.mainloop()
