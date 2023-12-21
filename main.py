from flask import Flask, render_template, request, jsonify
from google.cloud import firestore
import os
import requests
import cv2
import numpy as np
import face_recognition

app = Flask(__name__)

# Initialize Firebase Admin SDK
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\renat\OneDrive\Desktop\IOT-Py\firebase.json"
db = firestore.Client()

@app.route('/')
def index():
    return render_template('index.html', message=None, match_message=None)

@app.route('/', methods=['POST'])
def check_roll_number():
    roll_number = request.form.get('roll_number')

    # Query Firestore for the roll number
    students_ref = db.collection('students')
    query = students_ref.where('rollNumber', '==', roll_number).limit(1)
    result = query.stream()

    message = None

    for doc in result:
        data = doc.to_dict()
        image_url = data.get('imageUrl', '')

        if image_url:
            # Save the image locally in the "images" folder with the name "databaseimage.jpg"
            save_image_locally(image_url, 'databaseimage')
            message = f"Roll number {roll_number} found. "

    # If roll number not found, delete the databaseimage.jpg file and set the message
    if not message:
        delete_image('databaseimage')
        message = f"Roll number {roll_number} not found."

    return render_template('index.html', message=message, match_message=None)

@app.route('/capture_image', methods=['POST'])
def capture_image():
    # Capture image from the laptop camera
    camera = cv2.VideoCapture(0)
    return_code, frame = camera.read()
    camera.release()

    if return_code:
        # Save the captured image as "liveimage.jpg" in the "images" folder
        image_path = os.path.join(app.root_path, 'images', 'liveimage.jpg')
        cv2.imwrite(image_path, frame)
        message = "Live image captured and saved."
    else:
        message = "Failed to capture live image."

    return render_template('index.html', message=message, match_message=None)

@app.route('/domatching', methods=['POST'])
def do_matching():
    # Load the face encodings for the saved images
    database_image_path = os.path.join(app.root_path, 'images', 'databaseimage.jpg')
    live_image_path = os.path.join(app.root_path, 'images', 'liveimage.jpg')

    if os.path.exists(database_image_path) and os.path.exists(live_image_path):
        # Load face encodings
        database_image = face_recognition.load_image_file(database_image_path)
        live_image = face_recognition.load_image_file(live_image_path)

        # Get face encodings
        database_encodings = face_recognition.face_encodings(database_image)
        live_encodings = face_recognition.face_encodings(live_image)

        if database_encodings and live_encodings:
            # Use the first face encoding in each image (assuming one face per image)
            database_encoding = database_encodings[0]
            live_encoding = live_encodings[0]

            # Compare the face encodings
            results = face_recognition.compare_faces([database_encoding], live_encoding)

            # Check if the images match
            if results[0]:
                match_message = "Images matched! Attendance Added"
            else:
                match_message = "Images not matched. Absent Added"
        else:
            match_message = "No faces found in one or both images."
    else:
        match_message = "Images not available for matching."

    return render_template('index.html', message=None, match_message=match_message)

def save_image_locally(image_url, filename):
    try:
        # Download the image content
        image_content = requests.get(image_url).content

        # Save the image with the specified filename in the "images" folder
        images_folder = os.path.join(app.root_path, 'images')
        image_path = os.path.join(images_folder, f'{filename}.jpg')

        with open(image_path, 'wb') as f:
            f.write(image_content)

        print(f"Image '{filename}.jpg' saved locally.")
        return image_path
    except Exception as e:
        print(f"Error saving image '{filename}.jpg': {e}")
        return None

def delete_image(filename):
    # Delete the specified image file from the "images" folder
    image_path = os.path.join(app.root_path, 'images', f'{filename}.jpg')
    
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Image '{filename}.jpg' deleted.")
    else:
        print(f"Image '{filename}.jpg' not found.")

if __name__ == '__main__':
    app.run(debug=True)
