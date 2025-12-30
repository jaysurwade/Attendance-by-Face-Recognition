import cv2
import os
from datetime import datetime
import numpy as np 

# --- 1. Function to capture images of a student with a given roll number ---
def capture_images(roll_number, student_name, num_images=30):
    """Captures face images for a student and saves them to the dataset directory."""
    
    # Ensure dataset/RollNumber directory exists
    path = os.path.join('dataset', str(roll_number))
    if not os.path.exists(path):
        os.makedirs(path)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    count = 0
    
    print(f"\n--- CAPTURE MODE ---\nCapturing images for {student_name} (Roll No: {roll_number}). Look at the camera.")

    while count < num_images: 
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting capture.")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Only detect large faces to ensure quality
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Save the captured face region
            face_roi = gray[y:y+h, x:x+w]
            file_name = f"{student_name}_{roll_number}_{count}.jpg"
            cv2.imwrite(os.path.join(path, file_name), face_roi)
            count += 1
            cv2.putText(frame, f"Captured: {count}/{num_images}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
            
        cv2.imshow('Image Capture', frame)
        # Wait a bit longer to prevent too many similar images
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Image capturing complete.")


# --- 2. Function to train the face recognition model ---
def train_model():
    """Reads all images from the dataset and trains the LBPH face recognizer."""
    path = 'dataset'
    image_paths = []
    labels = []
    
    if not os.path.exists(path):
        print(f"Error: Dataset directory '{path}' not found. Capture images first.")
        return

    # Loop through all roll number directories
    for roll_number_dir in os.listdir(path):
        label_path = os.path.join(path, roll_number_dir)
        
        if os.path.isdir(label_path):
            try:
                # The roll number is the label (must be integer)
                roll_number = int(roll_number_dir) 
            except ValueError:
                print(f"Skipping non-integer directory: {roll_number_dir}")
                continue
                
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                # Ensure it's a file
                if os.path.isfile(file_path): 
                    image_paths.append(file_path)
                    labels.append(roll_number)

    # Convert lists to NumPy arrays
    faces = []
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if the image was read successfully and is not empty
        if img is not None and img.size > 0:
            faces.append(img)
            
    if not faces:
        print("Error: No valid face images found in the dataset to train the model.")
        return

    print(f"\n--- TRAINING MODE ---\nFound {len(faces)} images for training. Starting model training...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.write('face_recognizer.yml')
    print("✅ Model trained and saved as face_recognizer.yml")


# --- 3. Function to mark attendance for a given roll number (FIXED: handles missing CSV file) ---
def mark_attendance_for_roll_number(name, roll_number):
    """Marks attendance in the CSV file, ensuring no duplicate entries for the same day."""
    attendance_file = "attendance.csv"
    current_date = datetime.now().strftime("%Y-%m-%d")

    # FIX: Ensure the CSV file exists and has a header if it's new
    if not os.path.exists(attendance_file) or os.path.getsize(attendance_file) == 0:
        with open(attendance_file, "w") as file:
            file.write("Name,RollNumber,Timestamp\n")
            
    entry_exists = False
    
    # Check if an entry for the same roll number and day already exists
    with open(attendance_file, "r") as file:
        # Read all lines and skip the header (first line)
        existing_entries = file.readlines()[1:] 
        
        for entry in existing_entries:
            entry_data = entry.strip().split(',')
            
            # Check for: correct number of columns (3), matching Roll Number, and matching Date part of Timestamp
            if len(entry_data) == 3 and entry_data[1] == str(roll_number) and entry_data[2].startswith(current_date):
                entry_exists = True
                break

    # If no existing entry is found for today, add the new entry
    if not entry_exists:
        with open(attendance_file, "a") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"{name},{roll_number},{timestamp}\n")
            print(f"Attendance marked: {name} ({roll_number}) at {timestamp}")


# --- 4. Function to mark attendance using the trained model (FIXED: robust name lookup) ---
def mark_attendance():
    """Uses the trained model to recognize faces and mark attendance."""
    
    # Check for necessary files/directories before starting
    if not os.path.exists("face_recognizer.yml"):
        print("\nError: 'face_recognizer.yml' not found. Please run train_model() first.")
        return
    if not os.path.exists("dataset"):
        print("\nError: 'dataset' directory not found. Please run capture_images() first.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_recognizer.yml")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n--- ATTENDANCE MODE ---\nScanning faces. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image from camera.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Enforce minimum face size for better accuracy
            if w > 150 and h > 150: 
                face_roi = gray_frame[y:y+h, x:x+w]

                # Predict the label (roll number) for the face
                label, confidence = recognizer.predict(face_roi)
                
                # Draw the bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Default text
                display_text = "UnKnown"

                if confidence < 70:  # Confidence threshold (lower is better, around 50-70 is common)
                    person_name = "Unknown Name" # Default
                    
                    # CRITICAL FIX: Robust way to get the name from the dataset
                    try:
                        label_path = os.path.join('dataset', str(label))
                        
                        if os.path.exists(label_path):
                            # Find the first image file (assuming file naming convention is "Name_RollNumber_Index.jpg")
                            files = os.listdir(label_path)
                            imgfile = next((f for f in files if os.path.isfile(os.path.join(label_path, f))), None)

                            if imgfile:
                                # Extract name from the file name (e.g., "JohnDoe_101_1.jpg" -> "JohnDoe")
                                person_name = imgfile.split("_")[0]
                                display_text = f"Name: {person_name} | Roll No: {label} ({round(confidence)}%)"
                                
                                # Mark attendance only if a known face is confidently detected
                                mark_attendance_for_roll_number(person_name, label)
                            else:
                                display_text = f"Roll No: {label} (No image file)"
                        else:
                            display_text = f"Roll No: {label} (Dir missing)"
                            
                    except Exception as e:
                        print(f"Error processing predicted label {label}: {e}")
                        display_text = "Error in lookup"
                else:
                    display_text = f"UnKnown ({round(confidence)}%)"
                
                # Display the determined text
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, display_text, (x + 6, y + h - 6), font, 0.5, (255, 255, 255), 1)
    
        cv2.imshow("Mark Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --- EXECUTION STEPS ---

# STEP 1: Capture images for all students (Run this part first, commenting out the others)
# NOTE: Run this line, perform the capture, then comment it out before moving to STEP 2.
# capture_images(roll_number=101, student_name='JohnDoe')
# capture_images(roll_number=102, student_name='JaneSmith')


# STEP 2: Train the model (Run this once after all captures are done)
# NOTE: Run this line, ensure 'face_recognizer.yml' is created, then comment it out.
# train_model()


# STEP 3: Mark attendance (Run this only after the model is trained)
mark_attendance()