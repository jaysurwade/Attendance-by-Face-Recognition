import cv2
import os
import numpy as np

def train_model(dataset_folder="dataset"):
    images = []
    labels = []

    for label in os.listdir(dataset_folder):
        label_path = os.path.join(dataset_folder, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                images.append(img)
                labels.append(int(label))

    labels = np.array(labels)

    # Use LBPH (Local Binary Pattern Histogram) recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)

    recognizer.save("face_recognizer.yml")

    print("Model trained successfully.")


def capture_images(name,roll_number,num_images=30, output_folder="dataset"):
    sname = name.split()
    sname = "_".join(sname)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create output folder if it doesn't exist
    output_path = os.path.join(output_folder, str(roll_number))
    os.makedirs(output_path, exist_ok=True)

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    print(f"Capturing {num_images} images for Roll Number {roll_number}...")

    count = 0
    delay = 0
    while count < num_images:
        ret, frame = cap.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            if w >= 150 and h>=150:
                face_roi = gray_frame[y:y+h, x:x+w]
                img_name = f"{sname}_{roll_number}_{count}.png"
                img_path = os.path.join(output_path, img_name)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Image Capture: {count}", (x + 6, y + h - 6), font, 1, (255, 255, 255), 2)
                cv2.imwrite(img_path, face_roi)
                count += 1

        cv2.imshow("Capture Images", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"{num_images} images captured for Roll Number {roll_number}")

while True:
    sname = input("Enter Student Name").lower().strip()
    if(sname):
        roll_number = input("Enter Roll Number").lower().strip()
        if(roll_number):
            
            capture_images(sname,roll_number)
            train_model()
            break
        else:
            print("entries should not be blank")
    else:
        print("entries should not be blank")      