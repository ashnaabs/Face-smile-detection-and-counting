import cv2
import numpy as np
import os


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')




image_path = '/home/ashna/jest/download (4).jpeg'
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

smile_count=0


#face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Drawing rectangles 
for (x, y, w, h) in faces:
    # Extract the face region
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # Perform smile detection in the face region
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)

    # Increment smile count if a smile is detected
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
        #smile_count += 1
    
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

output_folder_path = "/home/ashna/jest/smile"

if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
# Loop through detected faces and analyze smiles
for i, (x, y, w, h) in enumerate(faces):
    # Extract the face region
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # Perform smile detection in the face region
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
    if len(smiles) > 0:
        smile_count += 1
        smiling_face_path = os.path.join(output_folder_path, f'smiling_face_{i}.jpg')
        
        # Crop and save the smiling face
        cv2.imwrite(smiling_face_path, roi_color)

# Result
cv2.imshow('Group Photo with Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Printing the number of detected faces
num_faces = len(faces)
print(f'Number of faces detected: {num_faces}')
print(f'Number of smiling faces detected: {smile_count}')

