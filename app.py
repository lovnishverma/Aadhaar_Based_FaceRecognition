from PIL import Image
import numpy as np
import cv2
import requests
import face_recognition
import os
import streamlit as st

# Set page title and description
st.set_page_config(
    page_title="Aadhaar Based Face Recognition Attendance System",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.title("Aadhaar-Based Face Recognition Attendance System 📷")
st.markdown("This app recognizes faces in an image, verifies Aadhaar card details, and updates attendance records with the current timestamp.")

# Load images for face recognition
Images = []   # List to store Images
classnames = []  # List to store classnames
aadhar_numbers = []  # List to store Aadhaar numbers

directory = "photos"
myList = os.listdir(directory)

for cls in myList:
    if os.path.splitext(cls)[1] in [".jpg", ".jpeg"]:
        img_path = os.path.join(directory, cls)
        curImg = cv2.imread(img_path)
        Images.append(curImg)
        classnames.append(os.path.splitext(cls)[0])
        # Assume Aadhaar number is part of the image filename (e.g., "123456_Sarwan.jpg")
        aadhar_numbers.append(cls.split('_')[0])

# Function to validate Aadhaar card number
def validate_aadhaar(aadhaar):
    # Implement your Aadhaar card validation logic here
    # For simplicity, let's assume any 6-digit number is a valid Aadhaar card
    return len(aadhaar) == 6 and aadhaar.isdigit()

# Function to update Aadhaar data
def update_data(name, aadhaar_number):
    url = "https://attendanceviaface.000webhostapp.com"
    url1 = "/update.php"
    data = {'name': name, 'aadhaar': aadhaar_number}
    response = requests.post(url + url1, data=data)
    
    if response.status_code == 200:
        st.success("Data updated on: " + url)
    else:
        st.warning("Data not updated")

# Function to display image with overlay
def display_image_with_overlay(image, name):
    # Add overlay to the image (e.g., bounding box and name)
    # ...

    # Apply styling with CSS
    st.markdown('<style>img { animation: pulse 2s infinite; }</style>', unsafe_allow_html=True)
    st.image(image, use_column_width=True, output_format="PNG")

# Take input Aadhaar card details
aadhaar_number = st.text_input("Enter your Last 6-digits Aadhaar Number:")
    
# Take picture using the camera 
img_file_buffer = st.camera_input("Take a picture")

# Load images for face recognition
encodeListknown = [face_recognition.face_encodings(img)[0] for img in Images]

if img_file_buffer is not None:
    # Validate Aadhaar card number
    if validate_aadhaar(aadhaar_number):
        test_image = Image.open(img_file_buffer)
        image = np.asarray(test_image)

        imgS = cv2.resize(image, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        name = "Unknown"  # Default name for unknown faces
        match_found = False  # Flag to track if a match is found

        # Checking if faces are detected
        if len(encodesCurFrame) > 0:
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                # Assuming that encodeListknown is defined and populated in your code
                matches = face_recognition.compare_faces(encodeListknown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListknown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classnames[matchIndex].upper()
                    
                    # Check if Aadhaar number is found in the database
                    if aadhaar_number not in aadhar_numbers:
                        st.error("Face recognized, but Aadhaar number not found in the database.")
                    else:
                        # Update data only if a known face is detected and Aadhaar number is valid
                        update_data(name, aadhaar_number)
                        match_found = True  # Set the flag to True
                    
                else:
                    # Face recognized, but not matched with Aadhaar number
                    st.error("Face recognized, but Aadhaar number does not match.")

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            display_image_with_overlay(image, name)

            # Display the name corresponding to the entered Aadhaar number
            if not match_found:
                # Match Aadhaar number with the list
                aadhar_index = aadhar_numbers.index(aadhaar_number) if aadhaar_number in aadhar_numbers else None
                if aadhar_index is not None:
                    st.success(f"Match found: {classnames[aadhar_index]}")
                else:
                    st.warning("Attendance Not Updated, Aadhaar number not found in the database.")
            else:
                st.success(f"Face recognized: {name}")

        else:
            st.warning("No faces detected in the image. Face recognition failed.")

    else:
        st.error("Invalid Aadhaar card number. Please enter a valid 6-digit Aadhaar number.")
