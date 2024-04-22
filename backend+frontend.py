import cv2
import numpy as np
import easyocr
import imutils
from tkinter import *
from PIL import Image, ImageTk
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import tensorflow as tf
import pywhatkit as pwk  # Import pywhatkit

# Initialize Tkinter
root = Tk()
root.title("License Plate Recognition")

# Create a new client and connect to the server
uri = "mongodb+srv://minorproject123:minorproject123@cluster0.leufw9o.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["licenseplate"]
details_collection = db["details"]
departments_collection = db["department"]

# Function to allocate parking space based on department
def allocate_parking_space(department):
    # Retrieve the department document from MongoDB
    department_doc = departments_collection.find_one({"department": department})

    if department_doc:
        available_spaces = department_doc.get("available_spaces", 0)
        if available_spaces > 0:
            # Deduct one parking space
            departments_collection.update_one({"department": department}, {"$inc": {"available_spaces": -1}})
            return True
        else:
            return False  # No available parking space
    else:
        return False  # Department not found

# Load the input image
image_path = 'imagesused\FiestaNov2013.JPG'  # Replace with the actual image file path
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter for noise removal
bfilter = cv2.bilateralFilter(gray, 11, 11, 17)

# Apply Canny edge detection
edged = cv2.Canny(bfilter, 30, 200)

# Find contours
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Find the contour representing the number plate location
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

# Create a blank mask
mask = np.zeros(gray.shape, np.uint8)

# Draw contours on the mask
new_image = cv2.drawContours(mask, [location], 0, 255, -1)

# Apply bitwise AND operation to get the number plate region
new_image = cv2.bitwise_and(image, image, mask=mask)

# Find non-black pixels in the mask to crop the number plate region
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))

# Add buffer to the cropped region
cropped_image = gray[x1:x2 + 3, y1:y2 + 3]

# Apply thresholding to the license plate region
_, thresh_plate = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours of characters
contours, _ = cv2.findContours(thresh_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area
min_area = 50
character_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Draw bounding boxes around segmented characters
segmented_characters_with_boxes = cropped_image.copy()
for contour in character_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(segmented_characters_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

def preprocess_character(character_img):
    # Resize the character image to match the input shape of the model
    character_img_resized = cv2.resize(character_img, (32, 32))
    # Convert the resized image to RGB
    character_img_rgb = cv2.cvtColor(character_img_resized, cv2.COLOR_GRAY2RGB)
    # Normalize the image
    character_img_normalized = character_img_rgb.astype('float32') / 255.0
    # Add batch dimension
    character_img_preprocessed = np.expand_dims(character_img_normalized, axis=0)
    return character_img_preprocessed

# Load the saved model
loaded_model = tf.keras.models.load_model("character_recognition_model.h5")

target_names = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}
# List to store recognized characters
recognized_characters = []

# Iterate through segmented character contours
for contour in character_contours:
    # Extract character image from the cropped image
    x, y, w, h = cv2.boundingRect(contour)
    character_img = cropped_image[y:y+h, x:x+w]
    
    # Preprocess character image
    preprocessed_character_img = preprocess_character(character_img)
    
    # Predict the character using the loaded model
    predicted_class_idx = np.argmax(loaded_model.predict(preprocessed_character_img))
    predicted_character = target_names[predicted_class_idx]
    
    # Append the predicted character to the list of recognized characters
    recognized_characters.append(predicted_character)

recognized_characters.reverse()
recognized_characters_str = ''.join(recognized_characters)
rs=recognized_characters_str[1:len(recognized_characters)]
print(recognized_characters_str[1:len(recognized_characters)])

# Query MongoDB for car owner, department, and parking number
query = {"licenseplate": rs}
document = details_collection.find_one(query)

if document:
    owner_name = document.get("name", "Unknown")
    department = document.get("department", "Unknown")
    phone_number = document.get("phoneno", "Unknown")  # Get phone number from document
else:
    owner_name = "Unknown"
    department = "Unknown"
    phone_number = "Unknown"

# Debugging: Print department info
print("Department:", department)

# Allocate parking space based on department
allocation_success = allocate_parking_space(department)

# Get parking number from the department collection
parking_number = "Unavailable"
department_doc = departments_collection.find_one({"department": department})
if department_doc:
    parking_number = department_doc.get("available_spaces", "Unavailable")

# Debugging: Print allocation status
print("Allocation Success:", allocation_success)

# Function to send WhatsApp message
def send_whatsapp_message(phone_number, message):
    try:
        pwk.sendwhatmsg_instantly(f"+{phone_number}", message)
        print("WhatsApp message sent successfully.")
        return True
    except Exception as e:
        print("Error sending WhatsApp message:", e)
        return False

# Send WhatsApp message with parking details
whatsapp_message = f"Hello, {owner_name}!\nYour parking details:\nDepartment: {department}\nParking Number: {parking_number}"
send_whatsapp_message(phone_number, whatsapp_message)

# Display the processed image on the left side
processed_image = Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
processed_image.thumbnail((400, 400))  # Adjust size if needed
photo = ImageTk.PhotoImage(processed_image)
image_label = Label(root, image=photo)
image_label.grid(row=0, column=0, padx=10, pady=10)

# Display OCR result, owner name, department, parking allocation status, and parking number on the right side
ocr_result_label = Label(root, text="OCR result: " + rs)
ocr_result_label.grid(row=0, column=1, padx=10, pady=10, sticky=W)

owner_name_label = Label(root, text="Owner Name: " + owner_name)
owner_name_label.grid(row=1, column=1, padx=10, pady=10, sticky=W)

department_label = Label(root, text="Department: " + department)
department_label.grid(row=2, column=1, padx=10, pady=10, sticky=W)

allocation_status_label = Label(root, text="Parking Allocation Status: " + ("Success" if allocation_success else "Failed"))
allocation_status_label.grid(row=3, column=1, padx=10, pady=10, sticky=W)

parking_number_label = Label(root, text="Parking Number: " + str(parking_number))
parking_number_label.grid(row=4, column=1, padx=10, pady=10, sticky=W)

root.mainloop()
