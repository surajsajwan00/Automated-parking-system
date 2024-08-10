# Automated Parking System

## Overview

The Automated Parking System is designed to streamline parking management by accurately reading driver’s licenses using a custom Optical Character Recognition (OCR) model based on Convolutional Neural Networks (CNNs). The system captures license plates with a camera, extracts and processes the relevant information using the CNN model, and stores it securely. It then integrates with Google Sheets for easy data management and allocates parking spaces efficiently, notifying drivers via email.

## Features

- **Custom OCR Model**: Utilizes a CNN-based OCR model for accurate license plate recognition.
- **License Plate Capture**: Captures license plates using a camera.
- **Data Extraction**: Processes and extracts relevant information from captured images.
- **Secure Storage**: Converts and stores the extracted data in a secure file format.
- **Google Sheets Integration**: Enables easy access and editing of data.
- **Automated Parking Allocation**: Assigns parking spaces to drivers.
- **Email Notifications**: Sends parking space allocation details to drivers.

## Technologies Used

- **Custom OCR with CNN**: For recognizing and extracting text from license plates.
- **Camera**: For capturing license plate images.
- **Google Sheets**: For managing and editing extracted data.
- **Email Service**: For sending parking space allocation notifications.

## Setup and Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/surajsajwan00/Automated-parking-system.git
   cd Automated-parking-system
