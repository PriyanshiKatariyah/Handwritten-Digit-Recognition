# Handwritten Digit Recognition using CNN

## Project Overview
This project recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

It uses TensorFlow and Keras for model development and Streamlit to create an interactive web application.

## Technologies Used
- Python
- TensorFlow
- Keras
- CNN (Convolutional Neural Network)
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- Seaborn
- Streamlit
- Pillow

## Features
- Draw handwritten digits using an interactive canvas.
- Predict digits from 0 to 9.
- Display prediction confidence.
- Visualize prediction probabilities.

## Project Structure
- `app.py` – Streamlit application.
- `handwritten_digit_model.keras` – Trained CNN model.
- `digit_recognition.ipynb` – Model training notebook.
- `requirements.txt` – Required Python libraries.

## How to Run

1. Clone this repository.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Run the application:

   `streamlit run app.py`

4. Open the local URL displayed in the terminal.

## Model
The CNN model is trained to classify handwritten digits from 0 to 9 using the MNIST dataset.