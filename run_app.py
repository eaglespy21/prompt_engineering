import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from PIL import Image

# Load the model from a file
with open('model.pkl', 'rb') as f:
  model = pickle.load(f)

# Create a main function
def main():
  # Create a sidebar with an uploader widget
  st.sidebar.markdown('### Upload an image')
  uploaded_file = st.sidebar.file_uploader('Choose a 8x8 grayscale image (png, jpg, or jpeg)', type='jpg')
  
  # Check if a file has been uploaded
  if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = image.convert('L')
    image = image.resize((8, 8))
    image = np.array(image).flatten() / 16.0
    # Load the image and scale it
    # image = plt.imread(uploaded_file)
    # image = image / 16.0
    
    # Reshape the image and make a prediction
    image = image.reshape(1, -1)
    prediction = model.predict(image)
    
    # Convert the prediction to a class label
    label = np.argmax(prediction[0])
    
    # Display the image and the prediction
    st.markdown(f'### Prediction: {label}')
    st.image(image.reshape(8, 8), width=200)

# Run the app
if __name__ == '__main__':
  main()