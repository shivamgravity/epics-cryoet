!pip install torch

import streamlit as st
import numpy as np
import torch
import zarr
from skimage import measure
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# Load the full PyTorch model
model = torch.load('unet3d_model.pth', map_location=torch.device('cpu'))
model.eval()  # Set model to evaluation mode

# Function to normalize data
def min_max_normalize(data, new_min=-1, new_max=1):
    min_val = data.min()
    max_val = data.max()
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

# Function to load and preprocess the Zarr file (tomogram data)
def load_zarr_data(file_path):
    zarr_file = zarr.open(file_path, mode='r')
    data = zarr_file['1'][:]  # Assuming your data is stored under the key '1'
    return data

# Function to perform inference
def run_inference(model, input_data):
    # Convert input data to PyTorch tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Get the prediction (assuming binary classification, convert output to binary mask)
    prediction = output.squeeze().numpy()
    return prediction

# Function to display the 3D volume and slice
def display_3d_slice(image_data, slice_idx):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_data[slice_idx, :, :], cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

# Streamlit app layout
st.title("3D Tomogram Prediction Using U-Net")

# Upload Zarr file (tomogram data)
uploaded_file = st.file_uploader("Upload Zarr File", type=['zarr'])
if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_data.zarr", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load the uploaded data
    data = load_zarr_data("temp_data.zarr")

    # Normalize the data
    normalized_data = min_max_normalize(data, -1, 1)

    # Run inference
    prediction = run_inference(model, normalized_data)

    # Display results
    st.subheader("Predicted 3D Mask")

    # Show a 3D slice of the predicted output
    slice_idx = st.slider("Select Slice", min_value=0, max_value=prediction.shape[0] - 1, value=prediction.shape[0] // 2)
    display_3d_slice(prediction, slice_idx)
