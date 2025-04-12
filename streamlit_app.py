import os

import streamlit as st
import numpy as np

# Set Streamlit page config
st.set_page_config(page_title="3D Tomogram Segmentation", layout="wide")
st.title("3D Tomogram Segmentation with U-Net (TensorFlow)")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("unet3d_model_tf")
    return model

model = load_model()

# Normalize input volume to [-1, 1]
def normalize_data(volume):
    min_val = volume.min()
    max_val = volume.max()
    if max_val - min_val == 0:
        return np.zeros_like(volume)
    return (volume - min_val) / (max_val - min_val) * 2 - 1

# File uploader
uploaded_file = st.file_uploader("Upload a 3D tomogram (.npy file)", type=["npy"])

if uploaded_file:
    volume = np.load(uploaded_file)
    st.write(f"Volume shape: {volume.shape}")

    # Normalize and pad input if needed
    input_volume = normalize_data(volume).astype(np.float32)
    pad_width = [(0, 0)] * 3
    for i in range(3):
        if input_volume.shape[i] % 16 != 0:
            pad_width[i] = (0, 16 - (input_volume.shape[i] % 16))
    input_volume = np.pad(input_volume, pad_width, mode='constant', constant_values=0)

    # Add batch and channel dimensions
    input_tensor = np.expand_dims(np.expand_dims(input_volume, axis=0), axis=-1)

    # Prediction
    prediction = model.predict(input_tensor)[0, ..., 0]

    # Remove padding
    for i in range(3):
        if pad_width[i][1] > 0:
            prediction = np.take(prediction, indices=range(volume.shape[i]), axis=i)

    # Visualize middle slice
    st.subheader("Middle Slice Comparison")
    slice_index = volume.shape[0] // 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(volume[slice_index], cmap='gray')
    axes[0].set_title("Original Tomogram Slice")
    axes[1].imshow(prediction[slice_index], cmap='viridis')
    axes[1].set_title("Predicted Mask Slice")
    for ax in axes:
        ax.axis('off')
    st.pyplot(fig)
