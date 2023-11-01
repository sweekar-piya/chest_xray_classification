import streamlit as st
import glob
import os

from torch import nn
from src.utils.constants import MODEL_DIR
from src.utils.helper import load_model, preprocess_image, return_pred_and_prob

def main():
    """Streamlit UI code and logic"""
    
    # Title
    st.markdown(
        "<h1 style='text-align: center; color: red;'>Rudimentary Chest X-Ray Classification App</h1>",
        unsafe_allow_html=True
    )
    
    # Get available models
    available_models = [os.path.basename(path) for path in glob.glob(str(MODEL_DIR/ "*.pth"))]
    
    # Load drop down to select the model to use.
    model = None
    selected_model = st.selectbox(
        "Choose model to use for prediction:",
        options=["Select an option"] + available_models,
    )
    
    if selected_model != "Select an option":
        st.write(f'You selected: {selected_model}')
        # Get the model
        model = load_model(selected_model)
        model.eval()
    else:
        st.write('Please make a selection')
    
    # If model is selected, load file Uploader to load image after selecting model
    if model:
        uploaded_img = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])
        
        if uploaded_img is not None:
            # Display uploaded image in UI
            filename = uploaded_img.name
            st.image(uploaded_img, caption=filename, use_column_width=True)
            
            # Get the prediction from the model
            preprocessed_image = preprocess_image(uploaded_img)
            pos_pred_prob = model(preprocessed_image)
            if selected_model == "resnet_transfer.pth":
                pos_pred_prob = nn.Sigmoid()(pos_pred_prob) # hack for fine-tuned model.
            pred, prob = return_pred_and_prob(pos_pred_prob)
            
            # Display the prediction in UI
            st.markdown(
                ("<h3 style='text-align: center;'>"
                f"Predicted as"
                f"</br>{pred}"
                f"</br>with {prob:.2%} probability."
                "</h3>"),
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()