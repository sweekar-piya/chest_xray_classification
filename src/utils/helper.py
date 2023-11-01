from PIL import Image
from typing import Tuple

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import torch
from torchvision import transforms

from src.utils.constants import MODEL_DIR, IMAGE_SIZE, CLASS_MAP
from src.models.baseline_cnn import return_model as return_baseline_model
from src.models.first_model import return_model as return_first_model
from src.models.resnet_transfer import return_model as return_resnet_model 


def return_pred_and_prob(value: torch.Tensor) -> Tuple[str, float]:
    """Takes the output from model and returns string output with
    prediction probabiltiy"""
    
    # process the value
    pred = torch.round(value)
    pred = CLASS_MAP[pred.item()]
    prob = value.item()
    
    # compute prob for -ve class prediction
    if pred == "Normal":
        prob = 1-prob

    return pred, prob
    
# TODO: Change to be dynamic.
@st.cache_resource
def load_model(selected_model: str) -> torch.nn.Module:
    """Loads pretrained model from under ./models/ selected
    from streamlit UI."""
    
    # get model object
    if selected_model=="first_model.pth":
        model = return_first_model()
    elif selected_model=="baseline_cnn.pth":
        model = return_baseline_model()
    elif selected_model=="resnet_transfer.pth":
        model = return_resnet_model()
    
    try:
        # load model weights
        model_weight = torch.load(MODEL_DIR / selected_model)["model_state_dict"]
        model.load_state_dict(model_weight)
    except Exception as e:
        st.write(f"Assure that model exists under project_root/models directory: {e}")
        raise(e)

    return model

@st.cache_data
def preprocess_image(image: UploadedFile) -> torch.Tensor:
    """Preprocesses image uploaded from streamlit UI."""
    
    # set mean and std for standardization
    mean = [0.4823, 0.4823, 0.4823]
    std = [0.1363, 0.1363, 0.1363]
    
    # TODO: Did blunder during experimentation where I loaded the image as RGB -> change models training?
    pil_img = Image.open(image).convert("RGB")
    
    # preprocessing pipeline
    preprocess_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # convert from [C, H, W] -> [1, C, H, W] (1 being the batch)
    preprocessed_img = preprocess_transforms(pil_img).unsqueeze(0)
    
    return preprocessed_img
    
    