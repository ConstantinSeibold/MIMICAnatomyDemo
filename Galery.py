import streamlit as st
import time
from pathlib import Path
from st_clickable_images import clickable_images
import base64
import glob
import json
import pandas as pd
import numpy as np
from PIL import Image

# Page Navigation
st.set_page_config(page_title="Multi-Page App", layout="wide")


def display_text_animated(text, sleep_time=0.1):
    """Displays text character by character."""
    for char in text:
        st.write(char, end="", unsafe_allow_html=True)
        time.sleep(sleep_time)


# Session State to Track Current Image
if "selected_image" not in st.session_state:
    st.session_state["selected_image"] = None


def get_coco_annotations(images):
    return


# Utility Functions
def load_images(folder_path):
    """Loads images from the specified folder."""
    return [str(img) for img in glob.glob(f"{folder_path}/**/*.jpg", recursive=True)]


def gallery_page():

    if not "image_info" in st.session_state.keys():
        images_json = "images.json"
        st.session_state["image_info"] = pd.DataFrame(json.load(open(images_json)))
        categories = "categories.json"
        st.session_state["categories"] = pd.DataFrame(json.load(open(categories)))

    st.title("Image Gallery")
    images_ = load_images("images")  # Replace with your folder path
    print(images_)
    images = []
    for file in images_:
        with open(file, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
            images.append(f"data:image/jpeg;base64,{encoded}")

    clicked = clickable_images(
        images,
        titles=[f"Image #{str(i)}" for i in range(len(images))],
        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        img_style={"margin": "5px", "height": "200px"},
    )

    annotated_images = get_coco_annotations(images_)

    st.markdown(f"Image #{clicked} clicked" if clicked > -1 else "No image clicked")

    if clicked > -1:
        st.session_state["images"] = images
        st.session_state["selected_image"] = clicked
        st.session_state["image_path"] = images_[clicked]
        st.session_state["current_image"] = np.array(
            Image.open(st.session_state["image_path"])
        )

        st.switch_page("pages/Report.py")


gallery_page()
