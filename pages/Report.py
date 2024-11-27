import streamlit as st
import time
from pathlib import Path
from st_clickable_images import clickable_images
import base64
import json
import pandas as pd
from PIL import Image
import numpy as np
from util.visualize import *
from streamlit_image_coordinates import streamlit_image_coordinates

# Page Navigation
st.set_page_config(page_title="Multi-Page App", layout="wide")


def display_text_animated(text, sleep_time=0.1):
    """Displays text character by character."""
    out_text = ""
    for char in text:
        out_text += char
        st.write(out_text, end="", unsafe_allow_html=True)
        time.sleep(sleep_time)

        
# Page 2: Selected Image
def image_detail_page():
    print(st.session_state["selected_image"])
    if st.session_state["selected_image"] > -1:
        st.image(
            st.session_state["images"][st.session_state["selected_image"]],
            use_column_width=True,
        )
    else:
        st.write("No image selected.")

    if st.button("Back to Gallery"):
        st.switch_page("app.py")

def coordinate_in_bbox(coordinate, bbox):
    bbox = np.array(bbox)
    bbox_mean = np.array([bbox[0]+bbox[2]//2, bbox[1]+bbox[3]//2])
    
    coordinate2 = np.array(list(coordinate.values()))[:4]
    # import pdb; pdb.set_trace()
    # print(coordinate2)
    # st.write(coordinate2)
    
    coordinate2 = np.array([coordinate2[0]/coordinate2[2]*512, coordinate2[1]/coordinate2[3]*512])
    is_in = (coordinate2[0]>bbox[0]) *(coordinate2[0]<(bbox[0]+bbox[2])) * \
            (coordinate2[1]>bbox[1]) *(coordinate2[1]<(bbox[1]+bbox[3]))
    dist = np.sqrt(np.pow((coordinate2 - bbox_mean), 2)).sum()
    if is_in:
        return dist
    else:
        return 1000


# Custom function to add a button with a specific type
def styled_button(label, button_type, key):
    # Wrap the button in a div with the button_type class
    button_html = f"""
    <div class="stButton {button_type}">
        <button>{label}</button>
    </div>
    """
    st.markdown(button_html, unsafe_allow_html=True)
    
# Page 3: Interactive Experience
def interactive_page():
    if st.session_state.get("image_path", -1) == -1:
        st.switch_page("Gallery.py")
    
    if st.session_state.get("coordinate_clicked", -1) == -1:
        st.session_state["coordinate_clicked"] = None
        
    st.markdown(
        """
        <style>
        div.stButton > button.primary {
            background-color: blue;
            color: white;
            border: 1px solid blue;
            border-radius: 5px;
            padding: 5px 10px;
        }
        div.stButton > button.primary:hover {
            background-color: darkblue;
            color: lightgray;
        }
        div.stButton > button.secondary {
            background-color: gray;
            color: white;
            border: 1px solid gray;
            border-radius: 5px;
            padding: 5px 10px;
        }
        div.stButton > button.secondary:hover {
            background-color: darkgray;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    annotation = json.load(
        open(st.session_state["image_path"].replace(".jpg", ".json"))
    )

    # import pdb; pdb.set_trace()
    text_annotations = [ann for ann in annotation if ann["sentences"] != "None"]
    text_annotations = pd.DataFrame(text_annotations)
    
    annotation = pd.DataFrame(annotation)

    sentences = text_annotations.explode("sentences")
    groups_by_sentences = sentences.groupby("sentences").apply(lambda x: x["id"])

    left, right = st.columns(2)

    if not "current_image" in st.session_state.keys():
        st.session_state["current_image"] = st.session_state["images"][
            st.session_state["selected_image"]
        ]

    with left:
        if st.session_state["selected_image"] > -1:
            value = streamlit_image_coordinates(st.session_state["current_image"], use_column_width=True)
            print(value)
            if (st.session_state["coordinate_clicked"] != value) and not value is None:
                st.session_state["coordinate_clicked"] = value
                
                bbox = annotation["bbox"]
                
                dist = coordinate_in_bbox(value, bbox.iloc[0])
                
                distances = bbox.apply(lambda x: coordinate_in_bbox(value,x))
                
                des_annotation = annotation.iloc[distances.argmin()]
                
                image = Image.open(st.session_state["image_path"])
                
                st.session_state["current_image"] = np.array(
                    visualize_coco_annotations_pil(
                        image, [des_annotation.to_dict()], st.session_state["categories"], True, False
                    )
                )
                
                st.session_state["clicked_category_id"] = des_annotation["category_id"]

                st.rerun()
                
                
                # sent = "Partial Finding: " + sent

                # if st.button(sent, key=str(i) + sent + st.session_state["image_path"]):
                

            
            left_left, left_right = st.columns(2)
            with left_left:
                if st.button("Clear Annotations"):
                    st.session_state["current_image"] = st.session_state["images"][
                        st.session_state["selected_image"]
                    ]
                    st.rerun()
            with left_right:
                if st.button("Back to the Galery"):
                    st.switch_page("Galery.py")

        else:
            st.write("No image selected.")

    with right:
        # Top Row

        st.subheader("Anatomical Findings")
        st.markdown("""
                    
                    Click on the individual findings to visualize the originating region in the image.
                    
                    ---
                    """)

        for i in range(len(sentences.sentences.unique())):

            sent = sentences.sentences.unique()[i]

            max_ana = 4
            cur_ann = text_annotations[
                text_annotations.id.isin(
                    groups_by_sentences[sentences.sentences.unique()[i]].tolist()
                )
            ]
            anatomies = cur_ann.category_id.tolist()
            anatomies = st.session_state["categories"][
                st.session_state["categories"].id.isin(anatomies)
            ].name.tolist()
            anatomies = ", ".join(anatomies[:max_ana]) + (
                "..." if len(anatomies) > max_ana else ""
            )

            sent = "Partial Finding: " + sent
                    
            if st.button(sent, key=str(i) + sent + st.session_state["image_path"], 
                         type="primary" if st.session_state.get("clicked_category_id", False) in cur_ann.category_id.values \
                             else "secondary"
                         ):
                
                if "clicked_category_id" in st.session_state.keys():
                    st.session_state["clicked_category_id"] = None
                    
                cur_ann = text_annotations[
                    text_annotations.id.isin(
                        groups_by_sentences[sentences.sentences.unique()[i]].tolist()
                    )
                ]

                image = Image.open(st.session_state["image_path"])

                st.session_state["current_image"] = np.array(
                    visualize_coco_annotations_pil(
                        image, cur_ann.to_dict(orient="records"), [1]*159, False, False
                    )
                )

                st.rerun()

        # st.subheader("Radiological Report")

        # image_dict = st.session_state["image_info"][
        #     st.session_state["image_info"].file_name
        #     == st.session_state["image_path"].split("/")[-1]
        # ]

        # # import pdb;pdb.set_trace()
        # styled_sent1 = f"""
        # <p style='font-size:20px; font-weight:bold; display:inline;'>Impression:</p> {image_dict.get('impression:', '-1').item()}
        # """
        # st.markdown(styled_sent1, unsafe_allow_html=True)


interactive_page()
