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


# Page 3: Interactive Experience
def interactive_page():
    if st.session_state.get("image_path", -1) == -1:
        st.switch_page("Galery.py")
    
    st.markdown(
        """
            <style>
            div.stButton > button {
                background-color: transparent;
                color: white;
                border: none;
                padding: 0;
                cursor: pointer;
                text-align: left;
            }
            div.stButton > button:hover {
                color: darkblue;
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

    sentences = text_annotations.explode("sentences")
    groups_by_sentences = sentences.groupby("sentences").apply(lambda x: x["id"])

    left, right = st.columns(2)

    if not "current_image" in st.session_state.keys():
        st.session_state["current_image"] = st.session_state["images"][
            st.session_state["selected_image"]
        ]

    with left:
        if st.session_state["selected_image"] > -1:
            st.image(st.session_state["current_image"], use_column_width=True)
            
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

            if st.button(sent, key=str(i) + sent + st.session_state["image_path"]):
                cur_ann = text_annotations[
                    text_annotations.id.isin(
                        groups_by_sentences[sentences.sentences.unique()[i]].tolist()
                    )
                ]

                print(groups_by_sentences[sentences.sentences.unique()[i]].tolist())

                image = Image.open(st.session_state["image_path"])

                st.session_state["current_image"] = np.array(
                    visualize_coco_annotations_pil(
                        image, cur_ann.to_dict(orient="records"), 159, False, False
                    )
                )

                st.rerun()

            # anatomies= f"Related structures: {anatomies}\n"
            # if st.button(anatomies, key=str(i)+anatomies+st.session_state["image_path"]):
            #     cur_ann = text_annotations[
            #         text_annotations.id.isin(
            #         groups_by_sentences[sentences.sentences.unique()[i]].tolist()
            #         )
            #     ]

            #     image = Image.open(st.session_state["image_path"])

            #     st.session_state["current_image"] = \
            #         np.array(visualize_coco_annotations_pil(image, cur_ann.to_dict(orient="records"), 159, False, False))

            #     st.rerun()
            # st.markdown(f"<p style='color:rgb(255,87,51); font-weight:bold;'>{anatomies}</p>", unsafe_allow_html=True)

        st.subheader("Radiological Report")

        image_dict = st.session_state["image_info"][
            st.session_state["image_info"].file_name
            == st.session_state["image_path"].split("/")[-1]
        ]

        # import pdb;pdb.set_trace()
        styled_sent1 = f"""
        <p style='font-size:20px; font-weight:bold; display:inline;'>Findings:</p> {image_dict.get('findings:', '-1').item()}
        """
        st.markdown(styled_sent1, unsafe_allow_html=True)

        # styled_sent2 = f"""
        # <p style='font-size:20px; font-weight:bold; display:inline;'>Impression:</p> {image_dict.get('impression:', '-1').item()}
        # """
        # st.markdown(styled_sent2, unsafe_allow_html=True)


interactive_page()
