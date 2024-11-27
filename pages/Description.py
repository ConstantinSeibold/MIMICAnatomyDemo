import streamlit as st
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from PIL.Image import Resampling
# Function to convert PDF to images
def pdf_to_images(pdf_file):
    # Convert PDF to images (one image per page)
    images = convert_from_path(pdf_file, dpi=300)
    return images

# Tabs for subsections
tab1, tab2, tab3, tab4 = st.tabs(["Overview of the Medical Report", 
                            "Large Scale Dataset Generation",
                            "Segmentation of Anatomical Structures", 
                            "Identifying Relevant Findings", 
                            ])

with tab1:
    st.markdown("""
        We follow general
        structure defined in the reporting consensus of the RSNA Radiology Reporting Committee.
        
        The automatic generation of the "Observations" section is an accumulation of 
        - (a) identifying relevant findings
        - (b)retrieving corresponding anchors based on established priors         
        - (c) quantification of findings based on anchors.
        """)
    st.image("content/medical_report.png")

with tab2:
    
    st.markdown("""

Clinical radiology reports capture critical insights about a patient’s health by combining expert knowledge of anatomy with the ability to identify abnormalities.

Many medical image-processing systems lack this anatomical understanding, as they are narrowly trained on specific tasks due to limited annotations. 

We introduce **PAXRay**, a large-scale dataset generated from 10,021 projected thoracic CTs with 157 labels, allowing the training of detailed semantic segmentation models achieving human-level performances for CXR segmentations without any manual annotation effort. 

                """)
    image = np.array(Image.open("content/DatasetGeneration.png").resize((1024,768), resample=Resampling.LANCZOS).convert("RGB"))
    st.image(image)
    

with tab3:
    st.markdown("""
  Producing densely annotated data is a difficult and tedious task for medical imaging applications. To address this problem, we propose a **novel approach** to generate supervision for **semi-supervised semantic segmentation**. 

We argue that visually similar regions between labeled and unlabeled images likely contain the same semantics and therefore should share their label. Following this thought, we use a **small number of labeled images** as reference material and match pixels in an unlabeled image to the semantic of the best fitting pixel in a reference set. 

This way, we avoid pitfalls such as **confirmation bias**, common in purely prediction-based pseudo-labeling. Since our method does not require any architectural changes or accompanying networks, one can easily insert it into existing frameworks.

The segmentation of anatomical structures, which we apply this method to, is highly relevant for generating precise and detailed medical reports.
""")
    image = Image.open("content/method.png").resize((1024,512), resample=Resampling.LANCZOS)
    white_background = Image.new("RGB", image.size, (255, 255, 255))

    # Composite the RGBA image onto the white background
    image = Image.alpha_composite(white_background.convert("RGBA"), image).convert("RGB")
    image = np.array(image)
    st.image(image)


    
with tab4:
    st.markdown(
                """
Radiologists generate detailed text reports to describe findings in medical images. However, current diagnostic AI tools are limited by their reliance on fixed categories extracted from these reports, making them unable to identify anomalies outside predefined sets without retraining. This work breaks away from such limitations by leveraging **direct text supervision**, enabling models to learn directly from unstructured medical reports. Using a contrastive global-local dual-encoder architecture, the approach captures contextual information from free-form text, allowing for open-set recognition and flexible classification. 

By learning from free-form medical text, this approach lays the groundwork for AI systems capable of generating richer, more adaptable medical reports, enhancing their clinical utility.

                """
                )
    
    
    image = Image.open("content/elaborate.png")
    white_background = Image.new("RGB", image.size, (255, 255, 255))

    # Composite the RGBA image onto the white background
    image = Image.alpha_composite(white_background.convert("RGBA"), image).convert("RGB")
    image = np.array(image)
    st.image(image)
