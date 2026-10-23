
import streamlit as st
from PIL import Image
from transformers import pipeline

st.title("AI Image Classification and NLP")
st.write("Experiment 5: Cloud AI Services")

@st.cache_resource
def load_models():
    image_model = pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )
    sentiment_model = pipeline("sentiment-analysis")
    return image_model, sentiment_model

image_model, sentiment_model = load_models()

st.header("Part A: Image Classification")
uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Image"):
        with st.spinner("Analyzing image..."):
            results = image_model(image)
        st.subheader("Classification Results")
        for result in results[:3]:
            st.write(
                f"{result['label']}: "
                f"{result['score'] * 100:.2f}% confidence"
            )

st.header("Part B: NLP Sentiment Analysis")
text_input = st.text_area("Enter text to analyze")

if st.button("Analyze Sentiment"):
    if text_input.strip():
        with st.spinner("Analyzing sentiment..."):
            result = sentiment_model(text_input)[0]
        st.write("Sentiment:", result["label"])
        st.write(f"Confidence: {result['score'] * 100:.2f}%")
    else:
        st.warning("Please enter some text.")
