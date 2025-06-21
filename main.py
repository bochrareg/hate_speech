import json
import streamlit as st
from huggingface_hub import InferenceClient

# Streamlit secrets for Hugging Face Token
HF_TOKEN = st.secrets.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("Please add your Hugging Face token to Streamlit secrets (st.secrets['HF_TOKEN']).")
    st.stop()

# Initialize the Inference Client
@st.cache_resource
def get_client(token: str):
    return InferenceClient(token=token)

# Streamlit page config
st.set_page_config(
    page_title="Arabic Hate Speech Classifier",
    layout="centered"
)

# Generation parameters (fixed)
TEMPERATURE = 0.0  # fully deterministic
TOP_P = 1.0        # no nucleus sampling restriction
MAX_TOKENS = 128

# Main app
st.title("Arabic Hate Speech Classification")
st.write(
    "Enter a sentence in Arabic, and the Llama 3 model will classify it into one or more of the following categories and provide confidence scores for each:"
)
st.markdown(
    "**Categories:** hate, offensive, violent, vulgar"
)

# Input textbox
user_input = st.text_area("Enter your sentence:", "Enter a sentence...", height=150)

if st.button("Classify"):
    if not user_input.strip():
        st.error("Please enter a sentence to classify.")
    else:
        client = get_client(HF_TOKEN)

        # Prepare messages for classification
        system_msg = (
            "You are a classifier. Given the following Arabic sentence, classify it into the categories: hate, offensive, violent, vulgar. "
            "For each category, provide a confidence score between 0 and 1. "
            "Respond with EXACTLY the JSON object and nothing else, prefixed by the token ###RESULT."
        )
        user_msg = f"Sentence: \"{user_input}\""

        try:
            with st.spinner("Classifying..."):
                response = client.chat_completion(
                    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=MAX_TOKENS
                )
                text = response.choices[0].message.content

            # Parse JSON after marker
            marker = "###RESULT"
            if marker in text:
                json_str = text.split(marker, 1)[1].strip()
                results = json.loads(json_str)
                st.subheader("Classification Results")
                st.json(results)
            else:
                st.warning("Could not find JSON in the model output. Showing raw output below for debugging:")
                st.code(text)

        except Exception as e:
            st.error(f"Error during classification: {e}")
