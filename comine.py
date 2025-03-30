import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import torch
import os
import pickle
import traceback
import requests

from streamlit_lottie import st_lottie  # For animation

# For the Basic model
from Translation.models import Transformer, Encoder, Decoder
from Translation.dictionary import Dictionary
from Translation.utilities import normalizeString, tokenize

# For the mBART model
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# --------------------------------------------------------------------
#                   LOTTIE ANIMATION LOADER
# --------------------------------------------------------------------
def load_lottie_url(url: str):
    """
    Loads a Lottie animation JSON from a given URL.
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Example Lottie animation URL (replace with any you like from lottiefiles.com)
LOTTIE_URL = "https://assets4.lottiefiles.com/packages/lf20_gqk7snud.json"

# --------------------------------------------------------------------
#                   BASIC MODEL FUNCTIONS
# --------------------------------------------------------------------
SOS_TOKEN = 1
EOS_TOKEN = 2

@st.cache_data
def load_dictionary(dictionary_path):
    if not os.path.exists(dictionary_path):
        raise FileNotFoundError(f"Dictionary file not found at: {dictionary_path}")
    with open(dictionary_path, "rb") as f:
        return pickle.load(f)

def find_latest_model(directory):
    model_files = [
        f for f in os.listdir(directory)
        if f.startswith("transformer_model_epoch_") and f.endswith(".pt")
    ]
    if not model_files:
        raise FileNotFoundError(f"No model files found in directory: {directory}")
    latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(directory, latest_model)

def load_basic_model(models_directory="Website/Translation/saved_models/english2juhoansi"):
    """
    Loads the 'Default' (Basic Transformer) model.
    Returns the model, input_dic, output_dic, and device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = models_directory
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Basic model directory not found: {model_dir}")

    input_dic_path = os.path.join(model_dir, "input_dic.pkl")
    output_dic_path = os.path.join(model_dir, "output_dic.pkl")
    input_lang_dic = load_dictionary(input_dic_path)
    output_lang_dic = load_dictionary(output_dic_path)

    input_size = input_lang_dic.n_count
    output_size = output_lang_dic.n_count

    encoder_part = Encoder(input_size, 256, 3, 8, 512, 0.1, device)
    decoder_part = Decoder(output_size, 256, 3, 8, 512, 0.1, device)
    transformer_model = Transformer(encoder_part, decoder_part, device).to(device)

    latest_model_path = find_latest_model(model_dir)
    transformer_model.load_state_dict(torch.load(latest_model_path, map_location=device))

    return transformer_model, input_dic, output_dic, device

# --------------------------------------------------------------------
#                   MBART MODEL FUNCTIONS
# --------------------------------------------------------------------
@st.cache_resource
def load_mbart_model(model_path="Website/Translation/saved_models/mbart"):
    """
    Loads the fine-tuned MBART model from the given directory.
    Make sure all necessary files (config.json, model.safetensors, tokenizer.json, etc.)
    are placed in that 'mbart' folder.
    """
    try:
        tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
        model = MBartForConditionalGeneration.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # We trained with en_XX as source, xh_ZA as the placeholder
        tokenizer.src_lang = "en_XX"
        tokenizer.tgt_lang = "xh_ZA"

        return model, tokenizer, device
    except Exception:
        st.error("Could not load the MBART model and tokenizer.")
        st.text(traceback.format_exc())
        return None, None, None

def translate_mbart(input_text, model, tokenizer, device, max_len=128):
    """Translates an English sentence using the MBART model approach."""
    try:
        model.eval()
        inputs = tokenizer([input_text], return_tensors="pt").to(device)
        forced_bos_token_id = tokenizer.lang_code_to_id["xh_ZA"]  # Adjust if needed

        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                max_length=max_len,
                forced_bos_token_id=forced_bos_token_id
            )
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translation[0]
    except Exception:
        st.error("An error occurred during translation with MBART:")
        st.text(traceback.format_exc())
        return ""

# --------------------------------------------------------------------
#            UTILITY: SAVE CORRECTIONS TO TEXT FILE
# --------------------------------------------------------------------
def save_translation_pair(english_text, juhoansi_text, english_path, juhoansi_path):
    try:
        os.makedirs(os.path.dirname(english_path), exist_ok=True)
        os.makedirs(os.path.dirname(juhoansi_path), exist_ok=True)

        with open(english_path, "a", encoding="utf-8") as eng_file:
            eng_file.write(english_text + "\n")
        with open(juhoansi_path, "a", encoding="utf-8") as juhoansi_file:
            juhoansi_file.write(juhoansi_text + "\n")
        st.success("Languages updated. Thank you!")
    except Exception:
        st.error("Failed to save the correction.")
        st.text(traceback.format_exc())

# --------------------------------------------------------------------
#                  STREAMLIT LAYOUT
# --------------------------------------------------------------------

# --- Custom CSS for Big Bold Button ---
st.markdown(
    """
    <style>
    .big-button {
        font-size: 1.2em;
        padding: 0.75em 1em;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .centered-content {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Lottie animation JSON
lottie_json = load_lottie_url(LOTTIE_URL)

st.title("English → Ju/'hoansi Translation")
st.write("This tool helps you turn English sentences into Ju/'hoansi. Type in your words or record them, and we will show you a translation.")

# Optional: Display Lottie animation at the top
if lottie_json:
    st_lottie(lottie_json, height=200, key="lottie_intro")

# Session states
if "default_translation" not in st.session_state:
    st.session_state.default_translation = ""
if "mbart_translation" not in st.session_state:
    st.session_state.mbart_translation = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "correct_translation" not in st.session_state:
    st.session_state.correct_translation = ""
if "show_correct_input" not in st.session_state:
    st.session_state.show_correct_input = False

# Main input for English text
input_text = st.text_input("Enter English text for translation:", key="input_text")

# Big, bold "Translate" button with a speech-bubble icon (emoji or custom icon)
if st.button("💬 Translate", key="translate_button", help="Click to translate your text"):
    if input_text.strip():
        try:
            # --- Translate with the Default (Basic) model ---
            default_model, input_dic, output_dic, device_basic = load_basic_model()
            st.session_state.default_translation = translate_basic(input_text, default_model, input_dic, output_dic, device_basic)

            # --- Translate with the MBART model ---
            mbart_model, tokenizer, device_mbart = load_mbart_model()
            if mbart_model is not None and tokenizer is not None:
                st.session_state.mbart_translation = translate_mbart(input_text, mbart_model, tokenizer, device_mbart, max_len=128)
            else:
                st.session_state.mbart_translation = "MBART translation not available."

            # Reset feedback inputs
            st.session_state.show_correct_input = False
            st.session_state.feedback = None

        except Exception as e:
            st.error("An error occurred during processing:")
            st.text(traceback.format_exc())
    else:
        st.warning("Please enter some English text before translating.")

# Display translations
if st.session_state.default_translation or st.session_state.mbart_translation:
    st.write("## Translations")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Default Model Translation")
        st.success(st.session_state.default_translation)

    with col2:
        st.markdown("### MBART Model Translation")
        st.info(st.session_state.mbart_translation)

    # Feedback for accuracy
    st.write("#### Is either translation accurate?")
    feedback = st.radio(
        label="",
        options=("Yes, it's accurate", "No, it's not accurate"),
        index=0,
        key="feedback_toggle"
    )

    if feedback == "No, it's not accurate":
        st.session_state.show_correct_input = True
    else:
        st.session_state.show_correct_input = False

# If user says "No", let them provide a correction
if st.session_state.show_correct_input:
    st.subheader("What is the correct Ju/'hoansi translation?")
    st.session_state.correct_translation = st.text_input(
        "Enter the correct Ju/'hoansi translation:",
        key="correct_input",
        placeholder="Type the correct translation here"
    )
    if st.session_state.correct_translation.strip():
        if st.button("Submit Correction", key="submit_correction"):
            # Save correction
            english_path = "/home/ubuntu/Website/Translation/Data/english.txt"
            juhoansi_path = "/home/ubuntu/Website/Translation/Data/juhoansi.txt"

            save_translation_pair(input_text, st.session_state.correct_translation, english_path, juhoansi_path)

            # Reset inputs and thank the user
            st.session_state.show_correct_input = False
            st.success("Your correction has been saved successfully!")
