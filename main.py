import openai
import os
import streamlit as st
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import re

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

st.title("Audio Upload for Transcription and Analysis")
st.write("Upload an audio file to transcribe and analyze the conversation with diarization and summary.")


def format_speaker_labels(text):
    """
    Format speaker labels to ensure consistent capitalization and spacing
    """
    # Replace variations of speaker labels with consistent format
    formatted_text = re.sub(r'\[?(?i)speaker\s*(\d+)\]?\s*:', r'Speaker \1:', text)

    # Ensure there's a newline before each speaker change
    formatted_text = re.sub(r'(?<!^)(?=Speaker \d+:)', r'\n\n', formatted_text)

    return formatted_text


def process_audio_with_diarization(audio_path):
    # Step 1: Initial transcription with Whisper
    with open(audio_path, "rb") as audio_file:
        initial_transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )

    # Step 2: Use GPT-4 to identify speakers and segment the transcript
    diarization_prompt = """
    Analyze this transcript and segment it by speaker. Format the output following these EXACT rules:
    1. Start each speaker segment with "Speaker N:" (where N is the speaker number)
    2. Put each speaker's segment on a new line
    3. Use consistent capitalization: "Speaker 1:", "Speaker 2:", etc.
    4. Add a blank line between different speakers
    5. DO NOT use brackets or other formatting around speaker labels

    Example format:
    Speaker 1: [First speaker's text]

    Speaker 2: [Second speaker's text]

    Speaker 1: [First speaker speaks again]

    Original transcript:
    {transcript}
    """

    diarization_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": diarization_prompt},
            {"role": "user", "content": initial_transcript["text"]}
        ],
        temperature=0.3
    )

    # Get the diarized text and format it
    diarized_text = diarization_response["choices"][0]["message"]["content"]
    formatted_diarized_text = format_speaker_labels(diarized_text)

    return {
        "diarized_text": formatted_diarized_text,
        "raw_transcript": initial_transcript["text"]
    }


def analyze_conversation(diarized_text):
    analysis_prompt = """
    Analyze this conversation and provide:
    1. Summary: 2-3 sentences covering the main points
    2. Topics: Main categories discussed
    3. Speakers: Number of unique speakers and their characteristics
    4. Key Points: 3-4 bullet points of important information

    Conversation:
    {conversation}
    """

    analysis_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": diarized_text}
        ],
        temperature=0
    )

    return analysis_response["choices"][0]["message"]["content"]


# Audio file upload
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if audio_file:
    with st.spinner("Processing audio file..."):
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(audio_file.read())
            temp_file_path = temp_file.name

        try:
            # Process audio with diarization
            results = process_audio_with_diarization(temp_file_path)

            # Display results in separate sections using markdown for better formatting
            st.subheader("Diarized Transcript")
            st.markdown(results["diarized_text"])

            st.subheader("Analysis")
            analysis = analyze_conversation(results["diarized_text"])
            st.write(analysis)

            # Add download buttons for transcripts
            st.download_button(
                label="Download Diarized Transcript",
                data=results["diarized_text"],
                file_name="diarized_transcript.txt"
            )

            st.download_button(
                label="Download Raw Transcript",
                data=results["raw_transcript"],
                file_name="raw_transcript.txt"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        finally:
            # Cleanup temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

# Add some helpful information
st.sidebar.markdown("""
### About This App
This application processes audio files to:
1. Transcribe the audio
2. Identify different speakers
3. Provide conversation analysis
4. Generate downloadable transcripts

Supported file types: MP3, WAV, M4A
""")
