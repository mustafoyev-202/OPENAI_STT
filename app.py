import openai
import os
import streamlit as st
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import re
import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY topilmadi")
openai.api_key = OPENAI_API_KEY

# Streamlit configuration
st.set_page_config(
    page_title="Audio yozuvini qayta ishlash",
    page_icon="🎙️",
    layout="wide"
)


class AudioProcessor:
    @staticmethod
    def format_speaker_labels(text: str) -> str:
        """
        Format speaker labels consistently
        """
        try:
            text = re.sub(r'\n{3,}', '\n\n', text.strip())
            text = re.sub(r'\[?(?i)speaker\s*(\d+)\]?\s*:', r'Suxbatdosh \1:', text)
            text = re.sub(r'(?<!^)(?=Suxbatdosh \d+:)', r'\n\n', text)
            return text
        except Exception as e:
            logger.error(f"Formatlashda xatolik: {e}")
            raise

    @staticmethod
    def transcribe_audio(audio_path: str) -> Dict[str, str]:
        """
        Transcribe audio file with proper error handling
        """
        try:
            with open(audio_path, "rb") as audio_file:
                prompt = """
                Qozoqcha so'zlar aralashgan o'zbekcha suhbatni yozing.
                So'zlarni asl holida saqlang.
                """

                response = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    prompt=prompt
                )

                return {"text": response["text"]}
        except Exception as e:
            logger.error(f"Yozib olishda xatolik: {e}")
            raise

    @staticmethod
    def identify_speakers(transcript: str) -> str:
        """
        Identify and label speakers in the transcript
        """
        try:
            diarization_prompt = """
            Bu qozoqcha-o'zbekcha aralash suhbat. Har bir so'zlovchining gaplarini belgilang.
            Qoidalar:
            1. Asl so'zlarni aynan saqlang
            2. Har bir so'zlovchining gapini "Suxbatdosh N:" bilan boshlang
            3. Turli so'zlovchilar orasida bo'sh qator qo'shing
            4. Barcha asl imlo va tinish belgilarini saqlang
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": diarization_prompt},
                    {"role": "user", "content": transcript}
                ],
                temperature=0.3
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"So'zlovchilarni aniqlashda xatolik: {e}")
            raise

    @staticmethod
    def convert_to_uzbek(text: str) -> str:
        """
        Convert mixed Kazakh-Uzbek text to pure Uzbek while preserving structure
        """
        try:
            conversion_prompt = """
            Qozoqcha-o'zbekcha aralash matnni toza o'zbek tiliga o'giring.
            QATTIQ QOIDALAR:
            1. So'zlar tartibini aynan saqlang
            2. Har bir so'zni alohida o'zbek tiliga o'giring
            3. Gap tuzilishini o'zgartirmang
            4. To'g'ri o'zbek harflarini ishlating (ў, қ, ғ, ҳ)
            5. Ko'p uchraydigan o'zgarishlar:
               - сіз → сиз
               - біз → биз
               - үчін → учун
               - ғам → ҳам
               - мұғым → муҳим
               - қоңғырақ → қўнғироқ
               - әгер → агар
               - кәбір → каби
               - мәқсат → мақсад
               - қызмет → хизмат
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": conversion_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"O'zbek tiliga o'girishda xatolik: {e}")
            raise

    @staticmethod
    def create_summary(text: str) -> str:
        """
        Create summary of the conversation in Uzbek
        """
        try:
            summary_prompt = """
            Quyidagi matnni tahlil qiling:

            1. Asosiy mazmun (2-3 gap):
               - Suhbatning asosiy mavzusi
               - Muhim fikrlar

            2. Qisqacha tahlil:
               - Muhim nuqtalar (3-4 ta)
               - Asosiy xulosalar
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": summary_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Tahlil yaratishda xatolik: {e}")
            raise


def main():
    st.title("🎙️ Audio faylni qayta ishlash")
    st.write("Audio faylni yuklang va u avtomatik ravishda tarjima qilinadi va tahlil qilinadi.")

    processor = AudioProcessor()

    # File uploader
    audio_file = st.file_uploader(
        "Audio faylni yuklang",
        type=["mp3", "wav", "m4a"],
        help="MP3, WAV yoki M4A formatidagi faylni tanlang"
    )

    if audio_file:
        with st.spinner("Fayl qayta ishlanmoqda..."):
            try:
                # Save temporary file
                with NamedTemporaryFile(delete=False, suffix="." + audio_file.name.split(".")[-1]) as temp_file:
                    temp_file.write(audio_file.getvalue())
                    temp_path = temp_file.name

                # Process audio
                transcript_result = processor.transcribe_audio(temp_path)
                diarized_text = processor.identify_speakers(transcript_result["text"])
                formatted_text = processor.format_speaker_labels(diarized_text)
                uzbek_text = processor.convert_to_uzbek(formatted_text)
                summary = processor.create_summary(uzbek_text)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Asl matn")
                    st.text_area("", formatted_text, height=300)

                with col2:
                    st.subheader("O'zbek tilidagi matn")
                    st.text_area("", uzbek_text, height=300)

                st.subheader("Tahlil")
                st.markdown(summary)

                # Download buttons
                col3, col4 = st.columns(2)
                with col3:
                    st.download_button(
                        label="Asl matnni yuklab olish",
                        data=formatted_text,
                        file_name="original_text.txt",
                        mime="text/plain"
                    )

                with col4:
                    st.download_button(
                        label="Tarjima matnni yuklab olish",
                        data=uzbek_text,
                        file_name="uzbek_text.txt",
                        mime="text/plain"
                    )

            except Exception as e:
                st.error(f"Xatolik yuz berdi: {str(e)}")
                logger.error(f"Qayta ishlashda xatolik: {e}")

            finally:
                # Cleanup
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)

    # Sidebar information
    with st.sidebar:
        st.header("Dastur haqida")
        st.markdown("""
        ### Imkoniyatlar:
        1. Audio faylni yozib olish
        2. So'zlovchilarni aniqlash
        3. Matnni toza o'zbek tiliga o'girish
        4. Qisqacha tahlil tayyorlash

        ### Qo'llab-quvvatlanadigan formatlar:
        - MP3
        - WAV
        - M4A
        """)


if __name__ == "__main__":
    main()
