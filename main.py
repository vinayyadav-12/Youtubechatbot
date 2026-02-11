from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

#Document ingestion
youtube_id = "6ECKffxXCP0"

try:
    transcript = YouTubeTranscriptApi.get_transcript(youtube_id, languages=['en'])
    transcript_text = " ".join([entry['text'] for entry in transcript])
except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")
    transcript_text = ""

print(transcript_text)