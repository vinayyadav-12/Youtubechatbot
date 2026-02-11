from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

load_dotenv()

def load_transcript(youtube_id: str):
    api = YouTubeTranscriptApi()
    try:
        transcript = api.fetch(youtube_id, languages=['en'])
        transcripts = transcript.to_raw_data()
        return " ".join([entry['text'] for entry in transcripts])
    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.")
        return ""

def create_vectorstore(transcript_text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents([transcript_text])

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def answer_question(vectorstore, question: str) -> str:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = PromptTemplate(
        template=(
            "Use the following context to answer the question.\n"
            "If you don't know the answer, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}"
        ),
        input_variables=["context", "question"],
    )

    llm = ChatGroq(model="openai/gpt-oss-120b")

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "context": context,
        "question": question
    })