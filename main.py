import streamlit as st
from utils import load_transcript, create_vectorstore, answer_question


st.set_page_config(
    page_title="YouTube RAG Chatbot",
    layout="wide"
)

st.title("🎥 YouTube Video Chatbot (RAG)")
st.caption("Ask questions directly from YouTube videos")

with st.sidebar:
    st.header("Video Input")
    video_id = st.text_input(
        "Enter YouTube Video ID",
        placeholder="nAmC7SoVLd8"
    )

    process_btn = st.button("Process Video")
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if process_btn and video_id:
    with st.spinner("Fetching transcript & building index..."):
        transcript_text = load_transcript(video_id)

        if not transcript_text:
            st.error("Transcript not available for this video.")
        else:
            st.session_state.vectorstore = create_vectorstore(transcript_text)
            st.success("Video processed successfully!")
            
if st.session_state.vectorstore:
    st.subheader("💬 Ask Questions")

    user_question = st.text_input("Your question")

    if user_question:
        with st.spinner("Thinking..."):
            answer = answer_question(
                st.session_state.vectorstore,
                user_question
            )

        st.session_state.chat_history.append(
            (user_question, answer)
        )

    # Display chat history
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")
        st.markdown("---")