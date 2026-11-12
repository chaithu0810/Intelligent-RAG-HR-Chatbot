
import streamlit as st
import traceback

from rag_pipeline import initialize_rag, ask_hr_bot


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# 2. CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #111111;
}

.stApp {
    background-color: #111111;
    color: white;
}

h1, h2, h3 {
    color: #ffffff;
}

.stChatMessage {
    border-radius: 12px;
}

div[data-testid="stExpander"] {
    border: 1px solid #555555;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.title("🤖 Zyro Dynamics HR Help Desk")

st.write(
    "Your AI-powered assistant for Zyro Dynamics "
    "Human Resources policies."
)

st.caption(
    "Ask questions about leave, attendance, "
    "work from home, travel, compensation and HR policies."
)

st.divider()


# --------------------------------------------------
# 4. INITIALIZE SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# 5. LOAD RAG PIPELINE
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_rag_system():
    return initialize_rag()


try:

    with st.spinner("Connecting to HR Knowledge Base..."):
        retriever = load_rag_system()

    st.sidebar.success("HR Knowledge Base Connected!")

except Exception as error:

    st.error(
        "Unable to initialize the HR Knowledge Base."
    )

    st.code(str(error))

    with st.expander("Show technical error details"):
        st.code(traceback.format_exc())

    st.stop()


# --------------------------------------------------
# 6. SIDEBAR
# --------------------------------------------------

st.sidebar.title("HR Assistant")

st.sidebar.write(
    "Powered by Gemini, LangChain and FAISS."
)

st.sidebar.markdown("---")

st.sidebar.subheader("Available topics")

st.sidebar.markdown("""
- Leave policies
- Work from home
- Attendance
- Travel reimbursement
- Compensation and benefits
- Code of conduct
- Employee onboarding
""")

st.sidebar.markdown("---")

if st.sidebar.button("Clear Chat", use_container_width=True):

    st.session_state.messages = []
    st.session_state.chat_history = []

    st.rerun()


# --------------------------------------------------
# 7. DISPLAY PREVIOUS MESSAGES
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display citations only for assistant messages
        sources = message.get("sources", [])

        if sources:

            with st.expander("📄 View PDF citations"):

                for source in sources:

                    source_id = source.get(
                        "source_id", "Source"
                    )

                    filename = source.get(
                        "filename", "Unknown document"
                    )

                    page = source.get(
                        "page", "Unknown"
                    )

                    content = source.get(
                        "content", "No source text available"
                    )

                    st.markdown(
                        f"**{source_id} — {filename}**"
                    )

                    st.caption(f"Page: {page}")

                    st.write(content)

                    st.divider()


# --------------------------------------------------
# 8. CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask your HR-related question..."
)


# --------------------------------------------------
# 9. GENERATE ANSWER
# --------------------------------------------------

if question:

    # Display user question
    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    try:

        with st.chat_message("assistant"):

            with st.spinner("Searching HR policies..."):

                answer, sources = ask_hr_bot(
                    question=question,
                    retriever=retriever,
                    history=st.session_state.chat_history
                )

            st.markdown(answer)

            # Display PDF citations
            if sources:

                with st.expander("📄 View PDF citations"):

                    for source in sources:

                        source_id = source.get(
                            "source_id", "Source"
                        )

                        filename = source.get(
                            "filename", "Unknown document"
                        )

                        page = source.get(
                            "page", "Unknown"
                        )

                        content = source.get(
                            "content", "No source text available"
                        )

                        st.markdown(
                            f"**{source_id} — {filename}**"
                        )

                        st.caption(f"Page: {page}")

                        st.write(content)

                        st.divider()

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        # Save conversation history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as error:

        st.error(f"Error: {str(error)}")

        with st.expander("Show technical error details"):
            st.code(traceback.format_exc())
