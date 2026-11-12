
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION
# --------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = BASE_DIR / "faiss_index"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. Add it to your .env file."
    )


# --------------------------------------------------
# 2. INITIALIZE EMBEDDING MODEL AND LLM
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_retries=2,
    timeout=120
)


# --------------------------------------------------
# 3. LOAD PDFS AND CREATE / LOAD FAISS
# --------------------------------------------------

def initialize_rag():

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"PDF data folder not found: {DATA_DIR}"
        )

    pdf_files = list(DATA_DIR.rglob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            "No PDF files found inside the data folder."
        )

    print(f"Found {len(pdf_files)} PDF files.")

    index_file = FAISS_DIR / "index.faiss"
    metadata_file = FAISS_DIR / "index.pkl"

    if index_file.exists() and metadata_file.exists():
        print("Loading existing FAISS database...")

        vectorstore = FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:
        print("Loading HR policy PDFs...")

        loader = PyPDFDirectoryLoader(
            str(DATA_DIR),
            recursive=True
        )

        documents = loader.load()

        if not documents:
            raise ValueError(
                "No readable text found in the HR PDFs."
            )

        print(f"Loaded {len(documents)} PDF pages.")

        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError(
                "No text chunks were generated."
            )

        print(f"Created {len(chunks)} chunks.")

        # Create FAISS vector database
        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        FAISS_DIR.mkdir(parents=True, exist_ok=True)

        vectorstore.save_local(str(FAISS_DIR))

        print("FAISS database created and saved.")

    # MMR retrieval
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 12,
            "lambda_mult": 0.5
        }
    )

    print("HR RAG chatbot is ready!")

    return retriever


# --------------------------------------------------
# 4. HR SCOPE GUARDRAIL
# --------------------------------------------------

HR_KEYWORDS = [
    "leave", "casual leave", "sick leave",
    "earned leave", "holiday", "vacation",
    "work from home", "wfh", "remote work",
    "attendance", "working hours", "overtime",
    "salary", "compensation", "payroll",
    "benefits", "reimbursement", "travel",
    "expense", "code of conduct", "misconduct",
    "employee", "hr", "human resources",
    "onboarding", "recruitment", "resignation",
    "notice period", "probation", "appraisal",
    "performance review", "medical certificate",
    "maternity leave", "paternity leave",
    "insurance", "promotion", "transfer",
    "policy", "workplace", "manager"
]


def is_hr_question(question):

    question_lower = question.lower()

    return any(
        keyword in question_lower
        for keyword in HR_KEYWORDS
    )


# --------------------------------------------------
# 5. FORMAT RETRIEVED DOCUMENTS AND SOURCES
# --------------------------------------------------

def format_context(documents):

    context_parts = []
    sources = []

    for index, doc in enumerate(documents, start=1):

        source_id = f"S{index}"

        raw_source = doc.metadata.get(
            "source", "Unknown document"
        )

        filename = Path(str(raw_source)).name

        # PyPDFDirectoryLoader normally stores
        # page numbers starting from zero.
        page_value = doc.metadata.get("page", None)

        if page_value is not None:
            page_number = int(page_value) + 1
        else:
            page_number = "Unknown"

        content = doc.page_content.strip()

        context_parts.append(
            f"[{source_id}]\n"
            f"Document: {filename}\n"
            f"Page: {page_number}\n"
            f"Content: {content}"
        )

        # IMPORTANT:
        # All source dictionaries use the same keys.
        sources.append({
            "source_id": source_id,
            "filename": filename,
            "page": page_number,
            "content": content
        })

    context = "\n\n".join(context_parts)

    return context, sources


# --------------------------------------------------
# 6. LCEL ANSWER GENERATION CHAIN
# --------------------------------------------------

answer_prompt = ChatPromptTemplate.from_template(
    """
You are the Zyro Dynamics HR Help Desk Assistant.

Answer the employee's question using ONLY the
provided HR policy context.

RULES:

1. Do not invent company policies.
2. Do not use external knowledge.
3. If the context does not contain the answer,
   respond exactly with:
   "I could not find this information in the
   available HR policies."
4. Cite supporting sources using IDs such as [S1].
5. Only cite sources that actually support your answer.
6. Give a clear and professional answer.
7. If information is incomplete, explain the limitation.

HR POLICY CONTEXT:

{context}

CONVERSATION HISTORY:

{history}

EMPLOYEE QUESTION:

{question}

ANSWER:
"""
)

answer_chain = answer_prompt | llm | StrOutputParser()


# --------------------------------------------------
# 7. MAIN CHATBOT FUNCTION
# --------------------------------------------------

def ask_hr_bot(question, retriever, history=None):

    if history is None:
        history = []

    question = question.strip()

    if not question:
        return (
            "Please enter a question.",
            []
        )

    try:

        # Guardrail 1: HR-related question
        if not is_hr_question(question):

            return (
                "I'm sorry, I can only assist with "
                "Zyro Dynamics HR-related questions. "
                "Please ask about company HR policies.",
                []
            )

        # Retrieve relevant documents using MMR
        documents = retriever.invoke(question)

        if not documents:
            return (
                "I could not find relevant information "
                "in the available HR policies.",
                []
            )

        # Format context and citations
        context, sources = format_context(documents)

        if not context.strip():
            return (
                "I could not find relevant information "
                "in the available HR policies.",
                []
            )

        # Format recent conversation history
        history_text = "\n".join(
            f"{message.get('role', 'user')}: "
            f"{message.get('content', '')}"
            for message in history[-6:]
        )

        # Generate answer using LCEL
        answer = answer_chain.invoke({
            "context": context,
            "history": history_text,
            "question": question
        })

        if not answer or not answer.strip():
            answer = (
                "I could not generate an answer. "
                "Please try again."
            )

        return answer.strip(), sources

    except Exception as error:

        print("RAG pipeline error:", repr(error))

        raise
