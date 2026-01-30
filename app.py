"""
Streamlit RAG Application with Google Generative AI
A chat interface for querying PDF documents using RAG (Retrieval Augmented Generation)
"""

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


# Page configuration
st.set_page_config(
    page_title="📚 RAG Document Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 50%, #1e1e2e 100%);
    }
    
    /* Make all text more visible */
    .stApp, .stApp p, .stApp span, .stApp div {
        color: #e0e0e0 !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        color: #c0c0c0 !important;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.08) !important;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-bottom: 1rem;
    }
    
    .stChatMessage p {
        color: #f0f0f0 !important;
    }
    
    /* Expander styling for sources */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.15);
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.4);
        color: #e0e0e0 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(30, 30, 46, 0.98) !important;
    }
    
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #e0e0e0 !important;
    }
    
    /* Status indicators */
    .status-ready {
        color: #68d391 !important;
        font-weight: 600;
    }
    
    .status-loading {
        color: #f6ad55 !important;
        font-weight: 600;
    }
    
    /* Source card styling */
    .source-card {
        background: rgba(102, 126, 234, 0.15);
        border-left: 3px solid #667eea;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        color: #e0e0e0 !important;
    }
    
    .source-card small {
        color: #b0b0b0 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        color: #a0aec0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_and_process_documents():
    """Load PDFs from data folder and create vector store"""
    
    # Check if data folder exists
    data_path = "data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        return None, 0, 0
    
    # Load PDFs
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    
    if not documents:
        return None, 0, 0
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vectorstore, len(documents), len(chunks)


def get_rag_chain(vectorstore):
    """Create RAG chain with retriever and LLM"""
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",  # Używamy modelu z Twojej listy!
    google_api_key=api_key,
    transport="rest",
    temperature=0.3
)
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Create prompt template
    system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
    Use the following pieces of context to answer the question. 
    If you don't know the answer based on the context, say so honestly.
    Always be concise, accurate, and helpful.
    
    Context:
    {context}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create LCEL chain
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever


def format_sources(source_documents):
    """Format source documents for display"""
    sources = []
    seen = set()
    
    for doc in source_documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        
        # Extract just the filename
        filename = os.path.basename(source)
        
        key = f"{filename}-{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": filename,
                "page": page + 1 if isinstance(page, int) else page,  # Pages are 0-indexed
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            })
    
    return sources


def main():
    # Header
    st.markdown('<h1 class="main-header">📚 RAG Document Chat</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Chat with your PDF documents using AI-powered retrieval</p>', unsafe_allow_html=True)
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
    if "doc_count" not in st.session_state:
        st.session_state.doc_count = 0
    
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.divider()
        
        # Load documents button
        if st.button("🔄 Load/Reload Documents", use_container_width=True):
            with st.spinner("📖 Loading and processing documents..."):
                # Clear cache and reload
                load_and_process_documents.clear()
                vectorstore, doc_count, chunk_count = load_and_process_documents()
                st.session_state.vectorstore = vectorstore
                st.session_state.doc_count = doc_count
                st.session_state.chunk_count = chunk_count
                
                if vectorstore:
                    st.success(f"✅ Loaded {doc_count} documents ({chunk_count} chunks)")
                else:
                    st.warning("⚠️ No PDF files found in 'data/' folder")
        
        st.divider()
        
        # Document stats
        st.markdown("### 📊 Document Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Documents", st.session_state.doc_count)
        with col2:
            st.metric("🧩 Chunks", st.session_state.chunk_count)
        
        # Status indicator
        if st.session_state.vectorstore:
            st.markdown("**Status:** <span class='status-ready'>🟢 Ready</span>", unsafe_allow_html=True)
        else:
            st.markdown("**Status:** <span class='status-loading'>🟡 Not loaded</span>", unsafe_allow_html=True)
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Instructions
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Place PDF files in the `data/` folder
        2. Click **Load/Reload Documents**
        3. Start chatting with your documents!
        """)
        
        st.divider()
        st.markdown("### 🔧 Powered by")
        st.markdown("""
        - 🧠 Google Gemini Flash Latest
        - 📐 Text Embedding 004
        - 🗄️ ChromaDB
        - 🦜 LangChain
        """)
    
    # Auto-load documents on first run
    if st.session_state.vectorstore is None and st.session_state.doc_count == 0:
        with st.spinner("📖 Loading documents..."):
            vectorstore, doc_count, chunk_count = load_and_process_documents()
            st.session_state.vectorstore = vectorstore
            st.session_state.doc_count = doc_count
            st.session_state.chunk_count = chunk_count
    
    # Main chat area
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    with st.expander("📎 View Sources", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>📄 Source {i}:</strong> {source['file']} (Page {source['page']})<br>
                                <small style="color: #a0aec0;">{source['content']}</small>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if vectorstore is loaded
        if not st.session_state.vectorstore:
            st.error("⚠️ Please load documents first! Click 'Load/Reload Documents' in the sidebar.")
            st.stop()
        
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get RAG chain and retriever
                    rag_chain, retriever = get_rag_chain(st.session_state.vectorstore)
                    
                    # Get relevant documents for sources
                    source_documents = retriever.invoke(prompt)
                    
                    # Get response from chain
                    answer = rag_chain.invoke(prompt)
                    
                    # Format sources
                    sources = format_sources(source_documents)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources in expander
                    if sources:
                        with st.expander("📎 View Sources", expanded=False):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>📄 Source {i}:</strong> {source['file']} (Page {source['page']})<br>
                                    <small style="color: #a0aec0;">{source['content']}</small>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_message = f"❌ An error occurred: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    })


if __name__ == "__main__":
    main()
