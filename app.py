
import streamlit as st
import os
from typing import List, Dict, Any
from pathlib import Path
import tempfile

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RAGConfig:
    """Configuration settings for the RAG application."""
    CHUNK_SIZE = 5000
    CHUNK_OVERLAP = 500
    MODEL_NAME = "openai/gpt-oss-20b"
    MAX_ANSWER_SENTENCES = 3
    

class DocumentProcessor:
    """Handles PDF document processing and vector store creation."""
    
    def __init__(self, embeddings: OpenAIEmbeddings, config: RAGConfig):
        self.embeddings = embeddings
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
    
    def process_pdfs(self, uploaded_files: List) -> Chroma:
        """
        Process uploaded PDF files and create a vector store.
        
        Args:
            uploaded_files: List of uploaded PDF files from Streamlit
            
        Returns:
            Chroma vector store containing document embeddings
        """
        documents = []
        
        for uploaded_file in uploaded_files:
            # Use temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                documents.extend(docs)
            finally:
                # Clean up temporary file
                Path(tmp_path).unlink(missing_ok=True)
        
        # Split documents and create vector store
        splits = self.text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=self.embeddings)
        
        return vectorstore


class ConversationalRAG:
    """Manages the conversational RAG chain and chat history."""
    
    def __init__(self, llm: ChatGroq, retriever, config: RAGConfig):
        self.llm = llm
        self.retriever = retriever
        self.config = config
        self.chain = self._create_chain()
    
    def _create_chain(self):
        """Create the conversational RAG chain with history awareness."""
        
        # Contextualize questions based on chat history
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        
        # Answer questions using retrieved context
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            f"don't know. Use {self.config.MAX_ANSWER_SENTENCES} sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Wrap with message history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        return conversational_rag_chain
    
    @staticmethod
    def _get_session_history(session_id: str) -> BaseChatMessageHistory:
        """Retrieve or create chat history for a session."""
        if 'store' not in st.session_state:
            st.session_state.store = {}
        
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        
        return st.session_state.store[session_id]
    
    def get_response(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """
        Get a response for user input within a session.
        
        Args:
            user_input: User's question
            session_id: Session identifier
            
        Returns:
            Dictionary containing the answer and metadata
        """
        response = self.chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return response


def initialize_app() -> tuple:
    """Initialize application components and validate configuration."""
    
    # Validate OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("⚠️ OPENAI_API_KEY not found in environment variables")
        st.stop()
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    config = RAGConfig()
    
    return embeddings, config


def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="Conversational RAG",
        page_icon="📚",
        layout="wide"
    )
    
    # Header
    st.title("📚 Conversational RAG with PDF Processing")
    st.markdown("Upload PDF documents and have intelligent conversations about their content")
    
    # Initialize components
    embeddings, config = initialize_app()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Groq API Key:",
            type="password",
            help="Enter your Groq API key to enable the chatbot"
        )
        
        # Session management
        session_id = st.text_input(
            "Session ID:",
            value="default_session",
            help="Unique identifier for your chat session"
        )
        
        st.divider()
        
        # File upload
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF files:",
            type="pdf",
            accept_multiple_files=True,
            help="Select one or more PDF files to process"
        )
    
    # Main content area
    if not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to continue")
        st.info("💡 Get your API key from [Groq Console](https://console.groq.com)")
        return
    
    # Initialize LLM
    try:
        llm = ChatGroq(groq_api_key=api_key, model_name=config.MODEL_NAME)
    except Exception as e:
        st.error(f"❌ Error initializing LLM: {str(e)}")
        return
    
    # Process uploaded files
    if uploaded_files:
        with st.spinner("🔄 Processing PDF documents..."):
            try:
                doc_processor = DocumentProcessor(embeddings, config)
                vectorstore = doc_processor.process_pdfs(uploaded_files)
                retriever = vectorstore.as_retriever()
                
                # Create conversational RAG
                conversational_rag = ConversationalRAG(llm, retriever, config)
                
                st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)")
                
                # Store in session state
                st.session_state.conversational_rag = conversational_rag
                st.session_state.session_id = session_id
                
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")
                return
    
    # Chat interface
    if 'conversational_rag' in st.session_state:
        st.divider()
        st.header("💬 Chat with Your Documents")
        
        # Display chat history
        if 'store' in st.session_state and session_id in st.session_state.store:
            session_history = st.session_state.store[session_id]
            
            if session_history.messages:
                st.subheader("Chat History")
                for i, message in enumerate(session_history.messages):
                    role = "User" if i % 2 == 0 else "Assistant"
                    with st.chat_message(role.lower()):
                        st.write(message.content)
        
        # User input
        user_input = st.chat_input("Ask a question about your documents...")
        
        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.conversational_rag.get_response(
                            user_input, 
                            st.session_state.session_id
                        )
                        st.write(response['answer'])
                        
                        # Show source documents in expander
                        with st.expander("📖 View Source Documents"):
                            for i, doc in enumerate(response.get('context', []), 1):
                                st.markdown(f"**Source {i}:**")
                                st.text(doc.page_content[:500] + "...")
                                st.divider()
                                
                    except Exception as e:
                        st.error(f"❌ Error generating response: {str(e)}")
    
    else:
        st.info("👆 Upload PDF documents in the sidebar to start chatting")
        
        # Display features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📤 Upload")
            st.write("Upload one or more PDF documents")
        
        with col2:
            st.subheader("🤖 Ask")
            st.write("Ask questions about document content")
        
        with col3:
            st.subheader("💾 Remember")
            st.write("Maintains conversation context")


if __name__ == "__main__":
    main()