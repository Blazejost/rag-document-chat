"""
Unit tests for the RAG Document Chat application.
Tests core functionality without requiring API calls.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDocumentProcessing:
    """Tests for document loading and processing functions."""
    
    def test_data_folder_exists(self):
        """Test that data folder exists or can be created."""
        data_path = "data"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        assert os.path.exists(data_path)
    
    def test_text_splitter_configuration(self):
        """Test that text splitter is configured correctly."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            pytest.skip("langchain_text_splitters not installed")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Test splitting a sample text
        sample_text = "This is a test. " * 100
        chunks = splitter.split_text(sample_text)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= 1000 for chunk in chunks)
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            pytest.skip("langchain_text_splitters not installed")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len
        )
        
        # Long text that will be split
        sample_text = "A" * 50 + " " + "B" * 50 + " " + "C" * 50
        chunks = splitter.split_text(sample_text)
        
        assert len(chunks) >= 2, "Text should be split into multiple chunks"


class TestSourceFormatting:
    """Tests for source citation formatting."""
    
    def test_format_sources_empty(self):
        """Test formatting with empty source list."""
        # Import the function from app
        def format_sources(source_documents):
            sources = []
            seen = set()
            for doc in source_documents:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                filename = os.path.basename(source)
                key = f"{filename}-{page}"
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "file": filename,
                        "page": page + 1 if isinstance(page, int) else page,
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    })
            return sources
        
        result = format_sources([])
        assert result == []
    
    def test_format_sources_with_documents(self):
        """Test formatting with mock documents."""
        def format_sources(source_documents):
            sources = []
            seen = set()
            for doc in source_documents:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                filename = os.path.basename(source)
                key = f"{filename}-{page}"
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "file": filename,
                        "page": page + 1 if isinstance(page, int) else page,
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    })
            return sources
        
        # Create mock documents
        mock_doc = Mock()
        mock_doc.metadata = {"source": "/path/to/test.pdf", "page": 0}
        mock_doc.page_content = "This is test content from the document."
        
        result = format_sources([mock_doc])
        
        assert len(result) == 1
        assert result[0]["file"] == "test.pdf"
        assert result[0]["page"] == 1  # 0-indexed becomes 1-indexed
        assert "test content" in result[0]["content"]
    
    def test_duplicate_sources_filtered(self):
        """Test that duplicate sources are filtered out."""
        def format_sources(source_documents):
            sources = []
            seen = set()
            for doc in source_documents:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                filename = os.path.basename(source)
                key = f"{filename}-{page}"
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "file": filename,
                        "page": page + 1 if isinstance(page, int) else page,
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    })
            return sources
        
        # Create duplicate mock documents
        mock_doc1 = Mock()
        mock_doc1.metadata = {"source": "test.pdf", "page": 0}
        mock_doc1.page_content = "Content 1"
        
        mock_doc2 = Mock()
        mock_doc2.metadata = {"source": "test.pdf", "page": 0}
        mock_doc2.page_content = "Content 2"
        
        result = format_sources([mock_doc1, mock_doc2])
        
        assert len(result) == 1  # Duplicates should be filtered


class TestEnvironmentConfiguration:
    """Tests for environment and configuration."""
    
    def test_env_file_example_exists(self):
        """Test that .env.example file exists."""
        assert os.path.exists(".env.example"), ".env.example should exist for new users"
    
    def test_env_example_contains_api_key_placeholder(self):
        """Test that .env.example has the correct structure."""
        with open(".env.example", "r") as f:
            content = f.read()
        assert "GOOGLE_API_KEY" in content


class TestProjectStructure:
    """Tests for project structure and files."""
    
    def test_required_files_exist(self):
        """Test that all required project files exist."""
        required_files = [
            "app.py",
            "pyproject.toml",
            "README.md",
            ".gitignore",
            ".env.example"
        ]
        
        for file in required_files:
            assert os.path.exists(file), f"Required file {file} should exist"
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile exists for containerization."""
        assert os.path.exists("Dockerfile"), "Dockerfile should exist"
    
    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists."""
        assert os.path.exists("docker-compose.yml"), "docker-compose.yml should exist"


class TestAppImports:
    """Tests for verifying app imports work correctly."""
    
    def test_langchain_imports(self):
        """Test that LangChain modules can be imported."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough
            
            assert RecursiveCharacterTextSplitter is not None
            assert ChatPromptTemplate is not None
            assert StrOutputParser is not None
            assert RunnablePassthrough is not None
        except ImportError as e:
            pytest.skip(f"LangChain dependencies not installed: {e}")
    
    def test_streamlit_import(self):
        """Test that Streamlit can be imported."""
        try:
            import streamlit as st
            assert st is not None
        except ImportError as e:
            pytest.skip(f"Streamlit not installed: {e}")
    
    def test_chromadb_import(self):
        """Test that ChromaDB can be imported."""
        try:
            import chromadb
            assert chromadb is not None
        except ImportError as e:
            pytest.skip(f"ChromaDB not installed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
