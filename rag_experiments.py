"""
RAG Experiments Main Module
Uses LangChain, FAISS, and HuggingFace models (Llama 3.2 3B, Qwen 2.5 7B)
Supports multiple experiment types: closed-book, single vector store, 
shared vector store, and open-book (evidence)
"""

from __future__ import annotations

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
from collections import defaultdict
import torch

# bitsandbytes availability check (used for 8-bit quantization)
try:
    import bitsandbytes as bnb  # noqa: F401
    _BNB_AVAILABLE = True
except Exception:
    _BNB_AVAILABLE = False

# Optional OpenAI client (for HF router / nference style API)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# LangChain imports (try multiple possible packages / entrypoints and provide safe fallbacks)
_HAS_LANGCHAIN = False
from dataclasses import dataclass
Document = None
RecursiveCharacterTextSplitter = None
HuggingFaceEmbeddings = None
FAISS = None
BaseRetriever = None

_logger = logging.getLogger(__name__)

# 1) Text splitter: try langchain, then langchain_text_splitters
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    _HAS_LANGCHAIN = True
except Exception:
    try:
        # some installs expose splitters via this package
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        _HAS_LANGCHAIN = True
    except Exception:
        RecursiveCharacterTextSplitter = None

# 2) Embeddings: try langchain_community, then langchain.embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    # from langchain_community.embeddings import HuggingFaceEmbeddings
    _HAS_LANGCHAIN = True
except Exception:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        _HAS_LANGCHAIN = True
    except Exception:
        HuggingFaceEmbeddings = None

# 3) FAISS vectorstore: try langchain_community.vectorstores then langchain.vectorstores
try:
    from langchain_community.vectorstores import FAISS
    _HAS_LANGCHAIN = True
except Exception:
    try:
        from langchain.vectorstores import FAISS
        _HAS_LANGCHAIN = True
    except Exception:
        FAISS = None

# 4) Document + BaseRetriever
try:
    from langchain.docstore.document import Document
    from langchain.schema import BaseRetriever
    _HAS_LANGCHAIN = True
except Exception:
    # If not available, we'll provide a tiny Document fallback below
    Document = None
    BaseRetriever = None

if not _HAS_LANGCHAIN:
    _logger.warning("langchain or langchain_community not available (or some subpackages missing); using minimal fallbacks. Install 'langchain' and 'langchain-community' for full functionality.")

    @dataclass
    class Document:
        page_content: str
        metadata: dict = None

    class _MinimalTextSplitter:
        """Minimal fallback for LangChain's RecursiveCharacterTextSplitter.

        Only implements create_documents(texts, metadatas) used in this repo.
        """
        def __init__(self, chunk_size=512, chunk_overlap=50, length_function=len, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function

        def _chunk_text(self, text: str) -> List[str]:
            if not text:
                return []
            chunks = []
            i = 0
            step = self.chunk_size - self.chunk_overlap if self.chunk_size > self.chunk_overlap else self.chunk_size
            while i < len(text):
                chunks.append(text[i:i + self.chunk_size])
                i += step
            return chunks

        def create_documents(self, texts: List[str], metadatas: List[dict]):
            docs = []
            for text, meta in zip(texts, metadatas):
                for chunk in self._chunk_text(text):
                    docs.append(Document(page_content=chunk, metadata=meta or {}))
            return docs

    RecursiveCharacterTextSplitter = _MinimalTextSplitter

    # Provide a placeholder FAISS class that raises a helpful error if used
    if FAISS is None:
        class FAISS:
            @classmethod
            def from_documents(cls, *args, **kwargs):
                raise RuntimeError("FAISS vector store not available. Install 'langchain-community' and a faiss package (faiss-cpu or faiss-gpu) to use vector stores.")


# HuggingFace transformers for LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Import local modules (support running as script or as package)
try:
    from finance_bench.data_loader import FinanceBenchLoader
    from finance_bench.evaluator import Evaluator
except Exception:
    try:
        # When running as a package (e.g., python -m finance_bench.rag_experiments)
        from data_loader import FinanceBenchLoader
        from evaluator import Evaluator
    except Exception as e:
        logging.getLogger(__name__).exception(
            "Failed to import local modules data_loader/evaluator.\n"
            "Make sure you're running this script from the project root or install the package."
        )
        # As a last resort, try to dynamically load the modules by file path
        try:
            import importlib.util
            base_dir = Path(__file__).resolve().parent
            dl_path = base_dir / 'data_loader.py'
            ev_path = base_dir / 'evaluator.py'

            def load_module_from_path(path: Path, module_name: str):
                spec = importlib.util.spec_from_file_location(module_name, str(path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                return mod

            dl_mod = load_module_from_path(dl_path, 'data_loader_local')
            ev_mod = load_module_from_path(ev_path, 'evaluator_local')

            FinanceBenchLoader = getattr(dl_mod, 'FinanceBenchLoader')
            Evaluator = getattr(ev_mod, 'Evaluator')
        except Exception as e2:
            logging.getLogger(__name__).exception(f"Dynamic import fallback failed: {e2}")
            raise


# Set up logging
def setup_logging(experiment_name: str, log_dir: str = "logs"):
    """Setup comprehensive logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{experiment_name}_{timestamp}.log"
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler with simpler formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Root logger - avoid adding duplicate handlers if already configured
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    return log_file


class RAGExperiment:
    """Main RAG Experiment Runner using LangChain, FAISS, and HuggingFace LLMs"""
    
    # Experiment types
    CLOSED_BOOK = "closed_book"
    SINGLE_VECTOR = "single_vector"
    SHARED_VECTOR = "shared_vector"
    OPEN_BOOK = "open_book"
    
    # Available LLMs
    LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"
    QWEN_2_5_7B = "Qwen/Qwen2.5-7B-Instruct"
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"  # For judge
    
    def __init__(self, 
                 experiment_type: str = CLOSED_BOOK,
                 llm_model: str = LLAMA_3_2_3B,
                 judge_model: str = MISTRAL_7B,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 top_k: int = 3,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_bertscore: bool = False,
                 use_llm_judge: bool = False,
                 output_dir: str = "outputs",
                 vector_store_dir: str = "vector_stores",
                 device: str = None,
                 load_in_8bit: bool = True,
                 max_new_tokens: int = 256,
                 use_api: bool = False,
                 api_base_url: str = "https://router.huggingface.co/v1",
                 api_key_env: str = "HF_TOKEN"):
        """
        Initialize RAG Experiment
        
        Args:
            experiment_type: Type of experiment (closed_book, single_vector, shared_vector, open_book)
            llm_model: HuggingFace model for generation (Llama 3.2 3B or Qwen 2.5 7B)
            judge_model: HuggingFace model for LLM-as-judge (Mistral 7B)
            chunk_size: Size of text chunks for retrieval
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
            embedding_model: Model for embeddings
            use_bertscore: Whether to use BERTScore
            use_llm_judge: Whether to use LLM as judge
            output_dir: Directory for outputs
            vector_store_dir: Directory for vector store persistence
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            load_in_8bit: Whether to load models in 8-bit for memory efficiency
            max_new_tokens: Maximum tokens to generate
        """
        self.experiment_type = experiment_type
        self.llm_model_name = llm_model
        self.judge_model_name = judge_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.output_dir = output_dir
        self.vector_store_dir = vector_store_dir
        self.max_new_tokens = max_new_tokens
        self.load_in_8bit = load_in_8bit
        self.use_api = use_api
        self.api_base_url = api_base_url
        self.api_key_env = api_key_env
        self.api_client = None
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize components
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_loader = FinanceBenchLoader()
        self.evaluator = Evaluator(use_bertscore=use_bertscore, use_llm_judge=use_llm_judge)
        
        # LLM components (lazy loading)
        self.llm_tokenizer = None
        self.llm_model = None
        self.llm_pipeline = None
        
        # Judge LLM components (lazy loading)
        self.judge_tokenizer = None
        self.judge_model = None
        self.judge_pipeline = None
        
        # Initialize LangChain components
        self.embeddings = None
        self.text_splitter = None
        self.vector_stores = {}
        
        # Results storage
        self.results = []
        self.experiment_metadata = {
            'experiment_type': experiment_type,
            'llm_model': llm_model,
            'use_api': use_api,
            'api_base_url': api_base_url if use_api else None,
            'judge_model': judge_model if use_llm_judge else None,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'top_k': top_k,
            'embedding_model': embedding_model,
            'device': self.device,
            'load_in_8bit': load_in_8bit,
            'max_new_tokens': max_new_tokens,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(vector_store_dir, exist_ok=True)
        
        self.logger.info("=" * 80)
        self.logger.info(f"INITIALIZING RAG EXPERIMENT: {experiment_type.upper()}")
        self.logger.info("=" * 80)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  Experiment Type: {experiment_type}")
        self.logger.info(f"  LLM Model: {llm_model}")
        self.logger.info(f"  Judge Model: {judge_model if use_llm_judge else 'Disabled'}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  8-bit Loading: {load_in_8bit}")
        self.logger.info(f"  Chunk Size: {chunk_size}")
        self.logger.info(f"  Chunk Overlap: {chunk_overlap}")
        self.logger.info(f"  Top-K Retrieval: {top_k}")
        self.logger.info(f"  Embedding Model: {embedding_model}")
        self.logger.info(f"  Max New Tokens: {max_new_tokens}")
        self.logger.info(f"  Using LangChain + FAISS + HuggingFace LLMs")
        if self.use_api:
            self.logger.info("  Using API-based LLM via OpenAI client (HF router)")
        self.logger.info("=" * 80)
        
        # Initialize LangChain components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LangChain embeddings and text splitter"""
        self.logger.info("\nInitializing LangChain components...")
        
        # Initialize embeddings using LangChain
        self.logger.info(f"Loading embeddings: {self.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.logger.info("✓ Embeddings loaded")
        
        # Initialize text splitter using LangChain
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.logger.info(f"✓ Text splitter initialized (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
    
    def _initialize_llm(self):
        """Initialize HuggingFace LLM for generation"""
        # If configured to use API, initialize the API client and skip local model
        if self.use_api:
            if self.api_client is not None:
                return

            if OpenAI is None:
                self.logger.error("OpenAI client library not installed. Install 'openai' package to use API mode.")
                raise RuntimeError("OpenAI client not available")

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                self.logger.error(f"API key environment variable '{self.api_key_env}' not set")
                raise RuntimeError(f"Missing API key in env var {self.api_key_env}")

            try:
                self.logger.info("Initializing OpenAI API client (HF router)...")
                self.api_client = OpenAI(base_url=self.api_base_url, api_key=api_key)
                self.logger.info("✓ API client initialized")
                return
            except Exception as e:
                self.logger.error(f"Failed to initialize API client: {e}")
                raise

        if self.llm_pipeline is not None:
            return  # Already initialized
        
        self.logger.info(f"\nInitializing LLM: {self.llm_model_name}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  8-bit: {self.load_in_8bit}")
        
        try:
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            
            # Load model
            self.logger.info("Loading model (this may take a moment)...")
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }

            if self.load_in_8bit and self.device == "cuda":
                if not _BNB_AVAILABLE:
                    # bitsandbytes not available — fall back to full precision to avoid exception
                    self.logger.warning("bitsandbytes not available or outdated; falling back to full precision (disabling 8-bit).")
                    # Don't set load_in_8bit; continue with default dtype/device_map
                    self.load_in_8bit = False
                else:
                    # If available, request 8-bit (older transformers use load_in_8bit; if your
                    # transformers requires BitsAndBytesConfig, you can upgrade or set quantization_config)
                    model_kwargs["load_in_8bit"] = True
                    self.logger.info("  Using 8-bit quantization for memory efficiency")
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                **model_kwargs
            )
            
            # Create pipeline
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.llm_model,
                tokenizer=self.llm_tokenizer,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_full_text=False
            )
            
            self.logger.info(f"✓ LLM initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {str(e)}")
            self.logger.error("Make sure you have sufficient GPU memory or try load_in_8bit=True")
            raise
    
    def _initialize_judge_llm(self):
        """Initialize HuggingFace LLM for judging (if enabled)"""
        if not self.evaluator.use_llm_judge:
            return
        
        if self.judge_pipeline is not None:
            return  # Already initialized
        
        self.logger.info(f"\nInitializing Judge LLM: {self.judge_model_name}")
        
        try:
            # Load tokenizer
            self.judge_tokenizer = AutoTokenizer.from_pretrained(self.judge_model_name)
            
            # Load model
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            if self.load_in_8bit and self.device == "cuda":
                model_kwargs["load_in_8bit"] = True
            
            self.judge_model = AutoModelForCausalLM.from_pretrained(
                self.judge_model_name,
                **model_kwargs
            )
            
            # Create pipeline
            self.judge_pipeline = pipeline(
                "text-generation",
                model=self.judge_model,
                tokenizer=self.judge_tokenizer,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.1,
                return_full_text=False
            )
            
            self.logger.info(f"✓ Judge LLM initialized successfully")
            
            # Update evaluator with judge pipeline
            self.evaluator._judge_pipeline = self.judge_pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Judge LLM: {str(e)}")
            self.logger.warning("Continuing without LLM judge")
            self.evaluator.use_llm_judge = False
    
    def _chunk_text_langchain(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Chunk text using LangChain's RecursiveCharacterTextSplitter
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of LangChain Document objects
        """
        # Safely coerce bytes to string (some dataset fields may be bytes)
        if isinstance(text, (bytes, bytearray)):
            try:
                text = text.decode('utf-8')
            except Exception:
                text = text.decode('utf-8', errors='replace')

        if not text or len(text) == 0:
            return []
        
        metadata = metadata or {}
        
        # Use LangChain text splitter
        # Ensure we pass str objects into the text splitter
        documents = self.text_splitter.create_documents(
            texts=[str(text)],
            metadatas=[metadata]
        )
        
        # Log chunking statistics
        # Defensive: coerce any bytes page_content to str before computing lengths
        chunk_lengths = [
            len(doc.page_content)
            if not isinstance(doc.page_content, (bytes, bytearray))
            else len(doc.page_content.decode('utf-8', errors='replace'))
            for doc in documents
        ]

        self.logger.info(f"\nChunking Statistics (LangChain):")
        self.logger.info(f"  Total chunks: {len(documents)}")
        self.logger.info(f"  Avg chunk size: {np.mean(chunk_lengths):.2f} chars")
        self.logger.info(f"  Min chunk size: {np.min(chunk_lengths)} chars")
        self.logger.info(f"  Max chunk size: {np.max(chunk_lengths)} chars")
        self.logger.info(f"  Median chunk size: {np.median(chunk_lengths):.2f} chars")
        
        # Log first few chunks for inspection
        self.logger.debug(f"\nFirst 3 chunks preview:")
        for i, doc in enumerate(documents[:3]):
            content_preview = doc.page_content
            if isinstance(content_preview, (bytes, bytearray)):
                content_preview = content_preview.decode('utf-8', errors='replace')
            self.logger.debug(f"  Chunk {i}: {content_preview[:100]}...")
            self.logger.debug(f"    Metadata: {doc.metadata}")
        
        return documents
    
    def _create_vector_store_faiss(self, documents: List[Document], index_name: str = "default") -> FAISS:
        """
        Create FAISS vector store using LangChain
        
        Args:
            documents: List of LangChain Document objects
            index_name: Name for the vector store index
            
        Returns:
            FAISS vector store
        """
        self.logger.info(f"\nCreating FAISS vector store '{index_name}' with {len(documents)} documents...")
        
        # Create FAISS index using LangChain
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        self.logger.info(f"✓ FAISS vector store created with {vector_store.index.ntotal} vectors")
        
        # Optionally save to disk
        vector_store_path = os.path.join(self.vector_store_dir, f"{index_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        vector_store.save_local(vector_store_path)
        self.logger.info(f"✓ Vector store saved to: {vector_store_path}")
        
        return vector_store
    
    def _retrieve_chunks_faiss(self, query: str, vector_store: FAISS, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks using FAISS
        
        Args:
            query: Query text
            vector_store: FAISS vector store
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with scores and metadata
        """
        if top_k is None:
            top_k = self.top_k
        
        self.logger.info(f"\nRetrieving top-{top_k} chunks for query: {query[:100]}...")
        
        # Use LangChain's similarity_search_with_score
        # Ensure query is a str
        if isinstance(query, (bytes, bytearray)):
            try:
                query = query.decode('utf-8')
            except Exception:
                query = query.decode('utf-8', errors='replace')

        results = vector_store.similarity_search_with_score(query, k=top_k)
        
        # Format results
        retrieved = []
        for rank, (doc, score) in enumerate(results):
            # Defensive: ensure doc.page_content is a string
            text_content = doc.page_content
            if isinstance(text_content, (bytes, bytearray)):
                try:
                    text_content = text_content.decode('utf-8')
                except Exception:
                    text_content = text_content.decode('utf-8', errors='replace')

            chunk_data = {
                'rank': rank + 1,
                'text': text_content,
                'score': float(score),
                'length': len(text_content),
                'metadata': doc.metadata
            }
            retrieved.append(chunk_data)
        
        # Log retrieved chunks
        self.logger.info(f"Retrieved {len(retrieved)} chunks:")
        for chunk in retrieved:
            self.logger.info(f"  Rank {chunk['rank']}: Score={chunk['score']:.4f}, "
                           f"Length={chunk['length']} chars")
            self.logger.info(f"    Metadata: {chunk['metadata']}")
            self.logger.info(f"    Text preview: {chunk['text'][:150]}...")
        
        return retrieved
    
    def _normalize_evidence(self, evidence: Any) -> List[str]:
        """
        Normalize evidence payloads from the dataset into a list of clean strings.
        
        FinanceBench evidence may be a string, list/tuple of strings, bytes, or nested structures.
        This helper flattens and decodes everything into a list of non-empty strings.
        """
        normalized: List[str] = []
        
        if evidence is None:
            return normalized
        
        # Recursively flatten iterables
        if isinstance(evidence, (list, tuple, set)):
            for item in evidence:
                normalized.extend(self._normalize_evidence(item))
            return normalized
        
        if isinstance(evidence, dict):
            for item in evidence.values():
                normalized.extend(self._normalize_evidence(item))
            return normalized
        
        if isinstance(evidence, (bytes, bytearray)):
            try:
                text = evidence.decode('utf-8')
            except Exception:
                text = evidence.decode('utf-8', errors='replace')
        else:
            text = str(evidence)
        
        text = text.strip()
        if text:
            normalized.append(text)
        
        return normalized
      
    def _generate_answer(self, question: str, context: str = None) -> str:
        """
        Generate answer using HuggingFace LLM (Llama 3.2 3B or Qwen 2.5 7B)
        
        Args:
            question: Question to answer
            context: Context for answering (optional)
            
        Returns:
            Generated answer
        """
        # Initialize LLM if not already done
        self._initialize_llm()
        
        self.logger.info(f"\nGenerating answer for question: {question[:100]}...")
        
        # Format prompt based on model type
        if "llama" in self.llm_model_name.lower():
            # Llama 3.2 uses chat format
            if context:
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful financial analyst assistant. Answer questions based on the provided context accurately and concisely.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            else:
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful financial analyst assistant. Answer questions accurately and concisely based on your knowledge.<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        elif "qwen" in self.llm_model_name.lower():
            # Qwen 2.5 uses a different chat format
            if context:
                prompt = f"""<|im_start|>system
You are a helpful financial analyst assistant. Answer questions based on the provided context accurately and concisely.<|im_end|>
<|im_start|>user
Context: {context}

Question: {question}<|im_end|>
<|im_start|>assistant
"""
            else:
                prompt = f"""<|im_start|>system
You are a helpful financial analyst assistant. Answer questions accurately and concisely based on your knowledge.<|im_end|>
<|im_start|>user
Question: {question}<|im_end|>
<|im_start|>assistant
"""
        
        else:
            # Generic format for other models
            if context:
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            else:
                prompt = f"Question: {question}\n\nAnswer:"
        
        if context:
            self.logger.info(f"Using context of length: {len(context)} chars")
        
        # If configured to use the OpenAI-style API (HF router), call it here
        if self.use_api:
            if self.api_client is None:
                # Ensure client initialized (this will raise if missing config)
                self._initialize_llm()

            # Build a concise message for the chat API
            if context:
                message_content = f"Context: {context}\n\nQuestion: {question}"
            else:
                message_content = f"Question: {question}"

            try:
                self.logger.info("Calling API-based LLM (chat completion)...")
                completion = self.api_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[{"role": "user", "content": message_content}],
                )

                # Extract text robustly
                choice = completion.choices[0]
                answer = None
                # Try dict-style access
                if isinstance(choice, dict):
                    msg = choice.get('message') or {}
                    answer = msg.get('content') or choice.get('text') or str(choice)
                else:
                    # object-style
                    msg = getattr(choice, 'message', None)
                    if msg is not None and hasattr(msg, 'content'):
                        answer = msg.content
                    else:
                        answer = getattr(choice, 'text', str(choice))

                answer = (answer or '').strip()
                self.logger.info(f"API-generated answer: {answer[:200]}...")
                return answer

            except Exception as e:
                self.logger.error(f"API generation error: {e}")
                return f"[API error generating answer: {e}]"

        # Fallback to local pipeline generation
        try:
            # Generate answer
            self.logger.info("Generating response from LLM...")
            response = self.llm_pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=1,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )

            # Extract generated text
            answer = response[0]['generated_text'].strip()

            # Clean up answer (remove special tokens if any leaked through)
            answer = answer.replace("<|eot_id|>", "").replace("<|im_end|>", "").strip()

            self.logger.info(f"Generated answer: {answer[:200]}...")

            return answer

        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            return f"[Error generating answer: {str(e)}]"
    
    def run_closed_book(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run closed-book experiment (no retrieval)
        
        Args:
            data: List of data samples
            
        Returns:
            List of results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("RUNNING CLOSED-BOOK EXPERIMENT")
        self.logger.info("=" * 80)
        
        results = []
        
        for i, sample in enumerate(data):
            self.logger.info(f"\n--- Sample {i+1}/{len(data)} ---")
            
            question = sample['question']
            reference_answer = sample['answer']
            
            # Generate answer without context
            generated_answer = self._generate_answer(question, context=None)
            
            # Evaluate generation
            eval_results = self.evaluator.evaluate_generation(
                generated_answer, 
                reference_answer,
                question
            )
            
            result = {
                'sample_id': i,
                'question': question,
                'reference_answer': reference_answer,
                'generated_answer': generated_answer,
                'generation_length': len(generated_answer),
                'generation_evaluation': eval_results,
                'experiment_type': self.CLOSED_BOOK
            }
            
            results.append(result)
            self.logger.info(f"Completed sample {i+1}")
        
        return results
    
    def run_single_vector(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run single vector store experiment (separate FAISS index per document)
        
        Args:
            data: List of data samples
            
        Returns:
            List of results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("RUNNING SINGLE VECTOR STORE EXPERIMENT (LangChain + FAISS)")
        self.logger.info("=" * 80)
        
        results = []
        
        # Group by document
        doc_groups = defaultdict(list)
        for sample in data:
            doc_name = sample.get('doc_name', 'unknown')
            doc_groups[doc_name].append(sample)
        
        self.logger.info(f"Processing {len(doc_groups)} unique documents")
        
        for doc_name, samples in doc_groups.items():
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Document: {doc_name}")
            self.logger.info(f"Samples: {len(samples)}")
            self.logger.info(f"{'='*80}")
            
            # Aggregate evidence from all samples under this document
            doc_segments: List[str] = []
            for sample in samples:
                doc_segments.extend(self._normalize_evidence(sample.get('evidence')))
            
            # Deduplicate while preserving order
            unique_segments = list(dict.fromkeys(doc_segments))
            doc_content = "\n\n".join(unique_segments).strip()
            
            if not doc_content:
                self.logger.warning(f"No content available for document: {doc_name}")
                continue
            
            # Chunk document using LangChain
            documents = self._chunk_text_langchain(
                doc_content,
                metadata={'doc_name': doc_name, 'source': 'evidence'}
            )
            
            # Create FAISS vector store
            vector_store = self._create_vector_store_faiss(documents, index_name=doc_name)
            
            # Process each sample
            for i, sample in enumerate(samples):
                self.logger.info(f"\n--- Sample {i+1}/{len(samples)} for {doc_name} ---")
                
                question = sample['question']
                reference_answer = sample['answer']
                gold_evidence_segments = self._normalize_evidence(sample.get('evidence'))
                gold_evidence = "\n\n".join(gold_evidence_segments)
                
                # Retrieve relevant chunks using FAISS
                retrieved_chunks = self._retrieve_chunks_faiss(question, vector_store)
                
                # Combine retrieved chunks into context
                context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
                
                # Evaluate retrieval
                retrieved_texts = [chunk['text'] for chunk in retrieved_chunks]
                retrieval_eval = self.evaluator.compute_retrieval_metrics(
                    retrieved_texts, 
                    gold_evidence
                )
                
                # Generate answer with context
                generated_answer = self._generate_answer(question, context)
                
                # Evaluate generation
                generation_eval = self.evaluator.evaluate_generation(
                    generated_answer,
                    reference_answer,
                    question
                )
                
                result = {
                    'sample_id': len(results),
                    'doc_name': doc_name,
                    'question': question,
                    'reference_answer': reference_answer,
                    'gold_evidence': gold_evidence,
                    'retrieved_chunks': retrieved_chunks,
                    'num_retrieved': len(retrieved_chunks),
                    'context_length': len(context),
                    'generated_answer': generated_answer,
                    'generation_length': len(generated_answer),
                    'retrieval_evaluation': retrieval_eval,
                    'generation_evaluation': generation_eval,
                    'experiment_type': self.SINGLE_VECTOR,
                    'vector_store_type': 'FAISS'
                }
                
                results.append(result)
                self.logger.info(f"Completed sample {len(results)}")
        
        return results
    
    def run_shared_vector(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run shared vector store experiment (single FAISS index for all documents)
        
        Args:
            data: List of data samples
            
        Returns:
            List of results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("RUNNING SHARED VECTOR STORE EXPERIMENT (LangChain + FAISS)")
        self.logger.info("=" * 80)
        
        # Collect all unique document contents
        unique_docs = {}
        for sample in data:
            doc_name = sample.get('doc_name', 'unknown')
            if doc_name not in unique_docs:
                unique_docs[doc_name] = sample.get('evidence', '')
        
        self.logger.info(f"Collected {len(unique_docs)} unique documents")
        
        # Chunk all documents using LangChain
        all_documents = []
        for doc_name, content in unique_docs.items():
            if content:
                self.logger.info(f"\nChunking document: {doc_name}")
                docs = self._chunk_text_langchain(
                    content,
                    metadata={'doc_name': doc_name, 'source': 'evidence'}
                )
                all_documents.extend(docs)
        
        self.logger.info(f"\nTotal documents across all chunks: {len(all_documents)}")
        
        # Create shared FAISS vector store
        self.logger.info("\nCreating shared FAISS vector store...")
        shared_vector_store = self._create_vector_store_faiss(all_documents, index_name="shared_store")
        
        # Process each sample
        results = []
        for i, sample in enumerate(data):
            self.logger.info(f"\n--- Sample {i+1}/{len(data)} ---")
            
            question = sample['question']
            reference_answer = sample['answer']
            gold_evidence = sample.get('evidence', '')
            # Defensive: coerce bytes to str for gold evidence
            if isinstance(gold_evidence, (bytes, bytearray)):
                try:
                    gold_evidence = gold_evidence.decode('utf-8')
                except Exception:
                    gold_evidence = gold_evidence.decode('utf-8', errors='replace')
            doc_name = sample.get('doc_name', 'unknown')
            
            # Retrieve relevant chunks using FAISS
            retrieved_chunks = self._retrieve_chunks_faiss(question, shared_vector_store)
            
            # Combine retrieved chunks into context
            context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
            
            # Log which documents chunks came from
            source_docs = [chunk['metadata'].get('doc_name', 'unknown') for chunk in retrieved_chunks]
            self.logger.info(f"Retrieved chunks from documents: {source_docs}")
            
            # Evaluate retrieval
            retrieved_texts = [chunk['text'] for chunk in retrieved_chunks]
            retrieval_eval = self.evaluator.compute_retrieval_metrics(
                retrieved_texts,
                gold_evidence
            )
            
            # Generate answer with context
            generated_answer = self._generate_answer(question, context)
            
            # Evaluate generation
            generation_eval = self.evaluator.evaluate_generation(
                generated_answer,
                reference_answer,
                question
            )
            
            result = {
                'sample_id': i,
                'doc_name': doc_name,
                'question': question,
                'reference_answer': reference_answer,
                'gold_evidence': gold_evidence,
                'retrieved_chunks': retrieved_chunks,
                'retrieved_from_docs': source_docs,
                'num_retrieved': len(retrieved_chunks),
                'context_length': len(context),
                'generated_answer': generated_answer,
                'generation_length': len(generated_answer),
                'retrieval_evaluation': retrieval_eval,
                'generation_evaluation': generation_eval,
                'experiment_type': self.SHARED_VECTOR,
                'vector_store_type': 'FAISS'
            }
            
            results.append(result)
            self.logger.info(f"Completed sample {i+1}")
        
        return results
    
    def run_open_book(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run open-book experiment (use gold evidence directly)
        
        Args:
            data: List of data samples
            
        Returns:
            List of results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("RUNNING OPEN-BOOK EXPERIMENT (Gold Evidence)")
        self.logger.info("=" * 80)
        
        results = []
        
        for i, sample in enumerate(data):
            self.logger.info(f"\n--- Sample {i+1}/{len(data)} ---")
            
            question = sample['question']
            reference_answer = sample['answer']
            gold_evidence = sample.get('evidence', '')
            # Defensive: coerce bytes to str for gold evidence
            if isinstance(gold_evidence, (bytes, bytearray)):
                try:
                    gold_evidence = gold_evidence.decode('utf-8')
                except Exception:
                    gold_evidence = gold_evidence.decode('utf-8', errors='replace')

            # Use gold evidence as context
            context = gold_evidence
            
            self.logger.info(f"Using gold evidence as context (length: {len(context)} chars)")
            
            # Generate answer with gold evidence
            generated_answer = self._generate_answer(question, context)
            
            # Evaluate generation (no retrieval evaluation since we use gold evidence)
            generation_eval = self.evaluator.evaluate_generation(
                generated_answer,
                reference_answer,
                question
            )
            
            result = {
                'sample_id': i,
                'question': question,
                'reference_answer': reference_answer,
                'gold_evidence': gold_evidence,
                'context_length': len(context),
                'generated_answer': generated_answer,
                'generation_length': len(generated_answer),
                'generation_evaluation': generation_eval,
                'experiment_type': self.OPEN_BOOK
            }
            
            results.append(result)
            self.logger.info(f"Completed sample {i+1}")
        
        return results
    
    def run_experiment(self, num_samples: int = None, sample_indices: List[int] = None):
        """
        Run the configured experiment
        
        Args:
            num_samples: Number of samples to process (None for all)
            sample_indices: Specific sample indices to process
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING EXPERIMENT")
        self.logger.info("=" * 80)
        
        # Load data
        df = self.data_loader.load_data()
        
        # Select samples
        if sample_indices is not None:
            data = self.data_loader.get_batch(indices=sample_indices)
        elif num_samples is not None:
            data = self.data_loader.get_batch(start=0, end=num_samples)
        else:
            data = self.data_loader.get_batch()
        
        self.logger.info(f"Processing {len(data)} samples")
        
        # Run appropriate experiment type
        if self.experiment_type == self.CLOSED_BOOK:
            results = self.run_closed_book(data)
        elif self.experiment_type == self.SINGLE_VECTOR:
            results = self.run_single_vector(data)
        elif self.experiment_type == self.SHARED_VECTOR:
            results = self.run_shared_vector(data)
        elif self.experiment_type == self.OPEN_BOOK:
            results = self.run_open_book(data)
        else:
            raise ValueError(f"Unknown experiment type: {self.experiment_type}")
        
        self.results = results
        
        # Compute aggregate statistics
        self._compute_aggregate_stats()
        
        # Save results
        self._save_results()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT COMPLETE")
        self.logger.info("=" * 80)
    
    def _compute_aggregate_stats(self):
        """Compute and log aggregate statistics across all samples"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("AGGREGATE STATISTICS")
        self.logger.info("=" * 80)
        
        if not self.results:
            self.logger.warning("No results to aggregate")
            return
        
        # Generation length stats
        gen_lengths = [r['generation_length'] for r in self.results]
        self.logger.info(f"\nGenerated Answer Lengths:")
        self.logger.info(f"  Mean: {np.mean(gen_lengths):.2f} chars")
        self.logger.info(f"  Min: {np.min(gen_lengths)} chars")
        self.logger.info(f"  Max: {np.max(gen_lengths)} chars")
        self.logger.info(f"  Median: {np.median(gen_lengths):.2f} chars")
        
        # Retrieval stats (if applicable)
        if self.experiment_type in [self.SINGLE_VECTOR, self.SHARED_VECTOR]:
            context_lengths = [r['context_length'] for r in self.results]
            self.logger.info(f"\nContext Lengths:")
            self.logger.info(f"  Mean: {np.mean(context_lengths):.2f} chars")
            self.logger.info(f"  Min: {np.min(context_lengths)} chars")
            self.logger.info(f"  Max: {np.max(context_lengths)} chars")
            
            # Retrieval metrics
            exact_matches = [r['retrieval_evaluation']['exact_match'] for r in self.results]
            max_overlaps = [r['retrieval_evaluation']['max_token_overlap'] for r in self.results]
            
            self.logger.info(f"\nRetrieval Performance:")
            self.logger.info(f"  Exact Match Rate: {np.mean(exact_matches):.2%}")
            self.logger.info(f"  Mean Max Token Overlap: {np.mean(max_overlaps):.4f}")
        
        # Generation metrics
        bleu_4_scores = []
        rouge_l_scores = []
        
        for r in self.results:
            if 'generation_evaluation' in r:
                eval_data = r['generation_evaluation']
                if 'bleu' in eval_data and 'bleu_4' in eval_data['bleu']:
                    bleu_4_scores.append(eval_data['bleu']['bleu_4'])
                if 'rouge' in eval_data and 'rouge_l_f1' in eval_data['rouge']:
                    rouge_l_scores.append(eval_data['rouge']['rouge_l_f1'])
        
        if bleu_4_scores:
            self.logger.info(f"\nGeneration Performance:")
            self.logger.info(f"  Mean BLEU-4: {np.mean(bleu_4_scores):.4f}")
            self.logger.info(f"  Mean ROUGE-L F1: {np.mean(rouge_l_scores):.4f}")
        
        self.logger.info("=" * 80)
    
    def _save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{self.experiment_type}_{timestamp}.json"
        
        output_data = {
            'metadata': self.experiment_metadata,
            'num_samples': len(self.results),
            'framework': 'LangChain + FAISS',
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"\nResults saved to: {filename}")


def main():
    """Main entry point for experiments"""
    
    # ========== EXPERIMENT CONFIGURATION ==========
    # Change these parameters to configure your experiment
    
    EXPERIMENT_TYPE = RAGExperiment.SINGLE_VECTOR  # Options: CLOSED_BOOK, SINGLE_VECTOR, SHARED_VECTOR, OPEN_BOOK
    NUM_SAMPLES = None  # Number of samples to process (None for all)
    
    # LLM Configuration
    LLM_MODEL = RAGExperiment.LLAMA_3_2_3B  # Options: LLAMA_3_2_3B, QWEN_2_5_7B
    # To test both models, set RUN_BOTH_MODELS = True
    RUN_BOTH_MODELS = True  # Run experiments with both Llama and Qwen
    
    # RAG Configuration
    CHUNK_SIZE = 512  # Size of text chunks
    CHUNK_OVERLAP = 50  # Overlap between chunks
    TOP_K = 3  # Number of chunks to retrieve
    
    # Evaluation Configuration
    USE_BERTSCORE = False  # Enable BERTScore (slower)
    USE_LLM_JUDGE = False  # Enable LLM as judge (requires more memory)
    
    # Model Configuration
    DEVICE = None  # Auto-detect ('cuda' or 'cpu')
    LOAD_IN_8BIT = True  # Use 8-bit quantization (recommended for GPU)
    MAX_NEW_TOKENS = 256  # Maximum tokens to generate
    
    # ===============================================
    
    if RUN_BOTH_MODELS:
        # Run experiments with both models
        models_to_test = [
            ("llama_3_2_3b", RAGExperiment.LLAMA_3_2_3B),
            ("qwen_2_5_7b", RAGExperiment.QWEN_2_5_7B)
        ]
    else:
        # Run with single model
        model_name = "llama" if "llama" in LLM_MODEL.lower() else "qwen"
        models_to_test = [(model_name, LLM_MODEL)]
    
    for model_name, model_path in models_to_test:
        print("\n" + "="*80)
        print(f"RUNNING EXPERIMENT WITH {model_name.upper()}")
        print("="*80)
        
        # Setup logging
        log_file = setup_logging(f"rag_experiment_{EXPERIMENT_TYPE}_{model_name}")
        print(f"Logging to: {log_file}")
        
        try:
            # Create experiment
            experiment = RAGExperiment(
                experiment_type=EXPERIMENT_TYPE,
                llm_model=model_path,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                top_k=TOP_K,
                use_bertscore=USE_BERTSCORE,
                use_llm_judge=USE_LLM_JUDGE,
                device=DEVICE,
                load_in_8bit=LOAD_IN_8BIT,
                max_new_tokens=MAX_NEW_TOKENS
            )
            
            # Run experiment
            experiment.run_experiment(num_samples=NUM_SAMPLES)
            
            print("\n" + "="*80)
            print(f"EXPERIMENT WITH {model_name.upper()} COMPLETE!")
            print(f"Check logs: {log_file}")
            print(f"Check outputs: {experiment.output_dir}/")
            print("="*80)
            
        except Exception as e:
            print(f"\n✗ Error running experiment with {model_name}: {str(e)}")
            print(f"Check logs for details: {log_file}")
            continue
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\n📊 To compare results:")
    print("  python analyze_results.py")


if __name__ == "__main__":
    main()