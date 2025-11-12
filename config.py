"""
Experiment Configuration File
Modify this file to configure your experiments
"""

class ExperimentConfig:
    """Configuration for RAG experiments"""
    
    # ========== EXPERIMENT TYPE ==========
    # Options: "closed_book", "single_vector", "shared_vector", "open_book"
    EXPERIMENT_TYPE = "closed_book"
    
    # ========== DATA CONFIGURATION ==========
    NUM_SAMPLES = 5  # Number of samples to process (None for all)
    SAMPLE_INDICES = None  # Specific indices to process (e.g., [0, 1, 2, 3, 4])
    
    # ========== RETRIEVAL CONFIGURATION ==========
    CHUNK_SIZE = 512  # Size of text chunks
    CHUNK_OVERLAP = 50  # Overlap between chunks
    TOP_K = 3  # Number of chunks to retrieve
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
    
    # Alternative embedding models:
    # - "sentence-transformers/all-mpnet-base-v2"  (better quality, slower)
    # - "sentence-transformers/all-MiniLM-L6-v2"  (faster, smaller)
    # - "BAAI/bge-small-en-v1.5"  (good balance)
    
    # ========== EVALUATION CONFIGURATION ==========
    USE_BERTSCORE = False  # Enable BERTScore (requires bert-score package)
    USE_LLM_JUDGE = False  # Enable LLM as judge (requires API implementation)
    
    # ========== OUTPUT CONFIGURATION ==========
    OUTPUT_DIR = "outputs"  # Directory for output files
    LOG_DIR = "logs"  # Directory for log files
    
    # ========== LLM CONFIGURATION (if implementing generation) ==========
    LLM_MODEL = "gpt-4"  # Model for generation
    LLM_TEMPERATURE = 0.7  # Temperature for generation
    LLM_MAX_TOKENS = 512  # Max tokens for generation
    
    # OpenAI configuration
    OPENAI_API_KEY = None  # Set your API key or use environment variable
    
    # Anthropic configuration
    ANTHROPIC_API_KEY = None  # Set your API key or use environment variable


# Predefined experiment configurations
class ClosedBookConfig(ExperimentConfig):
    """Configuration for closed-book experiments"""
    EXPERIMENT_TYPE = "closed_book"
    NUM_SAMPLES = 10


class SingleVectorConfig(ExperimentConfig):
    """Configuration for single vector store experiments"""
    EXPERIMENT_TYPE = "single_vector"
    NUM_SAMPLES = 10
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TOP_K = 5


class SharedVectorConfig(ExperimentConfig):
    """Configuration for shared vector store experiments"""
    EXPERIMENT_TYPE = "shared_vector"
    NUM_SAMPLES = 20
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TOP_K = 5


class OpenBookConfig(ExperimentConfig):
    """Configuration for open-book experiments"""
    EXPERIMENT_TYPE = "open_book"
    NUM_SAMPLES = 10


class QuickTestConfig(ExperimentConfig):
    """Quick test configuration with minimal samples"""
    EXPERIMENT_TYPE = "closed_book"
    NUM_SAMPLES = 2
    USE_BERTSCORE = False
    USE_LLM_JUDGE = False


class FullEvaluationConfig(ExperimentConfig):
    """Full evaluation with all metrics"""
    EXPERIMENT_TYPE = "shared_vector"
    NUM_SAMPLES = 50
    USE_BERTSCORE = True
    USE_LLM_JUDGE = True
    CHUNK_SIZE = 512
    TOP_K = 5


# Select which configuration to use
ACTIVE_CONFIG = ExperimentConfig  # Change this to use different configs