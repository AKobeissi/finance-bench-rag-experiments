***WORK IN PROGRESS***

# FinanceBench RAG Experiments

A modular framework for evaluating Retrieval-Augmented Generation (RAG) systems on financial question-answering using the FinanceBench dataset.

## 🎯 Overview

This project implements and evaluates multiple RAG strategies for answering financial questions from documents like 10-K filings, earnings reports, and financial statements. The framework provides comprehensive evaluation metrics for both retrieval quality and answer generation accuracy.

### Experiment Modes

- **Closed Book**: LLM answers from knowledge alone (no retrieval, no context)
- **Open Book**: Oracle evaluation with gold evidence provided as context  
- **Single Vector**: Separate FAISS index per document (realistic single-doc QA)
- **Shared Vector**: Unified FAISS index across all documents (cross-document retrieval)

### Key Features

✅ Modular architecture with pluggable experiment types  
✅ Comprehensive evaluation: BLEU, ROUGE, BERTScore, Exact Match, F1, Recall@K  
✅ PDF text extraction for real-world document retrieval  
✅ JSON output format for easy analysis and visualization  

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip package manager
- (Optional) CUDA-capable GPU for faster embeddings and LLM inference

### Basic Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd financebench-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Requirements

**Core Dependencies:**
```txt
datasets>=2.14.0
pandas>=2.0.0
numpy>=1.24.0
langchain>=0.1.0
langchain-community>=0.0.20
faiss-cpu>=1.7.4  # or faiss-gpu for GPU support
sentence-transformers>=2.2.0
torch>=2.0.0
transformers>=4.30.0
```

**Optional (for full evaluation metrics):**
```txt
sacrebleu>=2.3.0
rouge-score>=0.1.2
bert-score>=0.3.13
```

**For PDF processing:**
```txt
pypdf>=3.0.0
pdfplumber>=0.10.0
```

---

## 🚀 Quick Start

### 1. Load the Dataset

```python
from data_loader import FinanceBenchLoader

# Initialize loader
loader = FinanceBenchLoader()

# Load dataset from HuggingFace
df = loader.load_data(split="train")

# Get a sample
sample = loader.get_sample(0)
print(f"Question: {sample['question']}")
print(f"Answer: {sample['answer']}")
```

### 2. Run a Simple Experiment

```python
from rag_experiments import RAGExperiment
from rag_closed_book import run_closed_book

# Initialize experiment with your config
experiment = RAGExperiment(config)

# Get test samples
samples = loader.get_batch(start=0, end=10)

# Run closed-book experiment (no retrieval)
results = run_closed_book(experiment, samples)

# Save results
import json
with open('results_closed_book.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### 3. Run Retrieval-Based Experiments

```python
from rag_single_vector import run_single_vector
from rag_shared_vector import run_shared_vector

# Single vector store (one FAISS index per document)
results_single = run_single_vector(experiment, samples)

# Shared vector store (one FAISS index for all documents)
results_shared = run_shared_vector(experiment, samples)
```

---

## 📊 Understanding Results

### Result Structure

Each experiment produces a list of result dictionaries with the following structure:

```python
{
    # Identifiers
    "sample_id": 0,
    "doc_name": "APPLE_10K_2022",
    "doc_link": "https://...",
    
    # Question & Answers
    "question": "What was Apple's revenue in Q4 2022?",
    "reference_answer": "$90.1 billion",
    "generated_answer": "$90.146 billion",
    
    # Retrieval Info (for retrieval modes)
    "retrieved_chunks": [
        {
            "text": "Revenue for Q4 2022 was $90.146 billion...",
            "metadata": {"doc_name": "...", "source": "pdf"},
            "score": 0.85
        }
    ],
    "num_retrieved": 5,
    "context_length": 2048,
    
    # Evaluation Metrics
    "retrieval_evaluation": {
        "recall@5": 0.80,
        "precision@5": 0.60,
        "mrr": 0.50,
        "coverage": 0.75
    },
    "generation_evaluation": {
        "exact_match": false,
        "f1": 0.857,
        "bleu": 0.723,
        "rouge_l": 0.882,
        "bertscore_f1": 0.912
    },
    
    # Metadata
    "experiment_type": "single_vector",
    "vector_store_type": "FAISS"
}
```

### Skipped Samples

Samples without available PDF text are marked as skipped:

```python
{
    "sample_id": 42,
    "skipped": true,
    "skipped_reason": "no_pdf_text",
    # ... minimal other fields
}
```

---

## 🏗️ Architecture

### Module Organization

```
finance_bench/
├── data_loader.py           # Dataset loading from HuggingFace
├── evaluator.py             # Evaluation metrics (BLEU, ROUGE, etc.)
├── rag_experiments.py       # Core RAGExperiment class
├── rag_closed_book.py       # No-retrieval baseline
├── rag_open_book.py         # Oracle with gold evidence
├── rag_single_vector.py     # Per-document FAISS indices
└── rag_shared_vector.py     # Cross-document FAISS index
```

### Data Flow

```
HuggingFace Dataset
    ↓
FinanceBenchLoader (data_loader.py)
    ↓
Sample Dictionaries
    ↓
RAGExperiment + Experiment Module (e.g., rag_single_vector.py)
    ├─→ PDF Text Loading
    ├─→ Chunking (LangChain)
    ├─→ Vector Store Creation (FAISS)
    ├─→ Retrieval
    ├─→ Answer Generation (HF models)
    └─→ Evaluation (evaluator.py)
    ↓
Results JSON
```

---

## 🔧 Configuration

### Experiment Configuration

The `RAGExperiment` class accepts a configuration object with the following key parameters:

```python
class ExperimentConfig:
    # Model settings
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking parameters
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Retrieval settings
    top_k: int = 5
    
    # Evaluation toggles
    use_bertscore: bool = True
    use_llm_judge: bool = False
    
    # Generation settings
    max_new_tokens: int = 200
    temperature: float = 0.7
```

---

## 📈 Evaluation Metrics

### Retrieval Metrics

- **Recall@K**: Proportion of gold evidence found in top-K retrieved chunks
- **Precision@K**: Proportion of retrieved chunks that contain gold evidence
- **MRR (Mean Reciprocal Rank)**: Reciprocal of rank of first relevant chunk
- **Coverage**: Fraction of gold evidence text covered by retrieved chunks

### Generation Metrics

- **Exact Match (EM)**: Binary - generated answer exactly matches reference
- **F1 Score**: Token-level overlap between generated and reference answers
- **BLEU**: N-gram precision with brevity penalty
- **ROUGE-L**: Longest common subsequence F1 score
- **BERTScore**: Contextual embedding similarity (requires bert-score package)

---

## 🎓 Common Use Cases

### 1. Baseline Performance Evaluation

Compare different experiment modes on the same dataset:

```python
# Run all modes
results_closed = run_closed_book(experiment, samples)
results_open = run_open_book(experiment, samples)
results_single = run_single_vector(experiment, samples)
results_shared = run_shared_vector(experiment, samples)

# Compare F1 scores
for mode, results in [("closed", results_closed), ("open", results_open), ...]:
    avg_f1 = sum(r['generation_evaluation']['f1'] for r in results) / len(results)
    print(f"{mode}: {avg_f1:.3f}")
```

### 2. Document-Specific Analysis

Evaluate performance on specific financial documents:

```python
# Filter to specific document
apple_df = loader.filter_by_doc("APPLE_10K_2022")
apple_samples = apple_df.to_dict('records')

# Run experiment
results = run_single_vector(experiment, apple_samples)
```

### 3. Error Analysis

Identify failure modes:

```python
# Find samples with low F1 scores
poor_results = [r for r in results if r['generation_evaluation']['f1'] < 0.5]

for result in poor_results:
    print(f"Q: {result['question']}")
    print(f"Expected: {result['reference_answer']}")
    print(f"Got: {result['generated_answer']}")
    print(f"Retrieval Recall@5: {result['retrieval_evaluation']['recall@5']}")
    print("---")
```

---

## ⚠️ Important Notes

### PDF Text Requirements

- **`single_vector` and `shared_vector` modes require PDF text** extraction
- If `doc_link` doesn't provide valid PDF text, samples are skipped
- Check for `skipped: true` in results when analyzing performance
- **Do not use the `evidence` field as fallback** - this violates realistic retrieval scenarios

### Memory Considerations

- Large datasets may require substantial RAM for FAISS indices
- Use `faiss-gpu` for faster indexing with CUDA support
- Consider batch processing for very large document collections

### Model Inference

- Default models (Llama 3.2 3B, Qwen 2.5 7B) require ~6-14GB GPU memory
- Use 8-bit quantization to reduce memory footprint
- CPU inference is supported but much slower

---

## 📚 Dataset Information

### FinanceBench Dataset

- **Source**: PatronusAI/financebench on HuggingFace
- **Size**: ~150 question-answer pairs
- **Document Types**: 10-K filings, earnings transcripts, financial reports
- **Companies**: Major public companies (Apple, Microsoft, Google, etc.)
- **Task**: Extractive and short-form question answering

### Sample Structure

```python
{
    "question": "What was Apple's total revenue in fiscal 2022?",
    "answer": "$394.3 billion",
    "evidence": "Total net sales were $394,328 million...",
    "doc_name": "APPLE_10K_2022",
    "doc_link": "https://www.sec.gov/Archives/edgar/data/..."
}
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional experiment modes (hybrid retrieval, reranking, etc.)
- More evaluation metrics (domain-specific financial metrics)
- Improved PDF extraction handling
- Support for additional document formats
- Visualization tools for result analysis

---

## 🙏 Acknowledgments

- FinanceBench dataset by PatronusAI
- HuggingFace Datasets and Transformers
- LangChain for RAG utilities
- FAISS for efficient vector search

---

## 📧 Contact

amine.kobeissi[at]umontreal.ca
