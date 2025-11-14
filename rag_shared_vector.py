from typing import List, Dict, Any
import logging


def run_shared_vector(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run shared vector store experiment (single FAISS index for all documents).
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING SHARED VECTOR STORE EXPERIMENT (LangChain + FAISS)")
    logger.info("=" * 80)

    # Collect all unique documents with their doc_link (PDF URL)
    # We'll prefer local PDFs in experiment.pdf_local_dir using the doc_name as filename
    unique_docs: dict[str, str] = {}
    for sample in data:
        doc_name = sample.get('doc_name', 'unknown')
        if doc_name not in unique_docs:
            unique_docs[doc_name] = sample.get('doc_link', '')

    logger.info(f"Collected {len(unique_docs)} unique documents")

    # Chunk all documents using LangChain, preferring local PDFs (matching doc_name)
    # and falling back to the provided doc_link (URL). Keep track of where the
    # PDF text was loaded from for sanity checks.
    all_documents = []
    available_docs = set()
    pdf_source_map: dict = {}
    for doc_name, doc_link in unique_docs.items():
        logger.info(f"\nChecking document for PDF text: {doc_name}")
        pdf_text = None
        used_source = 'none'

        # 1) Try local PDF by doc_name inside configured pdf_local_dir
        try:
            local_dir = getattr(experiment, 'pdf_local_dir', None)
            if local_dir is not None:
                from pathlib import Path
                pdir = Path(local_dir)
                if pdir.exists() and pdir.is_dir():
                    # Try several candidate filenames: exact doc_name, doc_name.pdf
                    candidates = [pdir / doc_name, pdir / f"{doc_name}.pdf"]
                    # Also try case-insensitive / fuzzy stem matches
                    for p in pdir.iterdir():
                        if p.is_file() and p.suffix.lower() == '.pdf' and doc_name.lower() in p.stem.lower():
                            candidates.append(p)

                    for cand in candidates:
                        if cand and cand.exists():
                            try:
                                logger.info(f"Attempting to load local PDF for doc '{doc_name}': {cand}")
                                pdf_text = experiment._load_pdf_text(str(cand))
                                if pdf_text:
                                    used_source = 'local'
                                    break
                            except Exception:
                                pdf_text = None
                                continue
        except Exception as e:
            logger.debug(f"Local PDF lookup for {doc_name} failed: {e}")

        # 2) Fall back to doc_link URL if local not found
        if not pdf_text and doc_link:
            try:
                logger.info(f"Attempting to load PDF from URL for doc '{doc_name}': {doc_link}")
                pdf_text = experiment._load_pdf_text(doc_link)
                if pdf_text:
                    used_source = 'url'
            except Exception:
                pdf_text = None

        pdf_source_map[doc_name] = used_source

        if not pdf_text:
            logger.warning(f"No PDF text available for document '{doc_name}' (doc_link={doc_link}). This doc will be excluded from the shared vector store.")
            continue

        # We have PDF text: chunk and include in the shared store
        logger.info(f"Chunking (from {used_source}) document: {doc_name}")
        docs = experiment._chunk_text_langchain(
            pdf_text,
            metadata={'doc_name': doc_name, 'source': 'pdf', 'doc_link': doc_link, 'pdf_source': used_source}
        )
        all_documents.extend(docs)
        available_docs.add(doc_name)

    logger.info(f"\nTotal documents across all chunks: {len(all_documents)}")

    # Create shared FAISS vector store
    logger.info("\nCreating shared FAISS vector store...")
    if not all_documents:
        logger.warning("No documents with PDF text were available for the shared vector store. All samples will be skipped.")
        # Return skipped results for all samples
        results = []
        for i, sample in enumerate(data):
            result = {
                'sample_id': i,
                'doc_name': sample.get('doc_name', 'unknown'),
                'doc_link': sample.get('doc_link', ''),
                'question': sample.get('question', ''),
                'reference_answer': sample.get('answer', ''),
                'gold_evidence': '',
                'retrieved_chunks': [],
                'num_retrieved': 0,
                'context_length': 0,
                'generated_answer': '',
                'generation_length': 0,
                'retrieval_evaluation': {},
                'generation_evaluation': {},
                'experiment_type': experiment.SHARED_VECTOR,
                'vector_store_type': 'FAISS',
                'skipped': True,
                'skipped_reason': 'no_pdf_text_available'
            }
            results.append(result)
        return results

    shared_vector_store = experiment._create_vector_store_faiss(all_documents, index_name="shared_store")

    # Sanity summary: how many documents loaded from local vs url
    try:
        from collections import Counter
        cnt = Counter(pdf_source_map.values())
        logger.info(f"PDF source summary for shared store: {dict(cnt)}")
    except Exception:
        pass

    # Process each sample
    results = []
    for i, sample in enumerate(data):
        logger.info(f"\n--- Sample {i+1}/{len(data)} ---")

        question = sample.get('question', '')
        reference_answer = sample.get('answer', '')
        gold_evidence = sample.get('evidence', '')
        # Normalize gold evidence to string
        gold_parts = experiment._normalize_evidence(gold_evidence)
        gold_evidence = "\n\n".join(gold_parts)
        doc_name = sample.get('doc_name', 'unknown')

        # If this sample's doc was not in the available_docs set, skip it (no PDF text)
        if doc_name not in available_docs:
            logger.info(f"Skipping sample for doc '{doc_name}' because no PDF text was available.")
            result = {
                'sample_id': i,
                'doc_name': doc_name,
                'doc_link': sample.get('doc_link', ''),
                'question': question,
                'reference_answer': reference_answer,
                'gold_evidence': '',
                'retrieved_chunks': [],
                'retrieved_from_docs': [],
                'num_retrieved': 0,
                'context_length': 0,
                'generated_answer': '',
                'generation_length': 0,
                'retrieval_evaluation': {},
                'generation_evaluation': {},
                'experiment_type': experiment.SHARED_VECTOR,
                'vector_store_type': 'FAISS',
                'skipped': True,
                'skipped_reason': 'no_pdf_text'
            }
            results.append(result)
            logger.info(f"Completed sample {i+1} (skipped)")
            continue

        # Include pdf_source indicator for this sample (sanity check)
        pdf_source = pdf_source_map.get(doc_name, 'none')

        # Retrieve relevant chunks using FAISS
        retrieved_chunks = experiment._retrieve_chunks_faiss(question, shared_vector_store)

        # Combine retrieved chunks into context
        context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])

        # Log which documents chunks came from (include doc_link if present)
        source_docs = [
            {
                'doc_name': chunk['metadata'].get('doc_name', 'unknown'),
                'doc_link': chunk['metadata'].get('doc_link', '')
            }
            for chunk in retrieved_chunks
        ]
        logger.info(f"Retrieved chunks from documents: {source_docs}")

        # Evaluate retrieval
        retrieved_texts = [chunk['text'] for chunk in retrieved_chunks]
        retrieval_eval = experiment.evaluator.compute_retrieval_metrics(
            retrieved_texts,
            gold_evidence
        )

        # Generate answer with context
        generated_answer = experiment._generate_answer(question, context)

        # Evaluate generation
        generation_eval = experiment.evaluator.evaluate_generation(
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
            'experiment_type': experiment.SHARED_VECTOR,
            'vector_store_type': 'FAISS'
            , 'pdf_source': pdf_source
        }

        results.append(result)
        logger.info(f"Completed sample {i+1}")

    return results
