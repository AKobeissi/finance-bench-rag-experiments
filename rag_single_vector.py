from typing import List, Dict, Any
import logging


def run_single_vector(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run single vector store experiment (separate FAISS index per document).
    This function mirrors the original method but is kept modular.
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING SINGLE VECTOR STORE EXPERIMENT (LangChain + FAISS)")
    logger.info("=" * 80)

    results = []

    # Group by document
    from collections import defaultdict
    doc_groups = defaultdict(list)
    for sample in data:
        doc_name = sample.get('doc_name', 'unknown')
        doc_groups[doc_name].append(sample)

    logger.info(f"Processing {len(doc_groups)} unique documents")

    for doc_name, samples in doc_groups.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Document: {doc_name}")
        logger.info(f"Samples: {len(samples)}")
        logger.info(f"{'='*80}")

        # Determine doc_link and prefer local PDFs (matching doc_name) over URLs.
        doc_link = samples[0].get('doc_link', '')

        pdf_text = None
        pdf_source = 'none'

        # Try local PDF first
        try:
            local_dir = getattr(experiment, 'pdf_local_dir', None)
            if local_dir is not None:
                from pathlib import Path
                pdir = Path(local_dir)
                if pdir.exists() and pdir.is_dir():
                    candidates = [pdir / doc_name, pdir / f"{doc_name}.pdf"]
                    for p in pdir.iterdir():
                        if p.is_file() and p.suffix.lower() == '.pdf' and doc_name.lower() in p.stem.lower():
                            candidates.append(p)
                    for cand in candidates:
                        if cand and cand.exists():
                            try:
                                logger.info(f"Attempting to load local PDF for doc '{doc_name}': {cand}")
                                pdf_text = experiment._load_pdf_text(str(cand))
                                if pdf_text:
                                    pdf_source = 'local'
                                    break
                            except Exception:
                                pdf_text = None
                                continue
        except Exception as e:
            logger.debug(f"Local PDF lookup for {doc_name} failed: {e}")

        # Fall back to doc_link URL
        if not pdf_text and doc_link:
            try:
                logger.info(f"Attempting to load PDF from URL for doc '{doc_name}': {doc_link}")
                pdf_text = experiment._load_pdf_text(doc_link)
                if pdf_text:
                    pdf_source = 'url'
            except Exception:
                pdf_text = None

        if not pdf_text:
            # No PDF text available for this document: mark all its samples as skipped
            logger.warning(f"No PDF text available for document '{doc_name}' (doc_link={doc_link}). Skipping {len(samples)} samples.")
            for sample in samples:
                question = sample.get('question', '')
                reference_answer = sample.get('answer', '')
                result = {
                    'sample_id': len(results),
                    'doc_name': doc_name,
                    'doc_link': doc_link,
                    'question': question,
                    'reference_answer': reference_answer,
                    'gold_evidence': '',
                    'retrieved_chunks': [],
                    'num_retrieved': 0,
                    'context_length': 0,
                    'generated_answer': '',
                    'generation_length': 0,
                    'retrieval_evaluation': {},
                    'generation_evaluation': {},
                    'experiment_type': experiment.SINGLE_VECTOR,
                    'vector_store_type': 'FAISS',
                    'skipped': True,
                    'skipped_reason': 'no_pdf_text',
                    'pdf_source': pdf_source
                }
                results.append(result)
                logger.info(f"Skipped sample (no pdf): {result['sample_id']}")
            # Proceed to next document
            continue

        # Log where the PDF text was loaded from for this document
        logger.info(f"PDF source for document '{doc_name}': {pdf_source}")

        # We have PDF text: chunk using LangChain (via experiment helper)
        chunk_source = 'pdf'
        text_to_chunk = pdf_text
        documents = experiment._chunk_text_langchain(
            text_to_chunk,
            metadata={'doc_name': doc_name, 'source': chunk_source, 'doc_link': doc_link, 'pdf_source': pdf_source}
        )

        # Create FAISS vector store
        vector_store = experiment._create_vector_store_faiss(documents, index_name=doc_name)

        # Process each sample
        for i, sample in enumerate(samples):
            logger.info(f"\n--- Sample {i+1}/{len(samples)} for {doc_name} ---")

            question = sample['question']
            reference_answer = sample['answer']
            gold_evidence = sample.get('evidence', '')
            # Normalize gold evidence to a string (handles lists, bytes, numpy arrays)
            gold_parts = experiment._normalize_evidence(gold_evidence)
            gold_evidence = "\n\n".join(gold_parts)

            # Retrieve relevant chunks using FAISS
            retrieved_chunks = experiment._retrieve_chunks_faiss(question, vector_store)

            # Combine retrieved chunks into context
            context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])

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
                'sample_id': len(results),
                'doc_name': doc_name,
                'doc_link': doc_link,
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
                'experiment_type': experiment.SINGLE_VECTOR,
                'vector_store_type': 'FAISS'
                , 'pdf_source': pdf_source
            }

            results.append(result)
            logger.info(f"Completed sample {len(results)}")

    return results
