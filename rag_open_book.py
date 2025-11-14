from typing import List, Dict, Any
import logging


def run_open_book(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run open-book experiment (use gold evidence directly).
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING OPEN-BOOK EXPERIMENT (Gold Evidence)")
    logger.info("=" * 80)

    results = []

    for i, sample in enumerate(data):
        logger.info(f"\n--- Sample {i+1}/{len(data)} ---")

        question = sample['question']
        reference_answer = sample['answer']
        gold_evidence = sample.get('evidence', '')
        # Normalize gold evidence to string
        gold_parts = experiment._normalize_evidence(gold_evidence)
        context = "\n\n".join(gold_parts)

        logger.info(f"Using gold evidence as context (length: {len(context)} chars)")

        # Generate answer with gold evidence
        generated_answer = experiment._generate_answer(question, context)

        # Evaluate generation (no retrieval evaluation since we use gold evidence)
        generation_eval = experiment.evaluator.evaluate_generation(
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
            'experiment_type': experiment.OPEN_BOOK
        }

        results.append(result)
        logger.info(f"Completed sample {i+1}")

    return results
