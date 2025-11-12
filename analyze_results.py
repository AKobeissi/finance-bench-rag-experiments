"""
Results Analysis Utility
Load and analyze results from multiple experiments
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyze and compare experiment results"""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize results analyzer
        
        Args:
            output_dir: Directory containing result JSON files
        """
        self.output_dir = output_dir
        self.results = []
    
    def load_all_results(self) -> List[Dict[str, Any]]:
        """Load all result files from output directory"""
        result_files = list(Path(self.output_dir).glob("*.json"))
        
        logger.info(f"Found {len(result_files)} result files")
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.results.append({
                        'file': file_path.name,
                        'data': data
                    })
                logger.info(f"Loaded: {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        return self.results
    
    def load_specific_results(self, filenames: List[str]) -> List[Dict[str, Any]]:
        """Load specific result files"""
        for filename in filenames:
            file_path = Path(self.output_dir) / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        self.results.append({
                            'file': filename,
                            'data': data
                        })
                    logger.info(f"Loaded: {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {str(e)}")
            else:
                logger.warning(f"File not found: {filename}")
        
        return self.results
    
    def extract_metrics(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from a result file"""
        metadata = result_data.get('metadata', {})
        results = result_data.get('results', [])
        
        if not results:
            return {}
        
        metrics = {
            'experiment_type': metadata.get('experiment_type', 'unknown'),
            'num_samples': len(results),
            'chunk_size': metadata.get('chunk_size'),
            'top_k': metadata.get('top_k'),
        }
        
        # Extract generation metrics
        bleu_4_scores = []
        rouge_l_scores = []
        bertscore_f1_scores = []
        gen_lengths = []
        
        # Extract retrieval metrics (if applicable)
        exact_matches = []
        max_overlaps = []
        context_lengths = []
        
        for r in results:
            # Generation metrics
            if 'generation_evaluation' in r:
                gen_eval = r['generation_evaluation']
                if 'bleu' in gen_eval and 'bleu_4' in gen_eval['bleu']:
                    bleu_4_scores.append(gen_eval['bleu']['bleu_4'])
                if 'rouge' in gen_eval and 'rouge_l_f1' in gen_eval['rouge']:
                    rouge_l_scores.append(gen_eval['rouge']['rouge_l_f1'])
                if 'bertscore' in gen_eval and gen_eval['bertscore'].get('f1'):
                    bertscore_f1_scores.append(gen_eval['bertscore']['f1'])
            
            if 'generation_length' in r:
                gen_lengths.append(r['generation_length'])
            
            # Retrieval metrics
            if 'retrieval_evaluation' in r:
                ret_eval = r['retrieval_evaluation']
                exact_matches.append(ret_eval.get('exact_match', 0))
                max_overlaps.append(ret_eval.get('max_token_overlap', 0))
            
            if 'context_length' in r:
                context_lengths.append(r['context_length'])
        
        # Aggregate metrics
        if bleu_4_scores:
            metrics['bleu_4_mean'] = np.mean(bleu_4_scores)
            metrics['bleu_4_std'] = np.std(bleu_4_scores)
        
        if rouge_l_scores:
            metrics['rouge_l_f1_mean'] = np.mean(rouge_l_scores)
            metrics['rouge_l_f1_std'] = np.std(rouge_l_scores)
        
        if bertscore_f1_scores:
            metrics['bertscore_f1_mean'] = np.mean(bertscore_f1_scores)
            metrics['bertscore_f1_std'] = np.std(bertscore_f1_scores)
        
        if gen_lengths:
            metrics['gen_length_mean'] = np.mean(gen_lengths)
            metrics['gen_length_std'] = np.std(gen_lengths)
        
        if exact_matches:
            metrics['exact_match_rate'] = np.mean(exact_matches)
        
        if max_overlaps:
            metrics['max_overlap_mean'] = np.mean(max_overlaps)
        
        if context_lengths:
            metrics['context_length_mean'] = np.mean(context_lengths)
        
        return metrics
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create a comparison table of all loaded results"""
        if not self.results:
            logger.warning("No results loaded")
            return pd.DataFrame()
        
        all_metrics = []
        for result in self.results:
            metrics = self.extract_metrics(result['data'])
            metrics['file'] = result['file']
            all_metrics.append(metrics)
        
        df = pd.DataFrame(all_metrics)

        # Reorder columns for better readability (only keep priority cols that exist)
        priority_cols = [c for c in ['file', 'experiment_type', 'num_samples', 'chunk_size', 'top_k'] if c in df.columns]
        other_cols = [col for col in df.columns if col not in priority_cols]
        df = df[priority_cols + other_cols]

        return df
    
    def print_summary(self):
        """Print a summary of all loaded results"""
        if not self.results:
            logger.warning("No results loaded")
            return
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT RESULTS SUMMARY")
        logger.info("="*80)
        
        for result in self.results:
            data = result['data']
            metadata = data.get('metadata', {})
            results_list = data.get('results', [])
            
            logger.info(f"\nFile: {result['file']}")
            logger.info(f"  Experiment Type: {metadata.get('experiment_type', 'unknown')}")
            logger.info(f"  Samples: {len(results_list)}")
            logger.info(f"  Timestamp: {metadata.get('timestamp', 'unknown')}")
            
            metrics = self.extract_metrics(data)
            
            if 'bleu_4_mean' in metrics:
                logger.info(f"  BLEU-4: {metrics['bleu_4_mean']:.4f} (±{metrics.get('bleu_4_std', 0):.4f})")
            
            if 'rouge_l_f1_mean' in metrics:
                logger.info(f"  ROUGE-L F1: {metrics['rouge_l_f1_mean']:.4f} (±{metrics.get('rouge_l_f1_std', 0):.4f})")
            
            if 'bertscore_f1_mean' in metrics:
                logger.info(f"  BERTScore F1: {metrics['bertscore_f1_mean']:.4f} (±{metrics.get('bertscore_f1_std', 0):.4f})")
            
            if 'exact_match_rate' in metrics:
                logger.info(f"  Exact Match Rate: {metrics['exact_match_rate']:.2%}")
            
            if 'max_overlap_mean' in metrics:
                logger.info(f"  Max Token Overlap: {metrics['max_overlap_mean']:.4f}")
        
        logger.info("\n" + "="*80)
    
    def compare_experiments(self, metric: str = 'bleu_4_mean'):
        """
        Compare experiments by a specific metric
        
        Args:
            metric: Metric to compare (e.g., 'bleu_4_mean', 'rouge_l_f1_mean')
        """
        if not self.results:
            logger.warning("No results loaded")
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPARISON BY {metric.upper()}")
        logger.info(f"{'='*80}\n")
        
        comparisons = []
        for result in self.results:
            metrics = self.extract_metrics(result['data'])
            if metric in metrics:
                comparisons.append({
                    'file': result['file'],
                    'experiment_type': metrics.get('experiment_type', 'unknown'),
                    'value': metrics[metric]
                })
        
        if not comparisons:
            logger.warning(f"No results found with metric: {metric}")
            return
        
        # Sort by metric value
        comparisons.sort(key=lambda x: x['value'], reverse=True)
        
        logger.info(f"{'Rank':<6} {'Experiment Type':<20} {'Value':<12} {'File'}")
        logger.info("-" * 80)
        
        for i, comp in enumerate(comparisons, 1):
            logger.info(f"{i:<6} {comp['experiment_type']:<20} {comp['value']:<12.4f} {comp['file']}")
        
        logger.info("\n" + "="*80)
    
    def export_comparison(self, output_file: str = "experiment_comparison.csv"):
        """Export comparison table to CSV"""
        df = self.create_comparison_table()
        if not df.empty:
            df.to_csv(output_file, index=False)
            logger.info(f"\nComparison table exported to: {output_file}")
        else:
            logger.warning("No data to export")
    
    def get_best_experiment(self, metric: str = 'bleu_4_mean') -> Dict[str, Any]:
        """
        Get the best performing experiment by a metric
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Dictionary with best experiment info
        """
        if not self.results:
            logger.warning("No results loaded")
            return {}
        
        best = None
        best_value = -float('inf')
        
        for result in self.results:
            metrics = self.extract_metrics(result['data'])
            if metric in metrics and metrics[metric] > best_value:
                best_value = metrics[metric]
                best = {
                    'file': result['file'],
                    'metrics': metrics,
                    'value': best_value
                }
        
        if best:
            logger.info(f"\nBest experiment by {metric}:")
            logger.info(f"  File: {best['file']}")
            logger.info(f"  Value: {best['value']:.4f}")
            logger.info(f"  Type: {best['metrics'].get('experiment_type', 'unknown')}")
        
        return best


def main():
    """Example usage"""
    
    # Create analyzer
    analyzer = ResultsAnalyzer()
    
    # Load all results
    analyzer.load_all_results()
    
    # Print summary
    analyzer.print_summary()
    
    # Create comparison table
    comparison_df = analyzer.create_comparison_table()
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(comparison_df.to_string())
    
    # Compare by specific metric
    analyzer.compare_experiments('bleu_4_mean')
    analyzer.compare_experiments('rouge_l_f1_mean')
    
    # Get best experiment
    best = analyzer.get_best_experiment('bleu_4_mean')
    
    # Export to CSV
    analyzer.export_comparison()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()