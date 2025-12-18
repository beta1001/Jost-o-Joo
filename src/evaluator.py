# -*- coding: utf-8 -*-
"""
Search Evaluator Module for Jost-o-Joo Search Engine

This module evaluates search engine performance using standard IR metrics.
It compares search results against ground truth relevance judgments to
calculate precision, recall, F1-score, and other evaluation metrics.

Key Features:
- Precision@k, Recall@k calculation
- F1-Score, Average Precision (AP), Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (nDCG)
- Ground truth management and validation
- Comprehensive evaluation reports

Created on Sat Dec 13 19:59:33 2025

@author: Rihem Touzi
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

class SearchEvaluator:
    """
    Evaluates search engine performance using ground truth relevance judgments.
    
    The evaluator compares search results against known relevant documents
    to calculate various information retrieval metrics. It supports binary
    relevance judgments (0 = not relevant, 1 = relevant).
    
    Attributes:
        engine (VectorSpaceSearchEngine): Search engine to evaluate
        test_queries_file (Path): File containing test queries
        ground_truth_file (Path): CSV file with relevance judgments
        ground_truth (defaultdict): query -> {doc_id: relevance}
        test_queries (list): Loaded test queries
    """
    def __init__(self, search_engine,
             test_queries_file=BASE_DIR / "tests" / "test_queries.txt",
             ground_truth_file=BASE_DIR / "ground_truth.csv"):
        self.engine = search_engine
        self.test_queries_file = Path(test_queries_file)
        self.ground_truth_file = Path(ground_truth_file)
        self.ground_truth = defaultdict(dict)  # query -> {doc_id: relevance}
        self.test_queries = []
        
    def load_test_queries(self):
        """
       Load test queries from text file.
       
       File format: One query per line
       
       Returns:
           bool: True if loaded successfully, False otherwise
       """
        
        try:
            with open(self.test_queries_file, 'r', encoding='utf-8') as f:
                self.test_queries = [line.strip() for line in f if line.strip()]
            print(f" Loaded {len(self.test_queries)} test queries from {self.test_queries_file}")
            return True
        except FileNotFoundError:
            print(f"❌ Test queries file not found: {self.test_queries_file}")
            return False
    
    def load_ground_truth(self):
        """
        Load ground truth relevance judgments from CSV file.
        
        CSV format:
            query, doc_id, relevance
        
        Relevance values:
            - 0: Not relevant
            - 1: Relevant
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not self.ground_truth_file.exists():
            print(f"⚠️  Ground truth file not found: {self.ground_truth_file}")
            print("   Creating sample ground truth based on search results...")
            self._create_sample_ground_truth()
            return
        
        try:
            with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    query = row['query']
                    doc_id = row['doc_id']
                    # Parse as integer (0 or 1)
                    try:
                        relevance = int(row['relevance'])
                        if relevance not in [0, 1]:
                            print(f"Warning: Relevance value '{relevance}' is not 0 or 1 for {doc_id}")
                            relevance = 0
                    except ValueError:
                        print(f"Warning: Invalid relevance value '{row['relevance']}' for {doc_id}, setting to 0")
                        relevance = 0
                    
                    if query not in self.ground_truth:
                        self.ground_truth[query] = {}
                    
                    self.ground_truth[query][doc_id] = relevance
            
            print(f" Loaded ground truth for {len(self.ground_truth)} queries")
            return True
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            return False
    
    def _create_sample_ground_truth(self):
        """
       Create sample ground truth based on search results.
       
       Used when no ground truth file exists. Marks top 3 results
       as relevant for the first 5 queries.
       """
        if not self.test_queries:
            self.load_test_queries()
        
        print("🔍 Creating sample ground truth by analyzing search results...")
        
        for query in self.test_queries[:5]:  # Use first 5 queries
            print(f"  Analyzing query: '{query}'")
            
            # Get search results
            results, _, _ = self.engine.search(query, top_k=10)
            
            # Mark top 3 results as relevant (relevance=1)
            relevant_docs = []
            for i, result in enumerate(results[:3]):
                doc_id = result['doc_id']
                self.ground_truth[query][doc_id] = 1
                relevant_docs.append(doc_id)
            
            # Mark next 3 results as partially relevant (relevance=0.5)
            for i, result in enumerate(results[3:6]):
                if i < len(results[3:6]):
                    doc_id = result['doc_id']
                    self.ground_truth[query][doc_id] = 0.5
            
            print(f"    Marked as relevant: {', '.join(relevant_docs)}")
        
        # Save to CSV
        self.save_ground_truth()
    
    def save_ground_truth(self):
        """
        Save ground truth to CSV file.
        
        Creates/overwrites the ground truth CSV file with current judgments.
        """
        rows = []
        for query, doc_dict in self.ground_truth.items():
            for doc_id, relevance in doc_dict.items():
                rows.append({
                    'query': query,
                    'doc_id': doc_id,
                    'relevance': relevance
                })
        
        with open(self.ground_truth_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['query', 'doc_id', 'relevance']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f" Ground truth saved to: {self.ground_truth_file.absolute()}")
    
    def evaluate_query(self, query, k=10):
        """
        Evaluate a single query using standard IR metrics.
        
        Metrics calculated:
            - Precision@k: Relevant documents in top k results / k
            - Recall@k: Relevant documents in top k / total relevant
            - F1-Score: Harmonic mean of precision and recall
            - Average Precision (AP): Precision at each relevant document
            - nDCG: Normalized Discounted Cumulative Gain
        
        Args:
            query (str): Query to evaluate
            k (int): Number of top results to consider
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        # Get search results
        results, search_time, _ = self.engine.search(query, top_k=k)
        
        if not results:
            return {
                'query': query,
                'retrieved': 0,
                'relevant': 0,
                'retrieved_relevant': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'average_precision': 0,
                'ndcg': 0,
                'search_time': search_time
            }
        
        # Extract document IDs from results
        retrieved_docs = [result['doc_id'] for result in results]
        
        # Get relevant documents from ground truth (binary: 0 or 1)
        relevant_docs_dict = self.ground_truth.get(query, {})
        
        # For binary metrics, we consider relevance == 1 as relevant
        binary_relevant_docs = {doc_id for doc_id, rel in relevant_docs_dict.items() if rel == 1}
        
        # Calculate binary relevance metrics
        retrieved_relevant = set(retrieved_docs) & binary_relevant_docs
        
        # Precision@k
        precision = len(retrieved_relevant) / len(retrieved_docs) if retrieved_docs else 0
        
        # Recall@k
        recall = len(retrieved_relevant) / len(binary_relevant_docs) if binary_relevant_docs else 0
        
        # F1-Score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Average Precision (AP) - for binary relevance
        ap = self._calculate_average_precision(retrieved_docs, binary_relevant_docs)
        
        # Normalized Discounted Cumulative Gain (nDCG) - for binary, nDCG = DCG/IDCG
        ndcg = self._calculate_ndcg(retrieved_docs, relevant_docs_dict, k)
        
        return {
            'query': query,
            'retrieved': len(retrieved_docs),
            'relevant': len(binary_relevant_docs),
            'retrieved_relevant': len(retrieved_relevant),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'average_precision': ap,
            'ndcg': ndcg,
            'search_time': search_time
        }
    
    def _calculate_average_precision(self, retrieved_docs, relevant_docs):
        """
        Calculate Average Precision (AP).
        
        AP = Σ Precision@i × rel(i) / total_relevant_documents
        
        Args:
            retrieved_docs (list): Retrieved document IDs in rank order
            relevant_docs (set): Set of relevant document IDs
            
        Returns:
            float: Average Precision score
        """
        if not relevant_docs:
            return 0
        
        relevant_count = 0
        precision_sum = 0
        
        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs)
    
    def _calculate_ndcg(self, retrieved_docs, relevant_docs_dict, k):
        """
        Calculate Normalized Discounted Cumulative Gain (nDCG).
        
        nDCG = DCG / IDCG
        DCG = Σ relevance_i / log2(i + 1)
        IDCG = Ideal DCG for top k relevant documents
        
        Args:
            retrieved_docs (list): Retrieved document IDs in rank order
            relevant_docs_dict (dict): doc_id -> relevance score
            k (int): Cutoff for calculation
            
        Returns:
            float: nDCG score between 0 and 1
        """
        if not relevant_docs_dict:
            return 0
        
        # Calculate DCG
        dcg = 0
        for i, doc_id in enumerate(retrieved_docs[:k], 1):
            relevance = relevant_docs_dict.get(doc_id, 0)
            dcg += relevance / np.log2(i + 1)  # log2(i+1) for discount
        
        # Calculate Ideal DCG (IDCG)
        ideal_relevances = sorted(relevant_docs_dict.values(), reverse=True)[:k]
        idcg = 0
        for i, relevance in enumerate(ideal_relevances, 1):
            idcg += relevance / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0
    
    def evaluate_all(self, k_values=[5, 10]):
        """
        Evaluate all test queries.
        
        Args:
            k_values (list): List of k values to evaluate at
            
        Returns:
            tuple: (detailed_results, summary_stats) or (None, None) if failed
        """
        # Load test queries
        if not self.load_test_queries():
            return None, None
        
        # Load ground truth
        self.load_ground_truth()
        
        if not self.test_queries:
            print("❌ No test queries found!")
            return None, None
        
        print(f"\n Evaluating {len(self.test_queries)} queries...")
        
        results = {}
        summary = {
            'total_queries': len(self.test_queries),
            'metrics': {}
        }
        
        # Evaluate each query
        for query in self.test_queries:
            print(f"\n  Query: '{query}'")
            query_results = {}
            
            for k in k_values:
                metrics = self.evaluate_query(query, k)
                query_results[f'k={k}'] = metrics
                
                print(f"    k={k}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                      f"F1={metrics['f1_score']:.3f}, AP={metrics['average_precision']:.3f}")
            
            results[query] = query_results
        
        # Calculate average metrics
        for k in k_values:
            avg_precision = np.mean([results[q][f'k={k}']['precision'] for q in self.test_queries])
            avg_recall = np.mean([results[q][f'k={k}']['recall'] for q in self.test_queries])
            avg_f1 = np.mean([results[q][f'k={k}']['f1_score'] for q in self.test_queries])
            avg_ap = np.mean([results[q][f'k={k}']['average_precision'] for q in self.test_queries])
            avg_ndcg = np.mean([results[q][f'k={k}']['ndcg'] for q in self.test_queries])
            
            summary['metrics'][f'k={k}'] = {
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1': avg_f1,
                'mean_average_precision': avg_ap,
                'avg_ndcg': avg_ndcg
            }
            
            print(f"\n AVERAGE METRICS (k={k}):")
            print(f"  Average Precision: {avg_precision:.4f}")
            print(f"  Average Recall: {avg_recall:.4f}")
            print(f"  Average F1-Score: {avg_f1:.4f}")
            print(f"  Mean Average Precision (MAP): {avg_ap:.4f}")
            print(f"  Average nDCG: {avg_ndcg:.4f}")
        
        # Save evaluation results
        self._save_evaluation_results(results, summary)
        
        return results, summary
    
    def _save_evaluation_results(self, results, summary):
        """
        Save evaluation results to JSON and CSV files.
        
        Creates:
            - evaluation_results.json: Detailed results
            - evaluation_summary.csv: Summary statistics
        """
        # Save detailed results as JSON
        results_file = BASE_DIR  / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_queries': self.test_queries,
                'detailed_results': results,
                'summary': summary
            }, f, indent=2, default=str)
        
        print(f"\n Detailed results saved to: {results_file.absolute()}")
        
        # Save summary as CSV
        summary_file = BASE_DIR / "evaluation_summary.csv"
        with open(summary_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'k=5', 'k=10'])
            
            metrics_to_save = [
                ('avg_precision', 'Average Precision'),
                ('avg_recall', 'Average Recall'),
                ('avg_f1', 'Average F1-Score'),
                ('mean_average_precision', 'Mean Average Precision (MAP)'),
                ('avg_ndcg', 'Average nDCG')
            ]
            
            for metric_key, metric_name in metrics_to_save:
                row = [metric_name]
                for k in [5, 10]:
                    value = summary['metrics'][f'k={k}'][metric_key]
                    row.append(f"{value:.4f}")
                writer.writerow(row)
        
        print(f" Summary saved to: {summary_file.absolute()}")
    
    def create_evaluation_report(self):
        """
        Create a comprehensive evaluation report.
        
        Prints and saves a formatted report with all evaluation metrics.
        """
        report = []
        report.append("=" * 70)
        report.append(" SEARCH ENGINE EVALUATION REPORT")
        report.append("=" * 70)
        
        results, summary = self.evaluate_all()
        
        if not results or not summary:
            report.append("\n❌ Evaluation failed!")
            return
        
        report.append(f"\n EVALUATION SETUP")
        report.append(f"  Test Queries: {summary['total_queries']}")
        report.append(f"  Ground Truth File: {self.ground_truth_file.name}")
        report.append(f"  Test Queries File: {self.test_queries_file.name}")
        
        report.append(f"\n OVERALL PERFORMANCE METRICS")
        
        for k in [5, 10]:
            metrics = summary['metrics'][f'k={k}']
            report.append(f"\n  Results for k = {k}:")
            report.append(f"    {'• Average Precision:':<35} {metrics['avg_precision']:.4f}")
            report.append(f"    {'• Average Recall:':<35} {metrics['avg_recall']:.4f}")
            report.append(f"    {'• Average F1-Score:':<35} {metrics['avg_f1']:.4f}")
            report.append(f"    {'• Mean Average Precision (MAP):':<35} {metrics['mean_average_precision']:.4f}")
            report.append(f"    {'• Average nDCG:':<35} {metrics['avg_ndcg']:.4f}")
        
        report.append(f"\n DETAILED QUERY RESULTS")
        for query in self.test_queries:
            metrics = results[query]['k=10']
            report.append(f"\n  Query: '{query}'")
            report.append(f"    Precision@10: {metrics['precision']:.3f}")
            report.append(f"    Recall@10: {metrics['recall']:.3f}")
            report.append(f"    F1-Score: {metrics['f1_score']:.3f}")
            report.append(f"    Average Precision: {metrics['average_precision']:.3f}")
        
        report.append("\n" + "=" * 70)
        
        # Save report to file
        report_file = BASE_DIR / "evaluation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n Evaluation report saved to: {report_file.absolute()}")
        
        # Print report
        print('\n'.join(report))

def main():
    """Main evaluation function"""
    print("=" * 70)
    print(" SEARCH ENGINE EVALUATOR")
    print("=" * 70)
    
    # Import here to avoid circular imports
    from search_engine import VectorSpaceSearchEngine
    
    # Initialize search engine
    engine = VectorSpaceSearchEngine()
    
    # Initialize evaluator
    evaluator = SearchEvaluator(engine)
    
    # Run evaluation
    evaluator.create_evaluation_report()

if __name__ == "__main__":
    main()