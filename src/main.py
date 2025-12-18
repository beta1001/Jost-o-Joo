# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 19:59:42 2025

@author: Rihem Touzi
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 19:59:42 2025

@author: Rihem Touzi
"""
import json
import click
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import print as rprint
import sys
from pathlib import Path
import numpy as np

from evaluator import SearchEvaluator
from search_engine import VectorSpaceSearchEngine
from data_collector import DataCollector
from indexer import SearchIndexer


BASE_DIR = Path(__file__).resolve().parent.parent


console = Console()

@click.group()
def cli():
    """Jost-o-Joo Search Engine - A vector space model search engine"""
    pass

@cli.command()
@click.option('--dir', default=str(BASE_DIR / "data" / "documents"))

def collect(dir):
    """Collect and process documents"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Collecting documents...", total=None)
        
        collector = DataCollector(documents_dir=dir)
        if collector.collect_metadata():
            progress.update(task, completed=100)
            console.print(Panel.fit(
                f"✅ Successfully collected metadata for {len(collector.metadata)} documents",
                title="Data Collection Complete",
                border_style="green"
            ))
        else:
            console.print("[red]❌ Data collection failed![/red]")

@cli.command()
@click.option('--metadata', default=str(BASE_DIR / "data" / "metadata.json"))
@click.option('--index', default=str(BASE_DIR / "index" / "inverted_index.json"))
def index(metadata, index):
    """Build search index from documents"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Building search index...", total=None)
        
        indexer = SearchIndexer(metadata_file=metadata, index_file=index)
        if indexer.build_inverted_index():
            indexer.save_index()
            progress.update(task, completed=100)
            
            stats = indexer.get_index_stats()
            
            table = Table(title="Index Statistics", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")
            
            table.add_row("Total Documents", str(stats['total_documents']))
            table.add_row("Total Unique Terms", str(stats['total_terms']))
            table.add_row("Average Terms per Document", f"{stats['avg_terms_per_doc']:.2f}")
            
            console.print(table)
        else:
            console.print("[red]❌ Index building failed![/red]")

@cli.command()
@click.argument('query')
@click.option('--top', '-t', default=10, help='Number of results to show')
@click.option('--explain', '-e', is_flag=True, help='Show search explanation')
def search(query, top, explain):
    """Search for documents"""
    engine = VectorSpaceSearchEngine()
    
    if explain:
        explanation = engine.explain_search(query)
        console.print(Panel.fit(
            str(explanation),
            title=f"Search Explanation: '{query}'",
            border_style="blue"
        ))
        return
    
    console.print(f"\n🔍 Searching for: [bold cyan]'{query}'[/bold cyan]")
    
    start_time = time.time()
    results, search_time, message = engine.search(query, top_k=top)
    total_time = time.time() - start_time
    
    if not results:
        console.print("[yellow]⚠️  No results found![/yellow]")
        return
    
    # Create results table
    table = Table(title=f"Search Results (Top {top})", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Document", style="green")
    table.add_column("Title", style="white", max_width=50)
    table.add_column("Score", justify="right", style="yellow")
    table.add_column("Words", justify="right")
    
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        table.add_row(
            str(i),
            result['doc_id'],
            metadata.get('title', 'Unknown')[:50],
            f"{result['score']:.4f}",
            str(metadata.get('word_count', 0))
        )
    
    console.print(table)
    
    # Show snippets for top 3 results
    console.print("\n [bold]Top Result Previews:[/bold]")
    for i, result in enumerate(results[:3], 1):
        query_terms = engine.preprocess_query(query)
        snippet = engine.get_document_snippet(result['doc_id'], query_terms)
        console.print(Panel.fit(
            snippet,
            title=f"[{i}] {result['doc_id']}: {result['metadata'].get('title', 'Unknown')[:40]}...",
            border_style="green" if i == 1 else "blue"
        ))
    
    # Performance info
    console.print(Panel.fit(
        f"  Search Time: {search_time:.3f}s\n"
        f"  Total Time: {total_time:.3f}s\n"
        f" {message}",
        title="Performance",
        border_style="yellow"
    ))

@cli.command()
@click.option('--k', default=10, help='Number of results to evaluate')
@click.option('--queries-file',
              default=str(BASE_DIR / "tests" / "test_queries.txt"))

def evaluate(k, queries_file):
    """Evaluate search engine performance"""
    engine = VectorSpaceSearchEngine()
    evaluator = SearchEvaluator(engine, test_queries_file=queries_file)
    
    console.print(f"\n [bold]Running Evaluation[/bold]")
    console.print(f"   Test Queries: {queries_file}")
    console.print(f"   k-value: {k}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Evaluating search engine...", total=None)
        
        # Run evaluation
        evaluator.load_test_queries()
        evaluator.load_ground_truth()
        
        if not evaluator.test_queries:
            console.print("[red]❌ No test queries found![/red]")
            return
        
        # Create evaluation table
        table = Table(title=f"Evaluation Results (k={k})", show_header=True, header_style="bold magenta")
        table.add_column("Query", style="cyan")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1-Score", justify="right")
        table.add_column("Avg Prec", justify="right")
        table.add_column("Time", justify="right")
        
        all_metrics = []
        
        for query in evaluator.test_queries:
            metrics = evaluator.evaluate_query(query, k)
            all_metrics.append(metrics)
            
            table.add_row(
                query[:30],
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1_score']:.3f}",
                f"{metrics['average_precision']:.3f}",
                f"{metrics['search_time']:.3f}s"
            )
        
        progress.update(task, completed=100)
    
    console.print(table)
    
    # Calculate averages
    avg_precision = np.mean([m['precision'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
    avg_ap = np.mean([m['average_precision'] for m in all_metrics])
    
    console.print(Panel.fit(
        f" [bold]Average Metrics:[/bold]\n"
        f"   • Average Precision: {avg_precision:.4f}\n"
        f"   • Average Recall: {avg_recall:.4f}\n"
        f"   • Average F1-Score: {avg_f1:.4f}\n"
        f"   • Mean Average Precision: {avg_ap:.4f}",
        title="Summary",
        border_style="green"
    ))

@cli.command()
def test_queries():
    """Show test queries for evaluation"""
    queries_file = BASE_DIR / "tests" / "test_queries.txt"

    
    if not queries_file.exists():
        console.print(f"[red]❌ Test queries file not found: {queries_file}[/red]")
        return
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    console.print(Panel.fit(
        "\n".join([f"{i+1:2d}. {query}" for i, query in enumerate(queries)]),
        title=f"Test Queries ({len(queries)} total)",
        border_style="blue"
    ))
    
    console.print(f"\n📁 File: {queries_file.absolute()}")

@cli.command()
def stats():
    """Show search engine statistics"""
    engine = VectorSpaceSearchEngine()
    
    # Load metadata for stats
    metadata_file = BASE_DIR / "data" / "metadata.json"

    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        console.print(f"[red]❌ Metadata file not found: {metadata_file}[/red]")
        return
    
    # Calculate statistics
    total_words = sum(doc['word_count'] for doc in metadata.values())
    avg_words = total_words / len(metadata)
    
    table = Table(title="Search Engine Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    table.add_row("Total Documents", str(len(metadata)))
    table.add_row("Total Unique Terms", str(len(engine.inverted_index)))
    table.add_row("Total Words", f"{total_words:,}")
    table.add_row("Average Words per Document", f"{avg_words:,.0f}")
    table.add_row("Index File", "index/inverted_index.json")
    table.add_row("Metadata File", "data/metadata.json")
    
    console.print(table)

if __name__ == "__main__":
    cli()