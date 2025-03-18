import os
import json
import argparse
from typing import Dict, Any, List, Optional
from collections import defaultdict

def process_result_file(filename: str, compensated: bool = False) -> Dict[str, Any]:
    """
    Process a single result file and return statistics.

    Args:
        filename: Path to the result file
        compensated: Whether to use compensated accuracy calculation

    Returns:
        Dictionary with statistics for the file
    """
    try:
        with open(filename, encoding='utf-8') as f:
            # Load the file as JSONL directly
            pred_data = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return {}

    # Initialize counters
    stats = {
        'easy': 0, 'hard': 0, 'short': 0, 'medium': 0, 'long': 0,
        'easy_acc': 0, 'hard_acc': 0, 'short_acc': 0, 'medium_acc': 0, 'long_acc': 0,
        'error_count': 0
    }

    # Process each prediction
    for pred in pred_data:
        if 'error' in pred:
            stats['error_count'] += 1
            continue

        acc = int(pred['judge'])
        if compensated and pred.get("pred") is None:
            acc = 0.25

        # Count by difficulty
        if pred["difficulty"] == "easy":
            stats['easy'] += 1
            stats['easy_acc'] += acc
        else:
            stats['hard'] += 1
            stats['hard_acc'] += acc

        # Count by length
        if pred['length'] == "short":
            stats['short'] += 1
            stats['short_acc'] += acc
        elif pred['length'] == "medium":
            stats['medium'] += 1
            stats['medium_acc'] += acc
        else:
            stats['long'] += 1
            stats['long_acc'] += acc

    # Calculate totals
    stats['num_successful_queries'] = stats['easy'] + stats['hard']
    stats['total_queries'] = stats['num_successful_queries'] + stats['error_count']

    return stats

def calculate_percentages(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate percentage accuracies from raw statistics.

    Args:
        stats: Dictionary with raw statistics

    Returns:
        Dictionary with percentage accuracies
    """
    results = {}

    # Overall accuracy
    if stats['num_successful_queries'] != 0:
        results['overall'] = round(100 * (stats['easy_acc'] + stats['hard_acc']) / stats['num_successful_queries'], 1)
    else:
        results['overall'] = 'nan'

    # Accuracy by difficulty
    results['easy'] = round(100 * stats['easy_acc'] / stats['easy'], 1) if stats['easy'] != 0 else 'none'
    results['hard'] = round(100 * stats['hard_acc'] / stats['hard'], 1) if stats['hard'] != 0 else 'none'

    # Accuracy by length
    results['short'] = round(100 * stats['short_acc'] / stats['short'], 1) if stats['short'] != 0 else 'none'
    results['medium'] = round(100 * stats['medium_acc'] / stats['medium'], 1) if stats['medium'] != 0 else 'none'
    results['long'] = round(100 * stats['long_acc'] / stats['long'], 1) if stats['long'] != 0 else 'none'

    # Error count
    results['error_count'] = stats['error_count']

    return results

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a jsonl file into a list of dictionaries."""
    results = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {file_path}")
    return results

def unify_results(
    results_dir: str = "results", 
    output_file: str = "unified_results.json",
    agent_names: Optional[List[str]] = None
) -> None:
    """
    Unify result files from different agents into one JSON file.
    
    Args:
        results_dir: Directory containing the result files
        output_file: Path to the output JSON file
        agent_names: List of agent names to include, or None for all available
    """
    # If no agent names are provided, discover all JSONL files in the results directory
    if agent_names is None:
        agent_files = [f for f in os.listdir(results_dir) 
                      if f.endswith('.jsonl') and os.path.isfile(os.path.join(results_dir, f))]
        agent_names = [os.path.splitext(f)[0] for f in agent_files]
    else:
        # Ensure all agent files exist
        for agent in agent_names:
            if not os.path.exists(os.path.join(results_dir, f"{agent}.jsonl")):
                print(f"Warning: {agent}.jsonl not found in {results_dir}")
    
    print(f"Processing agent results: {', '.join(agent_names)}")
    
    # Create a dictionary to store all results, grouped by _id
    all_results = defaultdict(dict)
    
    # Metadata fields (shared across all results for a given item)
    metadata_fields = [
        '_id', 'domain', 'sub_domain', 'difficulty', 'length', 
        'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D',
        'context', 'answer'
    ]
    
    # Agent-specific fields
    agent_fields = ['pred', 'judge', 'token_usage']
    
    # Load results for each agent
    for agent in agent_names:
        file_path = os.path.join(results_dir, f"{agent}.jsonl")
        agent_results = load_jsonl(file_path)
        
        # Process each result
        for result in agent_results:
            if '_id' in result:
                item_id = result['_id']
                
                # Initialize item entry if not already done
                if 'metadata' not in all_results[item_id]:
                    all_results[item_id]['metadata'] = {}
                    all_results[item_id]['agent_results'] = {}
                
                # Copy metadata fields (only once per item)
                for field in metadata_fields:
                    if field in result and field not in all_results[item_id]['metadata']:
                        all_results[item_id]['metadata'][field] = result[field]
                
                # Initialize agent result if not already done
                if agent not in all_results[item_id]['agent_results']:
                    all_results[item_id]['agent_results'][agent] = {
                        'error': ''  # Default to empty error string
                    }
                
                # Copy agent-specific fields
                for field in agent_fields:
                    if field in result:
                        all_results[item_id]['agent_results'][agent][field] = result[field]
                
                # Handle error case
                if 'error' in result:
                    all_results[item_id]['agent_results'][agent]['error'] = result['error']
            else:
                print(f"Warning: Skipping result without _id in {agent}.jsonl")
    
    # Convert to final format (list of dictionaries)
    unified_results = []
    
    # Count errors
    error_counts = defaultdict(int)
    
    for item_id, data in all_results.items():
        # Create a unified result entry
        unified_entry = data['metadata']
        
        # Add agent results
        for agent, agent_data in data['agent_results'].items():
            unified_entry[agent] = agent_data
            if agent_data.get('error'):
                error_counts[agent] += 1
        
        unified_results.append(unified_entry)
    
    # Save the unified results to a JSON file in the results_dir
    output_path = os.path.join(results_dir, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "agents": agent_names,
                "total_items": len(unified_results),
                "error_counts": error_counts
            },
            "results": unified_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Unified results saved to {output_path}")
    print(f"Total items: {len(unified_results)}")
    print("Error counts by agent:")
    for agent, count in error_counts.items():
        print(f"  {agent}: {count} errors")

def main():
    """Process results, generate statistics, and unify results from different agents."""
    parser = argparse.ArgumentParser(description="Process result files, generate statistics, and unify results")
    parser.add_argument(
        "--dir",
        type=str,
        default="results",
        help="Directory containing result files (default: results)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/result.txt",
        help="Output file path for statistics (default: results/result.txt)"
    )
    parser.add_argument(
        "--compensated",
        action="store_true",
        default=False,
        help="Use compensated accuracy calculation (default: False)"
    )
    parser.add_argument(
        "--unified_output", 
        type=str, 
        default="unified_results.json",
        help="Path to the unified output JSON file (default: unified_results.json)"
    )
    parser.add_argument(
        "--agents",
        nargs="*",
        help="List of agent names to include in unified results (default: all available)"
    )
    parser.add_argument(
        "--skip_unify",
        action="store_true",
        default=False,
        help="Skip generating unified results (default: False)"
    )

    args = parser.parse_args()

    # Generate statistics
    if not os.path.isdir(args.dir):
        print(f"Error: Directory {args.dir} not found")
        return

    # Get files from the main results directory, skipping non-JSONL files
    files = []
    for f in os.listdir(args.dir):
        full_path = os.path.join(args.dir, f)
        if f.endswith('.jsonl') and os.path.isfile(full_path) and f != "requests":
            files.append(f)

    if not files:
        print(f"Warning: No .jsonl files found in {args.dir}")
        return

    # Initialize output
    output = ["Model\tOverall\tEasy\tHard\tShort\tMedium\tLong\tErrors"]

    # Process each file
    for file in files:
        filename = os.path.join(args.dir, file)
        stats = process_result_file(filename, args.compensated)
        if not stats:
            continue

        # Use filename (without extension) as the model name
        name = '.'.join(file.split('.')[:-1])
        results = calculate_percentages(stats)

        # Add to output
        output.append(f"{name}\t{results['overall']}\t{results['easy']}\t{results['hard']}\t{results['short']}\t{results['medium']}\t{results['long']}\t{results['error_count']}")

    # Write results to output file
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))

    print(f"Results written to {args.output}")
    
    # Generate unified results
    if not args.skip_unify:
        unify_results(args.dir, args.unified_output, args.agents)

if __name__ == "__main__":
    main()