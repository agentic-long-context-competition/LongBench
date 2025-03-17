import os
import json
import argparse
from typing import Dict, Any

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

def main():
    """Process results and generate statistics."""
    parser = argparse.ArgumentParser(description="Process result files and generate statistics")
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
        help="Output file path (default: results/result.txt)"
    )
    parser.add_argument(
        "--compensated",
        action="store_true",
        default=False,
        help="Use compensated accuracy calculation (default: False)"
    )

    args = parser.parse_args()

    # Get list of all result files
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

if __name__ == "__main__":
    main()