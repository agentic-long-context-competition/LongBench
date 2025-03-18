#!/usr/bin/env python3
import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Any, Optional

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
    parser = argparse.ArgumentParser(description="Unify result files from different agents")
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="results",
        help="Directory containing the result files (default: results)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="unified_results.json",
        help="Path to the output JSON file (default: unified_results.json)"
    )
    parser.add_argument(
        "--agents",
        nargs="*",
        help="List of agent names to include (default: all available)"
    )
    
    args = parser.parse_args()
    unify_results(args.results_dir, args.output, args.agents)

if __name__ == "__main__":
    main()