import os
import json
import argparse
import asyncio
from typing import List, Dict, Any, Type

from tqdm import tqdm
from datasets import load_dataset
from openai_client_plusplus import AsyncOpenAIPlusPlus

# Import agent registry and types
from agents import get_agent, list_agents, AgentProtocol
from specified_id_orderings import shuffled_ids, ids_ordered_by_context_length

# ANSI color codes for colored output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'


async def write_result(output_file: str, result: Dict[str, Any]) -> None:
    """
    Write a result to the output file using proper async file handling.
    
    Args:
        output_file: Path to the output file
        result: The result to write
    """
    async with asyncio.Lock():
        with open(output_file, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            fout.flush()


async def process_single_item(
    item: Dict[str, Any], 
    output_file: str, 
    semaphore: asyncio.Semaphore,
    agent_class: Type[AgentProtocol],
    enable_logging: bool = False,
    save_dir: str = "results"
) -> Dict[str, Any]:
    """
    Process a single data item with the specified agent.
    
    Args:
        item: The data item to process
        output_file: Path to the output file
        semaphore: Semaphore for controlling concurrency
        agent_class: The agent implementation to use
        enable_logging: Whether to enable logging
        save_dir: Directory to save results and logs
        
    Returns:
        The processed result
    """
    # Create the OpenAI client wrapper with agent-specific log file in requests subfolder
    requests_dir = os.path.join(save_dir, "requests")
    os.makedirs(requests_dir, exist_ok=True)
    client = AsyncOpenAIPlusPlus(
        request_id=f"{item['_id']}", 
        logging_enabled=enable_logging, 
        log_file_path=os.path.join(requests_dir, f"{agent_class.name}.jsonl"),
        semaphore=semaphore
    )
    
    try:
        # Extract the necessary data for the problem
        question = item['question']
        context = item['context']
        choices = {
            'choice_A': item['choice_A'],
            'choice_B': item['choice_B'],
            'choice_C': item['choice_C'],
            'choice_D': item['choice_D']
        }
        
        # Run the agent
        prediction = await agent_class.run(question, context, choices, client)
        
        # Create result with token usage
        result = {
            '_id': item['_id'],
            'pred': prediction,
            'judge': prediction == item['answer'],
            'token_usage': client.get_token_usage(),
            'domain': item['domain'],
            'sub_domain': item['sub_domain'],
            'difficulty': item['difficulty'],
            'length': item['length'],
            'question': item['question'],
            'choice_A': item['choice_A'],
            'choice_B': item['choice_B'],
            'choice_C': item['choice_C'],
            'choice_D': item['choice_D'],
            'context': item['context'][:1000],  # Truncate context to 1000 characters
            'agent': agent_class.name  # Add agent name to result
        }
        
    except Exception as e:
        result = {
            '_id': item['_id'],
            'error': str(e),
            'agent': agent_class.name  # Include agent name even for errors
        }
    
    return result


async def process_with_agent(
    data_subset: List[Dict[str, Any]],
    output_file: str,
    agent_class: Type[AgentProtocol],
    max_concurrent: int = 5,
    enable_logging: bool = False,
    save_dir: str = "results"
) -> None:
    """
    Process a subset of data with the agent and write results to the output file in order.
    Displays progress with metrics: completed tasks, written results, correct/incorrect responses,
    API errors, and fraction of correct results.

    Args:
        data_subset: The data subset to process
        output_file: Path to the output file
        agent_class: The agent implementation to use
        max_concurrent: Maximum number of concurrent operations
        enable_logging: Whether to enable logging for OpenAI client wrappers
        save_dir: Directory to save results and logs
    """
    print(f"Processing {len(data_subset)} items with agent '{agent_class.name}' and limit of {max_concurrent} concurrent API calls")
    
    # Create semaphore for controlled concurrency (of API calls)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for each item
    tasks = []
    for i, item in enumerate(data_subset):
        task = asyncio.create_task(
            process_single_item(item, output_file, semaphore, agent_class, enable_logging, save_dir)
        )
        tasks.append((i, task))
    
    # Initialize counters for progress tracking
    counters = {
        'completed': 0,
        'written': 0,
        'correct': 0,
        'incorrect': 0,
        'api_errors': 0,
    }
    
    # Store completed results and track the next index to write
    completed_results = {}
    next_to_write = 0
    
    # Process tasks as they complete, with enhanced progress tracking
    pending_tasks = [task for _, task in tasks]
    indices = {task: idx for idx, task in tasks}
    progress_bar = tqdm(total=len(tasks))
    
    while pending_tasks:
        done, pending_tasks = await asyncio.wait(
            pending_tasks, return_when=asyncio.FIRST_COMPLETED
        )
        
        for future in done:
            result = future.result()
            index = indices[future]
            counters['completed'] += 1
            completed_results[index] = result
            
            # Write all consecutive results starting from next_to_write
            while next_to_write in completed_results:
                result = completed_results[next_to_write]
                # Write result in order here
                await write_result(output_file, result)
                counters['written'] += 1
                if 'error' in result:
                    counters['api_errors'] += 1
                elif result.get('judge', False):
                    counters['correct'] += 1
                else:
                    counters['incorrect'] += 1
                del completed_results[next_to_write]
                next_to_write += 1
            
            # Update progress bar with colored metrics
            progress_bar.update(1)
            total_responses = counters['correct'] + counters['incorrect']
            correct_fraction = counters['correct'] / total_responses if total_responses > 0 else 0.0
            # Format counters with colors
            written_str = f"{CYAN}{counters['written']} written{RESET}"
            api_errors_str = f"{RED}{counters['api_errors']} API errors{RESET}"
            correct_frac_str = f"{MAGENTA}{correct_fraction*100:.2f}% correct{RESET}"
            
            progress_bar.set_postfix_str(
                f"{written_str}, {correct_frac_str}, {api_errors_str}"
            )
    
    progress_bar.close()


def prepare_dataset(
    max_entries: int = -1, 
    processing_order: str = "shuffled"
) -> List[Dict[str, Any]]:
    """
    Load and prepare the dataset for processing.
    
    Args:
        max_entries: Maximum number of entries to process, -1 for all
        processing_order: Order to process items - 'shuffled' or 'context_length'
        
    Returns:
        List of dataset items to process
    """
    # Load the dataset
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    
    # Create a mapping from ID to dataset item
    id_to_item = {item["_id"]: item for item in dataset}
    
    # Determine IDs to process based on processing order
    ids_to_process = shuffled_ids
    if processing_order == "context_length":
        ids_to_process = ids_ordered_by_context_length
    
    # Limit by max_entries if specified
    if max_entries > 0:
        ids_to_process = ids_to_process[:max_entries]
    
    # Create the final data list
    data_all = [id_to_item[id] for id in ids_to_process if id in id_to_item]
    
    return data_all


async def async_main(args) -> None:
    """
    Main async function that loads data and processes it.
    
    Args:
        args: Command line arguments
    """
    # Get the agent class to use
    try:
        agent_class = get_agent(args.agent)
    except KeyError as e:
        print(f"{RED}{str(e)}{RESET}")
        return
        
    # Set up directories and file paths
    requests_dir = os.path.join(args.save_dir, "requests")
    os.makedirs(requests_dir, exist_ok=True)
    
    # Use agent-specific output files
    out_file = os.path.join(args.save_dir, f"{agent_class.name}.jsonl")
    log_file = os.path.join(requests_dir, f"{agent_class.name}.jsonl")
    
    # Check if output files already exist
    files_exist = False
    existing_files = []
    
    if os.path.exists(out_file):
        files_exist = True
        existing_files.append(out_file)
    
    if args.logging and os.path.exists(log_file):
        files_exist = True
        existing_files.append(log_file)
    
    if files_exist:
        if args.delete_old:
            print(f"{YELLOW}Deleting existing output files:{RESET}")
            for file in existing_files:
                print(f" - {file}")
                os.remove(file)
        else:
            print(f"{RED}Error: Output files already exist:{RESET}")
            for file in existing_files:
                print(f" - {file}")
            print(f"\n{YELLOW}Use --delete-old to delete existing files and continue.{RESET}")
            return
    
    # Prepare the dataset
    data_all = prepare_dataset(args.max_entries, args.processing_order)
    
    # Process data
    await process_with_agent(
        data_all, 
        out_file, 
        agent_class,
        args.max_concurrent, 
        args.logging,
        args.save_dir
    )


def main() -> None:
    """
    Entry point for the script.
    """
    # List available agents
    available_agents = list_agents()
    agent_names = [a["name"] for a in available_agents]
    agent_descriptions = "\n".join([f"  - {a['name']}: {a['description']}" for a in available_agents])
    
    parser = argparse.ArgumentParser(description="Process data with an agent")
    parser.add_argument(
        "--agent",
        type=str,
        default="oneshot",
        choices=agent_names,
        help=f"Agent to use for processing.\nAvailable agents:\n{agent_descriptions}"
    )
    parser.add_argument(
        "--processing_order", 
        type=str, 
        default="context_length", 
        choices=["shuffled", "context_length"], 
        help="Which order from specified_id_orderings.py to use for processing items (randomly shuffled, increasing context length) (default: context_length)"
    )
    parser.add_argument(
        "--max_entries", 
        type=int, 
        default=50, 
        help="Processes only the first N entries, -1 for all (default: 50)"
    )
    parser.add_argument(
        "--max_concurrent", 
        type=int, 
        default=5, 
        help="Maximum number of concurrent queries (default: 5)"
    )
    parser.add_argument(
        "--logging", 
        action="store_true", 
        default=False,
        help="Enable logging of requests and responses to API (default: False)"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--delete-old", 
        action="store_true", 
        default=False,
        help="Delete existing output files if they exist (default: False)"
    )
    
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()