from typing import Dict, List
import asyncio
from openai import AsyncOpenAI
from .oneshot import extract_answer

def split_into_chunks(context: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split the context into chunks of specified size with overlap."""
    words = context.split()
    total_words = len(words)
    step_size = chunk_size - overlap
    chunks = []
    
    for i in range(0, total_words, step_size):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
    
    return chunks

async def extract_quotes_from_chunk(question: str, chunk: str, total_words: int, client: AsyncOpenAI) -> str:
    """Extract relevant quotes from a single chunk with confidence estimates."""
    prompt = f"""You are given a part of a very long text with {total_words} words. This chunk is approximately 1000 words, with a 200-word overlap with adjacent chunks. Multiple agents are processing different chunks in parallel. Your task is to extract relevant quotes from this chunk that help answer the following question: {question}.

If you find relevant quotes, provide them with a brief explanation of why they are relevant and a confidence level (from 0 to 1). If there is no relevant information, state so with a confidence level.

Output format:
- If relevant:
  Quotes:
  1. "quote1" - explanation1 (confidence: 0.8)
  2. "quote2" - explanation2 (confidence: 0.9)
- Else:
  No relevant information (confidence: 0.7)

Chunk:
{chunk}

Question: {question}"""
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024
    )
    return response.choices[0].message.content

async def compress_quotes(quotes_chunk: str, question: str, client: AsyncOpenAI) -> str:
    """Compress a chunk of quotes into a concise summary."""
    prompt = f"""You are given a collection of quotes extracted from a long text, relevant to answering the following question: {question}. Concisely summarize the information in these quotes, focusing on key points and evidence that help determine the correct answer.

Quotes:
{quotes_chunk}

Question: {question}

Provide a concise summary:"""
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024
    )
    return response.choices[0].message.content

async def process_long_context(question: str, context: str, choices: Dict[str, str], client: AsyncOpenAI) -> str:
    """Process long context with chunking, quote extraction, compression, and final answer extraction."""
    # Step 1: Split context into chunks
    total_words = len(context.split())
    chunks = split_into_chunks(context, chunk_size=1000, overlap=200)
    
    # Step 2: Process chunks in parallel
    tasks = [extract_quotes_from_chunk(question, chunk, total_words, client) for chunk in chunks]
    outputs = await asyncio.gather(*tasks)
    
    # Step 3: Collect relevant quotes
    relevant_quotes = []
    for output in outputs:
        if "Quotes:" in output:
            quotes_part = output.split("Quotes:")[1].strip()
            relevant_quotes.append(quotes_part)
    
    # Step 4: Concatenate relevant quotes
    concatenated_quotes = "\n\n".join(relevant_quotes)
    concat_words = len(concatenated_quotes.split())
    
    # Step 5: Compress if necessary
    if concat_words > 8000:
        quote_chunks = split_into_chunks(concatenated_quotes, chunk_size=4000, overlap=0)
        compress_tasks = [compress_quotes(quote_chunk, question, client) for quote_chunk in quote_chunks]
        compressed_outputs = await asyncio.gather(*compress_tasks)
        final_context = "\n\n".join(compressed_outputs)
    else:
        final_context = concatenated_quotes
    
    # Step 6: Extract final answer
    final_prompt = f"""Please read the following quotes carefully and answer the question below.

<quotes>
{final_context}
</quotes>

What is the correct answer to this question: {question}

Choices:
(A) {choices['choice_A']}
(B) {choices['choice_B']}
(C) {choices['choice_C']}
(D) {choices['choice_D']}

Follow these steps to answer the question:
1. Analyze the quotes and how they relate to the question
2. Determine which answer choice is best supported by the quotes
3. Provide your final answer in this format: "The correct answer is (insert answer here)" """
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.1,
        max_tokens=1024
    )
    answer = extract_answer(response.choices[0].message.content)
    
    # Step 7: Log information
    log_info = {
        "question_preview": question[:100],
        "total_chunks": len(chunks),
        "relevant_chunks": len(relevant_quotes),
        "concatenated_words": concat_words,
        "compression_applied": concat_words > 8000,
        "final_context_words": len(final_context.split())
    }
    with open("log.txt", "a") as f:
        f.write(str(log_info) + "\n")
    
    return answer

# Example usage (assuming the original class structure)
class QuotesChunkedAgent:
    """Agent that extracts relevant quotes from the context in chunks"""
    name = "quotes_chunked"
    description = "Chunked quote extraction agent"
    
    @staticmethod
    async def run(question: str, context: str, choices: Dict[str, str], client: AsyncOpenAI) -> str:
        """Process a question with context and choices using quote extraction over chunks."""
        return await process_long_context(question, context, choices, client)