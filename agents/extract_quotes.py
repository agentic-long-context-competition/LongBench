from typing import Dict
from openai import AsyncOpenAI
from .oneshot import extract_answer

class ExtractQuotesAgent:
    """Agent that extracts relevant quotes from the context for better accuracy."""
    name = "quotes"
    description = "Quote extraction agent that focuses on finding supporting evidence"
    
    @staticmethod
    async def run(question: str, context: str, choices: Dict[str, str], client: AsyncOpenAI) -> str:
        """Process a question with context and choices using quote extraction."""
        prompt = f"""Please read the following text carefully and answer the question below.

<text>
{context.strip()}
</text>

What is the correct answer to this question: {question.strip()}

Choices:
(A) {choices['choice_A'].strip()}
(B) {choices['choice_B'].strip()}
(C) {choices['choice_C'].strip()}
(D) {choices['choice_D'].strip()}

Follow these steps to answer the question:
1. First, extract 2-3 relevant quotes from the text that help answer the question
2. For each quote, explain why it's relevant to the question
3. Analyze how each quote relates to the answer choices
4. Based on these quotes and your analysis, determine which answer choice is correct
5. Provide your final answer in this format: "The correct answer is (insert answer here)"."""
        
        response = await client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024  # Increased token limit for longer reasoning
        )
        
        return extract_answer(response.choices[0].message.content)