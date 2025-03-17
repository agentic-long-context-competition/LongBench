from typing import Dict
from openai import AsyncOpenAI
from .oneshot import extract_answer

class ChainOfThoughtAgent:
    """Agent that uses chain-of-thought prompting for better reasoning."""
    name = "cot"
    description = "Chain-of-thought prompting agent that encourages step-by-step reasoning"
    
    @staticmethod
    async def run(question: str, context: str, choices: Dict[str, str], client: AsyncOpenAI) -> str:
        """Process a question with context and choices using chain-of-thought prompting."""
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

Take your time to think through this step by step:
1. First, understand what the question is asking for
2. Identify relevant information from the text
3. Consider each option carefully
4. Explain your reasoning for each option 
5. Select the most accurate answer

After explaining your thought process, provide your final answer in this format: "The correct answer is (insert answer here)"."""
        
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024  # Increased token limit for longer reasoning
        )
        
        return extract_answer(response.choices[0].message.content)