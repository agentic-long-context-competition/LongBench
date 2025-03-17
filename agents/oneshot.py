import re
from typing import Dict
from openai import AsyncOpenAI

class OneshotAgent:
    """Basic oneshot agent implementation."""
    name = "oneshot"
    description = "Simple one-shot prompting agent"
    
    @staticmethod
    async def run(question: str, context: str, choices: Dict[str, str], client: AsyncOpenAI) -> str:
        """Process a question with context and choices; returns predicted answer (A, B, C, D, or N)."""
        prompt = f"""Please read the following text and answer the question below.

<text>
{context.strip()}
</text>

What is the correct answer to this question: {question.strip()}
Choices:
(A) {choices['choice_A'].strip()}
(B) {choices['choice_B'].strip()}
(C) {choices['choice_C'].strip()}
(D) {choices['choice_D'].strip()}

Format your response as follows: "The correct answer is (insert answer here)"."""
        
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=128
        )
        
        return extract_answer(response.choices[0].message.content)

# For backward compatibility
run_agent = OneshotAgent.run

def extract_answer(response: str) -> str:
    """Extract the answer (A, B, C, D) from response text or return N if not found."""
    # Fallback patterns for common variations
    patterns = [
        r"answer is \(?([A-D])\)?",
        r"answer: \(?([A-D])\)?",
        r"option \(?([A-D])\)?"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Last resort: check for options directly
    for option in ["A", "B", "C", "D"]:
        if f"({option})" in response:
            return option
    
    return "N"