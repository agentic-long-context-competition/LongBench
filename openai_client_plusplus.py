import openai
from openai import AsyncOpenAI
import backoff
import json
import datetime
from dotenv import load_dotenv

load_dotenv()

class AsyncOpenAIPlusPlus(AsyncOpenAI):
    """
    An enhanced AsyncOpenAI client that handles rate limiting with exponential backoff,
    tracks token usage per model, and optionally logs requests and responses.

    Features:
    - Reads OPENAI_API_KEY from environment variables.
    - Implements exponential backoff on rate limit errors.
    - Tracks input and output tokens per model based on API responses.
    - Optionally logs requests and responses to a JSONL file if logging is enabled.
    - Supports asynchronous operation with asyncio.
    - Supports limiting concurrency with an optional semaphore.
    """

    def __init__(self, request_id=None, logging_enabled=False, log_file_path="api_calls.jsonl", semaphore=None, **kwargs):
        """
        Initializes the enhanced OpenAI client.

        Args:
            request_id (str, optional): An optional request ID to associate with API calls.
            logging_enabled (bool, optional): If True, enables logging of requests and responses.
            log_file_path (str, optional): The file path for logging.
            semaphore (asyncio.Semaphore, optional): A semaphore to limit concurrent API calls.
            **kwargs: Additional arguments to pass to the AsyncOpenAI constructor.
        """
        # Initialize the parent AsyncOpenAI client
        super().__init__(**kwargs)
        
        self.model = "gpt-4o-mini"
        self.request_id = request_id
        self.logging_enabled = logging_enabled
        self.log_file_path = log_file_path
        self.semaphore = semaphore

        # Initialize token usage tracking
        self.token_usage = {}

    @backoff.on_exception(
        backoff.expo,
        openai.RateLimitError,
        max_time=60,
        max_tries=6,
        jitter=None
    )
    async def chat_completion(self, **kwargs):
        """
        Creates a chat completion using the OpenAI API with exponential backoff on rate limit errors.
        This is an asynchronous method.

        Args:
            **kwargs: arguments to pass to the OpenAI API.

        Returns:
            ChatCompletion: The OpenAI API response as a Pydantic object.
        """

        # Ensure the model is set to gpt-4o-mini if not explicitly provided
        if 'model' in kwargs and kwargs['model'] not in [self.model]:
            raise ValueError(f"Model {kwargs['model']} is not allowed, switch to {self.model}.")
        
        kwargs['model'] = self.model
            
        # Use semaphore if provided to limit concurrency
        if self.semaphore:
            async with self.semaphore:
                # Make the API call asynchronously using the parent class method
                response = await self.chat.completions.create(**kwargs)
        else:
            # Make the API call without semaphore
            response = await self.chat.completions.create(**kwargs)

        # Update token usage statistics
        self._update_token_usage(response)

        # Log the request and response if logging is enabled
        if self.logging_enabled:
            timestamp = datetime.datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'request_id': self.request_id,
                'input': kwargs,
                'output': response,  # Log the entire response object
            }
            
            with open(self.log_file_path, 'a') as f:
                f.write(json.dumps(log_entry, default=lambda o: o.model_dump() if hasattr(o, 'model_dump') else str(o)) + '\n')

        return response

    def _update_token_usage(self, response):
        """
        Updates the token usage statistics based on the API response.

        Args:
            response (ChatCompletion): The API response containing usage information.
        """
        usage = response.usage
        model = response.model  # Get the actual model used from the response
        model_usage = self.token_usage.setdefault(model, {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        })
        model_usage['prompt_tokens'] += usage.prompt_tokens
        model_usage['completion_tokens'] += usage.completion_tokens
        model_usage['total_tokens'] += usage.total_tokens

    def get_token_usage(self, model=None):
        """
        Retrieves token usage statistics.

        Args:
            model (str, optional): The model to get token usage for.
                If None, returns usage for all models.

        Returns:
            dict: Token usage statistics.
        """
        if model:
            return self.token_usage.get(model, {})
        else:
            return self.token_usage

# Export the client class for use in other modules
__all__ = ['AsyncOpenAIPlusPlus']

# Example usage (commented out)
'''
if __name__ == "__main__":
    async def main():
        # Create an instance of the enhanced OpenAI client
        client = OpenAIClientWrapper(
            request_id="example_request",
            logging_enabled=True,
            log_file_path="api_calls.jsonl"
        )

        # You can use the client just like a regular AsyncOpenAI client
        # For example, you can call methods like:
        # await client.chat.completions.create(...)
        # await client.embeddings.create(...)
        # await client.audio.transcriptions.create(...)

        # Or use the chat_completion helper method with backoff and tracking
        messages = [
            {"role": "user", "content": "Once upon a time,"}
        ]

        response = await client.chat_completion(messages=messages)
        print("Response:", response.choices[0].message.content)

        # Get token usage statistics
        token_usage = client.get_token_usage()
        print("Token usage:", token_usage)

    # Run the async main function
    asyncio.run(main())
'''
