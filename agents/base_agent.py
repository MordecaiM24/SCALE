import time
from typing import List, Dict
from openai import OpenAI
from pydantic import BaseModel

class BaseAgent:
    """A base class for all AI-powered agents."""
    def __init__(self, client: OpenAI, model: str, system_prompt: str):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.context: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]

    def _generate_answer(self, temperature: float = 0.0, response_format: BaseModel = None) -> BaseModel | str:
        """
        Generates a response from the LLM based on the current context.
        Includes retry logic for API errors. Tries max_retries times with 20 second delay between tries.
        """
        tries = 0
        max_retries = 5
        while tries < max_retries:
            try:
                kwargs = {
                    "model": self.model,
                    "messages": self.context,
                    "n": 1,
                    "temperature": temperature
                }
                if response_format:
                    kwargs["response_format"] = response_format
                    completion = self.client.chat.completions.parse(**kwargs)
                    return completion.choices[0].message.parsed
                else:
                    completion = self.client.chat.completions.create(**kwargs)
                    return completion.choices[0].message.content

            except Exception as e:
                tries += 1
                if tries > max_retries:
                    raise e
                print(f"Retrying {tries} due to an error: {e}")
                time.sleep(20)
                return self._generate_answer(temperature, response_format)

    def add_user_message(self, content: str):
        """Adds a user message to the agent's context."""
        self.context.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """Adds an assistant message to the agent's context."""
        if isinstance(content, BaseModel):
            content = content.model_dump_json()
        self.context.append({"role": "assistant", "content": content})
        
    def get_last_response(self) -> str:
        """Returns the last assistant response from the context."""
        for message in reversed(self.context):
            if message["role"] == "assistant":
                return message["content"]
        return ""

    def reset_context(self):
        """Resets the conversation context to just the system prompt."""
        self.context = [{"role": "system", "content": self.system_prompt}]