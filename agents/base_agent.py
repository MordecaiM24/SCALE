import time
from typing import List, Dict, Optional
from openai import OpenAI
from pydantic import BaseModel

class BaseAgent:
    """A base class for all AI-powered agents.
    
    Supports two types of context:
    1. Conversation context: The immediate back-and-forth with the LLM (reset per phase)
    2. Session memory: Long-term memory injected as context (persists across phases)
    """
    def __init__(self, client: OpenAI, model: str, system_prompt: str):
        self.client = client
        self.model = model
        self.base_system_prompt = system_prompt  # Original system prompt
        self.system_prompt = system_prompt  # Current system prompt (may include memory)
        self.session_memory_context: str = ""  # Long-term memory to inject
        self.context: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]

    def _build_effective_system_prompt(self) -> str:
        """Builds the effective system prompt including any session memory."""
        if self.session_memory_context:
            return f"{self.base_system_prompt}\n\n{self.session_memory_context}"
        return self.base_system_prompt

    def update_session_memory(self, memory_context: str):
        """
        Updates the long-term session memory that persists across context resets.
        This mirrors how human coders remember past discussions and learnings.
        """
        self.session_memory_context = memory_context
        self.system_prompt = self._build_effective_system_prompt()
        # Update the system message in current context
        if self.context and self.context[0]["role"] == "system":
            self.context[0]["content"] = self.system_prompt

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

    def reset_context(self, preserve_memory: bool = True):
        """
        Resets the conversation context.
        
        Args:
            preserve_memory: If True, session memory is preserved in the system prompt.
                           If False, resets to base system prompt only.
        """
        if preserve_memory:
            self.system_prompt = self._build_effective_system_prompt()
        else:
            self.system_prompt = self.base_system_prompt
            self.session_memory_context = ""
        self.context = [{"role": "system", "content": self.system_prompt}]
    
    def get_context_length(self) -> int:
        """Returns the number of messages in current context (for debugging)."""
        return len(self.context)