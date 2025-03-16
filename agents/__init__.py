"""
Agent registry module for LongBench.

This module provides a registry system for agent implementations.
"""
from typing import Dict, Type, List, Protocol
from openai import AsyncOpenAI
# Import agent implementations
from .oneshot import OneshotAgent
from .chain_of_thought import ChainOfThoughtAgent
from .extract_quotes import ExtractQuotesAgent


# Define the agent protocol that all agents must implement
class AgentProtocol(Protocol):
    """Protocol defining the interface for all agents."""
    @staticmethod
    async def run(question: str, context: str, choices: Dict[str, str], client: AsyncOpenAI) -> str:
        """Process a question with context and choices; returns predicted answer (A, B, C, D, or N)."""
        ...
# Registry of available agents
AGENT_REGISTRY: Dict[str, Type[AgentProtocol]] = {
    OneshotAgent.name: OneshotAgent,
    ChainOfThoughtAgent.name: ChainOfThoughtAgent,
    ExtractQuotesAgent.name: ExtractQuotesAgent,
}

def register_agent(agent_class: Type[AgentProtocol]) -> None:
    """
    Register a new agent implementation with the registry.
    
    Args:
        agent_class: The agent class to register.
    """
    if not hasattr(agent_class, 'name'):
        raise ValueError(f"Agent class {agent_class.__name__} must have a 'name' attribute")
    
    AGENT_REGISTRY[agent_class.name] = agent_class
    
def get_agent(agent_name: str) -> Type[AgentProtocol]:
    """
    Get an agent implementation by name.
    
    Args:
        agent_name: The name of the agent to retrieve.
        
    Returns:
        The agent class.
        
    Raises:
        KeyError: If the agent name is not found in the registry.
    """
    if agent_name not in AGENT_REGISTRY:
        available_agents = ", ".join(AGENT_REGISTRY.keys())
        raise KeyError(f"Agent '{agent_name}' not found. Available agents: {available_agents}")
    
    return AGENT_REGISTRY[agent_name]

def list_agents() -> List[Dict[str, str]]:
    """
    List all registered agents with their descriptions.
    
    Returns:
        A list of dictionaries containing agent name and description.
    """
    return [
        {
            "name": agent_class.name,
            "description": getattr(agent_class, "description", "No description available"),
        }
        for agent_class in AGENT_REGISTRY.values()
    ]