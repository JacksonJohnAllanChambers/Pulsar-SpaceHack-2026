"""
Base pipeline interface for pluggable edge processing architectures.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from applet.config import AppletConfig


class BasePipeline(ABC):
    """
    Abstract pipeline runner. Each track or processing module implements
    this interface to maintain clean modularity and separation of concerns.
    """

    def __init__(self, config: AppletConfig):
        self.config = config

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes pipeline processing on the provided context dictionary.
        Must return updated or enriched context with structured results.
        """
        pass
