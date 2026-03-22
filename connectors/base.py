"""
connectors/base.py

Base interface for market data connectors.
"""

from abc import ABC, abstractmethod
from typing import List

from models.market import UnifiedMarket


class BaseConnector(ABC):
    @abstractmethod
    def platform_name(self) -> str:
        """Name of the platform (e.g., 'polymarket', 'kalshi')."""
        pass

    @abstractmethod
    async def fetch_all(self) -> List[UnifiedMarket]:
        """Fetch all currently open, valid markets from the platform."""
        pass
