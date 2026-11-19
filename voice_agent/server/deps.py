"""Long-lived services shared by all requests, created once per process."""

from __future__ import annotations

import logging
import threading

from fastapi import Request, WebSocket

from voice_agent import config
from voice_agent.pipeline import VoiceAgent
from voice_agent.vector_store import VectorStore

logger = logging.getLogger(__name__)


class AgentUnavailableError(RuntimeError):
    """The voice agent cannot run (usually: no SARVAM_API_KEY)."""


class Services:
    """Lazily-built singletons. Tests pass ready-made fakes instead."""

    def __init__(self, vector_store: VectorStore | None = None, agent: VoiceAgent | None = None) -> None:
        self._vector_store = vector_store
        self._agent = agent
        self._retriever = None
        self._lock = threading.Lock()
        # Embedding function for evaluation indexes; None uses the configured model.
        self.eval_embedding_function = None

    @property
    def vector_store(self) -> VectorStore:
        with self._lock:
            if self._vector_store is None:
                self._vector_store = self._agent.rag.vs if self._agent is not None else VectorStore()
            return self._vector_store

    @property
    def retriever(self):
        """The agent's own hybrid retriever, so searches here match what callers get."""
        from voice_agent.rag_pipeline import HybridRetriever

        rag = getattr(self._agent, "rag", None)
        own = getattr(rag, "retriever", None)
        if isinstance(own, HybridRetriever):
            return own
        store = self.vector_store
        with self._lock:
            if self._retriever is None or self._retriever.vs is not store:
                self._retriever = HybridRetriever(store)
            return self._retriever

    @property
    def agent(self) -> VoiceAgent:
        if self._agent is None:
            if not config.SARVAM_API_KEY:
                raise AgentUnavailableError(
                    "The voice agent is not configured. Set SARVAM_API_KEY in .env and restart."
                )
            store = self.vector_store
            with self._lock:
                if self._agent is None:
                    from voice_agent.rag_pipeline import RAGPipeline
                    from voice_agent.sarvam_client import SarvamClient

                    client = SarvamClient()
                    self._agent = VoiceAgent(
                        rag_pipeline=RAGPipeline(vector_store=store, sarvam_client=client),
                        sarvam_client=client,
                    )
        return self._agent

    @property
    def agent_ready(self) -> bool:
        return self._agent is not None or bool(config.SARVAM_API_KEY)

    async def close(self) -> None:
        if self._agent is not None:
            await self._agent.close()


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_services_ws(websocket: WebSocket) -> Services:
    return websocket.app.state.services
