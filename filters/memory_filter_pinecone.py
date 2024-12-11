"""
title: Memory Filter for Open WebUI
author: aahadr
author_url: https://github.com/ay4t
funding_url: https://github.com/ay4t
version: 0.1.0
description: A memory filter plugin for Open WebUI, enhancing chat history management and improving conversation context retention for more effective interactions.
"""

from datetime import datetime
import json
import logging
import os
import time
import uuid
import asyncio
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, ConfigDict, Field


class Filter:
    """Memory Filter implementation."""

    class Valves(BaseModel):
        """Configuration parameters for the filter."""

        pinecone_api_key: str = Field(default="your-pinecone-api-key")
        pinecone_cloud: str = Field(default="aws")
        pinecone_region: str = Field(default="us-west-2")
        pinecone_index_name: str = Field(default="openwebui-memory")
        embedding_model: str = Field(default="sentence-transformers/all-mpnet-base-v2")
        max_context_length: int = Field(default=4000)
        max_memories: int = Field(default=3)
        similarity_threshold: float = Field(default=0.7)
        priority: int = Field(default=1)

    class UserValves(Valves):
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )

    class MemoryAPI(BaseModel):
        """APIs for interacting with the memory store."""

        pc: Optional[Any] = None
        index: Optional[Any] = None
        embeddings: Optional[HuggingFaceEmbeddings] = None

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def initialize_store(self, valves: "Filter.Valves"):
            """Initialize vector store with configuration from valves."""
            try:
                if not self.embeddings:
                    # Initialize embeddings
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name=valves.embedding_model
                    )
                
                if not self.pc:
                    # Initialize Pinecone client
                    if valves.pinecone_api_key == "your-pinecone-api-key":
                        raise ValueError("Please set a valid Pinecone API key in valves configuration")
                    self.pc = Pinecone(api_key=valves.pinecone_api_key)
                
                if not self.index:
                    # Create index if it doesn't exist
                    if valves.pinecone_index_name not in self.pc.list_indexes().names():
                        self.pc.create_index(
                            name=valves.pinecone_index_name,
                            dimension=768,  # dimension for all-mpnet-base-v2
                            metric='cosine',
                            spec=ServerlessSpec(
                                cloud=valves.pinecone_cloud,
                                region=valves.pinecone_region
                            )
                        )
                    
                    # Connect to index
                    self.index = self.pc.Index(valves.pinecone_index_name)
                    logging.info("✅ Vector store initialized successfully")
            except Exception as e:
                logging.error(f"❌ Error initializing vector store: {str(e)}")
                raise

        async def add_memory(
            self,
            text: str,
            metadata: Optional[Dict] = None,
            valves: Optional["Filter.Valves"] = None,
        ):
            """Add a memory to the vector store."""
            if not self.index or not self.embeddings:
                if not valves:
                    raise ValueError("Valves required for initialization")
                self.initialize_store(valves)

            if metadata is None:
                metadata = {}

            # Add timestamp
            metadata["timestamp"] = time.time()
            metadata["datetime"] = datetime.now().isoformat()
            metadata["content"] = text  # Store content in metadata for retrieval

            try:
                # Generate embedding
                embedding = self.embeddings.embed_query(text)
                
                # Generate unique ID
                vector_id = str(uuid.uuid4())
                
                # Upsert to Pinecone
                self.index.upsert(
                    vectors=[(vector_id, embedding, metadata)],
                    namespace="memory-filter"
                )
                logging.info(f"Added memory: {text[:100]}...")
            except Exception as e:
                logging.error(f"Error adding memory: {str(e)}")
                raise

        async def search_memories(
            self,
            query: str,
            k: int = 3,
            score_threshold: float = 0.7,
            valves: Optional["Filter.Valves"] = None,
        ) -> List[Dict]:
            """Search for relevant memories."""
            if not self.index or not self.embeddings:
                if not valves:
                    raise ValueError("Valves required for initialization")
                self.initialize_store(valves)

            try:
                # Generate query embedding
                query_embedding = self.embeddings.embed_query(query)
                
                # Search in Pinecone
                results = self.index.query(
                    vector=query_embedding,
                    top_k=k,
                    namespace="memory-filter",
                    include_metadata=True
                )

                memories = []
                for match in results.matches:
                    if match.score >= score_threshold:
                        memories.append({
                            "content": match.metadata.get("content", ""),
                            "metadata": match.metadata,
                            "relevance_score": match.score,
                        })
                return memories
            except Exception as e:
                logging.error(f"Error searching memories: {str(e)}")
                raise

    def __init__(self):
        """Initialize Memory Filter."""
        try:
            self.valves = self.Valves()
            self.memory_api = self.MemoryAPI()

            # Initialize logging
            logging.basicConfig(level=logging.INFO)

        except Exception as e:
            logging.error(f"❌ Error initializing Memory Filter: {str(e)}")
            raise

    def _ensure_initialized(self):
        """Ensure memory store is initialized before use"""
        if not self.memory_api.index or not self.memory_api.embeddings:
            self.memory_api.initialize_store(self.valves)

    def _format_memories_to_context(self, memories: List[Dict]) -> str:
        """Format memories into a context string.

        Args:
            memories (List[Dict]): List of memories with content and metadata

        Returns:
            str: Formatted context string
        """
        context = "Previous relevant conversations:\n\n"

        # Sort memories by timestamp if available
        memories = sorted(
            memories,
            key=lambda x: x.get("metadata", {}).get("timestamp", 0),
            reverse=True,
        )

        for i, memory in enumerate(memories, 1):
            content = memory["content"]
            metadata = memory.get("metadata", {})

            # Add timestamp if available
            timestamp_str = ""
            if "timestamp" in metadata:
                dt = datetime.fromtimestamp(metadata["timestamp"])
                timestamp_str = f" ({dt.strftime('%Y-%m-%d %H:%M:%S')})"

            # Format the memory entry
            context += f"{i}. {content}{timestamp_str}\n\n"

        context += "\nPlease consider these previous conversations in your response."

        return context

    def _filter_metadata(self, metadata: Dict) -> Dict:
        """Filter metadata to only include simple types that ChromaDB supports.

        Args:
            metadata (Dict): Original metadata

        Returns:
            Dict: Filtered metadata with only simple types
        """
        filtered = {}
        for key, value in metadata.items():
            # Only allow simple types
            if isinstance(value, (str, int, float, bool)):
                filtered[key] = value
            # Convert complex types to string representation
            elif isinstance(value, dict):
                # For user dict, extract only simple fields
                if key == "user":
                    user_data = {}
                    for k, v in value.items():
                        if isinstance(v, (str, int, float, bool)):
                            user_data[k] = v
                    filtered["user_id"] = user_data.get("id", "")
                    filtered["user_name"] = user_data.get("name", "")
                    filtered["user_email"] = user_data.get("email", "")
                else:
                    filtered[key] = str(value)
            else:
                filtered[key] = str(value)
        return filtered

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> dict:
        """Process incoming messages before they reach the LLM.

        Args:
            body (dict): Request body containing messages
            __user__ (dict, optional): User information
            __event_emitter__ (callable, optional): Event emitter function

        Returns:
            dict: Modified request body
        """
        try:
            # Get the latest message
            messages = body.get("messages", [])
            if not messages:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "warning",
                                "description": "⚠️ No messages found in request",
                                "done": False,
                            },
                        }
                    )
                return body

            latest_message = messages[-1]
            if not isinstance(latest_message, dict):
                return body

            content = latest_message.get("content", "")
            if not content:
                return body

            # Only initialize when needed and after we know we have content to process
            try:
                self._ensure_initialized()
            except ValueError as ve:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "warning",
                                "description": f"⚠️ Configuration incomplete: {str(ve)}",
                                "done": True,
                            },
                        }
                    )
                return body
            except Exception as e:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "error",
                                "description": f"❌ Error initializing memory store: {str(e)}",
                                "done": True,
                            },
                        }
                    )
                return body

            # Emit starting search
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "processing",
                            "description": "🔍 Searching for relevant memories...",
                            "done": False,
                        },
                    }
                )

                await asyncio.sleep(1)

            # Search for relevant memories
            memories = await self.memory_api.search_memories(
                content,
                k=self.valves.max_memories,
                score_threshold=self.valves.similarity_threshold,
                valves=self.valves,
            )

            if memories:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "success",
                                "description": f"✨ Found {len(memories)} relevant memories",
                                "done": False,
                            },
                        }
                    )

                # Format memories into context
                context = self._format_memories_to_context(memories)

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "processing",
                                "description": "📝 Adding context from memories...",
                                "done": False,
                            },
                        }
                    )

                # Add context to system message
                system_msg = {"role": "system", "content": f"<CONTEXT>\n{context}</CONTEXT>\n"}
                messages.insert(0, system_msg)
                body["messages"] = messages

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "success",
                                "description": "✅ Memory context added successfully",
                                "done": True,
                            },
                        }
                    )

                await __event_emitter__({
                    "type": "citation",
                    "data": {
                        "source": {"name": "Memory Context"},
                        "document": [context],
                        "metadata": [{"source": "Pinecone Vector Store"}],
                    },
                })

            else:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "info",
                                "description": "💭 No relevant memories found",
                                "done": True,
                            },
                        }
                    )

            return body

        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "error",
                            "description": f"❌ Error in memory processing: {str(e)}",
                            "done": True,
                        },
                    }
                )
            logging.error(f"Error in inlet: {str(e)}")
            return body

    async def outlet(
        self,
        body: dict,
        __name__: Optional[str] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> dict:
        """Process responses after they are generated.

        Args:
            body (dict): Response body containing messages
            __name__ (str, optional): Name identifier
            __user__ (dict, optional): User information
            __event_emitter__ (callable, optional): Event emitter function

        Returns:
            dict: Modified response body
        """

        # fitur auto update memory saat ini belum diaktifkan
        return body
