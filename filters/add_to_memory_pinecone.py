"""
title: Add to Memory Knowledge Base Icon (Pinecone Version)
author: aahadr
author_url: https://github.com/ay4t
funding_url: https://github.com/ay4t
version: 0.1.0
icon_url: data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgdmlld0JveD0iMCAwIDMyIDMyIj4KICA8cmVjdCB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHg9IjQiIHk9IjQiIHN0cm9rZT0iIzRjNGM0YyIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSJub25lIi8+CiAgPGxpbmUgeDE9IjgiIHkxPSIxMiIgeDI9IjE2IiB5Mj0iMTIiIHN0cm9rZT0iIzRjNGM0YyIgc3Ryb2tlLXdpZHRoPSIyIi8+CiAgPGxpbmUgeDE9IjgiIHkxPSIxNiIgeDI9IjE2IiB5Mj0iMTYiIHN0cm9rZT0iIzRjNGM0YyIgc3Ryb2tlLXdpZHRoPSIyIi8+CiAgPGxpbmUgeDE9IjgiIHkxPSIyMCIgeDI9IjE2IiB5Mj0iMjAiIHN0cm9rZT0iIzRjNGM0YyIgc3Ryb2tlLXdpZHRoPSIyIi8+PC9zdmc+
required_open_webui_version: 0.1.0
description: An action button to add response LLM to memory knowledge base using Pinecone.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from fastapi.requests import Request
from open_webui.apps.webui.models.users import Users
from open_webui.main import webui_app
import asyncio
import json
from pathlib import Path
from datetime import datetime
import logging
import openai
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
import uuid


class Action:
    class Valves(BaseModel):
        pinecone_api_key: str = Field(default="your-pinecone-api-key")
        pinecone_cloud: str = Field(default="aws")
        pinecone_region: str = Field(default="us-west-2")
        pinecone_index_name: str = Field(default="openwebui-memory")
        embedding_model: str = Field(default="sentence-transformers/all-mpnet-base-v2")

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation: True
        self.pc = None
        self.index = None
        self.embeddings = None

    def initialize_memory_store(self):
        """Initialize the memory store with Pinecone if not already initialized"""
        if (
            self.pc is not None
            and self.index is not None
            and self.embeddings is not None
        ):
            return

        try:
            if not self.embeddings:
                # Initialize embeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.valves.embedding_model
                )

            if not self.pc:
                # Initialize Pinecone client
                if self.valves.pinecone_api_key == "your-pinecone-api-key":
                    raise ValueError(
                        "Please set a valid Pinecone API key in valves configuration"
                    )
                self.pc = Pinecone(api_key=self.valves.pinecone_api_key)

            if not self.index:
                # Create index if it doesn't exist
                if (
                    self.valves.pinecone_index_name
                    not in self.pc.list_indexes().names()
                ):
                    self.pc.create_index(
                        name=self.valves.pinecone_index_name,
                        dimension=768,  # dimension for all-mpnet-base-v2
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud=self.valves.pinecone_cloud,
                            region=self.valves.pinecone_region,
                        ),
                    )

                # Connect to index
                self.index = self.pc.Index(self.valves.pinecone_index_name)
                logging.info("✅ Pinecone vector store initialized successfully")
        except ValueError as ve:
            logging.error(f"❌ Configuration error: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"❌ Error initializing Pinecone: {str(e)}")
            raise

    async def add_to_memory(self, content: str, metadata: Dict, __event_emitter__=None):
        """Add content to Pinecone with metadata"""
        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Initializing Pinecone connection...",
                            "done": False,
                        },
                    }
                )

            # Initialize if needed
            if not self.index or not self.embeddings or not self.pc:
                self.initialize_memory_store()

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Generating embeddings...",
                            "done": False,
                        },
                    }
                )

            # Generate embedding for the content
            embedding = self.embeddings.embed_query(content)

            # Filter metadata to ensure only simple types and add content
            filtered_metadata = {
                "content": content,  # Store the actual content in metadata
                "timestamp": int(datetime.now().timestamp()),
            }

            # Add other metadata fields if they are simple types
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    filtered_metadata[key] = value

            # Generate a unique ID for the vector
            vector_id = str(uuid.uuid4())

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Saving to Pinecone...", "done": False},
                    }
                )

            # Upsert to Pinecone
            self.index.upsert(
                vectors=[(vector_id, embedding, filtered_metadata)],
                namespace="memory-filter",
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "success",
                            "description": f"✅ Successfully saved to Knowledge Base (ID: {vector_id})",
                            "done": True,
                        },
                    }
                )
            return True
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "error",
                            "description": f"❌ Error saving to Knowledge Base: {str(e)}",
                            "done": True,
                        },
                    }
                )
            return False

    def summaryze_user_question(self, body: dict) -> str:
        try:
            # Get messages from body
            messages = body.get("messages", [])
            if not messages:
                raise ValueError("No messages found in the body")

            if len(messages) > 5:
                messages = messages[-5:]

            # Extract only the content and role from messages
            conversation = []
            for msg in messages:
                if not msg.get("content"):
                    continue
                conversation.append(
                    {"role": msg.get("role"), "content": msg.get("content")}
                )

            if not conversation:
                raise ValueError("No valid messages with content found")

            # Convert to JSON string
            try:
                conversation_string = json.dumps(conversation, ensure_ascii=False)
            except Exception as e:
                logging.error(f"Error converting conversation to JSON: {str(e)}")
                raise ValueError(f"JSON conversion error: {str(e)}")

            system_prompt = """
            You are a helpful assistant that summarizes based on data provided by the user.
            Your task is to summarize following those instructions:

            Intructions:
            1. Summarize the user's question based on the provided data.
            2. Your response should be a string of 20 words or less.
            3. Your response must state a question sentence, it can start with What, How, When and so on.
            4. Following whats language used in the user's question.
            """

            try:
                openai.base_url = "https://api.groq.com/openai/v1/"
                openai.api_key = (
                    "<your OpenAI API key here>"
                )
                response = openai.chat.completions.create(
                    model="gemma-7b-it",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": conversation_string},
                    ],
                    temperature=0,
                )

                if not response or not response.choices:
                    raise ValueError("No response received from LLM")

                return response.choices[0].message.content

            except openai.APIError as e:
                logging.error(f"OpenAI API error: {str(e)}")
                return f"Error getting summary: {str(e)}"
            except Exception as e:
                logging.error(f"Error in LLM call: {str(e)}")
                return f"Error processing summary: {str(e)}"

        except Exception as e:
            logging.error(f"Error in summaryze_user_question: {str(e)}")
            return f"Error generating summary: {str(e)}"

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        print(f"action:{__name__}")

        user_valves = __user__.get("valves")
        if not user_valves:
            user_valves = self.UserValves()

        user = Users.get_user_by_id(__user__["id"])

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Processing Knowledge Base Action",
                    "done": False,
                },
            }
        )

        try:
            # Get messages from body
            messages = body.get("messages", [])
            if len(messages) >= 2:
                # Get last two messages
                user_message = messages[-2]  # Second to last message (user)
                assistant_message = messages[-1]  # Last message (assistant)

                summaryze_user_question = self.summaryze_user_question(body)
                if not summaryze_user_question:
                    return body

                # Format the messages with summary
                formatted_content = f"""User Question: {summaryze_user_question} \nAssistant: {assistant_message['content']}"""

                # Prepare metadata
                metadata = {
                    "user_message_id": user_message.get("id"),
                    "assistant_message_id": assistant_message.get("id"),
                    "timestamp": int(datetime.now().timestamp()),
                    "model": body.get("model"),
                    "chat_id": body.get("chat_id"),
                    "id": body.get("id"),
                    "session_id": body.get("session_id"),
                }

                if __user__:
                    metadata["user_id"] = __user__.get("id")
                    metadata["user_email"] = __user__.get("email")
                    metadata["user_name"] = __user__.get("name")

                # Add to Pinecone
                success = await self.add_to_memory(
                    formatted_content, metadata, __event_emitter__
                )

                if success:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "success",
                                "description": "✅ Added to Knowledge Base",
                                "done": True,
                            },
                        }
                    )
                else:
                    raise Exception("Failed to add to knowledge base")

        except Exception as e:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Error User Action", "done": True},
                }
            )

            await __event_emitter__(
                {
                    "type": "citation",
                    "data": {
                        "source": {"name": "Error:user action"},
                        "document": [str(e)],
                        "metadata": [{"source": "Knowledge Base Action Button"}],
                    },
                }
            )

        return body
