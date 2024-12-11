"""
YouTube Transcript Tools - Using youtube-transcript-api
author: aahadr
author_url: https://github.com/ay4t
funding_url: https://github.com/ay4t
Version: 0.1.4
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
from youtube_transcript_api import YouTubeTranscriptApi
import asyncio
import inspect
import sys
import re


class Tools:
    def __init__(self):
        self.citation = True

    async def get_youtube_transcript(
        self,
        video_id: str,
        language: str = "en",
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Get YouTube video transcript using youtube-transcript-api.

        :param video_id: YouTube video ID or URL.
        :param language: Language to retrieve transcript in (default is "en").
        :param __event_emitter__: Optional event emitter for status updates.
        :return: The transcript of the video or an error message.
        """
        if __event_emitter__ is not None and not callable(__event_emitter__):
            raise ValueError("event_emitter must be callable")
        emitter = EventEmitter(__event_emitter__)

        await emitter.status(f"Fetching transcript for video: {video_id}")
        await asyncio.sleep(1)

        try:
            # Extract video ID if a URL is provided
            if not re.match(r"^[a-zA-Z0-9_-]{11}$", video_id):
                match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", video_id)
                if match:
                    video_id = match.group(1)
                    if emitter.debug:
                        print(
                            f"Debug: Extracted video ID from URL: {video_id}",
                            file=sys.stderr,
                        )
                else:
                    raise ValueError(
                        "Video ID is invalid or could not be extracted from the input."
                    )

            # Debugging information
            if emitter.debug:
                print(
                    f"Debug: Requesting transcript for video ID: {video_id} with language: {language}",
                    file=sys.stderr,
                )

            transcript_data = YouTubeTranscriptApi.get_transcript(
                video_id, languages=[language]
            )
            transcript = " ".join([item["text"] for item in transcript_data])

            await emitter.status(f"Transcript fetched successfully.", "complete", True)

            # Debugging information
            if emitter.debug:
                print(
                    f"Debug: Successfully fetched transcript: {transcript[:200]}...",
                    file=sys.stderr,
                )

            return transcript
        except ValueError as ve:
            await emitter.fail(f"Input Error: {str(ve)}")
            if emitter.debug:
                print(f"Debug: Input Error - {str(ve)}", file=sys.stderr)
            return f"Input Error: {str(ve)}"
        except Exception as e:
            await emitter.fail(f"Error: {str(e)}")
            if emitter.debug:
                print(f"Debug: Exception occurred - {str(e)}", file=sys.stderr)
            return f"Error: {str(e)}"


@dataclass
class EventEmitter:
    event_emitter: Optional[Callable[[dict], Any]] = None
    debug: bool = False
    _status_prefix: Optional[str] = None

    def __post_init__(self):
        if self.event_emitter is not None and not callable(self.event_emitter):
            raise ValueError("event_emitter must be callable")

    def set_status_prefix(self, status_prefix: str) -> None:
        """Set a prefix for all status messages."""
        self._status_prefix = status_prefix

    async def _emit(self, typ: str, data: dict) -> None:
        """Internal method to emit events."""
        if self.debug:
            print(f"Emitting {typ} event: {data}", file=sys.stderr)
        if self.event_emitter is None:
            return None
        maybe_future = self.event_emitter(
            {
                "type": typ,
                "data": data,
            }
        )
        if asyncio.isfuture(maybe_future) or inspect.isawaitable(maybe_future):
            return await maybe_future

    async def status(
        self,
        description: str = "Unknown state",
        status: str = "in_progress",
        done: bool = False,
    ) -> None:
        """Emit a status update event."""
        if self._status_prefix is not None:
            description = f"{self._status_prefix}{description}"
        await self._emit(
            "status",
            {
                "status": status,
                "description": description,
                "done": done,
            },
        )

    async def fail(self, description: str = "Unknown error") -> None:
        """Emit a failure event."""
        await self.status(description=description, status="error", done=True)

    async def message(self, content: str) -> None:
        """Emit a message event."""
        await self._emit(
            "message",
            {
                "content": content,
            },
        )

    async def citation(self, document: str, metadata: dict, source: str) -> None:
        """Emit a citation event."""
        await self._emit(
            "citation",
            {
                "document": document,
                "metadata": metadata,
                "source": source,
            },
        )

    async def code_execution_result(self, output: str) -> None:
        """Emit a code execution result event."""
        await self._emit(
            "code_execution_result",
            {
                "output": output,
            },
        )
