"""
Streaming Response System for OdorDiff-2

Implements efficient streaming for large molecule generation batches:
- Server-Sent Events (SSE) for real-time progress updates
- WebSocket connections for bidirectional communication
- Chunked HTTP responses for large datasets
- Memory-efficient result streaming
- Progress tracking and cancellation support
- Compression and filtering for bandwidth optimization
"""

import os
import json
import time
import asyncio
import uuid
from typing import Dict, List, Optional, Any, AsyncIterator, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import asynccontextmanager
import gzip
import zlib
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from starlette.types import Send

from ..models.molecule import Molecule
from ..utils.logging import get_logger
from .redis_config import get_redis_manager

logger = get_logger(__name__)


class StreamingFormat(Enum):
    """Streaming response formats."""
    JSON = "json"
    JSONL = "jsonl"  # JSON Lines
    SSE = "sse"      # Server-Sent Events
    WEBSOCKET = "websocket"
    BINARY = "binary"


class CompressionType(Enum):
    """Compression types for streaming."""
    NONE = "none"
    GZIP = "gzip"
    DEFLATE = "deflate"
    LZ4 = "lz4"


@dataclass
class StreamingConfig:
    """Configuration for streaming responses."""
    # Chunking
    chunk_size: int = 1024 * 8  # 8KB chunks
    max_chunk_buffer: int = 100  # Maximum chunks in buffer
    
    # Compression
    compression: CompressionType = CompressionType.GZIP
    compression_threshold: int = 1024  # Compress if data > threshold
    compression_level: int = 6  # 1-9, higher = better compression
    
    # Progress tracking
    progress_interval: float = 1.0  # seconds between progress updates
    include_metadata: bool = True
    include_timestamps: bool = True
    
    # Connection management
    heartbeat_interval: float = 30.0  # seconds
    connection_timeout: float = 300.0  # 5 minutes
    max_concurrent_streams: int = 100
    
    # Filtering and transformation
    enable_filtering: bool = True
    include_raw_data: bool = False
    format_output: bool = True


@dataclass
class StreamEvent:
    """Represents a streaming event."""
    event_type: str
    data: Any
    event_id: Optional[str] = None
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_sse_format(self) -> str:
        """Convert to Server-Sent Events format."""
        lines = []
        
        if self.event_id:
            lines.append(f"id: {self.event_id}")
        
        lines.append(f"event: {self.event_type}")
        
        # Handle data serialization
        if isinstance(self.data, dict):
            data_str = json.dumps(self.data)
        elif isinstance(self.data, str):
            data_str = self.data
        else:
            data_str = str(self.data)
        
        # Split multi-line data
        for line in data_str.split('\n'):
            lines.append(f"data: {line}")
        
        lines.append("")  # Empty line to end event
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Convert to JSON format."""
        return json.dumps({
            'event_type': self.event_type,
            'data': self.data,
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        })


class StreamCompressor:
    """Handles compression for streaming data."""
    
    def __init__(self, compression_type: CompressionType, level: int = 6):
        self.compression_type = compression_type
        self.level = level
        
        if compression_type == CompressionType.GZIP:
            self.compressor = None
        elif compression_type == CompressionType.DEFLATE:
            self.compressor = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
        elif compression_type == CompressionType.LZ4:
            try:
                import lz4.frame
                self.compressor = lz4.frame.LZ4FrameCompressor()
            except ImportError:
                logger.warning("LZ4 not available, falling back to gzip")
                self.compression_type = CompressionType.GZIP
                self.compressor = None
    
    def compress_chunk(self, data: bytes) -> bytes:
        """Compress a data chunk."""
        if self.compression_type == CompressionType.NONE:
            return data
        
        if len(data) < 100:  # Don't compress very small chunks
            return data
        
        try:
            if self.compression_type == CompressionType.GZIP:
                return gzip.compress(data, compresslevel=self.level)
            
            elif self.compression_type == CompressionType.DEFLATE:
                compressed = self.compressor.compress(data)
                compressed += self.compressor.flush(zlib.Z_SYNC_FLUSH)
                return compressed
            
            elif self.compression_type == CompressionType.LZ4:
                return self.compressor.compress(data)
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return data
        
        return data
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for compression."""
        if self.compression_type == CompressionType.GZIP:
            return {"Content-Encoding": "gzip"}
        elif self.compression_type == CompressionType.DEFLATE:
            return {"Content-Encoding": "deflate"}
        elif self.compression_type == CompressionType.LZ4:
            return {"Content-Encoding": "lz4"}
        return {}


class MoleculeStreamProcessor:
    """Processes molecule generation results for streaming."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.compressor = StreamCompressor(config.compression, config.compression_level)
    
    async def process_molecules_stream(
        self,
        molecules: AsyncIterator[Molecule],
        format_type: StreamingFormat,
        request_id: str,
        total_expected: Optional[int] = None
    ) -> AsyncIterator[bytes]:
        """Process molecule stream and yield formatted chunks."""
        
        chunk_buffer = []
        processed_count = 0
        start_time = time.time()
        last_progress_time = start_time
        
        # Send initial event
        initial_event = StreamEvent(
            event_type="stream_start",
            event_id=request_id,
            data={
                "request_id": request_id,
                "total_expected": total_expected,
                "format": format_type.value,
                "compression": self.config.compression.value
            },
            timestamp=start_time
        )
        
        yield await self._format_and_compress(initial_event, format_type)
        
        async for molecule in molecules:
            try:
                # Process molecule
                processed_molecule = self._format_molecule(molecule)
                chunk_buffer.append(processed_molecule)
                processed_count += 1
                
                # Send progress update if needed
                current_time = time.time()
                if (current_time - last_progress_time) >= self.config.progress_interval:
                    progress_event = StreamEvent(
                        event_type="progress",
                        event_id=f"{request_id}-progress-{processed_count}",
                        data={
                            "processed": processed_count,
                            "total": total_expected,
                            "percentage": (processed_count / total_expected * 100) if total_expected else None,
                            "rate": processed_count / (current_time - start_time),
                            "elapsed_time": current_time - start_time
                        },
                        timestamp=current_time
                    )
                    yield await self._format_and_compress(progress_event, format_type)
                    last_progress_time = current_time
                
                # Send chunk if buffer is full
                if len(chunk_buffer) >= self.config.max_chunk_buffer:
                    chunk_event = StreamEvent(
                        event_type="data_chunk",
                        event_id=f"{request_id}-chunk-{processed_count // self.config.max_chunk_buffer}",
                        data=chunk_buffer.copy(),
                        timestamp=current_time
                    )
                    yield await self._format_and_compress(chunk_event, format_type)
                    chunk_buffer.clear()
                    
            except Exception as e:
                error_event = StreamEvent(
                    event_type="error",
                    event_id=f"{request_id}-error",
                    data={
                        "error": str(e),
                        "processed_count": processed_count
                    },
                    timestamp=time.time()
                )
                yield await self._format_and_compress(error_event, format_type)
        
        # Send remaining molecules in buffer
        if chunk_buffer:
            final_chunk_event = StreamEvent(
                event_type="data_chunk",
                event_id=f"{request_id}-final-chunk",
                data=chunk_buffer,
                timestamp=time.time()
            )
            yield await self._format_and_compress(final_chunk_event, format_type)
        
        # Send completion event
        completion_event = StreamEvent(
            event_type="stream_complete",
            event_id=f"{request_id}-complete",
            data={
                "request_id": request_id,
                "total_processed": processed_count,
                "total_time": time.time() - start_time,
                "avg_rate": processed_count / (time.time() - start_time) if time.time() > start_time else 0
            },
            timestamp=time.time()
        )
        yield await self._format_and_compress(completion_event, format_type)
    
    def _format_molecule(self, molecule: Molecule) -> Dict[str, Any]:
        """Format molecule for streaming output."""
        base_data = {
            'smiles': molecule.smiles,
            'confidence': molecule.confidence,
            'safety_score': molecule.safety_score,
            'synth_score': molecule.synth_score,
            'estimated_cost': molecule.estimated_cost
        }
        
        if self.config.include_metadata:
            base_data.update({
                'odor_profile': {
                    'primary_notes': molecule.odor_profile.primary_notes,
                    'secondary_notes': molecule.odor_profile.secondary_notes,
                    'intensity': molecule.odor_profile.intensity,
                    'longevity_hours': molecule.odor_profile.longevity_hours
                },
                'properties': molecule._properties
            })
        
        if self.config.include_timestamps:
            base_data['generated_at'] = time.time()
        
        return base_data
    
    async def _format_and_compress(
        self, 
        event: StreamEvent, 
        format_type: StreamingFormat
    ) -> bytes:
        """Format event and apply compression."""
        
        # Format based on type
        if format_type == StreamingFormat.SSE:
            formatted_data = event.to_sse_format().encode('utf-8')
        elif format_type == StreamingFormat.JSON:
            formatted_data = event.to_json().encode('utf-8')
        elif format_type == StreamingFormat.JSONL:
            formatted_data = (event.to_json() + '\n').encode('utf-8')
        else:
            formatted_data = event.to_json().encode('utf-8')
        
        # Apply compression if configured
        if (self.config.compression != CompressionType.NONE and 
            len(formatted_data) >= self.config.compression_threshold):
            compressed_data = self.compressor.compress_chunk(formatted_data)
            return compressed_data
        
        return formatted_data


class WebSocketStreamManager:
    """Manages WebSocket connections for streaming."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_tasks: Dict[str, asyncio.Task] = {}
        self._redis_manager = None
    
    async def initialize(self):
        """Initialize WebSocket manager."""
        self._redis_manager = await get_redis_manager()
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(websocket, connection_id)
        )
        self.connection_tasks[connection_id] = heartbeat_task
        
        logger.info(f"WebSocket connected: {connection_id}")
    
    async def disconnect(self, connection_id: str):
        """Clean up disconnected WebSocket."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if connection_id in self.connection_tasks:
            task = self.connection_tasks[connection_id]
            task.cancel()
            del self.connection_tasks[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_to_connection(
        self, 
        connection_id: str, 
        event: StreamEvent
    ) -> bool:
        """Send event to specific WebSocket connection."""
        if connection_id not in self.active_connections:
            return False
        
        websocket = self.active_connections[connection_id]
        
        try:
            message = {
                'type': event.event_type,
                'data': event.data,
                'id': event.event_id,
                'timestamp': event.timestamp
            }
            
            await websocket.send_json(message)
            return True
            
        except WebSocketDisconnect:
            await self.disconnect(connection_id)
            return False
        except Exception as e:
            logger.error(f"Error sending to WebSocket {connection_id}: {e}")
            return False
    
    async def broadcast_event(self, event: StreamEvent, filter_func: Optional[Callable] = None):
        """Broadcast event to all connected WebSockets."""
        disconnected = []
        
        for connection_id, websocket in self.active_connections.items():
            # Apply filter if provided
            if filter_func and not filter_func(connection_id):
                continue
            
            success = await self.send_to_connection(connection_id, event)
            if not success:
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            await self.disconnect(connection_id)
    
    async def _heartbeat_loop(self, websocket: WebSocket, connection_id: str):
        """Send periodic heartbeat to maintain connection."""
        while connection_id in self.active_connections:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                heartbeat_event = StreamEvent(
                    event_type="heartbeat",
                    data={"timestamp": time.time()},
                    event_id=f"heartbeat-{int(time.time())}"
                )
                
                await self.send_to_connection(connection_id, heartbeat_event)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat failed for {connection_id}: {e}")
                await self.disconnect(connection_id)
                break
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            'active_connections': len(self.active_connections),
            'connection_ids': list(self.active_connections.keys()),
            'max_concurrent': self.config.max_concurrent_streams
        }


class StreamingResponseBuilder:
    """Builds streaming responses for different formats."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.processor = MoleculeStreamProcessor(config)
        self.websocket_manager = WebSocketStreamManager(config)
    
    async def create_sse_stream(
        self,
        molecules: AsyncIterator[Molecule],
        request_id: str,
        total_expected: Optional[int] = None
    ) -> StreamingResponse:
        """Create Server-Sent Events streaming response."""
        
        async def sse_generator():
            async for chunk in self.processor.process_molecules_stream(
                molecules, StreamingFormat.SSE, request_id, total_expected
            ):
                yield chunk
        
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            **self.processor.compressor.get_headers()
        }
        
        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers=headers
        )
    
    async def create_json_stream(
        self,
        molecules: AsyncIterator[Molecule],
        request_id: str,
        total_expected: Optional[int] = None
    ) -> StreamingResponse:
        """Create JSON streaming response."""
        
        async def json_generator():
            yield b'{"stream_id": "' + request_id.encode() + b'", "data": ['
            
            first_item = True
            async for chunk in self.processor.process_molecules_stream(
                molecules, StreamingFormat.JSON, request_id, total_expected
            ):
                if not first_item:
                    yield b','
                yield chunk
                first_item = False
            
            yield b']}'
        
        headers = {
            "Content-Type": "application/json",
            **self.processor.compressor.get_headers()
        }
        
        return StreamingResponse(
            json_generator(),
            media_type="application/json",
            headers=headers
        )
    
    async def handle_websocket_stream(
        self,
        websocket: WebSocket,
        connection_id: str,
        molecules: AsyncIterator[Molecule],
        request_id: str,
        total_expected: Optional[int] = None
    ):
        """Handle WebSocket streaming connection."""
        try:
            await self.websocket_manager.connect(websocket, connection_id)
            
            async for chunk in self.processor.process_molecules_stream(
                molecules, StreamingFormat.WEBSOCKET, request_id, total_expected
            ):
                # Convert chunk to StreamEvent and send
                try:
                    chunk_data = json.loads(chunk.decode('utf-8'))
                    event = StreamEvent(
                        event_type=chunk_data.get('event_type', 'data'),
                        data=chunk_data.get('data'),
                        event_id=chunk_data.get('event_id'),
                        timestamp=chunk_data.get('timestamp')
                    )
                    
                    success = await self.websocket_manager.send_to_connection(
                        connection_id, event
                    )
                    
                    if not success:
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing WebSocket chunk: {e}")
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket client {connection_id} disconnected")
        finally:
            await self.websocket_manager.disconnect(connection_id)


# Global streaming components
_streaming_config: Optional[StreamingConfig] = None
_response_builder: Optional[StreamingResponseBuilder] = None


def get_streaming_config() -> StreamingConfig:
    """Get or create streaming configuration."""
    global _streaming_config
    
    if _streaming_config is None:
        _streaming_config = StreamingConfig(
            chunk_size=int(os.getenv('STREAMING_CHUNK_SIZE', '8192')),
            compression=CompressionType[os.getenv('STREAMING_COMPRESSION', 'GZIP').upper()],
            compression_threshold=int(os.getenv('STREAMING_COMPRESSION_THRESHOLD', '1024')),
            progress_interval=float(os.getenv('STREAMING_PROGRESS_INTERVAL', '1.0')),
            max_concurrent_streams=int(os.getenv('STREAMING_MAX_CONCURRENT', '100'))
        )
    
    return _streaming_config


async def get_response_builder() -> StreamingResponseBuilder:
    """Get or create streaming response builder."""
    global _response_builder
    
    if _response_builder is None:
        config = get_streaming_config()
        _response_builder = StreamingResponseBuilder(config)
        await _response_builder.websocket_manager.initialize()
    
    return _response_builder