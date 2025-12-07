import asyncio
import websockets
import json

class MocapWebSocketServer:
    """WebSocket server to stream mocap data to Unity/Blender."""
    
    def __init__(self, host: str = 'localhost', port: int = 8765, format: str = 'json'):
        """
        Initializes the WebSocket server.
        
        Args:
            host: Host address
            port: Port number
            format: Data format - 'json' (default) or 'bvh'
        """
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
        self.loop = None
        self.format = format
        
        # BVH-specific state
        if self.format == 'bvh':
            from export import BVHExporter
            self.bvh_exporter = BVHExporter()
            self.bvh_frames = []
    
    async def handler(self, websocket):
        """Handles incoming WebSocket connections."""
        self.clients.add(websocket)
        print(f"Unity client connected. Total clients: {len(self.clients)}")
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            print(f"Unity client disconnected. Total clients: {len(self.clients)}")
    
    async def start(self):
        """Starts the WebSocket server."""
        self.loop = asyncio.get_event_loop()
        self.server = await websockets.serve(self.handler, self.host, self.port)
        print(f"WebSocket server listening on ws://{self.host}:{self.port}")
        await asyncio.Future()  # Run forever
    
    async def send_data(self, data: dict):
        """Sends mocap data to all connected clients."""
        if self.clients:
            if self.format == 'json':
                message = json.dumps(data)
            elif self.format == 'bvh':
                # For BVH streaming, send frame data as text
                body_landmarks = data.get('body', [])
                if body_landmarks and len(body_landmarks) == 33:
                    self.bvh_exporter.add_frame(body_landmarks)
                    # Send simplified frame info
                    message = json.dumps({
                        'format': 'bvh',
                        'frame': len(self.bvh_exporter.frames),
                        'timestamp': data.get('timestamp', 0)
                    })
                else:
                    return
            else:
                message = json.dumps(data)
            
            await asyncio.gather(*[client.send(message) for client in self.clients], return_exceptions=True)
    
    def get_bvh_data(self) -> str:
        """
        Get accumulated BVH data as string.
        Only available when format='bvh'.
        """
        if self.format == 'bvh' and self.bvh_exporter.frames:
            return self._build_bvh_string()
        return ""
    
    def _build_bvh_string(self) -> str:
        """Build complete BVH file content as string."""
        lines = [self.bvh_exporter.skeleton]
        lines.append("MOTION")
        lines.append(f"Frames: {len(self.bvh_exporter.frames)}")
        lines.append(f"Frame Time: {self.bvh_exporter.frame_time}")
        
        for frame in self.bvh_exporter.frames:
            frame_str = " ".join([f"{value:.6f}" for value in frame])
            lines.append(frame_str)
        
        return "\n".join(lines)
