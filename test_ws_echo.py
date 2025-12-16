import asyncio
import aiohttp
import json
import time
import wave
import struct

SERVER_URL = "ws://localhost:8000/v1/audio/speech/stream/ws"
OUTPUT_FILE = "output.wav"
SAMPLE_RATE = 44100  # Match Server Config

async def test_echo_ws():
    print(f"Connecting to {SERVER_URL}...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.ws_connect(SERVER_URL) as ws:
                print("Connected!")
                
                # Prepare a request
                text = "Hello! This is a test of the Echo TTS streaming WebSocket. How does it sound?"
                voice = "tara"
                segment_id = "test_seg_1"
                
                request = {
                    "input": text,
                    "voice": "maya_ref",
                    "segment_id": segment_id,
                    "continue": True
                }
                
                print(f"Sending request: {json.dumps(request, indent=2)}")
                await ws.send_json(request)
                
                # Variables to track performance
                start_time = time.perf_counter()
                first_chunk = True
                audio_data = bytearray()
                
                print("Waiting for response...")
                
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        
                        if msg_type == "start":
                            print(f"[Server] Started segment: {data.get('segment_id')}")
                        elif msg_type == "end":
                            print(f"[Server] Ended segment: {data.get('segment_id')}")
                            # Send close signal
                            close_req = {"continue": False}
                            print("Sending close signal...")
                            await ws.send_json(close_req)
                        elif "done" in data:
                            print("[Server] Done signal received. Closing.")
                            break
                        elif "error" in data:
                            print(f"[Server] Error: {data['error']}")
                            break
                        else:
                            print(f"[Server] Unknown JSON: {data}")
                            
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        if first_chunk:
                            ttfb = (time.perf_counter() - start_time) * 1000
                            print(f"⚡ TTFB: {ttfb:.2f} ms")
                            first_chunk = False
                        
                        chunk_len = len(msg.data)
                        if chunk_len % 2 != 0:
                            print(f"⚠️ Warning: Odd chunk size received: {chunk_len} bytes")
                        
                        # Debug: check first/last bytes of chunk
                        # print(f"Chunk: {chunk_len} bytes. Start: {msg.data[:4].hex()} End: {msg.data[-4:].hex()}")
                        
                        audio_data.extend(msg.data)
                    
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        print("WebSocket closed by server.")
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print("WebSocket error.")
                        break
                
                # Save audio to WAV
                if audio_data:
                    print(f"Saving {len(audio_data)} bytes to {OUTPUT_FILE}...")
                    try:
                        with wave.open(OUTPUT_FILE, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2) # 16-bit PCM
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes(audio_data)
                        print(f"✅ Saved audio to {OUTPUT_FILE}")
                    except Exception as e:
                        print(f"❌ Failed to save audio: {e}")
                else:
                    print("❌ No audio data received.")
                    
        except aiohttp.ClientConnectorError as e:
            print(f"❌ Connection failed: {e}")
            print("Make sure the server is running: python server.py")

if __name__ == "__main__":
    asyncio.run(test_echo_ws())
