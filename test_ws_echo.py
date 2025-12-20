import asyncio
import aiohttp
import json
import time
import wave
import struct

SERVER_URL = "ws://localhost:8000/v1/audio/speech/stream/ws"
OUTPUT_FILE = "test_output_ws.wav"
SAMPLE_RATE = 44100  # Match Server Config
# voice = "maya_long_ref"
voice = "maya_ref"
# voice = "expresso_02_ex03-ex01_calm_005"
# voice = "maya_voices"

# Test cases for conversational voice assistant - uncomment one to test:


# text = "(singing) Happy birthday to you. Hope you have an amazing day today?"
# text = "(singing) Happy birthday to you."
# text = "(singing) Hope you have an amazing day today."

# text = "Hello (laughs) This is a test of the Echo TTS streaming WebSocket. How does it sound?"
# text = "That's hilarious! (laughs) I can't believe that actually happened to you."
# text = "I'm so happy for you! (laughs) That's the best news I've heard all week."

# text = "(whispers) Hey, I think someone's at the door."
# text = "(whispers) Should I check who it is?"

# text = "(sighs) I've been trying to solve this problem all day."
# text = "(sighs) I tried so hard to prevent it."
# text = "(sighs) Everything feels different now."

# text = "(sobbing) I just can't handle this anymore."
# text = "(sobbing) Everything is falling apart."
# text = "(sobbing) Why did this have to happen?"

# text = "Wait, what (gasps) ? I had no idea that was even possible!"
# text = "(gasps) Oh my goodness! I wasn't expecting that at all."

# text = "(yawns) Sorry, I'm a bit tired."
# text = "(yawns) What were you saying about the meeting?"

# text = "Excuse me (coughs) Now, where were we in our conversation?"
# text = "I'm not feeling well today (coughs) ."

# text = "(sad) Is there anything I can do to help?"
# text = "(surprised) Really? That's not what I thought would happen."
# text = "(frustrated) This isn't working no matter what I try. I don't understand why."
# text = "(angry) I can't believe you did that! This is completely unacceptable."
# text = "(shouts) Watch out! Get out of the way right now!"
# text = "No way! (disgusted) I'm not going anywhere near that mess."
# text = "(excited) Wow, that's incredible! You really did an amazing job on this project."
# text = "(cheerful) The weather looks perfect today. Maybe we should go for a walk in the park?"



seed = 101  # Random seed for reproducibility
# Advanced sampler configuration (optional)
extra_body = {
    "chunking_enabled": False,
    # "speaker_kv_scale": 1.33,
    # "speaker_kv_min_t": 0.9,
    # "speaker_kv_max_layers": 24,
    # Uncomment to customize sampler settings:
    # "block_sizes": [32, 128, 480],
    # "num_steps": [8, 15, 20],
    # "cfg_scale_text": 3.0,
    # "cfg_scale_speaker": 8.0,
}

async def test_echo_ws():
    print(f"Connecting to {SERVER_URL}...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.ws_connect(SERVER_URL) as ws:
                print("Connected!")
                
                # Prepare a request
               
                segment_id = "test_seg_1"
                
                request = {
                    "input": text,
                    "voice": voice,
                    "segment_id": segment_id,
                    "seed": seed,
                    "extra_body": extra_body,
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
