#!/usr/bin/env python3
"""
Test script for Echo TTS API that measures TTFB and saves streaming audio chunks.
"""

import time
import requests
import sys
import wave
from pathlib import Path


def test_tts_api(
    url: str = "http://localhost:8000/v1/audio/speech",
    input_text: str = "[S1] Hello, this is Echo TTS streaming.",
    voice: str = "expresso_02_ex03-ex01_calm_005",
    stream: bool = True,
    seed: int = 0,
    output_file: str = "test_output.wav",
    extra_body: dict = None,
):
    """
    Test the TTS API with streaming enabled, measure TTFB, and save audio chunks.
    
    Args:
        url: API endpoint URL
        input_text: Text to synthesize
        voice: Voice name
        stream: Whether to use streaming
        seed: Random seed
        output_file: Path to save the audio file
        extra_body: Additional parameters
    """
    if extra_body is None:
        extra_body = {}
    
    payload = {
        "input": input_text,
        "voice": voice,
        "stream": stream,
        "seed": seed,
        "extra_body": extra_body,
    }
    
    print(f"Testing TTS API: {url}")
    print(f"Input text: {input_text}")
    print(f"Voice: {voice}")
    print(f"Streaming: {stream}")
    print(f"Seed: {seed}")
    print("-" * 60)
    
    # Record time just before sending the request
    request_start_time = time.perf_counter()
    
    try:
        # Make the POST request with streaming enabled
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=300,  # 5 minute timeout for long audio
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Record time when we start receiving data (first byte)
        first_byte_time = None
        ttfb = None
        
        # Collect all chunks
        chunks = []
        chunk_count = 0
        total_bytes = 0
        
        # Read the streaming response
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                # Record TTFB on first chunk received
                if first_byte_time is None:
                    first_byte_time = time.perf_counter()
                    ttfb = (first_byte_time - request_start_time) * 1000  # Convert to milliseconds
                    print(f"✓ First byte received!")
                    print(f"  TTFB: {ttfb:.2f} ms")
                
                chunks.append(chunk)
                chunk_count += 1
                total_bytes += len(chunk)
                
                # Print progress for first few chunks
                if chunk_count <= 3:
                    print(f"  Chunk {chunk_count}: {len(chunk)} bytes")
        
        # Record completion time
        request_end_time = time.perf_counter()
        total_time = (request_end_time - request_start_time) * 1000  # Convert to milliseconds
        
        # Get sample rate from headers if available
        sample_rate = response.headers.get("X-Audio-Sample-Rate", "44100")
        
        print("-" * 60)
        print(f"✓ Request completed successfully")
        print(f"  Total chunks received: {chunk_count}")
        print(f"  Total bytes: {total_bytes:,} bytes ({total_bytes / 1024:.2f} KB)")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Total time: {total_time:.2f} ms")
        if ttfb is not None:
            print(f"  TTFB: {ttfb:.2f} ms")
            print(f"  Time after first byte: {total_time - ttfb:.2f} ms")
        
        # Save all chunks to WAV file (this happens AFTER TTFB calculation)
        output_path = Path(output_file)
        sample_rate_int = int(sample_rate)
        
        # Combine all PCM chunks
        pcm_data = b''.join(chunks)
        
        # Write WAV file with header
        with wave.open(str(output_path), 'wb') as wav_file:
            # Set WAV parameters: 1 channel (mono), 2 bytes per sample (16-bit), sample rate
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes per sample
            wav_file.setframerate(sample_rate_int)
            wav_file.writeframes(pcm_data)
        
        print(f"\n✓ Audio saved to: {output_path.absolute()}")
        print(f"  File size: {output_path.stat().st_size:,} bytes")
        
        return {
            "success": True,
            "ttfb_ms": ttfb,
            "total_time_ms": total_time,
            "chunk_count": chunk_count,
            "total_bytes": total_bytes,
            "sample_rate": int(sample_rate),
            "output_file": str(output_path.absolute()),
        }
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Status code: {e.response.status_code}")
            try:
                print(f"  Response: {e.response.text}")
            except:
                pass
        return {
            "success": False,
            "error": str(e),
        }
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }


if __name__ == "__main__":
    # Configuration variables
    URL = "http://localhost:8000/v1/audio/speech"
    # INPUT_TEXT = "Wow, this place looks even better than I imagined. How did they set all this up so perfectly? The lights, the music, everything feels magical. I can't stop smiling right now.”"
    # INPUT_TEXT = "Hey What's up? How are you doing today? So happy to see you again!"
    INPUT_TEXT = "Hey, how are you doing today (whsiper) ? Hey, I'm building a voice agent and I'm facing some issues with the latency (coughs) . I don't know how can I get it some ideas to fix it."
    # VOICE = "expresso_02_ex03-ex01_calm_005"
    VOICE = "maya_ref"
    STREAM = True
    SEED = 0
    OUTPUT_FILE = "test_output2.wav"
    EXTRA_BODY = {}
    
    result = test_tts_api(
        url=URL,
        input_text=INPUT_TEXT,
        voice=VOICE,
        stream=STREAM,
        seed=SEED,
        output_file=OUTPUT_FILE,
        extra_body=EXTRA_BODY,
    )


