from __future__ import annotations

import asyncio
import json
import weakref
import time
from dataclasses import dataclass, replace

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

AUDIO_FRAME_SIZE_MS = 20

@dataclass
class _TTSOptions:
    voice: str
    language: str
    base_url: str

    def get_http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}{path}"


class EchoTTS(tts.TTS):
    def __init__(
        self,
        *,
        language: str = "en",
        voice: str = "tara",
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = "http://localhost:8000",
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=44100,
            num_channels=1,
        )

        self._opts = _TTSOptions(
            language=language,
            voice=voice,
            base_url=base_url,
        )
        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        url = self._opts.get_ws_url("/v1/audio/speech/stream/ws")
        return await asyncio.wait_for(session.ws_connect(url), timeout)

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def prewarm(self) -> None:
        self._pool.prewarm()

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = language
        if is_given(voice):
            self._opts.voice = voice

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the bytes endpoint"""

    def __init__(self, *, tts: EchoTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: EchoTTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # Note: Echo server expects standard request payload for HTTP POST
        json_payload = {
            "input": self._input_text,
            "voice": self._opts.voice,
            "stream": True,  # Echo requires explicit stream flag
            "response_format": "pcm", # Echo valid format
        }

        try:
            async with self._tts._ensure_session().post(
                self._opts.get_http_url("/v1/audio/speech"),
                json=json_payload,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=44100,
                    num_channels=1,
                    mime_type="audio/pcm",
                    frame_size_ms=AUDIO_FRAME_SIZE_MS,
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: EchoTTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: EchoTTS = tts
        self._opts = replace(tts._opts)
        self._sent_tokenizer_stream = tokenize.blingfire.SentenceTokenizer().stream()
        self._start_time: float | None = None
        self._segment_started = False

    def push_text(self, text: str) -> None:
        if self._start_time is None:
            self._start_time = time.perf_counter()
        return super().push_text(text)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=44100,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
            frame_size_ms=AUDIO_FRAME_SIZE_MS,
        )

        async def _sentence_stream_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            base_pkt = {"voice": self._opts.voice}
            async for ev in self._sent_tokenizer_stream:
                segment_id = utils.shortuuid()
                token_pkt = base_pkt.copy()
                token_pkt["input"] = ev.token
                token_pkt["continue"] = True
                token_pkt["segment_id"] = segment_id
                self._mark_started()
                await ws.send_str(json.dumps(token_pkt))

            final_pkt = base_pkt.copy()
            final_pkt["input"] = ""
            final_pkt["continue"] = False
            final_pkt["segment_id"] = "final"
            await ws.send_str(json.dumps(final_pkt))

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._sent_tokenizer_stream.flush()
                    continue

                self._sent_tokenizer_stream.push_text(data)

            self._sent_tokenizer_stream.end_input()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            segment_started = False
            first_chunk = True
            try:
                async for msg in ws:
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        break

                    if msg.type == aiohttp.WSMsgType.BINARY:
                        if not segment_started:
                            # Echo server does not send start event before binary? 
                            # Actually implementation says it WILL send start event.
                            # But if binary comes first, we can start segment.
                            output_emitter.start_segment(segment_id=utils.shortuuid())
                            segment_started = True

                        if first_chunk and self._start_time:
                            ttfb = time.perf_counter() - self._start_time
                            # print(
                            #     f"Time to first audio chunk (TTFB): {ttfb*1000:.2f} ms"
                            # )
                            first_chunk = False

                        output_emitter.push(msg.data)
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        if msg_type == "start":
                            # Use server provided segment_id if possible
                            if not segment_started:
                                output_emitter.start_segment(
                                    segment_id=data["segment_id"]
                                )
                                segment_started = True
                        elif msg_type == "end":
                            if segment_started:
                                output_emitter.end_segment()
                                segment_started = False
                        elif msg_type == "no_op":
                            pass # Just ignore empty inputs
                        elif msg_type == "error":
                            print(f"Error from TTS server: {data}")
                        else:
                            print(f"Unknown text message from TTS server: {data}")
            except asyncio.CancelledError:
                pass
            finally:
                if segment_started:
                    output_emitter.end_segment()

        ws = None
        try:
            ws = await self._tts._connect_ws(self._conn_options.timeout)
            tasks = [
                asyncio.create_task(_input_task()),
                asyncio.create_task(_sentence_stream_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError) as e:
            raise APIConnectionError(f"Failed to connect to Echo TTS: {e}") from e
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if ws is not None and not ws.closed:
                await self._tts._close_ws(ws)
