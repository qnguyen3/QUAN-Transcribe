"""Exercise the live WebSocket path against a running server.

Feeds 16 kHz mono float32 frames in the same 0.48 s steps the browser client
uses and prints every JSON event, so the live session surface is tested against
the real model rather than a stub.

    python scripts/ws_probe.py --url ws://127.0.0.1:8124/v1/audio/transcriptions/stream
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct

import websockets

SAMPLE_RATE = 16000


def tone(seconds: float, freq: float = 220.0) -> list[float]:
    return [
        0.25 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)
        for i in range(int(seconds * SAMPLE_RATE))
    ]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8124/v1/audio/transcriptions/stream",
    )
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    step = int(0.48 * SAMPLE_RATE)
    samples = tone(args.seconds)

    async with websockets.connect(args.url) as ws:
        print("created:", json.loads(await ws.recv())["type"])
        # Send every step first: the server decodes each frame on its model
        # thread and queues the events, so a tight recv between frames would
        # just time out on CPU. Draining afterwards reads them all in order.
        for offset in range(0, len(samples), step):
            block = samples[offset : offset + step]
            await ws.send(struct.pack(f"<{len(block)}f", *block))
        await ws.send("stop")
        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=60)
            except (TimeoutError, asyncio.TimeoutError):
                print("  (timed out waiting for the server)")
                break
            event = json.loads(message)
            print(" ", event["type"], repr(event.get("text", ""))[:48])
            if event["type"] == "session.closed":
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
