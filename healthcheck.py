"""Docker HEALTHCHECK: healthy once the server answers a Wyoming Describe (i.e. the model is loaded)."""

import asyncio
import os
import sys

from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info


async def main() -> None:
    async with AsyncTcpClient("127.0.0.1", int(os.getenv("WHISTLE_PORT", "10300"))) as client:
        await client.write_event(Describe().event())
        event = await asyncio.wait_for(client.read_event(), timeout=5)
        if event is None or not Info.is_type(event.type):
            raise RuntimeError(f"unexpected answer: {event}")


try:
    asyncio.run(asyncio.wait_for(main(), timeout=8))
except Exception as err:
    print(f"unhealthy: {err!r}")
    sys.exit(1)
