from __future__ import annotations

import asyncio
from typing import Callable, TypeVar

T = TypeVar("T")


async def to_thread(func: Callable[..., T], *args, **kwargs) -> T:
    """Compatibility wrapper for asyncio.to_thread.

    Python 3.11+ has asyncio.to_thread; for older versions (3.7~3.8) we fallback to
    loop.run_in_executor.
    """
    to_thread_fn = getattr(asyncio, "to_thread", None)
    if callable(to_thread_fn):
        return await to_thread_fn(func, *args, **kwargs)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
