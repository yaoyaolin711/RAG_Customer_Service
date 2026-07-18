import time
import functools
from typing import Callable, Any, Optional, List, Type
from typing import TypeVar, Callable


T = TypeVar('T')


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Optional[List[Type[Exception]]] = None
):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if exceptions is None:
                exceptions_to_catch = [Exception]
            else:
                exceptions_to_catch = exceptions

            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except tuple(exceptions_to_catch) as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            import asyncio
            if exceptions is None:
                exceptions_to_catch = [Exception]
            else:
                exceptions_to_catch = exceptions

            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except tuple(exceptions_to_catch) as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    return retry(max_attempts=max_attempts, delay=delay)


class RetryConfig:
    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        timeout: Optional[float] = None
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.timeout = timeout


import asyncio
