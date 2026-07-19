"""Exponential-backoff retry helper (same pattern as research-agent)."""

import time
from rich.console import Console

console = Console()


def retry_with_backoff(func, max_retries: int = 3, tracer=None):
    """Call *func* (a zero-arg callable) with exponential-backoff retries.

    Each retry waits 1 s, then 2 s, then 4 s, etc.
    Prints a warning via ``rich`` on each retry.

    Args:
        func: A zero-arg callable.
        max_retries: Max attempts before giving up.
        tracer: Optional ``Tracer`` instance to log retry events.

    Returns:
        The return value of *func* on success, *or* a string error
        message if all retries are exhausted.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt - 1)
                console.print(
                    f"[yellow]Retry {attempt}/{max_retries} after error: {e}[/]"
                )
                console.print(f"[dim]Waiting {wait}s ...[/]")
                if tracer:
                    tracer.log_retry(attempt, max_retries, str(e))
                time.sleep(wait)
            else:
                console.print(
                    f"[red]All {max_retries} attempts failed. Last error: {e}[/]"
                )
                if tracer:
                    tracer.log_error(
                        f"All {max_retries} attempts failed. Last error: {e}"
                    )
                return (
                    f"The operation failed after {max_retries} attempts. "
                    f"Last error: {e}. Please try a different approach."
                )
