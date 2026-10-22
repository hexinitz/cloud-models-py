#!/usr/bin/env python3
"""Terminal chat with Ollama cloud models (ollama.com)."""

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from ollama import Client, ResponseError
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.segment import Segment
from rich.spinner import Spinner
from rich.text import Text

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):  # avoid encoding crashes on Windows
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv(Path(__file__).resolve().with_name(".env"))  # .env next to this script
load_dotenv()  # fallback: .env in current folder

HOST = "https://ollama.com"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
MAX_RETRIES = 3  # attempts when a request fails before any text arrives
MAX_HISTORY_CHARS = 150_000  # oldest messages are dropped beyond this
CODE_THEME = "monokai"

DEFAULT_SYSTEM = (
    "You are a helpful assistant running in a terminal. Reply in clean Markdown. "
    "Use fenced code blocks with a language tag for code. Keep tables narrow "
    "(few, short columns). Do not use LaTeX; write math in plain text or Unicode."
)

HELP = """\
[bold]Commands[/bold]
  /help            show this help
  /clear           start a new conversation
  /retry           regenerate the last reply (or resend a message that failed)
  /models          list available cloud models
  /model <name>    switch model (plain name, e.g. gpt-oss:120b)
  /system <text>   set the system prompt  (/system off, /system reset)
  /think           show or hide the model's reasoning
  /save <file>     save the conversation as a Markdown file
  /exit            quit

[bold]Keys[/bold]
  Enter = send    Ctrl+J or Alt+Enter = new line    Ctrl+C = stop reply    Ctrl+D = quit
"""

console = Console()


@dataclass
class State:
    client: Client
    model: str = DEFAULT_MODEL
    system: str = DEFAULT_SYSTEM
    show_thinking: bool = False
    history: list = field(default_factory=list)  # user/assistant messages only
    pending: str | None = None  # last message that got no answer


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _field(obj, name):
    """Read a field from either a dict or an object (works across ollama versions)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def is_retryable(exc):
    if isinstance(exc, ResponseError):
        return exc.status_code in (408, 429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError))


def describe_error(exc, model):
    if isinstance(exc, ResponseError):
        code = exc.status_code
        if code in (401, 403):
            return "Authentication failed. Check OLLAMA_API_KEY in your .env file."
        if code == 404:
            return f"Model '{model}' was not found. Try /models, and use the plain name (no -cloud suffix)."
        if code == 429:
            return "Rate or usage limit reached. Wait a while and try again."
        if code in (500, 502, 503, 504):
            return f"The server had a problem ({code}). Try /retry in a moment."
        return f"Server error ({code}): {exc}"
    if isinstance(exc, httpx.TimeoutException):
        return "The request timed out. Try /retry."
    if isinstance(exc, httpx.TransportError):
        return (
            "Could not reach ollama.com. Check your internet connection, then /retry."
        )
    return f"{type(exc).__name__}: {exc}"


def trim_history(history):
    """Drop the oldest messages if the conversation is too large. Returns count dropped."""
    dropped = 0
    while (
        len(history) > 1 and sum(len(m["content"]) for m in history) > MAX_HISTORY_CHARS
    ):
        history.pop(0)
        dropped += 1
    while (
        len(history) > 1 and history[0]["role"] != "user"
    ):  # must start with a user turn
        history.pop(0)
        dropped += 1
    return dropped


def transcript(state):
    parts = [f"# Chat with {state.model}\n"]
    for m in state.history:
        who = "You" if m["role"] == "user" else "Model"
        parts.append(f"## {who}\n\n{m['content']}\n")
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Live streaming view (shows only the tail so it can never overflow the screen)
# ----------------------------------------------------------------------------
class StreamView:
    def __init__(self, show_thinking):
        self.answer = ""
        self.thinking = ""
        self.status = ""
        self.show_thinking = show_thinking
        self._spinner = Spinner("dots", style="cyan")
        self._cache_key = None
        self._cache_lines = []

    def _tail(self, console, options, kind, text, max_lines):
        key = (kind, text, options.max_width)
        if key != self._cache_key:  # only re-render when something changed
            if kind == "answer":
                renderable = Markdown(text, code_theme=CODE_THEME)
            else:
                renderable = Text(text, style="dim italic")
            self._cache_lines = console.render_lines(
                renderable, options.update(height=None), pad=False
            )
            self._cache_key = key
        return self._cache_lines[-max_lines:]

    def __rich_console__(self, console, options):
        answer, thinking = self.answer, self.thinking
        max_lines = max(5, console.size.height - 3)

        if answer.strip():
            kind, text = "answer", answer
        elif self.show_thinking and thinking.strip():
            kind, text = "thinking", thinking
        else:
            label = self.status or (
                "Thinking…" if thinking else "Waiting for response…"
            )
            self._spinner.update(text=Text(" " + label, style="dim"))
            yield self._spinner
            return

        for i, line in enumerate(
            self._tail(console, options, kind, text.strip(), max_lines)
        ):
            if i:
                yield Segment.line()
            yield from line


# ----------------------------------------------------------------------------
# Talking to the model
# ----------------------------------------------------------------------------
def stream_reply(state, view):
    """Stream one reply into `view`. Retries only if nothing has arrived yet."""
    messages = (
        [{"role": "system", "content": state.system}] if state.system else []
    ) + state.history
    attempt = 0
    while True:
        attempt += 1
        try:
            for chunk in state.client.chat(
                model=state.model, messages=messages, stream=True
            ):
                msg = _field(chunk, "message")
                view.thinking += _field(msg, "thinking") or ""
                view.answer += _field(msg, "content") or ""
                view.status = ""
            return
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if (
                view.answer
                or view.thinking
                or attempt >= MAX_RETRIES
                or not is_retryable(exc)
            ):
                raise
            delay = 2**attempt
            view.status = f"Connection problem, retrying in {delay}s ({attempt}/{MAX_RETRIES - 1})…"
            time.sleep(delay)


def run_turn(state):
    """Generate a reply to the last user message in state.history."""
    dropped = trim_history(state.history)
    if dropped:
        console.print(
            f"[dim]Dropped {dropped} old message(s) to stay within the context budget.[/dim]"
        )

    view = StreamView(state.show_thinking)
    error, interrupted = None, False
    try:
        with Live(
            view,
            console=console,
            transient=True,
            refresh_per_second=8,
            vertical_overflow="crop",
        ):
            stream_reply(state, view)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        error = exc

    answer = view.answer.strip()

    if state.show_thinking and view.thinking.strip():
        console.print(Text("Thinking:", style="dim bold"))
        console.print(Text(view.thinking.strip(), style="dim italic"))
        console.print()
    if answer:
        console.print(Markdown(answer, code_theme=CODE_THEME))

    if interrupted:
        console.print("[yellow]Stopped.[/yellow]")
    elif error:
        console.print(f"[red]{escape(describe_error(error, state.model))}[/red]")
    elif not answer:
        console.print("[yellow]The model returned an empty response.[/yellow]")
    console.print()

    if answer:
        state.history.append({"role": "assistant", "content": answer})
        state.pending = None
    else:
        state.pending = state.history.pop()["content"]  # keep it so /retry can resend


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------
def handle_command(state, cmd, arg):
    """Returns 'quit', 'run' (generate a reply), or 'done'."""
    if cmd in ("/exit", "/quit"):
        return "quit"

    if cmd == "/help":
        console.print(HELP)

    elif cmd == "/clear":
        state.history.clear()
        state.pending = None
        console.print("[yellow]Conversation cleared.[/yellow]\n")

    elif cmd == "/models":
        try:
            names = sorted(
                _field(m, "model") or _field(m, "name") or "?"
                for m in (_field(state.client.list(), "models") or [])
            )
            console.print(
                "\n".join(f"  {escape(n)}" for n in names) or "  (none found)"
            )
        except Exception as exc:
            console.print(f"[red]{escape(describe_error(exc, state.model))}[/red]")
        console.print()

    elif cmd == "/model":
        if arg:
            state.model = arg
            console.print(f"Model set to [bold cyan]{escape(arg)}[/bold cyan]\n")
        else:
            console.print(
                f"Current model: [bold cyan]{escape(state.model)}[/bold cyan]\n"
            )

    elif cmd == "/system":
        if not arg:
            console.print(
                Text(f"System prompt: {state.system or '(none)'}", style="dim")
            )
        elif arg.lower() == "off":
            state.system = ""
            console.print("System prompt disabled.")
        elif arg.lower() == "reset":
            state.system = DEFAULT_SYSTEM
            console.print("System prompt reset to default.")
        else:
            state.system = arg
            console.print("System prompt updated.")
        console.print()

    elif cmd == "/think":
        state.show_thinking = not state.show_thinking
        console.print(
            f"Showing reasoning: [bold]{'on' if state.show_thinking else 'off'}[/bold]\n"
        )

    elif cmd == "/save":
        if not state.history:
            console.print("[yellow]Nothing to save yet.[/yellow]\n")
        else:
            path = (
                Path(arg).expanduser()
                if arg
                else Path(f"chat_{datetime.now():%Y%m%d_%H%M%S}.md")
            )
            try:
                path.write_text(transcript(state), encoding="utf-8")
                console.print(f"Saved to [bold]{escape(str(path.resolve()))}[/bold]\n")
            except OSError as exc:
                console.print(f"[red]Could not save: {escape(str(exc))}[/red]\n")

    elif cmd == "/retry":
        if state.pending is not None:
            state.history.append({"role": "user", "content": state.pending})
            state.pending = None
        elif state.history and state.history[-1]["role"] == "assistant":
            state.history.pop()
        if not state.history or state.history[-1]["role"] != "user":
            console.print("[yellow]Nothing to retry.[/yellow]\n")
        else:
            return "run"

    else:
        console.print(
            f"[yellow]Unknown command {escape(cmd)}. Type /help. "
            "(To send text that starts with '/', put a space before it.)[/yellow]\n"
        )
    return "done"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def build_session():
    kb = KeyBindings()

    @kb.add("enter")
    def _send(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")  # Alt+Enter
    @kb.add("c-j")  # Ctrl+J
    def _newline(event):
        event.current_buffer.insert_text("\n")

    commands = [
        "/help",
        "/clear",
        "/retry",
        "/models",
        "/model",
        "/system",
        "/think",
        "/save",
        "/exit",
        "/quit",
    ]
    return PromptSession(
        multiline=True,
        key_bindings=kb,
        prompt_continuation=lambda width, line_number, wrap_count: "... ",
        completer=WordCompleter(commands, sentence=True),
        complete_while_typing=False,  # press Tab to complete commands
    )


def main():
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        console.print("This chat needs an interactive terminal.")
        sys.exit(1)

    api_key = (os.getenv("OLLAMA_API_KEY") or "").strip().strip("\"'")
    if not api_key:
        console.print(
            "[red]OLLAMA_API_KEY not found.[/red] Put it in a .env file "
            "next to this script:\n  OLLAMA_API_KEY=your_key_here"
        )
        sys.exit(1)

    client = Client(
        host=HOST,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(120.0, connect=15.0),
    )
    state = State(client=client)
    session = build_session()
    prompt = HTML("<ansigreen><b>You: </b></ansigreen>")

    console.print(
        f"Chatting with [bold cyan]{escape(state.model)}[/bold cyan]. "
        "Type [bold]/help[/bold] for commands."
    )
    console.print(
        "[dim]Enter = send | Ctrl+J or Alt+Enter = new line | Ctrl+D = quit[/dim]\n"
    )

    interrupts = 0
    while True:
        try:
            raw = session.prompt(prompt)
            interrupts = 0
        except KeyboardInterrupt:
            interrupts += 1
            if interrupts >= 2:
                console.print("Bye!")
                break
            console.print(
                "[dim]Input cleared. Press Ctrl+C again (or Ctrl+D) to quit.[/dim]"
            )
            continue
        except EOFError:
            console.print("Bye!")
            break

        if not raw.strip():
            continue
        text = raw.strip("\r\n").rstrip()  # keep leading indentation of pasted code

        is_command = "\n" not in text and (
            text.startswith("/") or text.lower() in ("exit", "quit", "clear")
        )
        if is_command:
            cmd, _, arg = text.partition(" ")
            cmd = cmd.lower()
            if not cmd.startswith("/"):
                cmd = "/" + cmd
            result = handle_command(state, cmd, arg.strip())
            if result == "quit":
                console.print("Bye!")
                break
            if result == "run":
                run_turn(state)
            continue

        state.history.append({"role": "user", "content": text})
        run_turn(state)


if __name__ == "__main__":
    main()
