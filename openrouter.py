#!/usr/bin/env python3
"""Terminal chat with OpenRouter free models. Switch models any time with /model.

Setup:
    pip install openai httpx python-dotenv prompt_toolkit rich
    .env (next to this script):  OPENROUTER_API_KEY=sk-or-...
"""

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from html import escape as html_escape
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import APIError, APIStatusError, OpenAI
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
from rich.table import Table
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

BASE_URL = "https://openrouter.ai/api/v1"
MODELS_URL = f"{BASE_URL}/models"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
AUTO_ROUTER = "openrouter/free"  # OpenRouter's own "pick any working free model"
MAX_RETRIES = 4  # attempts when a request fails before any text arrives
MAX_HISTORY_CHARS = 150_000  # upper limit of history sent per request
UNKNOWN_CTX_CHARS = 60_000  # budget when a model's context size is unknown
CODE_THEME = "monokai"

DEFAULT_SYSTEM = (
    "You are a helpful assistant running in a terminal. Reply in clean Markdown. "
    "Use fenced code blocks with a language tag for code. Keep tables narrow "
    "(few, short columns). Do not use LaTeX; write math in plain text or Unicode."
)

HELP = """\
[bold]Commands[/bold]
  /help              show this help
  /clear             start a new conversation
  /retry             regenerate the last reply (or resend a message that failed)
  /models [filter]   list free models, e.g. /models qwen
  /model             show the current model
  /model <x>         switch model: a number from /models, a full id, or part of a name
  /system <text>     set the system prompt  (/system off, /system reset)
  /think             show or hide the model's reasoning
  /save <file>       save the conversation as a Markdown file
  /exit              quit

[bold]Tips[/bold]
  Switching models keeps the whole conversation. Switch, then /retry, to get a
  second opinion on the same question. Press Tab after "/model " to complete ids.

[bold]Keys[/bold]
  Enter = send    Ctrl+J or Alt+Enter = new line    Ctrl+C = stop reply    Ctrl+D = quit
"""

console = Console()


@dataclass
class State:
    client: OpenAI
    api_key: str = ""
    model: str = DEFAULT_MODEL
    system: str = DEFAULT_SYSTEM
    show_thinking: bool = False
    # user/assistant messages only: {"role", "content"} (+ "model" on assistant turns)
    history: list = field(default_factory=list)
    pending: str | None = None  # last message that got no answer
    models: dict = field(default_factory=dict)  # id -> {"id","ctx","free"} (free only)
    listed: list = field(default_factory=list)  # ids shown by the last /models
    no_system: set = field(default_factory=set)  # models that reject a system role
    last_dropped: int = 0


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
class ProviderError(Exception):
    """An error reported inside the stream by OpenRouter or the upstream provider."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def _field(obj, name):
    """Read a field from either a dict or an object (works across SDK versions)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def error_code(exc):
    if isinstance(exc, APIStatusError):
        return exc.status_code
    candidates = [getattr(exc, "code", None)]
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        candidates.append(body.get("code"))
    for c in candidates:
        try:
            return int(c)
        except (TypeError, ValueError):
            continue
    return None


def error_text(exc):
    """Full lowercase text of an error, for keyword checks."""
    return f"{exc} {getattr(exc, 'body', '')}".lower()


def error_detail(exc):
    """Best human-readable message from an API error."""
    body = getattr(exc, "body", None)
    msg = None
    if isinstance(body, dict):
        msg = body.get("message")
        meta = body.get("metadata")
        if msg and isinstance(meta, dict) and meta.get("raw"):
            msg = f"{msg} ({str(meta['raw'])[:200]})"
    return str(msg or getattr(exc, "message", None) or exc)


def is_daily_limit(exc):
    text = error_text(exc)
    return "per-day" in text or "per day" in text or "daily" in text


def is_retryable(exc):
    if isinstance(exc, ProviderError):
        return error_code(exc) in (None, 408, 429, 500, 502, 503, 504)
    code = error_code(exc)
    if code is not None:
        if code == 429 and is_daily_limit(exc):
            return False  # retrying will not help until the limit resets
        return code in (408, 429, 500, 502, 503, 504)
    return isinstance(
        exc, (APIError, httpx.TransportError, ConnectionError, TimeoutError)
    )


def retry_delay(exc, attempt):
    """Seconds to wait: honour Retry-After if the server sent one, else back off."""
    try:
        header = exc.response.headers.get("retry-after")
        if header:
            return max(1.0, min(float(header), 30.0))
    except Exception:
        pass
    return float(2**attempt)


def describe_error(exc, model):
    code = error_code(exc)
    text = error_text(exc)
    detail = error_detail(exc)
    if code in (401, 403) and "moderation" not in text:
        return "Authentication failed. Check OPENROUTER_API_KEY in your .env file."
    if code == 402:
        return (
            "OpenRouter says payment/credits are required (402). Free models should "
            "not need credits; check that this model really ends in :free, and that "
            "your account balance is not negative."
        )
    if code == 404:
        if "data policy" in text or "guardrail" in text:
            return (
                "No endpoint matches your privacy settings. Free models need "
                "'Free model publication' enabled at https://openrouter.ai/settings/privacy"
            )
        return (
            f"Model '{model}' has no available endpoint (it may have been removed). "
            "Run /models, pick another with /model <number>, then /retry."
        )
    if code == 429:
        if is_daily_limit(exc):
            return (
                "Daily free-model limit reached. It resets daily (UTC); adding a small "
                "credit balance to OpenRouter raises the free limit a lot."
            )
        return "This model is rate-limited right now. Wait a bit, or /model to another one and /retry."
    if code == 400:
        if "context" in text and (
            "length" in text or "maximum" in text or "token" in text
        ):
            return "The conversation is too long for this model. Use /clear or switch to a larger-context model."
        return f"The request was rejected (400): {detail}"
    if code in (500, 502, 503, 504):
        return f"The provider had a problem ({code}). Try /retry, or switch model."
    if isinstance(exc, httpx.TimeoutException) or "timed out" in text:
        return "The request timed out. Try /retry or switch model."
    if isinstance(exc, httpx.TransportError) or "connection" in text:
        return "Could not reach openrouter.ai. Check your internet connection, then /retry."
    if code:
        return f"Error {code}: {detail}"
    return f"{type(exc).__name__}: {detail}"


def split_think(text):
    """Separate inline <think>...</think> reasoning (some models) from the answer."""
    if "<think>" not in text and "</think>" not in text:
        return "", text
    if (
        "<think>" not in text
    ):  # opening tag missing: everything before </think> is reasoning
        head, _, tail = text.partition("</think>")
        return head, tail
    thinking, out, pos = [], [], 0
    while True:
        start = text.find("<think>", pos)
        if start == -1:
            out.append(text[pos:])
            break
        out.append(text[pos:start])
        end = text.find("</think>", start + 7)
        if end == -1:  # still thinking
            thinking.append(text[start + 7 :])
            break
        thinking.append(text[start + 7 : end])
        pos = end + 8
    return "\n".join(thinking), "".join(out)


def window(history, budget):
    """Newest part of the history that fits `budget` chars. Never mutates history."""
    start = 0
    total = sum(len(m["content"]) for m in history)
    while len(history) - start > 1 and total > budget:
        total -= len(history[start]["content"])
        start += 1
    while len(history) - start > 1 and history[start]["role"] != "user":
        total -= len(history[start]["content"])
        start += 1  # must start with a user turn
    return history[start:], start


def budget_for(state):
    ctx = (state.models.get(state.model) or {}).get("ctx") or 0
    if not ctx:
        return UNKNOWN_CTX_CHARS
    # ~3 chars per token, keep ~40% of the window free for the reply and system prompt
    return max(8_000, min(MAX_HISTORY_CHARS, int(ctx * 3 * 0.6)))


def build_messages(state, hist):
    msgs = [{"role": m["role"], "content": m["content"]} for m in hist]
    if state.system:
        if (
            state.model in state.no_system
        ):  # fold system prompt into the first user turn
            for m in msgs:
                if m["role"] == "user":
                    m["content"] = f"{state.system}\n\n---\n\n{m['content']}"
                    break
        else:
            msgs.insert(0, {"role": "system", "content": state.system})
    return msgs


def transcript(state):
    parts = ["# Chat\n"]
    for m in state.history:
        if m["role"] == "user":
            parts.append(f"## You\n\n{m['content']}\n")
        else:
            parts.append(f"## {m.get('model') or 'Model'}\n\n{m['content']}\n")
    return "\n".join(parts)


def short_name(model):
    return model.split("/", 1)[-1].removesuffix(":free")


def fmt_ctx(n):
    if not n:
        return "?"
    return f"{n // 1_000_000}M" if n >= 1_000_000 else f"{round(n / 1000)}k"


# ----------------------------------------------------------------------------
# Model list
# ----------------------------------------------------------------------------
def _price(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_free_models(api_key):
    """Return {id: info} for every free text model currently on OpenRouter."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.get(MODELS_URL, headers=headers, timeout=15.0)
    resp.raise_for_status()
    found = {}
    for m in resp.json().get("data", []):
        mid = m.get("id")
        if not mid:
            continue
        outputs = (m.get("architecture") or {}).get("output_modalities")
        if outputs and set(outputs) != {"text"}:
            continue  # skip music/image/audio generators: chat needs text-only output
        if any(w in mid.lower() for w in ("safety", "guard", "moderation")):
            continue  # classifiers, not chat models
        pricing = m.get("pricing") or {}
        free = mid.endswith(":free") or (
            _price(pricing.get("prompt")) == 0
            and _price(pricing.get("completion")) == 0
        )
        if free:
            found[mid] = {"id": mid, "ctx": m.get("context_length") or 0}
    return found


def refresh_models(state):
    """Update the cached list. Returns True on success; never raises."""
    try:
        state.models = fetch_free_models(state.api_key)
        return True
    except Exception:
        return False


def resolve_model(state, arg):
    """Turn user input into (model_id, error_message)."""
    arg = arg.strip()
    if arg.isdigit():
        i = int(arg) - 1
        if 0 <= i < len(state.listed):
            return state.listed[i], None
        return None, "No such number. Run /models first, then /model <number>."
    if arg in state.models:
        return arg, None
    low = arg.lower()
    if arg.lower() in {k.lower() for k in state.models}:
        return next(k for k in state.models if k.lower() == low), None
    matches = [k for k in state.models if low in k.lower()]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        shown = "\n".join(f"  {escape(k)}" for k in sorted(matches)[:15])
        return (
            None,
            f"'{escape(arg)}' matches several models, be more specific:\n{shown}",
        )
    if "/" in arg:  # full OpenRouter id that is not in the free list
        return arg, "unlisted"
    return None, f"No free model matches '{escape(arg)}'. Try /models."


# ----------------------------------------------------------------------------
# Live streaming view (shows only the tail so it can never overflow the screen)
# ----------------------------------------------------------------------------
class StreamView:
    def __init__(self, show_thinking):
        self.content = ""  # raw text from the model
        self.reasoning = ""  # reasoning sent in the separate reasoning field
        self.status = ""
        self.show_thinking = show_thinking
        self._spinner = Spinner("dots", style="cyan")
        self._cache_key = None
        self._cache_lines = []

    @property
    def parts(self):
        inline, answer = split_think(self.content)
        return (self.reasoning + inline), answer

    @property
    def thinking(self):
        return self.parts[0]

    @property
    def answer(self):
        return self.parts[1]

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
        thinking, answer = self.parts
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
def stream_reply(state, view, hist):
    """Stream one reply into `view`. Retries only if nothing has arrived yet."""
    attempt = 0
    while True:
        attempt += 1
        try:
            stream = state.client.chat.completions.create(
                model=state.model,
                messages=build_messages(state, hist),
                stream=True,
            )
            for chunk in stream:
                err = _field(chunk, "error")
                if err:
                    raise ProviderError(
                        str(_field(err, "message") or err), _field(err, "code")
                    )
                choices = _field(chunk, "choices") or []
                if not choices:  # e.g. the final usage-only chunk
                    continue
                delta = _field(choices[0], "delta")
                view.reasoning += (
                    _field(delta, "reasoning")
                    or _field(delta, "reasoning_content")
                    or ""
                )
                view.content += _field(delta, "content") or ""
                view.status = ""
            return
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            got_text = bool(view.content or view.reasoning)

            # Some free models reject a system/developer message: fold it into the
            # first user turn instead, remember that for this model, and try again.
            if (
                not got_text
                and state.system
                and state.model not in state.no_system
                and error_code(exc) == 400
                and any(
                    k in error_text(exc)
                    for k in (
                        "developer instruction",
                        "system instruction",
                        "system message",
                        "system prompt",
                        "system role",
                        "instruction is not enabled",
                    )
                )
            ):
                state.no_system.add(state.model)
                view.status = "Model rejects system prompts, retrying without one…"
                attempt -= 1
                continue

            if got_text or attempt >= MAX_RETRIES or not is_retryable(exc):
                raise
            delay = retry_delay(exc, attempt)
            view.status = (
                f"Connection problem, retrying in {delay:.0f}s "
                f"({attempt}/{MAX_RETRIES - 1})…"
            )
            time.sleep(delay)


def run_turn(state):
    """Generate a reply to the last user message in state.history."""
    hist, dropped = window(state.history, budget_for(state))
    if dropped and dropped != state.last_dropped:
        console.print(
            f"[dim]Sending only the latest {len(hist)} message(s) to stay within "
            f"{escape(short_name(state.model))}'s context size "
            f"({dropped} older kept locally).[/dim]"
        )
    state.last_dropped = dropped

    view = StreamView(state.show_thinking)
    error, interrupted = None, False
    started = time.monotonic()
    try:
        with Live(
            view,
            console=console,
            transient=True,
            refresh_per_second=8,
            vertical_overflow="crop",
        ):
            stream_reply(state, view, hist)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        error = exc
    elapsed = time.monotonic() - started

    thinking, answer = view.parts
    thinking, answer = thinking.strip(), answer.strip()

    if state.show_thinking and thinking:
        console.print(Text("Thinking:", style="dim bold"))
        console.print(Text(thinking, style="dim italic"))
        console.print()
    if answer:
        console.print(Markdown(answer, code_theme=CODE_THEME))

    if interrupted:
        console.print("[yellow]Stopped.[/yellow]")
    elif error:
        console.print(f"[red]{escape(describe_error(error, state.model))}[/red]")
    elif not answer:
        console.print(
            "[yellow]The model returned an empty response. Try /retry or switch "
            "model with /model.[/yellow]"
        )
    if answer:
        console.print(
            Text(f"— {short_name(state.model)} · {elapsed:.1f}s", style="dim")
        )
    console.print()

    if answer:
        state.history.append(
            {"role": "assistant", "content": answer, "model": state.model}
        )
        state.pending = None
    else:
        state.pending = state.history.pop()["content"]  # keep it so /retry can resend


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------
def show_models(state, filt):
    ok = refresh_models(state)
    if not ok and not state.models:
        console.print(
            "[red]Could not load the model list. Check your connection.[/red]\n"
        )
        return
    if not ok:
        console.print("[dim](could not refresh, showing the last known list)[/dim]")
    low = filt.lower()
    ids = sorted(
        (k for k in state.models if low in k.lower()),
        key=lambda k: (k != AUTO_ROUTER, k.lower()),  # auto-router first
    )
    if not ids:
        console.print(f"[yellow]No free models match '{escape(filt)}'.[/yellow]\n")
        return
    state.listed = ids
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Model")
    table.add_column("Context", justify="right", style="dim")
    for i, mid in enumerate(ids, 1):
        current = mid == state.model
        table.add_row(
            str(i),
            Text(
                mid + ("  ◀ current" if current else ""),
                style="bold cyan" if current else "",
            ),
            fmt_ctx(state.models[mid]["ctx"]),
        )
    console.print(table)
    console.print(
        f"\n[dim]{len(ids)} free model(s). Switch with /model <number or name>. "
        f"'{AUTO_ROUTER}' lets OpenRouter pick any working free model.[/dim]\n"
    )


def switch_model(state, arg):
    if not arg:
        console.print(f"Current model: [bold cyan]{escape(state.model)}[/bold cyan]\n")
        return
    if not state.models:
        refresh_models(state)  # try once if the startup fetch failed
    model, note = resolve_model(state, arg)
    if model is None:
        console.print(f"[yellow]{note}[/yellow]\n")
        return
    state.model = model
    state.last_dropped = 0
    console.print(f"Model set to [bold cyan]{escape(model)}[/bold cyan]")
    if note == "unlisted":
        console.print(
            "[yellow]This id is not in the free list; it may be paid or no longer "
            "available. If the next message fails, try /models.[/yellow]"
        )
    elif state.history:
        console.print("[dim]The conversation so far carries over to this model.[/dim]")
    console.print()


def handle_command(state, cmd, arg):
    """Returns 'quit', 'run' (generate a reply), or 'done'."""
    if cmd in ("/exit", "/quit"):
        return "quit"

    if cmd == "/help":
        console.print(HELP)

    elif cmd == "/clear":
        state.history.clear()
        state.pending = None
        state.last_dropped = 0
        console.print("[yellow]Conversation cleared.[/yellow]\n")

    elif cmd == "/models":
        show_models(state, arg)

    elif cmd == "/model":
        switch_model(state, arg)

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
COMMANDS = [
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


def build_session(state):
    kb = KeyBindings()

    @kb.add("enter")
    def _send(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")  # Alt+Enter
    @kb.add("c-j")  # Ctrl+J
    def _newline(event):
        event.current_buffer.insert_text("\n")

    def words():  # evaluated on every Tab, so it follows the live model list
        return COMMANDS + [f"/model {m}" for m in sorted(state.models)]

    return PromptSession(
        multiline=True,
        key_bindings=kb,
        prompt_continuation=lambda width, line_number, wrap_count: "... ",
        completer=WordCompleter(words, sentence=True),
        complete_while_typing=False,  # press Tab to complete commands
    )


def main():
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        console.print("This chat needs an interactive terminal.")
        sys.exit(1)

    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip().strip("\"'")
    if not api_key:
        console.print(
            "[red]OPENROUTER_API_KEY not found.[/red] Put it in a .env file "
            "next to this script:\n  OPENROUTER_API_KEY=your_key_here"
        )
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=httpx.Timeout(180.0, connect=15.0),
        max_retries=0,  # the script does its own retrying
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Terminal Chat",
        },
    )
    state = State(client=client, api_key=api_key)
    session = build_session(state)

    def prompt():
        return HTML(
            f"<ansigreen><b>You</b></ansigreen> "
            f"<ansibrightblack>[{html_escape(short_name(state.model))}]</ansibrightblack>"
            f"<ansigreen><b>: </b></ansigreen>"
        )

    loaded = refresh_models(state)
    if loaded and state.model not in state.models:
        # The default (or OPENROUTER_MODEL) is no longer free/live: pick a working one.
        old = state.model
        state.model = (
            AUTO_ROUTER if AUTO_ROUTER in state.models else sorted(state.models)[0]
        )
        console.print(
            f"[yellow]'{escape(old)}' is not available on the free list right now, "
            f"using {escape(state.model)} instead.[/yellow]"
        )
    console.print(
        f"Chatting with [bold cyan]{escape(state.model)}[/bold cyan]. "
        "Type [bold]/help[/bold] for commands, [bold]/models[/bold] to browse."
    )
    if loaded:
        console.print(f"[dim]{len(state.models)} free models available.[/dim]")
    else:
        console.print(
            "[dim]Could not load the model list yet; /models will retry.[/dim]"
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
            try:
                result = handle_command(state, cmd, arg.strip())
            except KeyboardInterrupt:
                console.print("[yellow]Cancelled.[/yellow]\n")
                continue
            if result == "quit":
                console.print("Bye!")
                break
            if result == "run":
                run_turn(state)
            continue

        state.pending = None  # a new message replaces any failed one
        state.history.append({"role": "user", "content": text})
        run_turn(state)


if __name__ == "__main__":
    main()
