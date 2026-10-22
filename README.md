# Cloud Models Py

A personal Python project for experimenting with cloud AI models, APIs, and a terminal user interface (TUI).

This repository contains Python scripts for trying different models and providers, comparing approaches, and learning how to work with AI APIs in Python. I also used AI assistance to help build the project's TUI, exploring how to make model experimentation accessible through a terminal-based interface.

The project is part of my journey toward understanding Python, LLMs, and building AI-powered applications.

## AI Assistance

I used AI as a development aid while building the project's terminal user interface (TUI). This gave me an opportunity to explore AI-assisted development and learn how a terminal-based interface can bring different model experiments together.

I treat AI as a tool to support my learning and development, rather than as a replacement for understanding the code.

## Project Structure

```text
cloud-models-py/
├── src/
├── gemma4_31b.py
├── gpt_oss_120b-gq.py
├── gpt_oss_120b.py
├── gpt_oss_20b.py
├── openrouter.py
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

## Requirements

- Python version specified in `.python-version`
- Either [uv](https://docs.astral.sh/uv/) or `pip`
- API keys for the providers used by the scripts

## Setup

Clone the repository:

```bash
git clone https://github.com/hexinitz/cloud-models-py.git
cd cloud-models-py
```

### Option 1: Install with uv (recommended)

```bash
uv sync
```

This uses `pyproject.toml` and `uv.lock` to manage the project environment and dependencies.

### Option 2: Install with pip

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Some scripts require API credentials. Create a local `.env` file in the project root and add the environment variables expected by your code.

Example:

```dotenv
OPENROUTER_API_KEY=your_api_key_here
GROQ_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here
```

These variable names are examples; check the scripts for the actual names they use.

**Security:** Never commit your `.env` file or expose API keys in source code. Keep `.env` excluded through `.gitignore`.

## Running the Scripts

Run scripts from the project root with `uv`:

```bash
uv run python openrouter.py
uv run python gemma4_31b.py
uv run python gpt_oss_20b.py
uv run python gpt_oss_120b.py
uv run python gpt_oss_120b-gq.py
```

If you use the `pip` setup, activate `.venv` first and replace `uv run python` with `python`.

Check each script for its specific configuration and execution requirements.

## Learning Goals

Through this project, I'm exploring:

- Working with cloud AI APIs in Python.
- Experimenting with different models and providers.
- Managing API credentials and project dependencies.
- Building terminal-based interfaces.
- Using AI tools to assist with software development.
- Developing a better understanding of LLMs and AI-powered applications.

## Project Status

This is a personal learning and experimentation project. The code, model integrations, and interface may evolve as I continue learning Python and exploring AI development.
