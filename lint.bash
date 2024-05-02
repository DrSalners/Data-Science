#!/usr/bin/env bash
paths_to_lint=${@}
source .venv/bin/activate
ruff check $paths_to_lint
black $paths_to_lint
