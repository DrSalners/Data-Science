#!/usr/bin/env bash
paths_to_lint=${@}
ruff check $paths_to_lint
black $paths_to_lint
