# Seam Carving with Dynamic Programming

Content-aware image resizing (Avidan & Shamir, 2007), Assignment 2 Question 1.
Removes low-energy vertical and horizontal seams, found with dynamic programming,
until the image reaches the requested size.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # macOS / Linux
```

## Tests

```bash
.venv/Scripts/python -m pytest -q
```
