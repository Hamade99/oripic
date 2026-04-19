# oripic

A Discord bot for rendering origami crease patterns (`.cp` files) directly in chat. Upload a `.cp` file and oripic replies with both the crease pattern and the computed folded state as images.

Built by ani and aspen. `≽^•⩊•^≼`

## What it does

- **Crease-pattern rendering** — parses `.cp` line segments and draws them with PIL, using conventional colors (black borders, red mountains, blue valleys, green auxiliary).
- **Folded-state rendering** — computes the flat-folded geometry using a pure-Python port of the flat-folder constraint solver (taco-taco, taco-tortilla, tortilla-tortilla, and transitivity constraints).
- **Image → CP conversion** (`!plumb`) — turns any image into a `.cp` file via edge detection and contour tracing, with optional tonal crosshatching.
- **History search** — scans recent messages for `.cp` attachments by filename, with fuzzy matching fallback (e.g. `momoko` → `momoka.cp`).

Everything runs in-memory. No files are written to disk for normal paths.

## Setup

Requires Python 3.10+.

```bash
chmod +x setup.sh
./setup.sh
```

This creates a `.venv`, installs requirements, and creates an empty `.env` file. Open `.env` and add your bot token:

```
DISCORD_TOKEN=your_token_here
```

Then run the bot:

```bash
.venv/bin/python bot.py
```

The bot needs **Read Message History** and **Message Content Intent** enabled in the Discord Developer Portal.

## Usage

**Trigger:** mention the bot at the start of the message, then a command with a `!` prefix.

```
@oripic !help
```

Or just upload a `.cp` file — the bot auto-renders both views without any command.

### Commands

| Command | What it does |
|---|---|
| `!show "query"` | Crease-pattern render of the most recent matching `.cp` in channel history |
| `!fold "query"` | Folded-state render |
| `!show fold "query"` | Both renders (also `!fold show`, `!show !fold`, `!show!fold`, etc.) |
| `!vis "query"` | Alias for `!show fold` |
| `!find "query" [time] [N]` | List up to N matches in a time window |
| `!plumb` | Convert an attached (or replied-to) image into a `.cp` file |
| `!plumbbobfans <username>` | Grant another user `!plumb` access (session-only) |
| `!help` | Show in-channel help |

### Reply shortcut

Reply to a message containing a `.cp` attachment and run `!show`, `!fold`, `!vis`, etc. with **no query** — the bot uses that attachment directly without searching history.

### Search behavior

- `!show` / `!fold` / `!vis` scan up to 500 recent messages for a `.cp` whose filename contains every whitespace-separated token (case-insensitive). Filename metadata only — no downloads during the scan.
- If no exact match exists, a fuzzy fallback can still match close names (threshold ~0.75 similarity).
- `!find` searches by time window instead of message count. Supports `7d`, `12h`, `3w`, `2mo`, `1y`, `4 days`, `6 months`. Defaults: 30 days, 5 results. Max 10 results, max 8000 messages scanned.

## Project layout

```
bot.py              Discord bot — command dispatch, history search, replies
oripic.py           Crease-pattern renderer (PIL)
cpsketch.py           Image → .cp converter (OpenCV)
fold_impl/          Folded-state pipeline (Python port of flat-folder)
  ├── render.py       Top-level: CP text → PNG bytes
  ├── io.py           CP / FOLD file parsing
  ├── conversion.py   Geometry (line arrangement, face graph, constraints)
  ├── solver_mod.py   Constraint solver (propagation + backtracking)
  ├── constraints.py  Flat-fold constraint tables
  ├── math2d.py       2D geometry primitives
  ├── avl.py          AVL tree for line sweep
  └── note.py         Logging/timing helpers
setup.sh            One-shot venv + requirements setup
requirements.txt    Python dependencies
```

## CLI tools

Both renderers also work standalone:

```bash
# Crease pattern → PNG (stdin to stdout)
python oripic.py < input.cp > output.png

# Image → .cp
python cpsketch.py input.jpg                   # defaults (crosshatch shading)
python cpsketch.py input.jpg --shade-levels 0  # edges only
python cpsketch.py input.jpg --epsilon-frac 0.003  # finer polylines
```

## Notes on `!plumb`

`!plumb` and `!plumbbobfans` are gated by an allow-list of Discord usernames. The seed list is hardcoded in `bot.py` (`PLUMB_OWNER_USERNAMES`); anyone in that set can invite new users at runtime via `!plumbbobfans <username>`. Grants are **session-only** — restarting the bot wipes them back to the hardcoded owners. This is deliberate: nothing touches disk, so nothing leaks into the repo.

## How folding works (short version)

its flatfolder


