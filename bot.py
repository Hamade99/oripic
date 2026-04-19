"""
oripic — Discord bot entry point.

Python is the only runtime you need:
 * crease-pattern render: `oripic.py` (PIL),
 * folded-state render: `fold_impl/` (Python port of the former flat-folder pipeline).

Triggers
--------
Uploading a `.cp` file renders both views (crease pattern + folded state).

**Trigger:** Mention the bot at the **start** of the message, then a command with
a **`!` prefix** (e.g. `@oripic !help`).

**Search behavior:** `!show` / `!fold` / `!vis` with a **query** scan up to
`HISTORY_SCAN_LIMIT` recent messages for a `.cp` whose **filename** contains
every whitespace-separated word (case-insensitive; metadata only until a match).
`!find` searches by **time window** (and a large per-search message cap in
busy channels). **Uploads** and **reply-without-query** use the attachment
directly — no filename scan.

If nothing exact matches a search, a fuzzy fallback can still match close names
(e.g. `momoko` → `momoka.cp`):

    @oripic !show "query"        → crease-pattern render
    @oripic !fold "query"        → folded-state render
    @oripic !show fold "query"   → crease-pattern + folded state
    @oripic !show !fold "query"  → same (`!` on the second word is optional)
    @oripic !fold show "query"   → same (order of `show` / `fold` doesn't matter)
    @oripic !fold !show "query"  → same
    @oripic !show!fold "query"   → same (no space between words; common on mobile)
    @oripic !find "query" [time] [N] → list matches in a time window
                                    (e.g. 7d, 2h, 3mo, 1y); defaults 30d / 5;
                                    max 10; time and N can be either order;
                                    very active channels hit a per-search message cap
    @oripic !help                → usage in channel

`!vis` is an alias for `!show fold`.

**Reply shortcut:** reply to a message that has a `.cp` and run **`!show`**,
**`!fold`**, **`!show fold`** / **`!show !fold`** / **`!fold show`** / **`!fold !show`**,
or **`!vis`**, with **no query** after **@oripic** — uses that attachment (no history search).

All data flow is in-memory end to end: attachment bytes into buffers, both
renderers return PNG bytes. Nothing is written to disk for normal paths.

If you need to add a feature, add it here.
"""
from __future__ import annotations

import asyncio
import difflib
import io
import json
import logging
import os
import re
import shlex
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import discord
from dotenv import load_dotenv

import oripic
from cpshit import bytes_to_cp
from fold_impl import render_cp_folded_png

load_dotenv()
TOKEN = os.environ["DISCORD_TOKEN"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
log = logging.getLogger("oripic")

_CAT = "≽^•⩊•^≼"

# How many messages back to scan. The scan inspects only the attachment
# metadata that Discord already hands us with each message — no downloads
# happen during the scan — so this can be set high without much cost.
HISTORY_SCAN_LIMIT = 500

# Cap / default for `!find`. Ten lines fits comfortably in an embed
# description even with long filenames and two snowflakes per jump URL.
FIND_MAX = 10
FIND_DEFAULT = 5

# Default lookback when `!find` omits a time range (plain integer = max
# results only, same as before).
DEFAULT_FIND_LOOKBACK = timedelta(days=30)

# Safety cap: stop scanning after this many messages even inside the time window
# (very active channels).
FIND_SCAN_MAX_MESSAGES = 8000

# Regex: quantity + unit (longer unit names first). Month is `mo` / `month`…
# — not bare `m` (avoids ambiguity with minutes if we add those later).
_FIND_LOOKBACK_RE = re.compile(
    r"^\s*(\d+)\s*"
    r"(hours?|hrs?|h|days?|d|weeks?|wks?|w|months?|mon\b|mo\b|years?|yrs?|y)\s*$",
    re.IGNORECASE,
)

# Similarity cutoff for the fuzzy-fallback match. Only kicks in when no
# exact-substring matches exist. 0.75 catches a single-char typo in a
# 5-6 char word (momoko↔momoka, crane↔grane) without being so loose
# that it matches unrelated filenames.
FUZZY_THRESHOLD = 0.75

_VALID_SUBCOMMANDS = frozenset({"show", "fold", "find", "vis", "help", "plumb", "plumbbobfans"})

# ── plumb access control ─────────────────────────────────────────────────────
# `!plumb` and `!plumbbobfans` are gated by a self-expanding allow-list of
# Discord usernames (the new handle, `message.author.name` — no discriminators).
#
# PLUMB_OWNER_USERNAMES is the hardcoded seed: these users always have access
# even if the persisted file is missing or corrupt, and can't be removed.
# Everyone else with access was invited at runtime via `!plumbbobfans` and is
# persisted to PLUMB_ALLOWED_FILE (a plain JSON list of usernames).
PLUMB_OWNER_USERNAMES: frozenset[str] = frozenset({"fishisfordogs", "c9i34l6"})
PLUMB_ALLOWED_FILE: Path = Path(__file__).resolve().parent / "plumb_allowed.json"


def _load_plumb_allowed() -> set[str]:
    """Read the persisted username list. Return empty set if the file is missing
    or unreadable — owners still have access via the hardcoded seed, so a
    missing/broken file degrades gracefully rather than locking everyone out."""
    try:
        raw = PLUMB_ALLOWED_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    except OSError as e:
        log.warning("could not read %s: %r", PLUMB_ALLOWED_FILE, e)
        return set()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("invalid JSON in %s: %r", PLUMB_ALLOWED_FILE, e)
        return set()
    if not isinstance(data, list):
        return set()
    return {str(u) for u in data if isinstance(u, str)}


def _save_plumb_allowed(allowed: set[str]) -> bool:
    """Persist the allow-list. Returns True on success. Writes are
    atomic-ish (tmp file + rename) so a crash mid-write can't leave a
    half-written JSON that breaks startup."""
    tmp = PLUMB_ALLOWED_FILE.with_suffix(PLUMB_ALLOWED_FILE.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(sorted(allowed), indent=2), encoding="utf-8")
        os.replace(tmp, PLUMB_ALLOWED_FILE)
        return True
    except OSError as e:
        log.warning("could not write %s: %r", PLUMB_ALLOWED_FILE, e)
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _plumb_allowed_set() -> set[str]:
    """Union of hardcoded owners and the persisted list — the actual check set."""
    return set(PLUMB_OWNER_USERNAMES) | _load_plumb_allowed()


def _is_plumb_allowed(user: discord.abc.User) -> bool:
    return user.name in _plumb_allowed_set()

# Image extensions that cpshit (OpenCV) can read. Used to detect image
# attachments for `!plumb`, same way we detect `.cp` for render commands.
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _bang_lower(tok: str) -> str:
    """Lowercase token, stripping a single leading `!` (e.g. `!fold` → `fold`)."""
    if tok.startswith("!"):
        return tok[1:].lower()
    return tok.lower()


_NARROW_AND_IDEO_SPACE = re.compile(
    r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]"
)


def _normalize_command_text(s: str) -> str:
    """Make Discord/mobile unicode spaces ASCII so shlex splits `!show` / `!fold` reliably."""
    return _NARROW_AND_IDEO_SPACE.sub(" ", s)


def _unglue_bang_subcommands(s: str) -> str:
    """Insert spaces between glued `!show`-style tokens (e.g. `!show!fold` → `!show !fold`)."""
    return re.sub(
        r"!(show|fold)(?=!(?:show|fold)\b)",
        r"!\1 ",
        s,
        flags=re.IGNORECASE,
    )


def _parse_positive_count(s: str) -> int | None:
    s = s.strip()
    if not s.isdigit():
        return None
    v = int(s)
    return v if v > 0 else None


def _parse_find_lookback(token: str) -> timedelta | None:
    m = _FIND_LOOKBACK_RE.match(token.strip())
    if not m:
        return None
    n = int(m.group(1))
    if n <= 0:
        return None
    u = m.group(2).lower()
    if u.startswith("hour") or u.startswith("hr") or u == "h":
        return timedelta(hours=n)
    if u.startswith("day") or u == "d":
        return timedelta(days=n)
    if u.startswith("week") or u.startswith("wk") or u == "w":
        return timedelta(weeks=n)
    if u.startswith("month") or u in ("mo", "mon"):
        return timedelta(days=30 * n)
    if u.startswith("year") or u.startswith("yr") or u == "y":
        return timedelta(days=365 * n)
    return None


def _format_lookback(td: timedelta) -> str:
    """Short English phrase for embeds (not calendar-accurate for mo/y)."""
    total = int(td.total_seconds())
    if total <= 0:
        return "0 seconds"
    if total % 86400 == 0:
        d = total // 86400
        return f"{d} days" if d != 1 else "1 day"
    if total % 3600 == 0:
        h = total // 3600
        return f"{h} hours" if h != 1 else "1 hour"
    if total % 60 == 0:
        m = total // 60
        return f"{m} minutes" if m != 1 else "1 minute"
    return f"{total} seconds"


intents = discord.Intents.default()
intents.message_content = True


class OripicBot(discord.Client):
    def __init__(self) -> None:
        super().__init__(intents=intents)

    async def _reply(
        self,
        message: discord.Message,
        content: str | None = None,
        **kw: Any,
    ) -> discord.Message:
        return await message.reply(content or "", **kw)

    async def _edit(self, message: discord.Message, **kw: Any) -> discord.Message:
        return await message.edit(**kw)

    async def _safe_edit(self, message: discord.Message, content: str) -> None:
        try:
            await self._edit(
                message,
                content=content,
                attachments=[],
                embeds=[],
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except discord.HTTPException:
            pass
    def _content_after_bot_mention(self, message: discord.Message) -> str | None:
        if self.user is None:
            return None
        raw = message.content.strip()
        # Match user mention OR the bot's role mention
        m = re.match(rf"^<@[!&]?{self.user.id}>\s*", raw)
        if not m:
            # Also check all roles the bot has
            for role in message.guild.me.roles if message.guild else []:
                m = re.match(rf"^<@&{role.id}>\s*", raw)
                if m:
                    break
        if not m:
            return None
        return raw[m.end():].strip()
        # ── lifecycle ────────────────────────────────────────────────────────────
    async def on_ready(self) -> None:
        log.info("online as %s", self.user)

    # ── message handling ─────────────────────────────────────────────────────
    async def on_message(self, message: discord.Message) -> None:
   
        log.info("got message from %s: %r", message.author, message.content)  # add this
  
        if message.author.bot:
            return

        tail = self._content_after_bot_mention(message)
        if tail is not None:
            if not tail:
                await self._reply(
                    message,
                    f"Mention **@oripic** at the start of the message, then try **`!help`**. {_CAT}",
                )
                return
            await self._handle_command(message, tail)
            return

        if not message.attachments:
            return
        attachment = message.attachments[0]
        if not attachment.filename.lower().endswith(".cp"):
            return

        log.info("processing %s (%d bytes)", attachment.filename, attachment.size)
        status = await self._reply(
            message, f"⏳ Processing **{attachment.filename}**… {_CAT}"
        )
        try:
            cp_bytes = await attachment.read()
            cp_text = cp_bytes.decode("utf-8", errors="replace")
            await self._render_and_reply(cp_text, attachment.filename, status)
        except Exception:
            log.exception("unexpected error handling %s", attachment.filename)
            await self._safe_edit(
                status, f"❌ Something went wrong while processing the file. {_CAT}"
            )

    # ── command dispatch ─────────────────────────────────────────────────────
    async def _handle_command(self, message: discord.Message, command_text: str) -> None:
        try:
            parts = shlex.split(
                _unglue_bang_subcommands(_normalize_command_text(command_text.strip()))
            )
        except ValueError as e:
            await self._reply(
                message, f"❌ Couldn't parse command (unmatched quote?): {e}\n{_CAT}"
            )
            return

        if not parts or not parts[0].startswith("!"):
            await self._reply(
                message,
                f"Mention **@oripic** at the start, then a command like **`!help`** or **`!show \"query\"`**. {_CAT}",
            )
            return

        subcmd = parts[0][1:].lower()
        if subcmd not in _VALID_SUBCOMMANDS:
            await self._reply(
                message,
                f"Unknown **`!{subcmd}`**. Try **`!help`** after **@oripic**. {_CAT}",
            )
            return
        parts[0] = subcmd

        _reply_hint = (
            f" Reply to a message with a `.cp` attachment and omit the query to use that file. {_CAT}"
        )
        if subcmd == "show":
            # `!show`, `!show fold`, `!show !fold`, optional extra `!show` tokens, then query.
            i = 1
            do_fold = False
            while i < len(parts):
                w = _bang_lower(parts[i])
                if w == "fold":
                    do_fold = True
                    i += 1
                elif w == "show":
                    i += 1
                else:
                    break
            query = " ".join(parts[i:]).strip()
            if do_fold:
                usage = (
                    'Usage: **@oripic**, then `!show fold "query"` (or `!show !fold "query"`) '
                    "— crease pattern + folded state."
                    + _reply_hint
                )
                await self._do_render_cmd(
                    message, query, usage=usage, do_cp=True, do_fold=True
                )
            else:
                usage = (
                    'Usage: **@oripic**, then `!show "query"` — crease-pattern render.'
                    + _reply_hint
                )
                await self._do_render_cmd(
                    message, query, usage=usage, do_cp=True, do_fold=False
                )

        elif subcmd == "fold":
            # `!fold`, `!fold show`, `!fold !show`, optional extra `!fold` tokens, then query.
            i = 1
            do_cp = False
            while i < len(parts):
                w = _bang_lower(parts[i])
                if w == "show":
                    do_cp = True
                    i += 1
                elif w == "fold":
                    i += 1
                else:
                    break
            query = " ".join(parts[i:]).strip()
            if do_cp:
                usage = (
                    'Usage: **@oripic**, then `!fold show "query"` (or `!fold !show "query"`) '
                    "— folded state + crease pattern."
                    + _reply_hint
                )
                await self._do_render_cmd(
                    message, query, usage=usage, do_cp=True, do_fold=True
                )
            else:
                usage = (
                    'Usage: **@oripic**, then `!fold "query"` — folded-state render.'
                    + _reply_hint
                )
                await self._do_render_cmd(
                    message, query, usage=usage, do_cp=False, do_fold=True
                )

        elif subcmd == "vis":
            query = parts[1].strip() if len(parts) >= 2 else ""
            usage = (
                'Usage: **@oripic**, then `!vis "query"` — crease pattern + folded state.'
                + _reply_hint
            )
            await self._do_render_cmd(message, query, usage=usage,
                                      do_cp=True, do_fold=True)

        elif subcmd == "find":
            query = parts[1].strip() if len(parts) >= 2 else ""
            arg_a = parts[2] if len(parts) >= 3 else None
            arg_b = parts[3] if len(parts) >= 4 else None
            await self._do_find_cmd(message, query, arg_a, arg_b)

        elif subcmd == "plumb":
            await self._do_plumb_cmd(message)

        elif subcmd == "plumbbobfans":
            target = " ".join(parts[1:]).strip()
            await self._do_plumbbobfans_cmd(message, target)

        elif subcmd == "help":
            await self._do_help_cmd(message)

    async def _do_help_cmd(self, message: discord.Message) -> None:
        text = (
            "**Commands:** Mention **@oripic** at the **start** of the message, then **`!…`** "
            "(example: `@oripic !help`).\n"
            "\n"
            "Upload a `.cp` file and I'll render both the crease pattern and the "
            "folded state!\n"
            "\n"
            f'`!show "query"` — show the most recent crease pattern with this name in from the last '
            f"**{HISTORY_SCAN_LIMIT}** messages\n"
            "\n"
            f'`!fold "query"` — show the most recent crease pattern with this name in from the last '
            f"**{HISTORY_SCAN_LIMIT}** messages \n"
            "\n"
            "Reply to a message that has a `.cp` and use commands with no query"
            "after **@oripic** to run on that specific file.\n"
            "\n"
            f'`!find "query" [time] [N]` — list up to **N** matches (default {FIND_DEFAULT}, '
            f"max {FIND_MAX}) in the last **time** (e.g. `7d`, `12h`, `3mo`, `1y`; "
            f"default window {_format_lookback(DEFAULT_FIND_LOOKBACK)}). "
            "**Time** and **N** can be in either order. "
            f"In very active channels, search stops after **{FIND_SCAN_MAX_MESSAGES}** messages"
            "\n"
            "`!help` — show this message.\n"
            "\n"
            "For **`!show`** / **`!fold`** / **`!find`**, If I can't find exact matches, I'll show the closest ones,"
            "(e.g. `momoko` → `momoka.cp`).\n"
            "\n"
            "_Built by ani and aspen._\n"
            f"\n{_CAT}"
        )
        await self._reply(message, text, allowed_mentions=discord.AllowedMentions.none())

    async def _resolve_referenced_cp(
        self, message: discord.Message
    ) -> tuple[discord.Message, discord.Attachment] | None:
        """If `message` is a reply, return the first `.cp` on the referenced message."""
        ref = message.reference
        if ref is None or ref.message_id is None:
            return None

        ref_msg: discord.Message | None = None
        res = ref.resolved
        if isinstance(res, discord.Message):
            ref_msg = res
        elif res is not None:
            # DeletedReferencedMessage or unknown — can't read attachments
            return None
        else:
            try:
                if ref.channel_id is not None and ref.channel_id != message.channel.id:
                    ch = await self.fetch_channel(ref.channel_id)
                else:
                    ch = message.channel
                ref_msg = await ch.fetch_message(ref.message_id)
            except (discord.NotFound, discord.Forbidden, discord.HTTPException) as e:
                log.debug("could not load referenced message for cp reply: %r", e)
                return None

        att = _first_cp_attachment(ref_msg)
        if att is None:
            return None
        return ref_msg, att

    # ── render-style commands ────────────────────────────────────────────────
    async def _do_render_cmd(
        self,
        message: discord.Message,
        query: str,
        *,
        usage: str,
        do_cp: bool,
        do_fold: bool,
    ) -> None:
        if not query:
            ref_cp = await self._resolve_referenced_cp(message)
            if ref_cp is None:
                await self._reply(message, usage)
                return
            src_msg, attachment = ref_cp
            log.info(
                "render from reply: %s (cp=%s fold=%s)",
                attachment.filename,
                do_cp,
                do_fold,
            )
            status = await self._reply(
                message,
                f"📎 Using **{attachment.filename}** from {src_msg.author.display_name} "
                f"([jump]({src_msg.jump_url})). Rendering… {_CAT}",
            )
            try:
                cp_bytes = await attachment.read()
            except (discord.HTTPException, discord.NotFound) as e:
                log.warning("could not fetch reply attachment: %r", e)
                await self._safe_edit(
                    status,
                    f"❌ Couldn't download **{attachment.filename}** from Discord's CDN. {_CAT}",
                )
                return
            cp_text = cp_bytes.decode("utf-8", errors="replace")
            await self._render_and_reply(
                cp_text,
                attachment.filename,
                status,
                do_cp=do_cp,
                do_fold=do_fold,
                source=src_msg,
            )
            return

        log.info("render cmd (cp=%s fold=%s): %r", do_cp, do_fold, query)
        status = await self._reply(
            message,
            f"🔎 Searching recent history for `.cp` files matching **{_escape_md(query)}**… {_CAT}",
        )

        try:
            found = await self._find_recent_cp(message.channel, query, HISTORY_SCAN_LIMIT)
        except discord.Forbidden:
            await self._safe_edit(
                status,
                "❌ I don't have permission to read this channel's history. "
                "Grant me **Read Message History** and try again. "
                f"{_CAT}",
            )
            return
        except Exception:
            log.exception("history search failed")
            await self._safe_edit(
                status, f"❌ Something went wrong while searching history. {_CAT}"
            )
            return

        if found is None:
            await self._safe_edit(
                status,
                f"No recent `.cp` file matching **{_escape_md(query)}** found "
                f"in the last {HISTORY_SCAN_LIMIT} messages. {_CAT}",
            )
            return

        src_msg, attachment, is_fuzzy = found
        log.info("match: %s (msg %s, fuzzy=%s)", attachment.filename, src_msg.id, is_fuzzy)
        prefix = "Closest match" if is_fuzzy else "Found"
        await self._safe_edit(
            status,
            f"{prefix}: **{attachment.filename}** from {src_msg.author.display_name} "
            f"([jump]({src_msg.jump_url})). Rendering… {_CAT}",
        )

        # Only now, for the single chosen file, do we hit the CDN.
        try:
            cp_bytes = await attachment.read()
        except (discord.HTTPException, discord.NotFound) as e:
            log.warning("could not fetch matched attachment: %r", e)
            await self._safe_edit(
                status,
                f"❌ Couldn't download **{attachment.filename}** from Discord's CDN. {_CAT}",
            )
            return

        cp_text = cp_bytes.decode("utf-8", errors="replace")
        await self._render_and_reply(
            cp_text, attachment.filename, status,
            do_cp=do_cp, do_fold=do_fold, source=src_msg,
        )

    # ── find command ─────────────────────────────────────────────────────────
    async def _do_find_cmd(
        self,
        message: discord.Message,
        query: str,
        arg_a: str | None,
        arg_b: str | None,
    ) -> None:
        usage = (
            'Usage: **@oripic**, then `!find "query" [time] [N]` — list up to **N** `.cp` matches '
            f'(default {FIND_DEFAULT}, max {FIND_MAX}) from messages in the time window.\n'
            "**Time** (pick one style): `12h`, `7d`, `3w`, `2mo`, `1y`, `4 days`, `6 months`. "
            "You can put **time then N** or **N then time**.\n"
            f'Omit both to use {_format_lookback(DEFAULT_FIND_LOOKBACK)} and N={FIND_DEFAULT}. '
            f"Omit only time to keep the default window. {_CAT}"
        )
        if not query:
            await self._reply(message, usage)
            return

        lookback = DEFAULT_FIND_LOOKBACK
        count = FIND_DEFAULT

        if arg_a is not None:
            lb_a = _parse_find_lookback(arg_a)
            n_a = _parse_positive_count(arg_a)
            if arg_b is None:
                if lb_a is not None:
                    lookback = lb_a
                elif n_a is not None:
                    count = n_a
                else:
                    await self._reply(
                        message,
                        f"❌ `{arg_a}` isn't a valid time range or count.\n{usage}",
                    )
                    return
            else:
                lb_b = _parse_find_lookback(arg_b)
                n_b = _parse_positive_count(arg_b)
                if lb_a is not None and n_b is not None:
                    lookback, count = lb_a, n_b
                elif n_a is not None and lb_b is not None:
                    lookback, count = lb_b, n_a
                else:
                    await self._reply(
                        message,
                        f"❌ Expected one time range and one count; got `{arg_a!r}` and `{arg_b!r}`.\n"
                        f"{usage}",
                    )
                    return

        capped = count > FIND_MAX
        if capped:
            count = FIND_MAX

        log.info("find cmd: %r (count %d, lookback %s)", query, count, lookback)
        status = await self._reply(
            message,
            f"🔎 Searching recent history for `.cp` files matching **{_escape_md(query)}**… {_CAT}",
        )

        cutoff = datetime.now(timezone.utc) - lookback
        try:
            matches, is_fuzzy = await self._find_recent_cp_many(
                message.channel, query, count, after=cutoff,
            )
        except discord.Forbidden:
            await self._safe_edit(
                status,
                "❌ I don't have permission to read this channel's history. "
                "Grant me **Read Message History** and try again. "
                f"{_CAT}",
            )
            return
        except Exception:
            log.exception("history search failed")
            await self._safe_edit(
                status, f"❌ Something went wrong while searching history. {_CAT}"
            )
            return

        if not matches:
            await self._safe_edit(
                status,
                f"No `.cp` files matching **{_escape_md(query)}** found "
                f"in the last {_format_lookback(lookback)}. {_CAT}",
            )
            return

        lines = []
        for i, (msg, att) in enumerate(matches, start=1):
            ts = int(msg.created_at.timestamp())
            lines.append(
                f"{i}. [{_escape_md(att.filename)}]({msg.jump_url}) — "
                f"{_escape_md(msg.author.display_name)}, <t:{ts}:R>"
            )

        title = (
            f'{_CAT} · .cp files close to "{query}"' if is_fuzzy
            else f'{_CAT} · .cp files matching "{query}"'
        )
        embed = discord.Embed(
            title=title,
            description=f"{_CAT}\n" + "\n".join(lines),
        )
        footer_bits = [
            f"Most recent first • scanned last {_format_lookback(lookback)} "
            f"(≤{FIND_SCAN_MAX_MESSAGES} messages)",
        ]
        if is_fuzzy:
            footer_bits.append("no exact hits — showing closest filenames")
        if capped:
            footer_bits.append(f"capped at {FIND_MAX}")
        footer_bits.append(_CAT)
        embed.set_footer(text=" • ".join(footer_bits))

        await self._edit(
            status,
            content=None,
            embeds=[embed],
            attachments=[],
            allowed_mentions=discord.AllowedMentions.none(),
        )

    # ── plumb command ────────────────────────────────────────────────────────
    async def _do_plumb_cmd(self, message: discord.Message) -> None:
        """`!plumb` — run cpshit on an image (attached or replied-to) and upload the .cp."""
        if not _is_plumb_allowed(message.author):
            log.info(
                "plumb denied for %s (not in allow-list)", message.author.name
            )
            await self._reply(
                message,
                f"❌ **`!plumb`** is restricted. Ask someone with access to run "
                f"**`!plumbbobfans {message.author.name}`** to add you. {_CAT}",
            )
            return

        usage = (
            "Usage: attach an image and run **@oripic `!plumb`**, "
            "or reply to a message with an image and run **@oripic `!plumb`**. "
            f"{_CAT}"
        )

        # Precedence: image attached to this message first, then the reply target.
        att = _first_image_attachment(message)
        src_msg: discord.Message | None = None
        if att is None:
            ref_img = await self._resolve_referenced_image(message)
            if ref_img is not None:
                src_msg, att = ref_img

        if att is None:
            await self._reply(message, usage)
            return

        log.info("plumb: %s (%d bytes)", att.filename, att.size)
        header = f"**{att.filename}**"
        if src_msg is not None:
            header += f" (from {src_msg.author.display_name}, [jump]({src_msg.jump_url}))"
        status = await self._reply(
            message, f"⏳ Plumbing {header}… {_CAT}"
        )

        try:
            image_bytes = await att.read()
        except (discord.HTTPException, discord.NotFound) as e:
            log.warning("could not fetch image for plumb: %r", e)
            await self._safe_edit(
                status, f"❌ Couldn't download **{att.filename}** from Discord's CDN. {_CAT}"
            )
            return

        try:
            cp_text = await asyncio.to_thread(bytes_to_cp, image_bytes)
        except Exception as e:
            log.exception("plumb failed for %s", att.filename)
            await self._safe_edit(
                status, f"❌ Couldn't plumb **{att.filename}**: {e} {_CAT}"
            )
            return

        if not cp_text.strip():
            await self._safe_edit(
                status,
                f"⚠️ Plumbed **{att.filename}** but found no line segments. "
                f"Try a more contrasty image. {_CAT}",
            )
            return

        base = Path(att.filename).stem or "plumbed"
        out_name = f"{base}.cp"
        out_file = discord.File(
            io.BytesIO(cp_text.encode("utf-8")),
            filename=out_name,
        )
        num_lines = cp_text.count("\n")
        await self._edit(
            status,
            content=f"Plumbed {header} — **{num_lines}** segment(s). {_CAT}",
            attachments=[out_file],
            embeds=[],
            allowed_mentions=discord.AllowedMentions.none(),
        )
        log.info("plumb sent %s (%d lines)", out_name, num_lines)

    # ── plumbbobfans command ─────────────────────────────────────────────────
    async def _do_plumbbobfans_cmd(
        self, message: discord.Message, target: str
    ) -> None:
        """`!plumbbobfans <username>` — grant another user access to `!plumb`.

        Accepts either a bare username (`aspen`) or a user mention (`@aspen`,
        which comes through as `<@12345>` in message.content). We resolve the
        mention form to the actual username via message.mentions so the stored
        list stays username-keyed regardless of which form the caller typed.
        """
        usage = (
            'Usage: **@oripic `!plumbbobfans <username>`** — grant access to '
            "`!plumb`. Only existing plumb bob fans can invite new ones. "
            f"{_CAT}"
        )

        if not _is_plumb_allowed(message.author):
            log.info(
                "whats a plumb??",
                message.author.name,
            )
            await self._reply(
                message,
                f"❌ Only users with **`!plumb`** access can grant it to others. {_CAT}",
            )
            return

        target = target.strip()
        if not target:
            await self._reply(message, usage)
            return

        # If the caller typed @someone, Discord put a mention in the content.
        # message.mentions is the resolved list in the order they appeared.
        mention_match = re.fullmatch(r"<@!?(\d+)>", target)
        if mention_match and message.mentions:
            mentioned_id = int(mention_match.group(1))
            for m in message.mentions:
                if m.id == mentioned_id:
                    target = m.name
                    break
            else:
                await self._reply(
                    message,
                    f"❌ Couldn't resolve that mention to a user. {_CAT}",
                )
                return

        # Strip a stray leading `@` if the caller typed `@name` without it
        # becoming a real Discord mention (e.g. the user isn't in the server).
        if target.startswith("@"):
            target = target[1:]

        if not target or any(c.isspace() for c in target):
            await self._reply(
                message,
                f"❌ `{_escape_md(target)}` doesn't look like a valid username. "
                f"{usage}",
            )
            return

        allowed = _load_plumb_allowed()
        if target in PLUMB_OWNER_USERNAMES:
            await self._reply(
                message,
                f"**{_escape_md(target)}** is a permanent owner — already has access. {_CAT}",
            )
            return
        if target in allowed:
            await self._reply(
                message,
                f"**{_escape_md(target)}** already has **`!plumb`** access. {_CAT}",
            )
            return

        allowed.add(target)
        if not _save_plumb_allowed(allowed):
            await self._reply(
                message,
                f"❌ Couldn't save the allow-list to disk. Access not granted. {_CAT}",
            )
            return

        log.info("plumbbobfans: %s granted !plumb to %s",
                 message.author.name, target)
        await self._reply(
            message,
            f"✨ Granted **`!plumb`** to **{_escape_md(target)}**, by order of "
            f"**{_escape_md(message.author.name)}**. Welcome to the club. {_CAT}",
        )

    async def _resolve_referenced_image(
        self, message: discord.Message
    ) -> tuple[discord.Message, discord.Attachment] | None:
        """If `message` is a reply, return the first image on the referenced message."""
        ref = message.reference
        if ref is None or ref.message_id is None:
            return None

        ref_msg: discord.Message | None = None
        res = ref.resolved
        if isinstance(res, discord.Message):
            ref_msg = res
        elif res is not None:
            return None
        else:
            try:
                if ref.channel_id is not None and ref.channel_id != message.channel.id:
                    ch = await self.fetch_channel(ref.channel_id)
                else:
                    ch = message.channel
                ref_msg = await ch.fetch_message(ref.message_id)
            except (discord.NotFound, discord.Forbidden, discord.HTTPException) as e:
                log.debug("could not load referenced message for image reply: %r", e)
                return None

        att = _first_image_attachment(ref_msg)
        if att is None:
            return None
        return ref_msg, att

    # ── history scans (metadata only) ────────────────────────────────────────
    async def _find_recent_cp(
        self,
        channel: discord.abc.Messageable,
        query: str,
        limit: int,
    ) -> tuple[discord.Message, discord.Attachment, bool] | None:
        """Find the newest matching `.cp` attachment. Returns a 3-tuple
        `(message, attachment, is_fuzzy)`, or None if nothing matches.

        Two-tier match, one pass:
          1. Newest filename that contains every query token as a substring
             (case-insensitive) — returned immediately if found.
          2. Otherwise, newest filename close enough under `_fuzzy_match`.

        So an exact hit always beats a fuzzy hit, even an older exact hit
        beats a newer fuzzy one — fuzzy only surfaces when no exact exists.
        """
        tokens = _query_tokens(query)
        if not tokens:
            return None

        fuzzy_fallback: tuple[discord.Message, discord.Attachment] | None = None
        async for msg in channel.history(limit=limit):
            for attachment in msg.attachments:
                fname = attachment.filename.lower()
                if not fname.endswith(".cp"):
                    continue
                if all(t in fname for t in tokens):
                    return msg, attachment, False
                if fuzzy_fallback is None and _fuzzy_match(tokens, fname):
                    fuzzy_fallback = (msg, attachment)

        if fuzzy_fallback is not None:
            m, a = fuzzy_fallback
            return m, a, True
        return None

    async def _find_recent_cp_many(
        self,
        channel: discord.abc.Messageable,
        query: str,
        count: int,
        *,
        after: datetime,
        max_messages: int = FIND_SCAN_MAX_MESSAGES,
    ) -> tuple[list[tuple[discord.Message, discord.Attachment]], bool]:
        """Up to `count` newest matching `.cp` attachments since `after` (UTC).
        Returns `(results, is_fuzzy)`, where `is_fuzzy` is True only if no exact
        hits existed and the list is filled from close matches."""
        tokens = _query_tokens(query)
        if not tokens:
            return [], False

        exact: list[tuple[discord.Message, discord.Attachment]] = []
        fuzzy: list[tuple[discord.Message, discord.Attachment]] = []
        scanned = 0
        # With `after=`, discord.py defaults to oldest_first; we need newest-first so
        # early exit returns the latest matches, same as the old limit= scan.
        async for msg in channel.history(
            limit=None, after=after, oldest_first=False
        ):
            scanned += 1
            if scanned > max_messages:
                break
            for attachment in msg.attachments:
                fname = attachment.filename.lower()
                if not fname.endswith(".cp"):
                    continue
                if all(t in fname for t in tokens):
                    exact.append((msg, attachment))
                    if len(exact) >= count:
                        return exact, False
                elif len(fuzzy) < count and _fuzzy_match(tokens, fname):
                    fuzzy.append((msg, attachment))

        if exact:
            return exact, False
        return fuzzy, True

    # ── shared render path ───────────────────────────────────────────────────
    async def _render_and_reply(
        self,
        cp_text: str,
        base_filename: str,
        status: discord.Message,
        *,
        do_cp: bool = True,
        do_fold: bool = True,
        source: discord.Message | None = None,
    ) -> None:
        """Run the requested renders concurrently and edit `status` with the result."""
        tasks: list[tuple[str, asyncio.Future | asyncio.Task | "asyncio.coroutines.Coroutine"]] = []
        if do_cp:
            # CP render is CPU-bound PIL work (releases the GIL), so a thread.
            tasks.append(("cp", asyncio.to_thread(oripic.draw_cp, cp_text)))
        if do_fold:
            # Fold render is CPU-heavy Python (flat-folder port); keep off event loop.
            tasks.append(("fold", asyncio.to_thread(render_cp_folded_png, cp_text)))

        results = await asyncio.gather(
            *(t for _, t in tasks), return_exceptions=True
        )

        files: list[discord.File] = []
        errors: list[str] = []
        for (kind, _), result in zip(tasks, results):
            if kind == "cp":
                if isinstance(result, (bytes, bytearray)):
                    files.append(discord.File(
                        io.BytesIO(result),
                        filename=f"{base_filename}_cp.png",
                    ))
                else:
                    log.warning("CP render failed: %r", result)
                    errors.append(f"❌ Crease pattern render failed. {_CAT}")
            else:  # "fold"
                if isinstance(result, (bytes, bytearray)):
                    files.append(discord.File(
                        io.BytesIO(result),
                        filename=f"{base_filename}_folded.png",
                    ))
                else:
                    log.warning("fold render failed: %r", result)
                    if isinstance(result, ValueError):
                        detail = f" ({result})"
                    elif isinstance(result, Exception):
                        detail = f" ({result!r})"
                    else:
                        detail = " (CP may not be flat-foldable)"
                    errors.append(f"❌ Folded state render failed{detail}. {_CAT}")

        header = f"**{base_filename}**"
        if source is not None:
            header += f" (from {source.author.display_name}, [jump]({source.jump_url}))"

        if not files:
            content = f"Failed to render {header}. {_CAT}\n" + "\n".join(errors)
        elif do_cp and do_fold and len(files) == 2:
            content = f"Here are your renders for {header}: {_CAT}"
        elif do_cp and not do_fold:
            content = f"Here's the crease-pattern render for {header}: {_CAT}"
        elif do_fold and not do_cp:
            content = f"Here's the folded-state render for {header}: {_CAT}"
        else:
            # both requested, only one succeeded
            content = f"Here's your render for {header}: {_CAT}"
        if errors and files:
            content += "\n" + "\n".join(errors)

        await self._edit(
            status,
            content=content,
            attachments=files,
            embeds=[],
            allowed_mentions=discord.AllowedMentions.none(),
        )
        log.info("sent %d file(s)", len(files))


def _escape_md(s: str) -> str:
    """Escape characters Discord treats as Markdown when echoing user input."""
    return re.sub(r"([\\`*_~|>\[\]()])", r"\\\1", s)


def _first_cp_attachment(message: discord.Message) -> discord.Attachment | None:
    for att in message.attachments:
        if att.filename.lower().endswith(".cp"):
            return att
    return None


def _first_image_attachment(message: discord.Message) -> discord.Attachment | None:
    """First image attachment on the message. Uses Discord's content_type when
    available (covers weird extensions / stripped filenames) and falls back to
    a filename-extension check."""
    for att in message.attachments:
        ct = (att.content_type or "").lower()
        if ct.startswith("image/"):
            return att
        if att.filename.lower().endswith(_IMAGE_EXTS):
            return att
    return None


def _query_tokens(query: str) -> list[str]:
    """Split a query on whitespace and lowercase each piece. Every token
    must appear as a substring of the filename for the file to match."""
    return [t for t in query.lower().split() if t]


# Splits filenames into alphanumeric "words" for fuzzy matching. So
# `crane_v3.cp` → ["crane", "v3", "cp"]; we drop the trailing "cp".
_WORD_RE = re.compile(r"[a-z0-9]+")


def _fuzzy_match(
    tokens: list[str],
    lower_filename: str,
    threshold: float = FUZZY_THRESHOLD,
) -> bool:
    """Every query token must be close to *some* word in the filename."""
    words = _WORD_RE.findall(lower_filename)
    if words and words[-1] == "cp":          # drop the .cp extension
        words = words[:-1]
    if not words:
        return False
    for tok in tokens:
        best = 0.0
        for w in words:
            r = difflib.SequenceMatcher(None, tok, w).ratio()
            if r > best:
                best = r
                if best >= 1.0:
                    break
        if best < threshold:
            return False
    return True


if __name__ == "__main__":
    OripicBot().run(TOKEN, log_handler=None)