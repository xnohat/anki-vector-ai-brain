"""
Vector's memory — what turns him from a reactive pet into an AI Agent.

Files live in memory/:
  IDENTITY.md                 who Vector is (persona)
  USER-<name>.md              profile of a person he knows
  USER-PIC-<name>.jpeg        a photo he took of that person
  MEMORY-DD-MM-YYYY.md        today's journal (events + things learned)
  MEMORY.md                   consolidated long-term knowledge (Vector Brain writes it)

QMD search = semantic search over all of the above (OpenAI embeddings, cached on
disk). Before the brain answers, we recall() the most relevant memory chunks and
feed them in, so Vector remembers people, promises and what he has learned.
"""

import os
import re
import json
import time
import hashlib

import numpy as np

MEM_DIR = os.environ.get("VECTOR_MEM_DIR", "memory")
EMBED_MODEL = os.environ.get("VECTOR_EMBED_MODEL", "text-embedding-3-small")


def _today() -> str:
    return time.strftime("%d-%m-%Y")


def _now() -> str:
    return time.strftime("%H:%M")


def _stamp() -> str:
    """Full journal timestamp: dd.mm.yyyy hh:mm:ss"""
    return time.strftime("%d.%m.%Y %H:%M:%S")


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-").lower() or "unknown"


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


class MemoryStore:
    def __init__(self, client) -> None:
        self.client = client
        os.makedirs(MEM_DIR, exist_ok=True)
        self.index_path = os.path.join(MEM_DIR, ".qmd_index.json")
        self._cache = self._load_cache()      # {hash: [embedding floats]}

    # ------------------------------------------------------------------ files
    def journal_path(self, day: str = None) -> str:
        return os.path.join(MEM_DIR, f"MEMORY-{day or _today()}.md")

    def identity(self) -> str:
        return _read(os.path.join(MEM_DIR, "IDENTITY.md"))

    def longterm(self) -> str:
        return _read(os.path.join(MEM_DIR, "MEMORY.md"))

    # ---- daily chat thread (raw conversation, persisted + reloaded) ----
    def chat_path(self, day: str = None) -> str:
        return os.path.join(MEM_DIR, f"CHAT-{day or _today()}.jsonl")

    def append_chat(self, role: str, content: str) -> None:
        if not content:
            return
        try:
            with open(self.chat_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": _stamp(), "role": role, "content": content},
                                   ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[memory] append_chat failed: {exc}")

    def load_today_chat(self, max_msgs: int = 24) -> list:
        """Today's conversation as [{role, content}] (recent tail) for context."""
        out = []
        try:
            with open(self.chat_path(), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if d.get("role") in ("user", "assistant") and d.get("content"):
                        out.append({"role": d["role"], "content": d["content"]})
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[memory] load_today_chat failed: {exc}")
        return out[-max_msgs:]

    # ---- auto-loaded daily context (openclaw-style: long-term + recent notes) ----
    def journal_recent(self, max_lines: int = 25) -> str:
        """Today + yesterday's journal lines (recent tail)."""
        days = [time.strftime("%d-%m-%Y", time.localtime(time.time() - 86400)), _today()]
        lines = []
        for d in days:
            for ln in _read(self.journal_path(d)).splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#"):
                    lines.append(ln)
        return "\n".join(lines[-max_lines:])

    def today_context(self) -> str:
        """Compact always-on context: long-term memory + recent journal."""
        parts = []
        lt = self.longterm().strip()
        if lt:
            parts.append("# Long-term memory\n" + lt)
        recent = self.journal_recent()
        if recent:
            parts.append("# Recent journal (yesterday/today)\n" + recent)
        return "\n\n".join(parts)

    def remember(self, text: str, tag: str = "note") -> None:
        """Append one observation/event to today's journal."""
        text = (text or "").strip().replace("\n", " ")
        if not text:
            return
        p = self.journal_path()
        new = not os.path.exists(p)
        try:
            with open(p, "a", encoding="utf-8") as f:
                if new:
                    f.write(f"# Journal {_today()}\n\n")
                # one line: dd.mm.yyyy hh:mm:ss (tag) event/learning
                f.write(f"{_stamp()} ({tag}) {text}\n")
        except Exception as exc:
            print(f"[memory] remember failed: {exc}")

    # --------------------------------------------------------------- people
    def user_path(self, name: str) -> str:
        return os.path.join(MEM_DIR, f"USER-{_slug(name)}.md")

    def user_pic_path(self, name: str) -> str:
        return os.path.join(MEM_DIR, f"USER-PIC-{_slug(name)}.jpeg")

    def knows_user(self, name: str) -> bool:
        return os.path.exists(self.user_path(name))

    def get_user(self, name: str) -> str:
        return _read(self.user_path(name))

    def save_user(self, name: str, description: str, pic_jpeg: bytes = None) -> None:
        """Create/update a person's profile (+ optional photo he took of them)."""
        p = self.user_path(name)
        existing = _read(p)
        try:
            if not existing:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(f"# {name}\n\n*First met {_today()} at {_now()}.*\n\n{description}\n")
            else:
                with open(p, "a", encoding="utf-8") as f:
                    f.write(f"\n- [{_today()} {_now()}] {description}\n")
            if pic_jpeg:
                with open(self.user_pic_path(name), "wb") as f:
                    f.write(pic_jpeg)
        except Exception as exc:
            print(f"[memory] save_user failed: {exc}")

    # ----------------------------------------------------------- QMD search
    def _md_files(self) -> list:
        out = []
        for fn in sorted(os.listdir(MEM_DIR)):
            if fn.endswith(".md"):
                out.append(os.path.join(MEM_DIR, fn))
        return out

    def _chunks(self) -> list:
        """Split every memory .md into searchable chunks tagged with their source."""
        chunks = []
        for path in self._md_files():
            src = os.path.basename(path)
            text = _read(path)
            # Journals are one event per line -> index each line for precise recall.
            if src.startswith("MEMORY-"):
                for line in text.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        chunks.append((src, line))
                continue
            for block in re.split(r"\n\s*\n", text):
                block = block.strip()
                if not block:
                    continue
                # split very long blocks into ~500-char windows
                if len(block) <= 600:
                    chunks.append((src, block))
                else:
                    for i in range(0, len(block), 500):
                        chunks.append((src, block[i:i + 500]))
        return chunks

    def _embed(self, texts: list) -> list:
        try:
            r = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [d.embedding for d in r.data]
        except Exception as exc:
            print(f"[memory] embed failed: {exc}")
            return [None] * len(texts)

    def recall(self, query: str, k: int = 4) -> str:
        """QMD search: return the k most relevant memory chunks as a text block."""
        if not query:
            return ""
        chunks = self._chunks()
        if not chunks:
            return ""
        # embed any chunks not in the cache
        need = []
        for _, txt in chunks:
            h = hashlib.sha1(txt.encode("utf-8")).hexdigest()
            if h not in self._cache:
                need.append((h, txt))
        for i in range(0, len(need), 64):
            batch = need[i:i + 64]
            vecs = self._embed([t for _, t in batch])
            for (h, _), v in zip(batch, vecs):
                if v is not None:
                    self._cache[h] = v
        self._save_cache()

        qv = self._embed([query])[0]
        if qv is None:
            return ""
        qv = np.array(qv, dtype=np.float32)
        qn = qv / (np.linalg.norm(qv) + 1e-9)
        # hybrid: vector similarity + keyword overlap (catches names / exact terms)
        qwords = set(w for w in re.findall(r"\w+", query.lower()) if len(w) > 2)
        scored = []
        for src, txt in chunks:
            h = hashlib.sha1(txt.encode("utf-8")).hexdigest()
            v = self._cache.get(h)
            if not v:
                continue
            vv = np.array(v, dtype=np.float32)
            sim = float(np.dot(qn, vv / (np.linalg.norm(vv) + 1e-9)))
            kw = (len(qwords & set(re.findall(r"\w+", txt.lower()))) / (len(qwords) + 1)
                  if qwords else 0.0)
            scored.append((sim + 0.15 * kw, src, txt))
        scored.sort(reverse=True)
        top = [f"({src}) {txt}" for sc, src, txt in scored[:k] if sc > 0.18]
        return "\n".join(top)

    # ------------------------------------------------------- consolidation
    def consolidate(self, llm_model: str = None) -> bool:
        """Vector Brain reads recent journals + current long-term memory and
        rewrites MEMORY.md with durable, deduplicated knowledge."""
        journals = sorted(g for g in os.listdir(MEM_DIR)
                          if g.startswith("MEMORY-") and g.endswith(".md"))
        if not journals:
            return False
        recent = "\n\n".join(f"## {j}\n{_read(os.path.join(MEM_DIR, j))}"
                             for j in journals[-5:])
        current = self.longterm()
        prompt = (
            "You are Vector's memory-consolidation process. From the existing "
            "long-term memory and recent daily journals, write an updated, concise "
            "long-term memory in ENGLISH (this is an open-source project; keep the "
            "file readable to contributors, but preserve names and short quotes as-is). "
            "Keep durable facts about the people Vector loves (names, traits, "
            "preferences, promises), routines, and what he has learned about his world. "
            "Merge duplicates, drop trivial one-offs. Output ONLY the new MEMORY.md "
            "content (markdown).\n\n"
            f"=== CURRENT LONG-TERM MEMORY ===\n{current}\n\n"
            f"=== RECENT JOURNALS ===\n{recent}"
        )
        try:
            r = self.client.chat.completions.create(
                model=llm_model or "gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4)
            new = (r.choices[0].message.content or "").strip()
            if new:
                with open(os.path.join(MEM_DIR, "MEMORY.md"), "w", encoding="utf-8") as f:
                    f.write(new + f"\n\n*Consolidated {_today()} {_now()}.*\n")
                return True
        except Exception as exc:
            print(f"[memory] consolidate failed: {exc}")
        return False

    # ------------------------------------------------------------- cache io
    def _load_cache(self) -> dict:
        try:
            with open(self.index_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_cache(self) -> None:
        try:
            with open(self.index_path, "w") as f:
                json.dump(self._cache, f)
        except Exception:
            pass
