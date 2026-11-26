"""Wyoming STT server for Cactus Whistle, with keyword biasing from Home Assistant."""

import argparse
import array
import asyncio
import contextlib
import json
import logging
import os
import re
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import cactus
from cactus.agent.whistle import LANGUAGES as _WHISTLE_LANGUAGES, SAMPLE_RATE
from websockets.asyncio.client import connect
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger("whistle_wyoming")

LANGUAGES = list(_WHISTLE_LANGUAGES)

# Fallback filter, used only if HA reports no entities exposed to Assist.
VOICE_DOMAINS = {
    "light", "switch", "climate", "cover", "media_player", "fan", "lock",
    "vacuum", "scene", "script", "input_boolean",
}
# Base keywords per language: what people actually say to Assist, i.e. imperatives and question
# words, not infinitives (A/B test: infinitives biased "apaga" -> "apagar"; these forms fixed
# "enciende"/"encendida"). Stopwords are short/function words that add noise to biasing.
#
# The Spanish to-do list verbs are the ones HA's HassListAddItem sentences accept; both the tú
# ("agrega") and the voseo ("agregá") imperatives are in, as HA matches both. A real-voice
# "agregá" came out as "Averigá" / "Airega" without them. "añadí"/"añade" are left out on purpose:
# in an A/B test with synthetic speech they leaked into unrelated commands ("…lo añadí todos").
# "the"/"and" stay Spanish stopwords: entity names are often in English.
LANG_DATA = {
    "en": {
        "keywords": {
            "tell", "turn", "switch", "on", "off", "up", "down", "open", "close",
            "activate", "deactivate", "what", "which", "how", "is", "was",
            "time", "temperature", "status", "set", "play", "pause", "stop",
            "toggle", "lock", "unlock", "run", "execute", "weather", "lights",
            "thermostat", "blind", "shade", "curtain", "door", "window", "fan"
        },
        "stopwords": {
            "the", "and", "a", "an", "of", "to", "in", "for", "with", "on", "at", "by", "from", "please"
        }
    },
    "es": {
        "keywords": {
            "dime", "enciende", "prende", "apaga", "sube", "baja", "pon", "abre", "cierra",
            "activa", "desactiva", "qué", "cuál", "cuánto", "está", "encendida", "encendido",
            "apagada", "apagado", "hora",
            "agrega", "agregá", "poné", "sumá", "lista",
            "temperatura", "estado", "fija", "reproduce", "pausa",
            "detén", "bloquea", "desbloquea", "ejecuta", "clima", "luces", "persiana", "puerta", "ventana"
        },
        "stopwords": {
            "del", "las", "los", "con", "para", "por", "una", "uno", "de", "la", "el", "un",
            "the", "and"
        }
    },
    "de": {
        "keywords": {
            "schalte", "an", "aus", "auf", "zu", "wie", "ist", "zeit", "temperatur",
            "aktiviere", "deaktiviere", "öffne", "schließe", "sperre", "entsperre", "starte", "stoppe",
            "pausiere", "status", "grad", "wetter", "licht", "lampen", "rollo", "rollladen", "tür", "fenster", "lüfter"
        },
        "stopwords": {
            "der", "die", "das", "ein", "eine", "und", "zu", "von", "mit", "für", "im", "in", "den", "dem", "bitte"
        }
    },
    "fr": {
        "keywords": {
            "allume", "éteins", "eteins", "ouvre", "ferme", "active", "désactive", "desactive", "quel", "quelle", "temps", "heure",
            "température", "temperature", "statut", "monte", "baisse", "mets", "lance", "arrête", "arrete", "verrouille", "déverrouille",
            "météo", "meteo", "lumière", "lumiere", "lampe", "volet", "store", "porte", "fenêtre", "fenetre", "ventilateur"
        },
        "stopwords": {
            "le", "la", "les", "un", "une", "de", "et", "en", "pour", "dans", "par", "du", "au", "aux", "s'il", "te", "plaît", "plait"
        }
    },
    "it": {
        "keywords": {
            "accendi", "spegni", "apri", "chiudi", "attiva", "disattiva", "che", "quale", "ora", "tempo",
            "temperatura", "stato", "alza", "abbassa", "imposta", "avvia", "ferma", "blocca", "sblocca",
            "meteo", "luce", "luci", "tapparella", "tenda", "porta", "finestra", "ventilatore"
        },
        "stopwords": {
            "il", "lo", "la", "i", "gli", "le", "un", "una", "di", "e", "per", "in", "da", "con", "su", "per", "favore"
        }
    },
    "nl": {
        "keywords": {
            "zet", "doe", "schakel", "draai", "aan", "uit", "open", "dicht", "start", "stop",
            "activeer", "deactiveer", "verhoog", "verlaag", "wat", "hoe", "hoeveel", "is", "was",
            "tijd", "temperatuur", "status", "weer", "verlichting", "lamp", "knop", "omhoog", "omlaag",
            "vergrendel", "ontgrendel", "rolluik", "gordijn", "deur", "raam", "ventilator"
        },
        "stopwords": {
            "de", "het", "een", "en", "van", "in", "voor", "met", "op", "te", "om", "bij", "alsjeblieft"
        }
    },
    "pl": {
        "keywords": {
            "włącz", "wyłącz", "otwórz", "zamknij", "aktywuj", "deaktywuj", "co", "jak", "czas", "godzina",
            "temperatura", "status", "podnieś", "opuść", "ustaw", "graj", "pauza", "zatrzymaj", "zablokuj",
            "odblokuj", "pogoda", "światło", "światła", "lampa", "roleta", "drzwi", "okno", "wentylator"
        },
        "stopwords": {
            "i", "w", "z", "na", "do", "że", "o", "to", "a", "po", "za", "dla", "pod", "ze", "proszę"
        }
    }
}

def lang_profile(language: str) -> dict:
    return LANG_DATA.get(language, LANG_DATA["en"])


def normalize_language(language: str | None) -> str | None:
    """"de-DE" / "de_CH" / "DE" -> "de"; None if Whistle doesn't support it."""
    if not language:
        return None
    base = re.split(r"[-_]", language.strip().lower(), maxsplit=1)[0]
    return base if base in LANGUAGES else None


def _split_words(text: str, stopwords: set[str] = frozenset()) -> list[str]:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in cleaned.split() if len(w) > 2 and w not in stopwords and not w.isdigit()]


def _ws_url(ha_url: str) -> str:
    return re.sub(r"^http", "ws", ha_url.rstrip("/")) + "/api/websocket"


class _HAWebsocket:
    """One authenticated HA websocket connection; call() runs commands and returns their results."""

    def __init__(self, ws):
        self.ws = ws
        self.next_id = 1

    async def call(self, commands: list[str | dict]) -> list:
        """Each command is a message type, or a dict with "type" plus extra fields."""
        messages = {}
        for command in commands:
            message = command if isinstance(command, dict) else {"type": command}
            messages[self.next_id] = message
            await self.ws.send(json.dumps({**message, "id": self.next_id}))
            self.next_id += 1
        results: dict[int, object] = {}
        while len(results) < len(messages):
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=30))
            if msg.get("type") != "result" or msg.get("id") not in messages:
                continue
            if not msg.get("success"):
                raise RuntimeError(f"{messages[msg['id']]['type']} failed: {msg.get('error')}")
            results[msg["id"]] = msg["result"]
        return [results[i] for i in sorted(messages)]


@contextlib.asynccontextmanager
async def _ha_ws(ha_url: str, token: str):
    async with connect(_ws_url(ha_url), max_size=64 * 1024 * 1024, open_timeout=10) as ws:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if msg.get("type") != "auth_required":
            raise RuntimeError(f"unexpected HA hello: {msg.get('type')}")
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if msg.get("type") != "auth_ok":
            raise RuntimeError(f"HA auth failed: {msg.get('message', msg.get('type'))}")
        yield _HAWebsocket(ws)


async def _ha_ws_fetch(ha_url: str, token: str, commands: list[str | dict]) -> list:
    """Authenticate to the HA websocket API and run commands, returning their results in order."""
    async with _ha_ws(ha_url, token) as ha:
        return await ha.call(commands)


def parse_extra_keywords(value: str | None) -> set[str]:
    """Comma-separated keywords from WHISTLE_EXTRA_KEYWORDS, e.g. "yerba, dulce de leche"."""
    return {k.strip().lower() for k in (value or "").split(",") if k.strip()}


def _rank(keywords: set[str]) -> list[str]:
    # Full phrases (more words) first.
    return sorted(keywords, key=lambda k: (-len(k.split()), k))


async def fetch_ha_names(ha_url: str, token: str) -> tuple[list[str], list[str], int]:
    """Area names + aliases, and the names people use for entities exposed to Assist.

    For an entity with aliases only the aliases are used (they're what people actually say;
    entity names are often technical), otherwise its friendly name.
    Returns (area_names, entity_names, number_of_exposed_entities)."""
    async with _ha_ws(ha_url, token) as ha:
        exposure, areas, states = await ha.call([
            "homeassistant/expose_entity/list",
            "config/area_registry/list",
            "get_states",
        ])
        exposed = {eid for eid, opts in exposure.get("exposed_entities", {}).items()
                   if opts.get("conversation")}
        if not exposed:
            _LOGGER.warning("No entities exposed to Assist; falling back to domain filter")
            exposed = {s["entity_id"] for s in states if s["entity_id"].split(".", 1)[0] in VOICE_DOMAINS}
        # config/entity_registry/list omits aliases; get_entries returns the full entries.
        (entries,) = await ha.call([
            {"type": "config/entity_registry/get_entries", "entity_ids": sorted(exposed)},
        ])

    aliases_by_entity = {
        entity_id: [a for a in (entry.get("aliases") or []) if a]
        for entity_id, entry in entries.items() if entry
    }
    entity_names: list[str] = []
    for state in states:
        entity_id = state["entity_id"]
        if entity_id not in exposed:
            continue
        aliases = aliases_by_entity.get(entity_id)
        entity_names.extend(aliases or [(state.get("attributes") or {}).get("friendly_name")])
    area_names = [name for area in areas for name in [area.get("name"), *(area.get("aliases") or [])]]
    clean = lambda names: sorted({n.strip().lower() for n in names if n and n.strip()})
    return clean(area_names), clean(entity_names), len(exposed)


def build_keywords(language: str, area_names: list[str], entity_names: list[str],
                   extra: set[str], max_keywords: int) -> list[str]:
    """The keyword list for one language, at most `max_keywords` long (unless base + extra alone are more).

    Base (per language) and extra keywords are always kept. The HA-derived ones fill the rest, by
    priority: area names, then full entity names/aliases, then single words split out of entity
    names. Within a tier, longer phrases first. Without the tiers a large install would cut the
    single-word areas and names alphabetically (everything after "m..." gone)."""
    profile = lang_profile(language)
    fixed = set(profile["keywords"]) | set(extra)
    stopwords = profile["stopwords"]
    tiers = [set(area_names), set(entity_names),
             {w for name in entity_names for w in _split_words(name, stopwords)}]
    seen = set(fixed)
    ranked: list[str] = []
    for tier in tiers:
        tier -= seen
        seen |= tier
        ranked.extend(_rank(tier))
    return _rank(fixed | set(ranked[:max(0, max_keywords - len(fixed))]))


async def fetch_ha_keywords(ha_url: str, token: str, max_keywords: int,
                            extra: set[str] = frozenset(), language: str = "en") -> list[str]:
    area_names, entity_names, _ = await fetch_ha_names(ha_url, token)
    return build_keywords(language, area_names, entity_names, extra, max_keywords)


class KeywordStore:
    def __init__(self, ha_url: str | None, token: str | None, max_keywords: int,
                 extra: set[str] = frozenset(), language: str = "en"):
        self.ha_url, self.token, self.max_keywords = ha_url, token, max_keywords
        self.extra = set(extra)
        self.language = language
        self.area_names: list[str] = []
        self.entity_names: list[str] = []
        self._cache: dict[str, list[str]] = {}

    def keywords_for(self, language: str) -> list[str]:
        """Keyword list for `language`: its own base words and stopwords, the same HA names."""
        if language not in self._cache:
            self._cache[language] = build_keywords(language, self.area_names, self.entity_names,
                                                   self.extra, self.max_keywords)
        return self._cache[language]

    @property
    def keywords(self) -> list[str]:
        return self.keywords_for(self.language)

    async def refresh(self) -> None:
        if not (self.ha_url and self.token):
            _LOGGER.warning("HA_URL/HA_TOKEN not set: using only base/extra keywords")
            return
        try:
            area_names, entity_names, exposed = await fetch_ha_names(self.ha_url, self.token)
        except Exception as err:
            _LOGGER.error("Keyword refresh from HA failed (keeping previous list): %s", err)
            return
        if (area_names, entity_names) == (self.area_names, self.entity_names):
            _LOGGER.debug("Keywords from HA unchanged (%d entities exposed)", exposed)
            return
        self.area_names, self.entity_names = area_names, entity_names
        self._cache.clear()
        keywords = self.keywords
        _LOGGER.info("%d entities exposed to Assist, %d areas/aliases -> %d keywords, e.g. %s",
                     exposed, len(area_names), len(keywords), keywords[:10])

    async def refresh_forever(self, interval: int) -> None:
        while True:
            await asyncio.sleep(interval)
            await self.refresh()


def pcm16_to_float(raw: bytes) -> array.array:
    """16-bit little-endian PCM -> float32 samples in [-1, 1], as cactus wants them.

    Raw bytes would be silently misread as float32."""
    pcm = array.array("h")
    pcm.frombytes(raw[: len(raw) // 2 * 2])
    if sys.byteorder == "big":
        pcm.byteswap()
    return array.array("f", [s / 32768.0 for s in pcm])


def transcribe_pcm(raw: bytes, language: str, keywords: list[str]) -> dict:
    """Runs on the single inference thread: conversion included, so the event loop never waits on it."""
    return cactus.transcribe(pcm16_to_float(raw), language=language, keywords=keywords or None)


def save_recording(directory: Path, max_files: int, pcm: bytes, meta: dict) -> Path:
    """Write the audio exactly as Whistle got it (16 kHz mono 16-bit WAV) plus a JSON sidecar.

    The sidecar keeps the transcript and the keyword list used, so a run can be replayed later
    against another model or other settings. Only the newest `max_files` recordings are kept
    (0 = no limit)."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    # Open the file first: if wave.open() itself fails, Python 3.12 logs a spurious __del__ traceback.
    with open(stem.with_suffix(".wav"), "wb") as fh, wave.open(fh, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    stem.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    if max_files > 0:
        # Timestamped names sort chronologically.
        for old in sorted(directory.glob("*.wav"))[:-max_files]:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)
    return stem.with_suffix(".wav")


class WhistleEventHandler(AsyncEventHandler):
    def __init__(self, info: Info, args, store: KeywordStore, executor: ThreadPoolExecutor,
                 *handler_args, **handler_kwargs):
        super().__init__(*handler_args, **handler_kwargs)
        self.info = info
        self.args = args
        self.store = store
        self.executor = executor
        self.converter = AudioChunkConverter(rate=SAMPLE_RATE, width=2, channels=1)
        self.audio = bytearray()
        self.language = args.language
        self.max_bytes = max(0, int(getattr(args, "max_seconds", 0) * SAMPLE_RATE)) * 2
        self.truncated = False

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            language = normalize_language(transcribe.language)
            if language:
                self.language = language
            elif transcribe.language:
                _LOGGER.debug("Unsupported language %r, using %s", transcribe.language, self.language)
            return True

        if AudioStart.is_type(event.type):
            self.audio = bytearray()
            self.truncated = False
            return True

        if AudioChunk.is_type(event.type):
            if self.max_bytes and len(self.audio) >= self.max_bytes:
                if not self.truncated:
                    _LOGGER.warning("Audio longer than %ss, ignoring the rest", self.args.max_seconds)
                    self.truncated = True
                return True
            chunk = self.converter.convert(AudioChunk.from_event(event))
            self.audio.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            raw = bytes(self.audio[: self.max_bytes] if self.max_bytes else self.audio)
            self.audio = bytearray()
            seconds = len(raw) / 2 / SAMPLE_RATE

            keywords = self.store.keywords_for(self.language)
            text, error = "", None
            start = time.monotonic()
            if len(raw) < 2:
                _LOGGER.warning("AudioStop without audio")
            else:
                try:
                    result = await asyncio.get_running_loop().run_in_executor(
                        self.executor, partial(transcribe_pcm, raw, self.language, keywords))
                    text = (result.get("text") or "").strip()
                except Exception as err:
                    # Still answer (with an empty transcript), or HA waits for its own timeout.
                    error = str(err)
                    _LOGGER.exception("Transcription failed")
            elapsed_ms = (time.monotonic() - start) * 1000
            _LOGGER.info("[%s] %.1fs audio -> %r (%.0f ms)", self.language, seconds, text, elapsed_ms)

            if self.args.save_audio and raw:
                meta = {"text": text, "language": self.language, "seconds": round(seconds, 2),
                        "elapsed_ms": round(elapsed_ms), "keywords": keywords}
                if error:
                    meta["error"] = error
                try:
                    path = await asyncio.to_thread(save_recording, Path(self.args.save_audio),
                                                   self.args.save_audio_max, raw, meta)
                    _LOGGER.debug("Saved %s", path)
                except OSError as err:
                    # A full or read-only disk must not break voice commands.
                    _LOGGER.error("Could not save the recording: %s", err)
            try:
                await self.write_event(Transcript(text=text, language=self.language).event())
            except (ConnectionResetError, BrokenPipeError):
                # HA gave up on this pipeline run (timeout/cancel) before we answered.
                _LOGGER.warning("Client disconnected before the transcript was sent")
            return False  # one transcription per connection, like other Wyoming STT servers

        return True


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300")
    parser.add_argument("--language", default=os.getenv("WHISTLE_LANGUAGE", "en"), choices=LANGUAGES)
    parser.add_argument("--ha-url", default=os.getenv("HA_URL"))
    parser.add_argument("--max-keywords", type=int, default=int(os.getenv("WHISTLE_MAX_KEYWORDS", "200")))
    parser.add_argument("--refresh-interval", type=int, default=int(os.getenv("WHISTLE_REFRESH", "3600")),
                        help="Seconds between keyword refreshes from HA (0 = only at startup)")
    parser.add_argument("--extra-keywords", default=os.getenv("WHISTLE_EXTRA_KEYWORDS", ""),
                        help="Comma-separated keywords always added to the list")
    parser.add_argument("--save-audio", default=os.getenv("WHISTLE_SAVE_AUDIO", ""),
                        help="Directory to save each request's audio + transcript in (off when empty)")
    parser.add_argument("--save-audio-max", type=int, default=int(os.getenv("WHISTLE_SAVE_AUDIO_MAX", "500")),
                        help="Keep only the newest N recordings (0 = no limit)")
    parser.add_argument("--max-seconds", type=float, default=float(os.getenv("WHISTLE_MAX_SECONDS", "30")),
                        help="Ignore audio past this many seconds per request (0 = no limit)")
    parser.add_argument("--debug", action="store_true", default=os.getenv("WHISTLE_DEBUG", "") not in ("", "0"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Their DEBUG output is frame-level noise, and websockets' includes part of the HA token.
    for name in ("websockets", "httpcore", "httpcore2", "httpx", "httpx2", "filelock", "urllib3"):
        logging.getLogger(name).setLevel(max(logging.INFO, logging.getLogger(name).level))

    extra = parse_extra_keywords(args.extra_keywords)
    if extra:
        _LOGGER.info("Extra keywords: %s", sorted(extra))
    if args.save_audio:
        _LOGGER.info("Saving recordings to %s (newest %s kept)", args.save_audio,
                     args.save_audio_max or "all")
    store = KeywordStore(args.ha_url, os.getenv("HA_TOKEN"), args.max_keywords, extra, args.language)
    await store.refresh()
    refresh_task = None
    if args.refresh_interval > 0:
        # Keep a reference: the event loop holds tasks only weakly.
        refresh_task = asyncio.create_task(store.refresh_forever(args.refresh_interval))

    # Load engine + weights once (downloads on first run), off the event loop.
    # The engine holds one global model, so all inference goes through a single worker thread.
    executor = ThreadPoolExecutor(max_workers=1)
    await asyncio.get_running_loop().run_in_executor(
        executor, partial(cactus.transcribe, array.array("f", [0.0] * SAMPLE_RATE), language=args.language)
    )
    _LOGGER.info("Whistle model loaded")

    info = Info(asr=[AsrProgram(
        name="whistle",
        description="Cactus Whistle speech-to-text",
        attribution=Attribution(name="Cactus Compute", url="https://huggingface.co/Cactus-Compute/whistle"),
        installed=True,
        version=None,
        models=[AsrModel(
            name="whistle",
            description="Whistle 16.9 MB",
            attribution=Attribution(name="Cactus Compute", url="https://huggingface.co/Cactus-Compute/whistle"),
            installed=True,
            languages=LANGUAGES,
            version=None,
        )],
    )])

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Listening on %s", args.uri)
    try:
        await server.run(partial(WhistleEventHandler, info, args, store, executor))
    finally:
        if refresh_task:
            refresh_task.cancel()
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    asyncio.run(main())
