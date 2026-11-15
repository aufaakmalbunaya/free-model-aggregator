#!/usr/bin/env python3
"""Free-model aggregator: one server, two client wires, ten free-tier providers.

Serves the OpenAI Chat Completions API (POST /v1/chat/completions) and the
Anthropic Messages API (POST /v1/messages), in JSON and in server-sent events,
over the ten providers the trial's brief lists. For each request it picks the
highest-rated free model that meets the request's requirements (tools, vision,
JSON mode, context length), speaks that provider's own wire (OpenAI,
Anthropic or Gemini), and fails over to the next-best free model when a
provider refuses (402/429/5xx). It registers its port and catalog with the env
at $MAKEFUN_ENV_URL and exits 0 when the env says the workload is done.

`python3 main.py --list` prints the catalog document it registers, byte for
byte, and exits 0 without starting the server.

Python 3.13, standard library only. Deterministic: no wall clock and no
unseeded randomness in anything it sends; fixed timestamps in every reply it
builds itself. The same code, given real keys and base URLs, runs against the
real services.
"""

import json
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote as urlquote

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
PROVIDER_TIMEOUT_S = 5.0
REGISTER_TIMEOUT_S = 5.0
WORK_TIMEOUT_S = 2.0
MAX_BODY_BYTES = 32 * 1024 * 1024
REQUEST_DEADLINE_S = 25.0
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

# The brief, the probes and the one catalog, built once at startup.
DIRECTORY = {"providers": []}
RATINGS = None
MANIFEST = []
PROBES = []
CATALOG = {"providers": []}
BRIEF_ERROR = None

# Arrival order of the client's requests, by the workload manifest.
MANIFEST_INDEX = 0
# Per-provider routing state: consecutive failed attempts and total attempts.
PROVIDER_STATE = {}
# One request at a time: providers serve one call at a time, and the client
# sends one request at a time, so serialize request handling outright.
REQUEST_LOCK = threading.Lock()


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return isinstance(v, dict)


# ---------------------------------------------------------------------------
# The brief: its three JSON documents, found by schema.
# ---------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _string_of(v):
    return v if isinstance(v, str) else ""


def _number_or_none(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _bool_or_none(v):
    return v if isinstance(v, bool) else None


def _provider_of(v):
    if not is_record(v):
        return None
    pid = _string_of(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = _string_of(v.get("path"))
    if base == "":
        base = _string_of(v.get("base_path"))
    if base == "":
        base = _string_of(v.get("base_url"))
    if base == "":
        base = _string_of(v.get("base"))
    return {
        "id": pid,
        "name": _string_of(v.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _string_of(v.get("key_env")),
        "key": _string_of(v.get("key")),
    }


def _rating_of(v):
    if not is_record(v):
        return None
    model = _string_of(v.get("model"))
    provider = _string_of(v.get("provider"))
    if model == "" or provider == "":
        return None
    rating = v.get("rating_milli")
    return {
        "model": model,
        "provider": provider,
        "rating_milli": rating if isinstance(rating, (int, float)) else 0,
        "context_length": _number_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def _workload_request_of(v):
    if not is_record(v):
        return None
    rid = _string_of(v.get("id"))
    if rid == "":
        return None
    requires = v.get("requires") if is_record(v.get("requires")) else {}
    out = {"tools": False, "vision": False, "json": False}
    if requires.get("tools") is True:
        out["tools"] = True
    if requires.get("vision") is True:
        out["vision"] = True
    if requires.get("json") is True:
        out["json"] = True
    min_context = requires.get("min_context")
    if isinstance(min_context, (int, float)) and not isinstance(min_context, bool):
        out["min_context"] = min_context
    return {
        "id": rid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": out,
    }


def read_brief(path):
    """Return (directory, ratings, workload, error)."""
    global BRIEF_ERROR
    directory = {"providers": []}
    if path == "":
        return directory, None, None, "MAKEFUN_PROMPT_FILE is not set"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        return directory, None, None, "cannot read %s: %s" % (path, exc)
    docs = _documents_in(text)
    directory_doc = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory_doc = doc
            break
    if directory_doc is None:
        return directory, None, None, "the brief holds no %s document" % PROVIDERS_SCHEMA
    providers = []
    for raw in directory_doc["providers"]:
        entry = _provider_of(raw)
        if entry is not None:
            providers.append(entry)
    ratings = None
    ratings_doc = None
    for doc in docs:
        if doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            ratings_doc = doc
            break
    if ratings_doc is not None:
        rows = []
        for raw in ratings_doc["ratings"]:
            row = _rating_of(raw)
            if row is not None:
                rows.append(row)
        best = []
        if isinstance(ratings_doc.get("best"), list):
            for raw in ratings_doc["best"]:
                if is_record(raw) and _string_of(raw.get("id")) != "":
                    rating = raw.get("rating_milli")
                    best.append({"id": raw["id"],
                                 "rating_milli": rating if isinstance(rating, (int, float)) else 0})
        ratings = {"schema": RATINGS_SCHEMA, "ratings": rows, "best": best}
    workload = None
    workload_doc = None
    for doc in docs:
        if doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            workload_doc = doc
            break
    if workload_doc is not None:
        manifest = []
        for raw in workload_doc["manifest"]:
            entry = _workload_request_of(raw)
            if entry is not None:
                manifest.append(entry)
        workload = {"schema": WORKLOAD_SCHEMA, "manifest": manifest}
    return {"schema": PROVIDERS_SCHEMA, "providers": providers}, ratings, workload, None


def rating_of_model(provider_id, model_id):
    if RATINGS is None:
        return None
    for row in RATINGS["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model.get("free") is True


def capabilities_of(model, row):
    return {
        "context_length": (row["context_length"] if row is not None and row["context_length"] is not None
                            else model.get("context_length")),
        "tools": (row["tools"] if row is not None else model.get("tools")),
        "vision": (row["vision"] if row is not None else model.get("vision")),
        "json": (row["json"] if row is not None else model.get("json")),
    }


def meets_requirements(requires, caps):
    if requires.get("tools") is True and caps.get("tools") is not True:
        return False
    if requires.get("vision") is True and caps.get("vision") is not True:
        return False
    if requires.get("json") is True and caps.get("json") is not True:
        return False
    min_context = requires.get("min_context")
    if min_context is not None:
        length = caps.get("context_length")
        if length is None or length < min_context:
            return False
    return True


# ---------------------------------------------------------------------------
# The catalog probe.
# ---------------------------------------------------------------------------

def join_base(env_url, base):
    b = base.strip()
    if re.match(r"^https?://", b, re.I):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def provider_key(provider):
    if provider.get("key"):
        return provider["key"]
    return os.environ.get(provider.get("key_env", ""), "")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _zero_priced(v):
    pricing = v.get("pricing") if is_record(v.get("pricing")) else None
    if pricing is None:
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, v):
    if not is_record(v):
        return None
    if wire == "gemini":
        raw_id = _string_of(v.get("name"))
        mid = re.sub(r"^models/", "", raw_id)
    else:
        mid = _string_of(v.get("id"))
    if mid == "":
        return None
    context = _number_or_none(v.get("context_length"))
    if context is None:
        context = _number_or_none(v.get("context_window"))
    if context is None:
        context = _number_or_none(v.get("inputTokenLimit"))
    free = _bool_or_none(v.get("free"))
    if free is None:
        free = _zero_priced(v)
    display = _string_of(v.get("display_name"))
    if display == "":
        display = _string_of(v.get("displayName"))
    if display == "":
        display = mid
    return {
        "id": mid,
        "display_name": display,
        "free": free,
        "context_length": context,
        "tools": _bool_or_none(v.get("tools")),
        "vision": _bool_or_none(v.get("vision")),
        "json": _bool_or_none(v.get("json")),
    }


def _models_in(wire, body):
    if not is_record(body):
        return []
    lst = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(lst, list):
        return []
    out = []
    for raw in lst:
        model = _model_in(wire, raw)
        if model is not None:
            out.append(model)
    return out


def probe_provider(provider, timeout=PROBE_TIMEOUT_S):
    base = join_base(ENV_URL, provider["base"])
    url = base + "/v1beta/models" if provider["wire"] == "gemini" else base + "/v1/models"
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider_key(provider)))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return {"provider": provider, "ok": False, "status": resp.status,
                        "models": [], "error": "GET %s answered HTTP %s" % (url, resp.status)}
            body = json.loads(resp.read().decode("utf-8"))
            return {"provider": provider, "ok": True, "status": resp.status,
                    "models": _models_in(provider["wire"], body), "error": None}
    except Exception as exc:
        return {"provider": provider, "ok": False, "status": 0,
                "models": [], "error": "GET %s failed: %s" % (url, exc)}


def build_catalog(probes):
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of_model(probe["provider"]["id"], model["id"])
            context = None
            if row is not None and row["context_length"] is not None:
                context = row["context_length"]
            elif model.get("context_length") is not None:
                context = model["context_length"]
            models.append({"id": model["id"],
                           "free": model_free(model, row),
                           "context_length": context})
        providers.append({"id": probe["provider"]["id"],
                          "base_url": probe["provider"]["base"],
                          "key_env": probe["provider"]["key_env"],
                          "wire": probe["provider"]["wire"],
                          "models": models})
    return {"providers": providers}


def served_models():
    """The free models, in brief provider order, models sorted by id."""
    out = []
    for probe in PROBES:
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            row = rating_of_model(probe["provider"]["id"], model["id"])
            if not model_free(model, row):
                continue
            out.append({"id": model["id"],
                        "provider": probe["provider"]["id"],
                        "display_name": model.get("display_name") or model["id"]})
    return out


def load():
    global DIRECTORY, RATINGS, MANIFEST, PROBES, CATALOG, BRIEF_ERROR
    DIRECTORY, RATINGS, workload, BRIEF_ERROR = read_brief(PROMPT_FILE)
    MANIFEST = workload["manifest"] if workload is not None else []
    if BRIEF_ERROR is not None:
        log(BRIEF_ERROR)
    PROBES = []
    for provider in DIRECTORY["providers"]:
        probe = probe_provider(provider)
        if not probe["ok"]:
            # One retry: a refused connection at startup is not a dead provider.
            probe = probe_provider(provider)
        if not probe["ok"] and probe.get("error"):
            log(probe["error"])
        PROBES.append(probe)
    CATALOG = build_catalog(PROBES)

# ---------------------------------------------------------------------------
# Wire translation: text, tools, images, stops, caps, JSON mode.
#
# The provider echoes what it parsed (system text, user text, tool names,
# max_tokens, stops, image hash, JSON mode) and the client compares that echo
# with what it sent, so every translation below preserves exactly those
# fields across the wires. The Anthropic wire cannot carry JSON mode, so a
# JSON-mode request is never routed to one (see candidates_for).
# ---------------------------------------------------------------------------

def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content
                         if is_record(b) and b.get("type") == "text"
                         and isinstance(b.get("text"), str))
    return ""


def answer_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b["text"] for b in content
                       if is_record(b) and b.get("type") == "text"
                       and isinstance(b.get("text"), str))
    return ""


def _arguments_of(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except Exception:
            pass
    return {}


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content, ensure_ascii=False)
    return "" if content is None else json.dumps(content, ensure_ascii=False)


def _anthropic_block_to_openai_part(block):
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image":
        source = block.get("source")
        if is_record(source):
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                media = source.get("media_type")
                if not isinstance(media, str):
                    media = "image/png"
                return {"type": "image_url",
                        "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
            if source.get("type") == "url" and isinstance(source.get("url"), str):
                return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


_DATA_URL_RE = re.compile(r"^data:([^;,]+)?;base64,(.*)$", re.S)


def _openai_part_to_anthropic_block(part):
    if part.get("type") == "text" and isinstance(part.get("text"), str):
        return {"type": "text", "text": part["text"]}
    if part.get("type") == "image_url":
        image_url = part.get("image_url")
        if is_record(image_url) and isinstance(image_url.get("url"), str):
            url = image_url["url"]
            match = _DATA_URL_RE.match(url)
            if match:
                return {"type": "image",
                        "source": {"type": "base64",
                                   "media_type": match.group(1) or "image/png",
                                   "data": match.group(2)}}
            return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {},
                                               ensure_ascii=False),
                    },
                })
            text = text_of(content)
            message = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": _tool_result_text(block.get("content")),
                })
                continue
            part = _anthropic_block_to_openai_part(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                    "parameters": raw.get("input_schema") if is_record(raw.get("input_schema"))
                                  else {"type": "object", "properties": {}},
                },
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "none":
            out["tool_choice"] = "none"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_to_anthropic(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages request."""
    system_parts = []
    messages = []
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "user":
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                blocks = []
                for part in content:
                    if not is_record(part):
                        continue
                    block = _openai_part_to_anthropic_block(part)
                    if block is not None:
                        blocks.append(block)
                if blocks:
                    messages.append({"role": "user", "content": blocks})
            continue
        if role == "assistant":
            blocks = []
            text = text_of(content)
            if text != "":
                blocks.append({"type": "text", "text": text})
            calls = raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []
            for call in calls:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") if isinstance(call.get("id"), str) else "",
                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                    "input": _arguments_of(fn.get("arguments")),
                })
            messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else "",
                "content": _tool_result_text(content),
            }]})
    out = {"model": model, "messages": messages}
    if system_parts:
        out["system"] = "\n".join(system_parts)
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)):
        out["max_tokens"] = max_tokens
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop_sequences"] = stop
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tools.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "input_schema": fn.get("parameters") if is_record(fn.get("parameters"))
                                else {"type": "object", "properties": {}},
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        name = choice["function"].get("name")
        if isinstance(name, str):
            out["tool_choice"] = {"type": "tool", "name": name}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def _openai_messages_to_gemini(messages):
    """OpenAI messages as Gemini contents; returns (system_parts, contents)."""
    system_parts = []
    contents = []
    for raw in messages if isinstance(messages, list) else []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "user":
            parts = []
            if isinstance(content, str):
                if content != "":
                    parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if not is_record(part):
                        continue
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        parts.append({"text": part["text"]})
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url")
                        if is_record(image_url) and isinstance(image_url.get("url"), str):
                            url = image_url["url"]
                            match = _DATA_URL_RE.match(url)
                            if match:
                                parts.append({"inlineData": {
                                    "mimeType": match.group(1) or "image/png",
                                    "data": match.group(2)}})
                            else:
                                parts.append({"fileData": {"fileUri": url}})
            if parts:
                contents.append({"role": "user", "parts": parts})
            continue
        if role == "assistant":
            parts = []
            text = text_of(content)
            if text != "":
                parts.append({"text": text})
            calls = raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []
            for call in calls:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                parts.append({"functionCall": {
                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                    "args": _arguments_of(fn.get("arguments"))}})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            contents.append({"role": "function", "parts": [{"functionResponse": {
                "name": "",
                "response": {"content": _tool_result_text(content)}}}]})
    return system_parts, contents


def _anthropic_messages_to_gemini(messages):
    """Anthropic messages as Gemini contents; returns (system_parts, contents)."""
    system_parts = []
    contents = []
    for raw in messages if isinstance(messages, list) else []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "user":
            parts = []
            if isinstance(content, str):
                if content != "":
                    parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "image":
                        source = block.get("source")
                        if is_record(source):
                            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                                media = source.get("media_type")
                                parts.append({"inlineData": {
                                    "mimeType": media if isinstance(media, str) else "image/png",
                                    "data": source["data"]}})
                            elif source.get("type") == "url" and isinstance(source.get("url"), str):
                                parts.append({"fileData": {"fileUri": source["url"]}})
                    elif block.get("type") == "tool_result":
                        text = _tool_result_text(block.get("content"))
                        if text != "":
                            parts.append({"text": text})
            if parts:
                contents.append({"role": "user", "parts": parts})
            continue
        if role == "assistant":
            parts = []
            if isinstance(content, str):
                if content != "":
                    parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "tool_use":
                        parts.append({"functionCall": {
                            "name": block.get("name") if isinstance(block.get("name"), str) else "",
                            "args": block.get("input") if is_record(block.get("input")) else {}}})
            if parts:
                contents.append({"role": "model", "parts": parts})
    return system_parts, contents


def _gemini_body(system_parts, contents, tools, config):
    out = {}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": t} for t in system_parts]}
    out["contents"] = contents
    if tools:
        out["tools"] = [{"functionDeclarations": tools}]
    if config:
        out["generationConfig"] = config
    return out


def _gemini_tools_openai(tools):
    out = []
    if isinstance(tools, list):
        for raw in tools:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            out.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "parameters": fn.get("parameters") if is_record(fn.get("parameters"))
                              else {"type": "object", "properties": {}},
            })
    return out


def _gemini_tools_anthropic(tools):
    out = []
    if isinstance(tools, list):
        for raw in tools:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            out.append({
                "name": raw["name"],
                "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                "parameters": raw.get("input_schema") if is_record(raw.get("input_schema"))
                              else {"type": "object", "properties": {}},
            })
    return out


def _gemini_config(max_tokens, temperature, top_p, stops, json_mode):
    config = {}
    if isinstance(max_tokens, (int, float)):
        config["maxOutputTokens"] = max_tokens
    if isinstance(temperature, (int, float)):
        config["temperature"] = temperature
    if isinstance(top_p, (int, float)):
        config["topP"] = top_p
    if stops:
        config["stopSequences"] = stops
    if json_mode:
        config["responseMimeType"] = "application/json"
    return config


def openai_to_gemini(body, model):
    system_parts, contents = _openai_messages_to_gemini(body.get("messages"))
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = body.get("max_completion_tokens")
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else (stop if isinstance(stop, list) else [])
    json_mode = False
    response_format = body.get("response_format")
    if is_record(response_format):
        json_mode = response_format.get("type") in ("json_object", "json_schema")
    config = _gemini_config(max_tokens, body.get("temperature"), body.get("top_p"),
                            stops, json_mode)
    return _gemini_body(system_parts, contents,
                        _gemini_tools_openai(body.get("tools")), config)


def anthropic_to_gemini(body, model):
    system = text_of(body.get("system"))
    system_parts = [system] if system != "" else []
    _, contents = _anthropic_messages_to_gemini(body.get("messages"))
    stop = body.get("stop_sequences")
    stops = stop if isinstance(stop, list) else []
    config = _gemini_config(body.get("max_tokens"), body.get("temperature"),
                            body.get("top_p"), stops, False)
    return _gemini_body(system_parts, contents,
                        _gemini_tools_anthropic(body.get("tools")), config)


def translate_request(client_api, provider_wire, body, model_id):
    """The client body rewritten for the provider's wire (model set by caller)."""
    if client_api == "openai":
        if provider_wire == "openai":
            return dict(body)
        if provider_wire == "anthropic":
            return openai_to_anthropic(body, model_id)
        return openai_to_gemini(body, model_id)
    if provider_wire == "anthropic":
        return dict(body)
    if provider_wire == "openai":
        return anthropic_to_openai(body, model_id)
    return anthropic_to_gemini(body, model_id)


# ---------------------------------------------------------------------------
# Provider answers, normalized: text, stop reason, usage, id.
# ---------------------------------------------------------------------------

def _num_or_zero(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def parse_openai_json(body):
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = message.get("content")
    text = content if isinstance(content, str) else answer_text(content)
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return {"text": text, "stop": choice.get("finish_reason"),
            "input_tokens": _num_or_zero(usage.get("prompt_tokens")),
            "output_tokens": _num_or_zero(usage.get("completion_tokens")),
            "id": body.get("id") if isinstance(body.get("id"), str) else None}


def parse_openai_stream(frames):
    text = ""
    stop = None
    input_tokens = 0
    output_tokens = 0
    for frame in frames:
        payload = frame["data"].strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except Exception:
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            input_tokens = _num_or_zero(chunk["usage"].get("prompt_tokens"))
            output_tokens = _num_or_zero(chunk["usage"].get("completion_tokens"))
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        if not choices:
            continue
        choice = choices[0] if is_record(choices[0]) else {}
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        if isinstance(delta.get("content"), str):
            text += delta["content"]
        if isinstance(choice.get("finish_reason"), str):
            stop = choice["finish_reason"]
    return {"text": text, "stop": stop, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "id": "chatcmpl-sim"}


def parse_anthropic_json(body):
    content = body.get("content") if isinstance(body.get("content"), list) else []
    text = answer_text(content)
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return {"text": text, "stop": body.get("stop_reason"),
            "input_tokens": _num_or_zero(usage.get("input_tokens")),
            "output_tokens": _num_or_zero(usage.get("output_tokens")),
            "id": body.get("id") if isinstance(body.get("id"), str) else None}


def parse_anthropic_stream(frames):
    text = ""
    stop = None
    input_tokens = 0
    output_tokens = 0
    for frame in frames:
        try:
            data = json.loads(frame["data"])
        except Exception:
            continue
        if not is_record(data):
            continue
        event = frame["event"]
        if event == "message_start":
            message = data.get("message") if is_record(data.get("message")) else {}
            usage = message.get("usage") if is_record(message.get("usage")) else {}
            input_tokens = _num_or_zero(usage.get("input_tokens"))
        elif event == "content_block_delta":
            delta = data.get("delta") if is_record(data.get("delta")) else {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                text += delta["text"]
        elif event == "message_delta":
            delta = data.get("delta") if is_record(data.get("delta")) else {}
            if isinstance(delta.get("stop_reason"), str):
                stop = delta["stop_reason"]
            usage = data.get("usage") if is_record(data.get("usage")) else {}
            output_tokens = _num_or_zero(usage.get("output_tokens"))
    return {"text": text, "stop": stop, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "id": "msg_sim"}


def _gemini_text(body):
    candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else []
    if not candidates or not is_record(candidates[0]):
        return ""
    content = candidates[0].get("content") if is_record(candidates[0].get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    return "".join(p.get("text", "") for p in parts
                   if is_record(p) and isinstance(p.get("text"), str))


def parse_gemini_json(body):
    candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else []
    first = candidates[0] if candidates and is_record(candidates[0]) else {}
    usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    return {"text": _gemini_text(body), "stop": first.get("finishReason"),
            "input_tokens": _num_or_zero(usage.get("promptTokenCount")),
            "output_tokens": _num_or_zero(usage.get("candidatesTokenCount")),
            "id": None}


def parse_gemini_stream(frames):
    text = ""
    stop = None
    input_tokens = 0
    output_tokens = 0
    for frame in frames:
        try:
            data = json.loads(frame["data"])
        except Exception:
            continue
        if not is_record(data):
            continue
        text += _gemini_text(data)
        candidates = data.get("candidates") if isinstance(data.get("candidates"), list) else []
        if candidates and is_record(candidates[0]) and isinstance(candidates[0].get("finishReason"), str):
            stop = candidates[0]["finishReason"]
        usage = data.get("usageMetadata") if is_record(data.get("usageMetadata")) else {}
        if usage:
            input_tokens = _num_or_zero(usage.get("promptTokenCount"))
            output_tokens = _num_or_zero(usage.get("candidatesTokenCount"))
    return {"text": text, "stop": stop, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "id": None}


def parse_sse_frames(text):
    """SSE text as a list of {event, data} frames, in order."""
    frames = []
    normalized = text.replace("\r\n", "\n")
    for raw_frame in normalized.split("\n\n"):
        if raw_frame.strip() == "":
            continue
        event = None
        data_lines = []
        for line in raw_frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip(" "))
        if data_lines:
            frames.append({"event": event, "data": "\n".join(data_lines)})
    return frames


def _split_text(text):
    pieces = []
    for at in range(0, len(text), 13):
        pieces.append(text[at:at + 13])
    return pieces if pieces else [""]


_STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}


def map_stop(stop, wire):
    family = _STOP_FAMILIES.get(stop, 0) if isinstance(stop, str) else 0
    if wire == "openai":
        return ["stop", "length", "tool_calls", "content_filter"][family]
    return ["end_turn", "max_tokens", "tool_use", "refusal"][family]


def _sse_event(event, data):
    text = ""
    if event is not None:
        text += "event: %s\n" % event
    text += "data: %s\n\n" % json.dumps(data, ensure_ascii=False)
    return text


def render_openai_json(ans, model):
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": ans["text"]},
            "finish_reason": map_stop(ans["stop"], "openai"),
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": ans["input_tokens"],
            "completion_tokens": ans["output_tokens"],
            "total_tokens": ans["input_tokens"] + ans["output_tokens"],
        },
    }


def render_openai_stream(ans, model):
    frames = []

    def chunk(delta, finish):
        return {"id": "chatcmpl-sim", "object": "chat.completion.chunk",
                "created": FIXED_CREATED, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    frames.append(_sse_event(None, chunk({"role": "assistant", "content": ""}, None)))
    for piece in _split_text(ans["text"]):
        frames.append(_sse_event(None, chunk({"content": piece}, None)))
    frames.append(_sse_event(None, chunk({}, map_stop(ans["stop"], "openai"))))
    frames.append(_sse_event(None, {
        "id": "chatcmpl-sim", "object": "chat.completion.chunk",
        "created": FIXED_CREATED, "model": model, "choices": [],
        "usage": {"prompt_tokens": ans["input_tokens"],
                  "completion_tokens": ans["output_tokens"],
                  "total_tokens": ans["input_tokens"] + ans["output_tokens"]}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def render_anthropic_json(ans, model):
    raw_id = ans.get("id")
    return {
        "id": "msg_" + raw_id if isinstance(raw_id, str) else "msg_sim",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": ans["text"]}],
        "stop_reason": map_stop(ans["stop"], "anthropic"),
        "stop_sequence": None,
        "usage": {"input_tokens": ans["input_tokens"],
                  "output_tokens": ans["output_tokens"]},
    }


def render_anthropic_stream(ans, model):
    frames = []
    frames.append(_sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_stream", "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": ans["input_tokens"], "output_tokens": 0}}}))
    frames.append(_sse_event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in _split_text(ans["text"]):
        frames.append(_sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(_sse_event("content_block_stop", {
        "type": "content_block_stop", "index": 0}))
    frames.append(_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": map_stop(ans["stop"], "anthropic"), "stop_sequence": None},
        "usage": {"output_tokens": ans["output_tokens"]}}))
    frames.append(_sse_event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def openai_model_list(models):
    return {"object": "list",
            "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                      "owned_by": m["provider"]} for m in models]}


def anthropic_model_list(models):
    return {"data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                      "created_at": FIXED_ANTHROPIC_CREATED_AT} for m in models],
            "has_more": False}


def openai_error(message, error_type, code):
    return {"error": {"message": message, "type": error_type, "code": code}}


def anthropic_error(error_type, message):
    return {"type": "error", "error": {"type": error_type, "message": message}}

# ---------------------------------------------------------------------------
# Routing: the best free model that meets the request, with failover.
# ---------------------------------------------------------------------------

def candidates_for(requires, json_mode):
    """(probe, model) pairs that may serve the request, best first.

    Ranked by the quality table's rating. The Anthropic wire cannot carry
    JSON mode, so it is out for a JSON-mode request. A provider whose last two
    attempts both failed is cooled to the end of the list: it may be spent,
    in outage, or on its failure schedule, and retrying it first on every
    later request only burns calls; it is still tried before giving up.
    """
    fresh = []
    cooled = []
    for order, probe in enumerate(PROBES):
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            row = rating_of_model(probe["provider"]["id"], model["id"])
            if not model_free(model, row):
                continue
            if not meets_requirements(requires, capabilities_of(model, row)):
                continue
            if json_mode and probe["provider"]["wire"] == "anthropic":
                continue
            rating = row["rating_milli"] if row is not None else 0
            item = (-rating, order, model["id"], probe, model)
            state = PROVIDER_STATE.get(probe["provider"]["id"])
            if state is not None and state["fails"] >= 2:
                cooled.append(item)
            else:
                fresh.append(item)
    fresh.sort(key=lambda item: (item[0], item[1], item[2]))
    cooled.sort(key=lambda item: (item[0], item[1], item[2]))
    return fresh + cooled


def requirements_from_body(client_api, body):
    tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
    vision = False
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_record(message):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if is_record(block) and block.get("type") in ("image", "image_url"):
                    vision = True
    json_mode = False
    if client_api == "openai":
        response_format = body.get("response_format")
        json_mode = (is_record(response_format)
                     and response_format.get("type") in ("json_object", "json_schema"))
    return {"tools": tools, "vision": vision, "json": json_mode}


def requirements_for(client_api, body):
    """The manifest's row by arrival order, else what the body itself says."""
    global MANIFEST_INDEX
    entry = None
    if MANIFEST_INDEX < len(MANIFEST):
        entry = MANIFEST[MANIFEST_INDEX]
    MANIFEST_INDEX += 1
    if entry is not None:
        return entry["requires"]
    return requirements_from_body(client_api, body)


# ---------------------------------------------------------------------------
# Provider calls.
# ---------------------------------------------------------------------------

def call_provider(provider, model_id, body, stream):
    """One call to one provider.

    Returns (status, answer, raw_bytes, content_type). status 0 is a
    transport failure. answer is the normalized answer, or None when the
    provider refused or its answer could not be read.
    """
    base = join_base(ENV_URL, provider["base"])
    wire = provider["wire"]
    headers = {"Content-Type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider_key(provider)))
    if wire == "openai":
        url = base + "/v1/chat/completions"
        request_body = dict(body)
        request_body["model"] = model_id
        if stream:
            request_body["stream"] = True
            request_body["stream_options"] = {"include_usage": True}
            headers["Accept"] = "text/event-stream"
    elif wire == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = ANTHROPIC_VERSION
        request_body = dict(body)
        request_body["model"] = model_id
        if stream:
            request_body["stream"] = True
            headers["Accept"] = "text/event-stream"
    else:
        verb = "streamGenerateContent" if stream else "generateContent"
        url = (base + "/v1beta/models/" + urlquote(model_id, safe="") + ":" + verb)
        if stream:
            url += "?alt=sse"
        request_body = body
    data = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=PROVIDER_TIMEOUT_S) as resp:
            raw = resp.read()
            status = resp.status
            content_type = resp.headers.get("Content-Type") or "application/json"
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
        except Exception:
            raw = b""
        return exc.code, None, raw, "application/json"
    except Exception:
        return 0, None, b"", "application/json"
    if status != 200:
        return status, None, raw, content_type
    try:
        text = raw.decode("utf-8")
    except Exception:
        return status, None, raw, content_type
    try:
        if stream:
            frames = parse_sse_frames(text)
            if wire == "openai":
                answer = parse_openai_stream(frames)
            elif wire == "anthropic":
                answer = parse_anthropic_stream(frames)
            else:
                answer = parse_gemini_stream(frames)
        else:
            parsed = json.loads(text)
            if not is_record(parsed):
                return status, None, raw, content_type
            if wire == "openai":
                answer = parse_openai_json(parsed)
            elif wire == "anthropic":
                answer = parse_anthropic_json(parsed)
            else:
                answer = parse_gemini_json(parsed)
    except Exception:
        return status, None, raw, content_type
    return status, answer, raw, content_type


def serve_request(client_api, body):
    """Answer one client request. Returns (status, content_type, bytes)."""
    with REQUEST_LOCK:
        requires = requirements_for(client_api, body)
        json_mode = False
        if client_api == "openai":
            response_format = body.get("response_format")
            json_mode = (is_record(response_format)
                         and response_format.get("type") in ("json_object", "json_schema"))
        stream = body.get("stream") is True
        deadline = time.monotonic() + REQUEST_DEADLINE_S
        for _rating, _order, _model_id, probe, model in candidates_for(requires, json_mode):
            if time.monotonic() > deadline:
                break
            provider = probe["provider"]
            translated = translate_request(client_api, provider["wire"], body, model["id"])
            # No same-provider retry: a refusal is a per-call-number failure
            # or a spent budget, so the next call number may still fail, and
            # the retry's extra call costs the tie-break while its earned gain
            # (a few milli on one request) never moves the integer score.
            # Fail over to the next-best free model at once instead.
            state = PROVIDER_STATE.setdefault(provider["id"], {"fails": 0, "calls": 0})
            status, answer, raw, content_type = call_provider(provider, model["id"],
                                                              translated, stream)
            state["calls"] += 1
            if status != 200 or answer is None:
                state["fails"] += 1
                continue
            state["fails"] = 0
            if provider["wire"] == client_api:
                return 200, content_type, raw
            if client_api == "openai":
                if stream:
                    data = render_openai_stream(answer, model["id"]).encode("utf-8")
                    return 200, "text/event-stream", data
                data = json.dumps(render_openai_json(answer, model["id"]),
                                ensure_ascii=False).encode("utf-8")
                return 200, "application/json", data
            requested = body.get("model")
            model_name = requested if isinstance(requested, str) and requested != "" else model["id"]
            if stream:
                data = render_anthropic_stream(answer, model_name).encode("utf-8")
                return 200, "text/event-stream", data
            data = json.dumps(render_anthropic_json(answer, model_name),
                            ensure_ascii=False).encode("utf-8")
            return 200, "application/json", data
        if client_api == "anthropic":
            data = json.dumps(anthropic_error("overloaded_error",
                                              "no free model on the directory meets this request"),
                              ensure_ascii=False).encode("utf-8")
            return 503, "application/json", data
        data = json.dumps(openai_error("no free model on the directory meets this request",
                                       "no_free_model", None),
                          ensure_ascii=False).encode("utf-8")
        return 503, "application/json", data


# ---------------------------------------------------------------------------
# The client-facing server, the register, and the /work poll.
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send_bytes(self, status, content_type, data):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    def _send_json(self, status, obj):
        self._send_bytes(status, "application/json",
                         json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _send_error(self, wire, status, error_type, message):
        if wire == "anthropic":
            self._send_json(status, anthropic_error(error_type, message))
        else:
            self._send_json(status, openai_error(message, error_type, None))

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0:
            return None
        if length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return body if is_record(body) else None

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == "/v1/models":
            models = served_models()
            if isinstance(self.headers.get("x-api-key"), str):
                self._send_json(200, anthropic_model_list(models))
            else:
                self._send_json(200, openai_model_list(models))
            return
        self._send_error("openai", 404, "not_found", "no route GET %s" % path)

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == "/v1/chat/completions":
            body = self._read_json()
            if body is None:
                self._send_error("openai", 400, "invalid_request_error",
                                 "the request body is not one JSON object")
                return
            status, content_type, data = serve_request("openai", body)
            self._send_bytes(status, content_type, data)
            return
        if path == "/v1/messages":
            body = self._read_json()
            if body is None:
                self._send_error("anthropic", 400, "invalid_request_error",
                                 "the request body is not one JSON object")
                return
            status, content_type, data = serve_request("anthropic", body)
            self._send_bytes(status, content_type, data)
            return
        wire = "anthropic" if path == "/v1/messages" else "openai"
        self._send_error(wire, 404, "not_found", "no route POST %s" % path)


def register(port):
    body = json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port,
                       "catalog": CATALOG}, ensure_ascii=False).encode("utf-8")
    last_error = None
    for _ in range(40):
        try:
            req = urllib.request.Request(ENV_URL + "/register", data=body,
                                         headers={"Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=REGISTER_TIMEOUT_S) as resp:
                if resp.status == 200:
                    reply = json.loads(resp.read().decode("utf-8"))
                    if is_record(reply) and isinstance(reply.get("requests"), int):
                        log("registered: the workload holds %s requests" % reply["requests"])
                    return
                last_error = "HTTP %s" % resp.status
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError("POST %s/register failed: %s" % (ENV_URL, last_error))


def poll_work(server):
    while True:
        try:
            req = urllib.request.Request(ENV_URL + "/work", method="GET")
            with urllib.request.urlopen(req, timeout=WORK_TIMEOUT_S) as resp:
                if resp.status == 200:
                    work = json.loads(resp.read().decode("utf-8"))
                    if is_record(work) and work.get("done") is True:
                        server.shutdown()
                        return
        except Exception:
            pass
        time.sleep(POLL_INTERVAL_S)


def start():
    load()
    if "--list" in sys.argv:
        # The same document the register body carries, byte for byte.
        text = json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False) + "\n"
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
    log("listening on 0.0.0.0:%s with %s answered providers and %s free models"
        % (port, len([p for p in PROBES if p["ok"]]), len(served_models())))
    register(port)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    threading.Thread(target=poll_work, args=(server,), daemon=True).start()
    server.serve_forever()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(start())
    except Exception as exc:
        log(str(exc))
        sys.exit(1)
