"""
Hermes Agent — Railway admin server.

Serves an admin UI on $PORT, manages the Hermes gateway as a subprocess.
The gateway is started automatically on boot if a provider API key is present.
"""

import asyncio
import base64
import json
import os
import re
import secrets
import signal
import time
import logging
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.authentication import (
    AuthCredentials,
    AuthenticationBackend,
    AuthenticationError,
    SimpleUser,
)
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Route
from starlette.templating import Jinja2Templates

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# In Docker, we prefer a fixed /app/data path if possible, or fallback to ~/.hermes
HERMES_HOME = os.environ.get("HERMES_HOME", "/app/data" if os.path.exists("/app") else str(Path.home() / ".hermes"))
ENV_FILE = Path(HERMES_HOME) / ".env"
PAIRING_DIR = Path(HERMES_HOME) / "pairing"
PAIRING_TTL = 3600

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = secrets.token_urlsafe(16)
    print(f"[server] Admin credentials — username: {ADMIN_USERNAME}  password: {ADMIN_PASSWORD}", flush=True)
else:
    print(f"[server] Admin username: {ADMIN_USERNAME}", flush=True)

# ── Env var registry ──────────────────────────────────────────────────────────
# (key, label, category, is_secret)
ENV_VARS = [
    ("LLM_MODEL",               "Model",                    "model",     False),
    ("OPENROUTER_API_KEY",       "OpenRouter",               "provider",  True),
    ("DEEPSEEK_API_KEY",         "DeepSeek",                 "provider",  True),
    ("DASHSCOPE_API_KEY",        "DashScope",                "provider",  True),
    ("GLM_API_KEY",              "GLM / Z.AI",               "provider",  True),
    ("KIMI_API_KEY",             "Kimi",                     "provider",  True),
    ("MINIMAX_API_KEY",          "MiniMax",                  "provider",  True),
    ("HF_TOKEN",                 "Hugging Face",             "provider",  True),
    ("PARALLEL_API_KEY",         "Parallel (search)",        "tool",      True),
    ("FIRECRAWL_API_KEY",        "Firecrawl (scrape)",       "tool",      True),
    ("TAVILY_API_KEY",           "Tavily (search)",          "tool",      True),
    ("FAL_KEY",                  "FAL (image gen)",          "tool",      True),
    ("BROWSERBASE_API_KEY",      "Browserbase key",          "tool",      True),
    ("BROWSERBASE_PROJECT_ID",   "Browserbase project",      "tool",      False),
    ("GITHUB_TOKEN",             "GitHub token",             "tool",      True),
    ("VOICE_TOOLS_OPENAI_KEY",   "OpenAI (voice/TTS)",       "tool",      True),
    ("HONCHO_API_KEY",           "Honcho (memory)",          "tool",      True),
    ("TELEGRAM_BOT_TOKEN",       "Bot Token",                "telegram",  True),
    ("TELEGRAM_ALLOWED_USERS",   "Allowed User IDs",         "telegram",  False),
    ("DISCORD_BOT_TOKEN",        "Bot Token",                "discord",   True),
    ("DISCORD_ALLOWED_USERS",    "Allowed User IDs",         "discord",   False),
    ("SLACK_BOT_TOKEN",          "Bot Token (xoxb-...)",     "slack",     True),
    ("SLACK_APP_TOKEN",          "App Token (xapp-...)",     "slack",     True),
    ("WHATSAPP_ENABLED",         "Enable WhatsApp",          "whatsapp",  False),
    ("EMAIL_ADDRESS",            "Email Address",            "email",     False),
    ("EMAIL_PASSWORD",           "Email Password",           "email",     True),
    ("EMAIL_IMAP_HOST",          "IMAP Host",                "email",     False),
    ("EMAIL_SMTP_HOST",          "SMTP Host",                "email",     False),
    ("MATTERMOST_URL",           "Server URL",               "mattermost",False),
    ("MATTERMOST_TOKEN",         "Bot Token",                "mattermost",True),
    ("MATRIX_HOMESERVER",        "Homeserver URL",           "matrix",    False),
    ("MATRIX_ACCESS_TOKEN",      "Access Token",             "matrix",    True),
    ("MATRIX_USER_ID",           "User ID",                  "matrix",    False),
    ("GATEWAY_ALLOW_ALL_USERS",  "Allow all users",          "gateway",   False),
    ("ADMIN_USERNAME",           "Admin username",           "admin",     False),
    ("ADMIN_PASSWORD",           "Admin password",           "admin",     True),
]

SECRET_KEYS  = {k for k, _, _, s in ENV_VARS if s}
PROVIDER_KEYS = [k for k, _, c, _ in ENV_VARS if c == "provider"]
CHANNEL_MAP  = {
    "Telegram":    "TELEGRAM_BOT_TOKEN",
    "Discord":     "DISCORD_BOT_TOKEN",
    "Slack":       "SLACK_BOT_TOKEN",
    "WhatsApp":    "WHATSAPP_ENABLED",
    "Email":       "EMAIL_ADDRESS",
    "Mattermost":  "MATTERMOST_TOKEN",
    "Matrix":      "MATRIX_ACCESS_TOKEN",
}


# ── .env helpers ──────────────────────────────────────────────────────────────
def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        out[k.strip()] = v
    return out


def write_config_yaml(data: dict[str, str]) -> None:
    """Write a minimal config.yaml so hermes picks up the model and provider."""
    model = data.get("LLM_MODEL", "")

    # For Custom Providers, use standard openai/ prefix.
    # Hermes explicitly extracts OPENAI_API_KEY when this prefix is used.
    if data.get("ACTIVE_CUSTOM_PROVIDER"):
        if not model.startswith("openai/") and "/" not in model:
            model = f"openai/{model}"

    config_path = Path(HERMES_HOME) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(f"""\
model:
  default: "{model}"
  provider: "auto"

terminal:
  backend: "local"
  timeout: 60
  cwd: "/tmp"

agent:
  max_iterations: 50

data_dir: "{HERMES_HOME}"
""")


def write_env(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cat_order = ["model", "provider", "tool",
                 "telegram", "discord", "slack", "whatsapp",
                 "email", "mattermost", "matrix", "gateway"]
    cat_labels = {
        "model": "Model", "provider": "Providers", "tool": "Tools",
        "telegram": "Telegram", "discord": "Discord", "slack": "Slack",
        "whatsapp": "WhatsApp", "email": "Email",
        "mattermost": "Mattermost", "matrix": "Matrix", "gateway": "Gateway",
    }
    key_cat = {k: c for k, _, c, _ in ENV_VARS}
    grouped: dict[str, list[str]] = {c: [] for c in cat_order}
    grouped["other"] = []

    for k, v in data.items():
        if not v:
            continue
        cat = key_cat.get(k, "other")
        grouped.setdefault(cat, []).append(f"{k}={v}")

    lines: list[str] = []
    for cat in cat_order:
        entries = sorted(grouped.get(cat, []))
        if entries:
            lines.append(f"# {cat_labels.get(cat, cat)}")
            lines.extend(entries)
            lines.append("")
    if grouped["other"]:
        lines.append("# Other")
        lines.extend(sorted(grouped["other"]))
        lines.append("")

    path.write_text("\n".join(lines))


def is_secret_key(k: str) -> bool:
    if k in SECRET_KEYS:
        return True
    k_upper = k.upper()
    return any(x in k_upper for x in ("_KEY", "_TOKEN", "SECRET", "PASSWORD"))

def mask(data: dict[str, str]) -> dict[str, str]:
    return {
        k: (v[:8] + "***" if len(v) > 8 else "***") if is_secret_key(k) and v else v
        for k, v in data.items()
    }


def unmask(new: dict[str, str], existing: dict[str, str]) -> dict[str, str]:
    return {
        k: (existing.get(k, "") if is_secret_key(k) and v.endswith("***") else v)
        for k, v in new.items()
    }


# ── Auth (Custom Session Manager) ─────────────────────────────────────────────
SESSIONS: dict[str, str] = {} # token -> username

def guard(request: Request):
    token = request.cookies.get("session_id")
    if not token or token not in SESSIONS:
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return RedirectResponse(url="/login", status_code=303)


async def page_login(request: Request):
    token = request.cookies.get("session_id")
    if token and token in SESSIONS:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"error": ""})


async def api_login(request: Request):
    try:
        data = await request.form()
        user = data.get("username")
        pw = data.get("password")
        if user == ADMIN_USERNAME and pw == ADMIN_PASSWORD:
            token = secrets.token_urlsafe(32)
            SESSIONS[token] = user
            resp = RedirectResponse(url="/", status_code=303)
            # Cookie expires in 1 day
            resp.set_cookie("session_id", token, httponly=True, max_age=86400, samesite="lax")
            return resp
        return templates.TemplateResponse(request, "login.html", {"error": "Invalid username or password"})
    except Exception:
        return RedirectResponse(url="/login", status_code=303)


async def api_logout(request: Request):
    token = request.cookies.get("session_id")
    if token in SESSIONS:
        del SESSIONS[token]
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie("session_id")
    return resp


# ── Gateway manager ───────────────────────────────────────────────────────────
class Gateway:
    def __init__(self):
        self.proc: asyncio.subprocess.Process | None = None
        self.state = "stopped"
        self.logs: deque[str] = deque(maxlen=500)
        self.started_at: float | None = None
        self.restarts = 0

    async def start(self):
        if self.proc and self.proc.returncode is None:
            return
        self.state = "starting"
        try:
            # Build a clean environment for the subprocess.
            # Strategy: start from os.environ but STRIP all known provider API keys completely.
            # Then re-inject ONLY what is in our .env file.
            # This prevents Docker/Railway static env vars from "haunting" old providers.
            PROVIDER_STRIP_SUFFIXES = ('_API_KEY', '_TOKEN', '_API_BASE', '_BASE_URL', '_SECRET')
            PROVIDER_STRIP_PREFIXES = ('OPENROUTER', 'OPENAI', 'DEEPSEEK', 'DASHSCOPE',
                                       'GLM', 'KIMI', 'MINIMAX', 'HF_TOKEN', 'ANTHROPIC',
                                       'GEMINI', 'COHERE', 'MISTRAL', 'GROQ', 'TOGETHER',
                                       'ACTIVE_CUSTOM_PROVIDER')
            env = {}
            for k, v in os.environ.items():
                skip = (any(k.startswith(p) for p in PROVIDER_STRIP_PREFIXES) or
                        any(k.endswith(s) for s in PROVIDER_STRIP_SUFFIXES))
                if not skip:
                    env[k] = v

            env["HERMES_HOME"] = HERMES_HOME

            # Load saved .env and inject everything — this is the single source of truth
            saved_vars = read_env(ENV_FILE)
            env.update(saved_vars)
            
            # If custom provider is active, map to OPENAI_BASE_URL (not OPENAI_API_BASE!)
            # The Hermes docs say the correct var is OPENAI_BASE_URL
            # Also set HERMES_INFERENCE_PROVIDER=openai to bypass auto-detection
            active = saved_vars.get("ACTIVE_CUSTOM_PROVIDER", "")
            if active:
                pfx = active.upper()
                base_url = saved_vars.get("OPENAI_BASE_URL") or saved_vars.get(f"{pfx}_API_BASE", "")
                api_key  = saved_vars.get("OPENAI_API_KEY") or saved_vars.get(f"{pfx}_API_KEY", "")
                if base_url:
                    env["OPENAI_BASE_URL"] = base_url
                if api_key:
                    env["OPENAI_API_KEY"] = api_key
                # Use auto provider, Litellm will route based on the openai/ prefix
                env["HERMES_INFERENCE_PROVIDER"] = "auto"
                print(f"[gateway] custom active={active} | base={base_url} | key={'set' if api_key else 'MISSING'}", flush=True)
            
            model = env.get("LLM_MODEL", "")
            # Debug: print all provider-relevant env vars being passed to subprocess
            print(f"[gateway] Starting with HERMES_HOME={HERMES_HOME}", flush=True)
            print(f"[gateway] LLM_MODEL={model or '⚠ NOT SET'}", flush=True)
            print(f"[gateway] OPENAI_API_BASE={env.get('OPENAI_API_BASE', '⚠ NOT SET')}", flush=True)
            print(f"[gateway] OPENAI_API_KEY={'SET (' + env['OPENAI_API_KEY'][:8] + '...)' if env.get('OPENAI_API_KEY') else '⚠ NOT SET'}", flush=True)
            print(f"[gateway] ACTIVE_CUSTOM_PROVIDER={env.get('ACTIVE_CUSTOM_PROVIDER', '⚠ NOT SET')}", flush=True)
            print(f"[gateway] OPENROUTER_API_KEY={'SET' if env.get('OPENROUTER_API_KEY') else 'NOT SET (good)'}", flush=True)
            print(f"[gateway] .env contents: {list(saved_vars.keys())}", flush=True)
            # Write config.yaml so hermes picks up the model (env vars alone aren't always enough)
            write_config_yaml(read_env(ENV_FILE))
            self.proc = await asyncio.create_subprocess_exec(
                "hermes", "gateway",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            self.state = "running"
            self.started_at = time.time()
            asyncio.create_task(self._drain())
        except Exception as e:
            self.state = "error"
            self.logs.append(f"[error] Failed to start: {e}")

    async def stop(self):
        if not self.proc or self.proc.returncode is not None:
            self.state = "stopped"
            return
        self.state = "stopping"
        self.proc.terminate()
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            self.proc.kill()
            await self.proc.wait()
        self.state = "stopped"
        self.started_at = None

    async def restart(self):
        await self.stop()
        self.restarts += 1
        await self.start()

    async def _drain(self):
        assert self.proc and self.proc.stdout
        async for raw in self.proc.stdout:
            line = ANSI_ESCAPE.sub("", raw.decode(errors="replace").rstrip())
            self.logs.append(line)
        if self.state == "running":
            self.state = "error"
            self.logs.append(f"[error] Gateway exited (code {self.proc.returncode})")

    def status(self) -> dict:
        uptime = int(time.time() - self.started_at) if self.started_at and self.state == "running" else None
        return {
            "state":    self.state,
            "pid":      self.proc.pid if self.proc and self.proc.returncode is None else None,
            "uptime":   uptime,
            "restarts": self.restarts,
        }


gw = Gateway()
cfg_lock = asyncio.Lock()


# ── Route handlers ────────────────────────────────────────────────────────────
async def page_index(request: Request):
    if err := guard(request): return err
    token = request.cookies.get("session_id")
    user = SESSIONS.get(token) if token else None
    return templates.TemplateResponse(request, "index.html", {"user": user})


async def route_health(request: Request):
    return JSONResponse({"status": "ok", "gateway": gw.state})


async def api_config_get(request: Request):
    if err := guard(request): return err
    async with cfg_lock:
        data = read_env(ENV_FILE)
    defs = [{"key": k, "label": l, "category": c, "secret": s} for k, l, c, s in ENV_VARS]
    return JSONResponse({"vars": mask(data), "defs": defs})


async def api_config_put(request: Request):
    if err := guard(request): return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    try:
        restart = body.pop("_restart", False)
        new_vars = body.get("vars", {})
        async with cfg_lock:
            existing = read_env(ENV_FILE)

            # Keys that are safe to keep from existing (not provider-related)
            SAFE_CATEGORIES = {"telegram", "discord", "slack", "whatsapp",
                               "email", "mattermost", "matrix", "gateway", "admin"}
            safe_keys = {k for k, _, c, _ in ENV_VARS if c in SAFE_CATEGORIES}

            # Provider/dynamic keys suffixes - these must NEVER be preserved from old state
            PROVIDER_SUFFIXES = ('_API_KEY', '_API_BASE', '_BASE_URL', '_SECRET')
            PROVIDER_PREFIXES = ('OPENROUTER', 'OPENAI', 'DEEPSEEK', 'DASHSCOPE',
                                 'GLM', 'KIMI', 'MINIMAX', 'HF_TOKEN', 'ANTHROPIC',
                                 'GEMINI', 'COHERE', 'MISTRAL', 'GROQ', 'TOGETHER',
                                 'ACTIVE_CUSTOM_PROVIDER')

            def is_provider_key(k: str) -> bool:
                return (any(k.startswith(p) for p in PROVIDER_PREFIXES) or
                        (any(k.endswith(s) for s in PROVIDER_SUFFIXES) and k not in safe_keys))

            # Start fresh: only keep safe non-provider keys from existing
            merged: dict[str, str] = {}
            for k, v in existing.items():
                if k in safe_keys and v:
                    merged[k] = v
                elif not is_provider_key(k) and k not in {kk for kk, _, _, _ in ENV_VARS} and v:
                    merged[k] = v  # unknown keys (custom user vars)

            # Apply new_vars on top (unmask secrets)
            applied = unmask(new_vars, existing)
            for k, v in applied.items():
                if v:  # only write non-empty
                    merged[k] = v

            write_env(ENV_FILE, merged)
            # Persist inference provider directive so Hermes dotenv load doesn't override it
            if merged.get("ACTIVE_CUSTOM_PROVIDER"):
                merged["HERMES_INFERENCE_PROVIDER"] = "auto"
                
                # To override the in-memory load correctly, we should prefix it
                # directly here if it's not already
                mod = merged.get("LLM_MODEL", "")
                if mod and not mod.startswith("openai/") and "/" not in mod:
                    merged["HERMES_MODEL"] = f"openai/{mod}"
                else:
                    merged["HERMES_MODEL"] = mod

                write_env(ENV_FILE, merged)
            write_config_yaml(merged)
        if restart:
            asyncio.create_task(gw.restart())
        return JSONResponse({"ok": True, "restarting": restart})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_reset_providers(request: Request):
    """Nuke all provider and custom api keys from .env, keeping only channels/tools."""
    if err := guard(request): return err
    async with cfg_lock:
        existing = read_env(ENV_FILE)
        SAFE_CATEGORIES = {"telegram", "discord", "slack", "whatsapp",
                           "email", "mattermost", "matrix", "gateway", "admin", "tool"}
        safe_keys = {k for k, _, c, _ in ENV_VARS if c in SAFE_CATEGORIES}
        cleaned = {k: v for k, v in existing.items() if k in safe_keys and v}
        write_env(ENV_FILE, cleaned)
        write_config_yaml(cleaned)
    return JSONResponse({"ok": True, "removed": len(existing) - len(cleaned)})


async def api_status(request: Request):
    if err := guard(request): return err
    data = read_env(ENV_FILE)
    providers = {
        k.replace("_API_KEY","").replace("_TOKEN","").replace("HF_","HuggingFace ").replace("_"," ").title():
        {"configured": bool(data.get(k))}
        for k in PROVIDER_KEYS
    }
    
    # Custom providers: only show _API_KEY / _TOKEN entries (not _BASE/_URL which are not credentials)
    known_tool_keys = {k for k, _, c, _ in ENV_VARS if c != "provider"}
    for k, v in data.items():
        if k not in PROVIDER_KEYS and k not in known_tool_keys and k != "LLM_MODEL":
            if k.endswith("_API_KEY") or k.endswith("_TOKEN"):
                name = k.replace("_API_KEY","").replace("_TOKEN","").replace("_"," ").title()
                providers[name] = {"configured": bool(v)}
    channels = {
        name: {"configured": bool(v := data.get(key,"")) and v.lower() not in ("false","0","no")}
        for name, key in CHANNEL_MAP.items()
    }
    return JSONResponse({"gateway": gw.status(), "providers": providers, "channels": channels})


async def api_logs(request: Request):
    if err := guard(request): return err
    return JSONResponse({"lines": list(gw.logs)})


async def api_gw_start(request: Request):
    if err := guard(request): return err
    asyncio.create_task(gw.start())
    return JSONResponse({"ok": True})


async def api_gw_stop(request: Request):
    if err := guard(request): return err
    asyncio.create_task(gw.stop())
    return JSONResponse({"ok": True})


async def api_gw_restart(request: Request):
    if err := guard(request): return err
    asyncio.create_task(gw.restart())
    return JSONResponse({"ok": True})


async def api_config_reset(request: Request):
    if err := guard(request): return err
    asyncio.create_task(gw.stop())
    async with cfg_lock:
        if ENV_FILE.exists():
            ENV_FILE.unlink()
        write_config_yaml({})
    return JSONResponse({"ok": True})


# ── Pairing ───────────────────────────────────────────────────────────────────
def _pjson(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def _wjson(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    try: os.chmod(path, 0o600)
    except OSError: pass


def _platforms(suffix: str) -> list[str]:
    if not PAIRING_DIR.exists(): return []
    return [f.stem.rsplit(f"-{suffix}", 1)[0] for f in PAIRING_DIR.glob(f"*-{suffix}.json")]


async def api_pairing_pending(request: Request):
    if err := guard(request): return err
    now = time.time()
    out = []
    for p in _platforms("pending"):
        for code, info in _pjson(PAIRING_DIR / f"{p}-pending.json").items():
            if now - info.get("created_at", now) <= PAIRING_TTL:
                out.append({"platform": p, "code": code,
                            "user_id": info.get("user_id",""), "user_name": info.get("user_name",""),
                            "age_minutes": int((now - info.get("created_at", now)) / 60)})
    return JSONResponse({"pending": out})


async def api_pairing_approve(request: Request):
    if err := guard(request): return err
    try: body = await request.json()
    except Exception: return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    platform, code = body.get("platform",""), body.get("code","").upper().strip()
    if not platform or not code:
        return JSONResponse({"error": "platform and code required"}, status_code=400)
    pending_path = PAIRING_DIR / f"{platform}-pending.json"
    pending = _pjson(pending_path)
    if code not in pending:
        return JSONResponse({"error": "Code not found"}, status_code=404)
    entry = pending.pop(code)
    _wjson(pending_path, pending)
    approved = _pjson(PAIRING_DIR / f"{platform}-approved.json")
    approved[entry["user_id"]] = {"user_name": entry.get("user_name",""), "approved_at": time.time()}
    _wjson(PAIRING_DIR / f"{platform}-approved.json", approved)
    return JSONResponse({"ok": True})


async def api_pairing_deny(request: Request):
    if err := guard(request): return err
    try: body = await request.json()
    except Exception: return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    platform, code = body.get("platform",""), body.get("code","").upper().strip()
    p = PAIRING_DIR / f"{platform}-pending.json"
    pending = _pjson(p)
    if code in pending:
        del pending[code]
        _wjson(p, pending)
    return JSONResponse({"ok": True})


async def api_pairing_approved(request: Request):
    if err := guard(request): return err
    out = []
    for p in _platforms("approved"):
        for uid, info in _pjson(PAIRING_DIR / f"{p}-approved.json").items():
            out.append({"platform": p, "user_id": uid,
                        "user_name": info.get("user_name",""), "approved_at": info.get("approved_at",0)})
    return JSONResponse({"approved": out})


async def api_pairing_revoke(request: Request):
    if err := guard(request): return err
    try: body = await request.json()
    except Exception: return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    platform, uid = body.get("platform",""), body.get("user_id","")
    if not platform or not uid:
        return JSONResponse({"error": "platform and user_id required"}, status_code=400)
    p = PAIRING_DIR / f"{platform}-approved.json"
    approved = _pjson(p)
    if uid in approved:
        del approved[uid]
        _wjson(p, approved)
    return JSONResponse({"ok": True})


# ── App lifecycle ─────────────────────────────────────────────────────────────
async def auto_start():
    data = read_env(ENV_FILE)
    
    has_provider = any(data.get(k) for k in PROVIDER_KEYS)
    if not has_provider:
        for k, v in data.items():
            if is_secret_key(k) and v and "TELEGRAM" not in k and "DISCORD" not in k and "SLACK" not in k and "MATTERMOST" not in k and "MATRIX" not in k:
                has_provider = True
                break

    if has_provider or data.get("LLM_MODEL"):
        asyncio.create_task(gw.start())
    else:
        print("[server] No provider key found — gateway not started. Configure one in the admin UI.", flush=True)


@asynccontextmanager
async def lifespan(app):
    await auto_start()
    yield
    await gw.stop()


routes = [
    Route("/",                          page_index),
    Route("/health",                    route_health),
    Route("/api/config",                api_config_get,      methods=["GET"]),
    Route("/api/config",                api_config_put,      methods=["PUT"]),
    Route("/api/status",                api_status),
    Route("/api/logs",                  api_logs),
    Route("/api/gateway/start",         api_gw_start,        methods=["POST"]),
    Route("/api/gateway/stop",          api_gw_stop,         methods=["POST"]),
    Route("/api/gateway/restart",       api_gw_restart,      methods=["POST"]),
    Route("/api/config/reset",          api_config_reset,    methods=["POST"]),
    Route("/api/config/reset-providers", api_reset_providers, methods=["POST"]),
    Route("/api/pairing/pending",       api_pairing_pending),
    Route("/api/pairing/approve",       api_pairing_approve, methods=["POST"]),
    Route("/api/pairing/deny",          api_pairing_deny,    methods=["POST"]),
    Route("/api/pairing/approved",      api_pairing_approved),
    Route("/api/pairing/revoke",        api_pairing_revoke,  methods=["POST"]),
    Route("/login",                     page_login,          methods=["GET"]),
    Route("/login",                     api_login,           methods=["POST"]),
    Route("/logout",                    api_logout),
]

app = Starlette(
    routes=routes,
    middleware=[],
    lifespan=lifespan,
)

if __name__ == "__main__":
    import uvicorn
    
    # Silence repetitive polling logs
    logging.getLogger("uvicorn.access").addFilter(
        lambda record: all(path not in record.getMessage() for path in ["/api/logs", "/api/status", "/api/pairing"])
    )

    port = int(os.environ.get("PORT", "8080"))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info", loop="asyncio")
    server = uvicorn.Server(config)

    def _shutdown():
        loop.create_task(gw.stop())
        server.should_exit = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown)

    loop.run_until_complete(server.serve())
