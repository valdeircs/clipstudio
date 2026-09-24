"""Private local API configuration. Public status never contains credentials."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading

APP_DIR = Path(__file__).resolve().parent
CONFIG_FILE = APP_DIR / 'data' / 'connections.json'
LOCK = threading.RLock()
OWNED_ACTOR = 'agency-shift/youtube-transcript-scraper'
LEGACY_DEFAULT_ACTOR = 'pintostudio/youtube-transcript-scraper'
DEFAULTS = {'pipeline': 'local_ai', 'ai_provider': 'gemini', 'ai_model': 'gemini-3.5-flash-lite', 'apify_actor': OWNED_ACTOR, 'apify_max_charge_usd': 0.02, 'apify_paid_allowed': False}


def _effective_config(stored):
    config = {**DEFAULTS, **stored}
    actor = config.get('apify_actor')
    if isinstance(actor, str) and actor in {LEGACY_DEFAULT_ACTOR, LEGACY_DEFAULT_ACTOR.replace('/', '~')}:
        config['apify_actor'] = OWNED_ACTOR
    return config


def requires_usage_consent(config):
    actor = config.get('apify_actor', OWNED_ACTOR)
    return not isinstance(actor, str) or actor.replace('~', '/') != OWNED_ACTOR


def _stored():
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _apify_cli_token():
    # The user requested their existing Apify connection. Use its documented
    # OS credential store, keeping the result in memory only.
    node = shutil.which('node') or '/opt/homebrew/bin/node'
    helper = APP_DIR / 'integrations' / 'read-apify-token.cjs'
    if not Path(node).is_file() or not helper.is_file():
        return ''
    try:
        auth = json.loads((Path.home() / '.apify' / 'auth.json').read_text())
        if auth.get('secretsBackend') != 'keyring':
            return ''
        result = subprocess.run([node, str(helper)], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else ''
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return ''


def load(*, resolve_apify=True):
    with LOCK:
        config = _effective_config(_stored())
    apify_env = os.environ.get('APIFY_TOKEN', '') if resolve_apify else ''
    apify_saved = config.get('apify_token', '') if resolve_apify else ''
    token = (apify_env or apify_saved or _apify_cli_token()) if resolve_apify else ''
    config['apify_token'] = token
    config['apify_token_source'] = 'environment' if apify_env else ('saved locally' if apify_saved else ('Apify CLI' if token else 'not connected'))
    provider = config['ai_provider']
    env_key = (os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')) if provider == 'gemini' else os.environ.get('OPENAI_API_KEY')
    config['ai_api_key'] = env_key or config.get(provider + '_api_key', '')
    return config


def public_status():
    config = load()
    local_ai_ready = bool(config.get('ai_api_key'))
    apify_ai_ready = bool(config.get('apify_token') and local_ai_ready)
    ready = {'local': True, 'local_ai': local_ai_ready, 'apify_ai': apify_ai_ready}.get(config['pipeline'], False)
    return {key: config[key] for key in DEFAULTS} | {'apify_connected': bool(config.get('apify_token')), 'apify_token_source': config['apify_token_source'], 'ai_connected': local_ai_ready, 'ready': ready, 'local_ai_ready': local_ai_ready, 'apify_ai_ready': apify_ai_ready, 'requires_usage_consent': requires_usage_consent(config)}


def save(body):
    if not isinstance(body, dict):
        raise ValueError('Invalid connection settings.')
    with LOCK:
        config = _effective_config(_stored())
        provider = body.get('ai_provider', config['ai_provider'])
        if provider not in {'gemini', 'openai'}:
            raise ValueError('Choose Gemini or OpenAI.')
        if provider != config['ai_provider']:
            config['ai_model'] = 'gemini-3.5-flash-lite' if provider == 'gemini' else 'gpt-4.1-mini'
        config['ai_provider'] = provider
        pipeline = body.get('pipeline', config['pipeline'])
        if pipeline not in {'local_ai', 'apify_ai', 'local'}:
            raise ValueError('Choose local captions with AI, Apify with AI, or local analysis.')
        config['pipeline'] = pipeline
        if 'apify_actor' in body:
            actor = body['apify_actor']
            if not isinstance(actor, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}(?:[/~][A-Za-z0-9_-]{1,100})?', actor):
                raise ValueError('Enter a valid Apify actor identifier.')
            actor = actor.replace('~', '/')
            if actor != config['apify_actor']:
                config['apify_paid_allowed'] = False
            config['apify_actor'] = actor
        if 'apify_paid_allowed' in body:
            if not isinstance(body['apify_paid_allowed'], bool):
                raise ValueError('Invalid Apify usage preference.')
            config['apify_paid_allowed'] = body['apify_paid_allowed']
        if 'ai_model' in body:
            model = str(body['ai_model']).strip()
            if not re.fullmatch(r'[a-zA-Z0-9._-]{1,100}', model):
                raise ValueError('Enter a valid model name.')
            config['ai_model'] = model
        for field, saved_name in [('apify_token', 'apify_token'), ('ai_api_key', provider + '_api_key')]:
            value = body.get(field)
            if value:
                if not isinstance(value, str) or len(value) > 1000 or '\n' in value or '\r' in value:
                    raise ValueError('Invalid API key format.')
                config[saved_name] = value.strip()
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = CONFIG_FILE.with_suffix('.tmp')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump(config, out, ensure_ascii=False, indent=2)
        os.chmod(temporary, 0o600)
        os.replace(temporary, CONFIG_FILE)
    return public_status()
