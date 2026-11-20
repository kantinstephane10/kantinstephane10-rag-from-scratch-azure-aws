"""Local-only OpenAI settings; role routes are editable in the UI.

Windows: API key is encrypted using Windows DPAPI for the logged-in account.
Elsewhere: API key is memory-only (or OPENAI_API_KEY); never persist cleartext.
Only serve this local configuration API to trusted loopback clients.
"""
import ctypes
import json
import os
import re
import threading
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import MODELS, ARCHITECT, BUSINESS, ROUTINE
from roles import ROLES
from store import StoreError

MODEL_ID = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._:-]{1,120}$')
FAMILIES = ('architecture', 'engineering', 'business', 'routine')


def family_for(role):
    if role in ARCHITECT: return 'architecture'
    if role in BUSINESS: return 'business'
    if role in ROUTINE: return 'routine'
    return 'engineering'


def _dpapi(value, decrypt=False):
    if os.name != 'nt':
        raise StoreError('Encrypted key storage is supported only on Windows. On other systems keys stay in memory.')
    # Windows CryptProtectData encrypts for the current logged-in Windows account.
    import ctypes.wintypes
    from ctypes import c_void_p, c_char, c_uint, POINTER, byref
    class BLOB(ctypes.Structure):
        _fields_ = [('cbData', ctypes.wintypes.DWORD), ('pbData', POINTER(ctypes.c_ubyte))]
    inp = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
    source = BLOB(len(value), ctypes.cast(inp, POINTER(ctypes.c_ubyte)))
    output = BLOB()
    crypt = ctypes.windll.crypt32
    fn = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    fn.restype = ctypes.wintypes.BOOL
    if decrypt:
        fn.argtypes = [POINTER(BLOB), POINTER(c_void_p), POINTER(BLOB), c_void_p, c_void_p, ctypes.wintypes.DWORD, POINTER(BLOB)]
        ok = fn(byref(source), None, None, None, None, 0, byref(output))
    else:
        fn.argtypes = [POINTER(BLOB), ctypes.wintypes.LPCWSTR, POINTER(BLOB), c_void_p, c_void_p, ctypes.wintypes.DWORD, POINTER(BLOB)]
        ok = fn(byref(source), 'AI Product Builder local key', None, None, None, 0, byref(output))
    if not ok:
        raise StoreError('Windows could not encrypt/decrypt the API key for this account.')
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree.argtypes = [c_void_p]
        ctypes.windll.kernel32.LocalFree(ctypes.cast(output.pbData, c_void_p))


class RuntimeSettings:
    def __init__(self, base):
        self.lock = threading.RLock()
        self.directory = Path(base)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'runtime-settings.json'
        self.key_path = self.directory / 'openai-key.dpapi'
        self.data = {
            'mode': 'demo',
            'models': dict(MODELS),
            'role_models': {},
            'meeting_model': '',
        }
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding='utf8'))
                if isinstance(saved, dict):
                    self._update_data(saved, allow_mode=True)
            except (ValueError, OSError, StoreError):
                # Fail closed: do not silently run live mode after corrupted config.
                self.data['mode'] = 'demo'
        self._key = os.getenv('OPENAI_API_KEY', '').strip()
        if not self._key and self.key_path.exists() and os.name == 'nt':
            try:
                self._key = _dpapi(self.key_path.read_bytes(), decrypt=True).decode('utf8')
            except (ValueError, OSError, UnicodeError, StoreError):
                self.data['mode'] = 'demo'
        if not self._key:
            self.data['mode'] = 'demo'

    @property
    def key(self):
        with self.lock: return self._key

    def snapshot(self):
        with self.lock:
            data = json.loads(json.dumps(self.data))
            data['api_key_configured'] = bool(self._key)
            data['key_storage'] = 'Windows DPAPI' if os.name == 'nt' else 'session only'
            data['roles'] = {role: self.model_for(role) for role in ROLES}
            return data

    def _update_data(self, supplied, allow_mode=False):
        data = dict(self.data)
        if 'models' in supplied:
            models = supplied['models']
            if not isinstance(models, dict) or set(models) != set(FAMILIES):
                raise StoreError('Supply all four model groups: architecture, engineering, business, routine.')
            if any(not isinstance(m, str) or not MODEL_ID.fullmatch(m.strip()) for m in models.values()):
                raise StoreError('All model IDs must be valid API model identifiers.')
            data['models'] = {k: models[k].strip() for k in FAMILIES}
        if 'role_models' in supplied:
            roles = supplied['role_models']
            if not isinstance(roles, dict) or any(r not in ROLES for r in roles):
                raise StoreError('Unknown agent role.')
            if any(not isinstance(m, str) or (m.strip() and not MODEL_ID.fullmatch(m.strip())) for m in roles.values()):
                raise StoreError('Invalid role-specific model ID.')
            data['role_models'] = {r: m.strip() for r, m in roles.items() if m.strip()}
        if 'meeting_model' in supplied:
            value = supplied['meeting_model']
            if not isinstance(value, str) or (value.strip() and not MODEL_ID.fullmatch(value.strip())):
                raise StoreError('Invalid meeting model ID.')
            data['meeting_model'] = value.strip()
        # Legacy token/call caps are ignored, including values saved by older versions.
        for old_limit in ('max_output_tokens','max_calls_per_project','max_total_tokens_per_project'):
            data.pop(old_limit, None)
        if allow_mode and 'mode' in supplied:
            if supplied['mode'] not in ('demo', 'openai'):
                raise StoreError('Mode must be demo or openai.')
            data['mode'] = supplied['mode']
        self.data = data

    def update(self, supplied):
        if not isinstance(supplied, dict): raise StoreError('Expected a settings object.')
        with self.lock:
            old = json.loads(json.dumps(self.data))
            self._update_data(supplied, allow_mode=True)
            if self.data['mode'] == 'openai' and not self._key:
                self.data = old
                raise StoreError('Save an OpenAI API key before switching to Live GPT.')
            try:
                temporary = self.path.with_suffix('.tmp')
                temporary.write_text(json.dumps(self.data, indent=2), encoding='utf8')
                os.replace(temporary, self.path)
            except OSError as exc:
                self.data = old
                raise StoreError(f'Could not save runtime configuration: {type(exc).__name__}') from exc
            return self.snapshot()

    def save_key(self, key):
        if not isinstance(key, str) or not key.startswith('sk-') or not 20 <= len(key) <= 500:
            raise StoreError('Enter a valid-looking OpenAI API key; it is never echoed back.')
        with self.lock:
            if os.name == 'nt':
                encrypted = _dpapi(key.encode('utf8'))
                tmp = self.key_path.with_suffix('.tmp')
                tmp.write_bytes(encrypted)
                os.replace(tmp, self.key_path)
            self._key = key
            return self.snapshot()

    def clear_key(self):
        with self.lock:
            self._key = ''
            self.data['mode'] = 'demo'
            if self.key_path.exists(): self.key_path.unlink()
            self.update({'mode': 'demo'})
            return self.snapshot()

    def model_for(self, role, *, meeting=False, profile='standard'):
        with self.lock:
            if role not in ROLES: raise StoreError('Unknown agent role.')
            if meeting and self.data.get('meeting_model'):
                return self.data['meeting_model']
            # Per-agent overrides take precedence over group defaults.
            return self.data['role_models'].get(role) or self.data['models'][family_for(role)]

    def available_models(self):
        with self.lock:
            key = self._key
        if not key:
            raise StoreError('Save the OpenAI API key first.')
        req = Request('https://api.openai.com/v1/models', headers={'Authorization': f'Bearer {key}'})
        try:
            with urlopen(req, timeout=12) as response:
                data = json.load(response)
        except HTTPError as exc:
            raise StoreError(f'OpenAI model lookup failed: HTTP {exc.code}. Check key or permissions.') from None
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise StoreError(f'OpenAI model lookup failed: {type(exc).__name__}.') from exc
        return sorted([m['id'] for m in data.get('data', []) if isinstance(m.get('id'), str) and m['id'].startswith(('gpt-', 'o1', 'o3', 'o4'))])
