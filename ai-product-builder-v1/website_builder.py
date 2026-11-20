"""Isolated, deterministic static website builder for an authorized project.

No model output is executed. Sites are versioned, checksum verified and exported as
portable HTML/CSS, without externally loaded scripts, trackers, or deployment.
"""
import hashlib
import html
import io
import json
import os
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path
from uuid import uuid4

from store import StoreError

THEMES = {'indigo': '#727bff', 'teal': '#20bfa8', 'amber': '#f4a53a', 'rose': '#fa6c97'}
TEMPLATES = {'business', 'portfolio', 'documentation'}
SITE_FILES = ('index.html', 'about.html', 'contact.html', 'styles.css', 'README.md', 'site.json')
_LOCKS = {}
_LOCK_GUARD = threading.Lock()


def project_lock(pid):
    with _LOCK_GUARD:
        return _LOCKS.setdefault(pid, threading.RLock())


def _root(store, pid):
    if not isinstance(pid, str) or not re.fullmatch(r'[a-f0-9]{12}', pid):
        raise StoreError('Invalid project ID.')
    return Path(store.filename).parent / 'websites' / pid


def _short(value, limit, label):
    if not isinstance(value, str):
        raise StoreError(f'{label} must be text.')
    value = ' '.join(value.split())
    if not value or len(value) > limit:
        raise StoreError(f'{label} must contain 1–{limit} characters.')
    if any(ord(x) < 32 for x in value):
        raise StoreError(f'{label} contains invalid control characters.')
    return value


def _clean_brief(text):
    # Show no more than a small, printable preview of the user's brief.
    text = re.sub(r'[#*_`\[\]]', '', text)
    words = ' '.join(text.split())
    return words[:720].rsplit(' ', 1)[0] or 'Explore our work and learn more.'


def _nav():
    return ('<nav aria-label="Primary navigation"><a href="index.html">Home</a>'
            '<a href="about.html">About</a><a href="contact.html">Contact</a></nav>')


def _page(title, description, content, brand, accent, template, active):
    heading = html.escape(title, quote=True)
    brand_safe = html.escape(brand, quote=True)
    description_safe = html.escape(description[:160], quote=True)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="{description_safe}"><meta name="color-scheme" content="dark light">
<title>{heading} | {brand_safe}</title><link rel="stylesheet" href="styles.css"></head>
<body data-template="{template}"><a class="skip-link" href="#main">Skip to content</a>
<header class="site-header"><div class="wrap header-inner"><a class="brand" href="index.html"><span class="brand-mark">✦</span> {brand_safe}</a>{_nav()}</div></header>
<main id="main">{content}</main><footer class="site-footer"><div class="wrap footer-inner"><strong>{brand_safe}</strong><span>Built for clarity. Designed to grow.</span><nav aria-label="Footer"><a href="about.html">About</a><a href="contact.html">Contact</a></nav></div></footer>
</body></html>'''


def _styles(accent):
    return '''/* Portable static website. No external CDNs, trackers or scripts. */
:root{color-scheme:dark;--accent:'''+accent+''';--bg:#080e1d;--panel:#121c30;--text:#f2f6ff;--muted:#a9b8d0;--line:#29354d;font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 90% 10%,#202e59 0%,transparent 37%),var(--bg);color:var(--text);line-height:1.65}
a{color:inherit;text-underline-offset:.25em}a:focus-visible{outline:3px solid var(--accent);outline-offset:4px;border-radius:3px}.wrap{width:min(1120px,calc(100% - 48px));margin:auto}
.site-header{position:relative;border-bottom:1px solid var(--line);background:#0b1325d9}.header-inner,.footer-inner{display:flex;align-items:center;justify-content:space-between;gap:24px;min-height:82px}.brand{font-size:1.2rem;text-decoration:none;font-weight:800;letter-spacing:-.025em}.brand-mark{color:var(--accent)}nav{display:flex;gap:24px;flex-wrap:wrap}nav a{text-decoration:none;color:var(--muted)}nav a:hover{color:var(--text)}
.hero{padding:100px 0 86px}.eyebrow{color:var(--accent);letter-spacing:.15em;text-transform:uppercase;font-size:.78rem;font-weight:800}.hero h1{font-size:clamp(2.5rem,6vw,5.6rem);line-height:1.04;letter-spacing:-.055em;max-width:860px;margin:20px 0}.hero p,.intro{color:var(--muted);font-size:1.15rem;max-width:740px}.actions{display:flex;gap:14px;flex-wrap:wrap;margin-top:30px}.button{display:inline-block;padding:13px 24px;background:var(--accent);color:#071226;border-radius:10px;text-decoration:none;font-weight:750}.button.secondary{background:transparent;color:var(--text);border:1px solid var(--line)}
.section{padding:60px 0 95px}.section h2{font-size:clamp(1.8rem,3vw,2.7rem);letter-spacing:-.04em}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:26px;min-height:190px}.card .num{color:var(--accent);font-weight:800}.card h3{font-size:1.16rem;line-height:1.35;margin:14px 0 10px}.card p{color:var(--muted);margin:0}.page-title{padding:85px 0 24px}.page-title h1{font-size:clamp(2.5rem,5vw,4.4rem);letter-spacing:-.05em;margin:12px 0}.content-panel{padding:28px 32px;border:1px solid var(--line);border-radius:16px;background:var(--panel);margin:18px 0 100px}.content-panel p{white-space:pre-wrap;color:var(--muted)}
.site-footer{border-top:1px solid var(--line);padding:25px 0}.footer-inner{color:var(--muted);font-size:.9rem}.skip-link{position:absolute;left:-1000px;top:0;background:var(--accent);color:#071226;padding:8px}.skip-link:focus{left:10px;z-index:3}
@media(max-width:760px){.header-inner,.footer-inner{align-items:flex-start;flex-direction:column;padding:22px 0}.hero{padding:72px 0 50px}.grid{grid-template-columns:1fr}.wrap{width:min(100% - 32px,1120px)}nav{gap:15px}.section{padding:35px 0 65px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
'''


def _make_files(project, payload):
    if not isinstance(payload, dict):
        raise StoreError('Website settings must be an object.')
    title = _short(payload.get('title') or project['name'], 90, 'Website title')
    tagline = _short(payload.get('tagline') or f'A clearer way to explore {project["name"]}.', 160, 'Tagline')
    template = payload.get('template', 'business')
    theme = payload.get('theme', 'indigo')
    if template not in TEMPLATES or theme not in THEMES:
        raise StoreError('Choose a supported website template and color theme.')
    brief = _clean_brief(project['brief'])
    name, tag, short = (html.escape(x, quote=True) for x in (title, tagline, brief))
    cards = {'business': [('What we offer', 'A focused foundation for your goals.'), ('Our approach', 'Thoughtful design with practical results.'), ('Get started', 'See how this idea comes together.')],
             'portfolio': [('Selected work', 'Explore what we create and why it matters.'), ('Our process', 'From an idea to a well-crafted result.'), ('Collaborate', 'Start a new project together.')],
             'documentation': [('Overview', 'Understand the core concept and objectives.'), ('Getting started', 'Explore the information you need first.'), ('Support', 'Find answers and contact the project owner.')]}[template]
    card_markup = ''.join(f'<article class="card"><span class="num">0{i}</span><h3>{html.escape(a)}</h3><p>{html.escape(b)}</p></article>' for i, (a, b) in enumerate(cards, 1))
    home_content = (f'<section class="hero"><div class="wrap"><p class="eyebrow">{html.escape(template.title())} website</p>'
                    f'<h1>{name}</h1><p>{tag}</p><div class="actions"><a class="button" href="about.html">Explore the project</a>'
                    '<a class="button secondary" href="contact.html">Get in touch</a></div></div></section>'
                    f'<section class="section"><div class="wrap"><p class="eyebrow">At a glance</p><h2>Built around your vision.</h2>'
                    f'<div class="grid">{card_markup}</div></div></section>')
    about_content = (f'<div class="wrap"><section class="page-title"><p class="eyebrow">About</p><h1>Our story</h1>'
                     f'<p class="intro">{tag}</p></section><section class="content-panel"><h2>What we are building</h2><p>{short}</p>'
                     '</section></div>')
    contact_content = ('<div class="wrap"><section class="page-title"><p class="eyebrow">Contact</p><h1>Let’s connect.</h1>'
                       '<p class="intro">Have a question or an idea? Reach out using your preferred contact method.</p>'
                       '</section><section class="content-panel"><h2>Contact information</h2>'
                       '<p>This website is a static template. Configure your real contact address before publication; no messages are collected by this page.</p>'
                       '</section></div>')
    files = {
        'index.html': _page(title, tagline, home_content, title, THEMES[theme], template, 'home'),
        'about.html': _page('About', tagline, about_content, title, THEMES[theme], template, 'about'),
        'contact.html': _page('Contact', tagline, contact_content, title, THEMES[theme], template, 'contact'),
        'styles.css': _styles(THEMES[theme]),
        'README.md': f'# {title}\n\nStatic website generated by AI Product Builder.\n\nOpen `index.html` locally to preview.\n\nTemplate: {template}\nTheme: {theme}\n\nBefore public deployment, add real contact details, your privacy policy and accessibility review.\nNo tracking or external scripts are included.\n',
        'site.json': json.dumps({'title': title, 'tagline': tagline, 'template': template, 'theme': theme}, ensure_ascii=False, indent=2),
    }
    return {name: data.encode('utf-8') for name, data in files.items()}, {'title': title, 'template': template, 'theme': theme}


def build_website(store, pid, settings):
    with project_lock(pid):
        project = store.project(pid)
        if project['intake']['product_type'] != 'website':
            raise StoreError('This project is not a website project.')
        files, meta = _make_files(project, settings)
        root = _root(store, pid)
        version = uuid4().hex
        path = root / 'generations' / version
        path.mkdir(parents=True, exist_ok=False)
        checksums = {}
        try:
            for name, data in files.items():
                (path / name).write_bytes(data)
                checksums[name] = hashlib.sha256(data).hexdigest()
            document = {'generation': version, 'generated_at': int(time.time()), 'files': checksums, **meta}
            # A deletion in progress cannot be overridden by a completed build.
            store.project(pid)
            tmp = root / ('active.' + uuid4().hex + '.tmp')
            tmp.write_text(json.dumps(document, ensure_ascii=False), encoding='utf-8')
            os.replace(tmp, root / 'active.json')
        except Exception:
            shutil.rmtree(path, ignore_errors=True)
            raise
        return {**document, 'built': True}


def _active(store, pid):
    project = store.project(pid)
    if project['intake']['product_type'] != 'website':
        raise StoreError('This project is not a website project.')
    root = _root(store, pid)
    try:
        manifest = json.loads((root / 'active.json').read_text('utf-8'))
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise StoreError('No verified website build exists. Generate a website first.') from exc
    if not isinstance(manifest, dict) or not re.fullmatch(r'[a-f0-9]{32}', str(manifest.get('generation', ''))):
        raise StoreError('Website generation metadata is invalid.')
    if not isinstance(manifest.get('files'), dict) or set(manifest['files']) != set(SITE_FILES):
        raise StoreError('Website file manifest is invalid.')
    folder = root / 'generations' / manifest['generation']
    return manifest, folder


def website_status(store, pid):
    project = store.project(pid)
    if project['intake']['product_type'] != 'website':
        raise StoreError('This project is not a website project.')
    try:
        manifest, _ = _active(store, pid)
        return {'built': True, **manifest}
    except StoreError as exc:
        if 'No verified website build' in str(exc):
            return {'built': False, 'files': {}}
        raise


def website_file(store, pid, name):
    manifest, folder = _active(store, pid)
    if name not in SITE_FILES:
        raise StoreError('Website file is not available.')
    file = folder / name
    if file.is_symlink():
        raise StoreError('Website file integrity failed.')
    try:
        raw = file.read_bytes()
    except OSError as exc:
        raise StoreError('Website file missing; rebuild the site.') from exc
    if hashlib.sha256(raw).hexdigest() != manifest['files'][name]:
        raise StoreError('Website file integrity failed; rebuild the site.')
    return raw


def website_zip(store, pid):
    manifest, _ = _active(store, pid)
    buff = io.BytesIO()
    with zipfile.ZipFile(buff, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(manifest['files']):
            z.writestr(name, website_file(store, pid, name))
    return buff.getvalue()


def delete_website(store, pid):
    with project_lock(pid):
        root = _root(store, pid)
        if root.is_symlink():
            root.unlink()
        elif root.exists():
            shutil.rmtree(root)