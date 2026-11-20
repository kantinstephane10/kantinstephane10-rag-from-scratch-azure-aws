"""Verified project-scoped static website delivery and localhost HTTP routes."""
import io
import json
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from engine import Engine
from server import make_server
from store import Store, StoreError
from website_builder import build_website, website_file, website_status, website_zip


class WebsiteBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(str(Path(self.tmp.name) / 'state.sqlite'))
        self.website = self.store.create_project('My <Studio> & Co',
            'Create a smart <script>alert(1)</script> service site with an intro and clear value proposition.',
            intake={'product_type': 'website'})
        self.pid = self.website['id']

    def tearDown(self):
        self.tmp.cleanup()

    def test_website_builder_creates_versioned_and_verified_files(self):
        a = build_website(self.store, self.pid, {'template': 'portfolio', 'theme': 'teal'})
        self.assertTrue(a['built'])
        self.assertEqual(len(a['files']), 6)
        self.assertEqual(website_status(self.store, self.pid)['generation'], a['generation'])
        home = website_file(self.store, self.pid, 'index.html').decode('utf-8')
        self.assertIn('My &lt;Studio&gt; &amp; Co', home)
        self.assertIn('styles.css', home)
        self.assertNotIn('<script>alert(1)</script>', home)
        self.assertNotIn('<script', home.lower())
        self.assertIn(':root', website_file(self.store, self.pid, 'styles.css').decode())
        zipped = zipfile.ZipFile(io.BytesIO(website_zip(self.store, self.pid)))
        self.assertEqual(set(zipped.namelist()), set(a['files']))
        for filename in zipped.namelist():
            self.assertEqual(zipped.read(filename), website_file(self.store, self.pid, filename))

    def test_failed_rebuild_does_not_replace_prior_generation(self):
        a = build_website(self.store, self.pid, {'theme': 'amber'})
        with self.assertRaises(StoreError):
            build_website(self.store, self.pid, {'theme': 'url(http://evil.test)'})
        self.assertEqual(website_status(self.store, self.pid)['generation'], a['generation'])
        self.assertIn(b'--accent:#f4a53a', website_file(self.store, self.pid, 'styles.css'))

    def test_tampering_prevents_reuse_and_export(self):
        data = build_website(self.store, self.pid, {})
        path = Path(self.store.filename).parent / 'websites' / self.pid / 'generations' / data['generation'] / 'index.html'
        path.write_text('unverified replacement')
        with self.assertRaisesRegex(StoreError, 'integrity'):
            website_file(self.store, self.pid, 'index.html')
        with self.assertRaisesRegex(StoreError, 'integrity'):
            website_zip(self.store, self.pid)

    def test_cannot_build_on_non_website_project(self):
        rag = self.store.create_project('My RAG', 'Read my document', intake={'product_type': 'rag'})
        with self.assertRaisesRegex(StoreError, 'not a website'):
            build_website(self.store, rag['id'], {})

    def test_delete_removes_only_target_website(self):
        a = build_website(self.store, self.pid, {})
        other = self.store.create_project('Another website', 'Site 2', intake={'product_type': 'website'})
        build_website(self.store, other['id'], {})
        self.store.delete_project(self.pid, self.website['name'])
        self.assertFalse((Path(self.store.filename).parent / 'websites' / self.pid).exists())
        self.assertTrue(website_status(self.store, other['id'])['built'])

    def test_invalid_values_and_traversal_rejected(self):
        for payload in ({'title': '<' * 91}, {'template': '../secret'}, {'theme': 'blue'}, []):
            with self.subTest(payload=payload):
                with self.assertRaises(StoreError):
                    build_website(self.store, self.pid, payload)
        build_website(self.store, self.pid, {})
        with self.assertRaises(StoreError):
            website_file(self.store, self.pid, '../../secret.txt')


class WebsiteHttpTests(WebsiteBuilderTests):
    def setUp(self):
        super().setUp()
        self.engine = Engine(self.store, client=object(), demo=True)
        self.server = make_server(self.store, self.engine, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_address[1]}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)
        super().tearDown()

    def test_website_is_generated_automatically_on_project_creation(self):
        data = json.dumps({'name':'A New Site','brief':'Develop a cloud documentation site.',
                           'intake':{'product_type':'website'}, 'kickoff':False}).encode()
        request = Request(self.base + '/api/projects', data=data, method='POST',
                          headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=10) as response:
            self.assertEqual(response.status,201)
            created=json.load(response)
        self.assertTrue(website_status(self.store,created['id'])['built'])
        self.assertIn(b'A New Site', website_file(self.store,created['id'],'index.html'))

    def test_http_preview_and_zip_download(self):
        path = f'/api/projects/{self.pid}/website/build'
        req = Request(self.base + path, method='POST',
                      data=json.dumps({'template': 'business', 'theme': 'rose'}).encode(),
                      headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=10) as response:
            self.assertEqual(response.status, 201)
            self.assertTrue(json.load(response)['built'])
        with urlopen(self.base + f'/api/projects/{self.pid}/website/preview/index.html') as response:
            page = response.read().decode()
            self.assertIn('frame-ancestors', response.headers.get('Content-Security-Policy'))
            self.assertIn("script-src 'none'", response.headers.get('Content-Security-Policy'))
            self.assertIn('My &lt;Studio&gt;', page)
        with urlopen(self.base + f'/api/projects/{self.pid}/website/download') as response:
            self.assertEqual(response.headers.get_content_type(), 'application/zip')
            self.assertIn('attachment', response.headers.get('Content-Disposition'))
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                self.assertIn('index.html', z.namelist())
        with urlopen(self.base + f'/api/projects/{self.pid}/website/status') as response:
            self.assertTrue(json.load(response)['built'])

    def test_readiness_is_transparent(self):
        with urlopen(self.base + '/api/system/readiness', timeout=10) as response:
            data = json.load(response)
        self.assertFalse(data['production_ready'])
        self.assertTrue(data['production_gaps'])
        self.assertEqual(data['website_builder'], 'available (no additional Python packages)')
        self.assertIn('libraries', data)

    def test_cross_product_and_bad_requests_are_blocked(self):
        rag = self.store.create_project('Not a site', 'RAG', intake={'product_type': 'rag'})
        url = self.base + f'/api/projects/{rag["id"]}/website/build'
        with self.assertRaises(HTTPError) as err:
            urlopen(Request(url, data=b'{}', method='POST',
                            headers={'Content-Type': 'application/json'}))
        self.assertEqual(err.exception.code, 400)
        req = Request(self.base + f'/api/projects/{self.pid}/website/build',
                      data=b'{}', method='POST', headers={
                          'Content-Type': 'application/json',
                          'Origin': 'https://untrusted.example'})
        with self.assertRaises(HTTPError) as err:
            urlopen(req)
        self.assertEqual(err.exception.code, 403)


if __name__ == '__main__':
    unittest.main()