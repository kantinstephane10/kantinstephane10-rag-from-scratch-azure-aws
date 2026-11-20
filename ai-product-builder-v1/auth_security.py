"""Opt-in local multiuser authentication, scoped ACLs and server-side session revocation.

Run behind a trusted HTTPS reverse proxy on loopback. No privilege from user data,
forwarded identity headers, GPT messages or source files is accepted.
"""
import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from pathlib import Path

from store import StoreError

ROLES = {'viewer': 1, 'editor': 2, 'owner': 3}
USER_RE = re.compile(r'[a-zA-Z][a-zA-Z0-9_.-]{2,39}\Z')
SESSION_SECONDS = 8 * 3600


class AccessDenied(StoreError):
    pass


class UserAuth:
    def __init__(self, store):
        self.store = store
        self.init()

    def init(self):
        with self.store.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS app_users (
                  id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                  salt TEXT NOT NULL, password_hash TEXT NOT NULL,
                  admin INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1,
                  created_at INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS user_sessions (
                  token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                  csrf_hash TEXT NOT NULL, expires_at INTEGER NOT NULL, created_at INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS project_memberships (
                  project_id TEXT NOT NULL, user_id TEXT NOT NULL,
                  role TEXT NOT NULL CHECK(role IN ('owner','editor','viewer')),
                  PRIMARY KEY(project_id,user_id));
                CREATE INDEX IF NOT EXISTS ix_membership_user ON project_memberships(user_id);
            ''')

    @staticmethod
    def _secret(password, salt):
        return hashlib.scrypt(password.encode('utf-8'), salt=bytes.fromhex(salt),
                              n=2**14, r=8, p=1, dklen=32).hex()

    def add_user(self, username, password, *, admin=False):
        if not isinstance(username, str) or not USER_RE.fullmatch(username):
            raise StoreError('Username must be 3–40 letters, numbers, dots, dashes or underscores.')
        if not isinstance(password, str) or len(password) < 14 or len(password) > 1024:
            raise StoreError('Password must contain at least 14 characters.')
        salt = secrets.token_hex(16)
        uid = secrets.token_hex(16)
        with self.store.db() as db:
            if admin and db.execute('SELECT 1 FROM app_users LIMIT 1').fetchone():
                raise StoreError('Admin elevation requires an existing administrator.')
            try:
                db.execute('INSERT INTO app_users VALUES(?,?,?,?,?,?,?)',
                           (uid,username,salt,self._secret(password,salt),int(admin),1,int(time.time())))
            except sqlite3.IntegrityError as exc:
                raise StoreError('Username already registered.') from exc
        return {'id':uid,'username':username,'admin':bool(admin)}

    def _find(self, db, username):
        return db.execute('SELECT * FROM app_users WHERE username=?', (username,)).fetchone()

    def login(self, username, password):
        # Always perform one expensive hash, even for invalid names, to reduce enumeration.
        with self.store.db() as db:
            row = self._find(db, username) if isinstance(username,str) else None
            salt = row['salt'] if row else '00'*16
            digest = self._secret(password if isinstance(password,str) else '', salt)
            if not row or not row['active'] or not hmac.compare_digest(digest,row['password_hash']):
                raise AccessDenied('Invalid username or password.')
            token=secrets.token_urlsafe(36)
            csrf=secrets.token_urlsafe(24)
            expiry=int(time.time())+SESSION_SECONDS
            db.execute('INSERT INTO user_sessions VALUES(?,?,?,?,?)',
                       (hashlib.sha256(token.encode()).hexdigest(),row['id'],
                        hashlib.sha256(csrf.encode()).hexdigest(),expiry,int(time.time())))
        return {'token':token,'csrf':csrf,'expires_at':expiry,'username':row['username'],
                'admin':bool(row['admin'])}

    def session(self, token, csrf=None, *, write=False):
        if not isinstance(token,str) or not token:return None
        digest=hashlib.sha256(token.encode()).hexdigest()
        with self.store.db() as db:
            row=db.execute('''SELECT s.csrf_hash,s.expires_at,u.id,u.username,u.admin
                           FROM user_sessions s JOIN app_users u ON u.id=s.user_id
                           WHERE s.token_hash=? AND u.active=1''',(digest,)).fetchone()
        if not row or row['expires_at'] <= int(time.time()):return None
        if write and (not isinstance(csrf,str) or
           not hmac.compare_digest(hashlib.sha256(csrf.encode()).hexdigest(),row['csrf_hash'])):
            return None
        return {'id':row['id'],'username':row['username'],'admin':bool(row['admin'])}

    def logout(self, token):
        with self.store.db() as db:
            db.execute('DELETE FROM user_sessions WHERE token_hash=?',
                       (hashlib.sha256(token.encode()).hexdigest(),))

    def grant(self, pid, username, role):
        if role not in ROLES:raise StoreError('Invalid project role.')
        self.store.project(pid)
        with self.store.db() as db:
            row=self._find(db,username)
            if not row or not row['active']:raise StoreError('Unknown active user.')
            db.execute('INSERT INTO project_memberships VALUES(?,?,?) ON CONFLICT(project_id,user_id) DO UPDATE SET role=excluded.role',
                       (pid,row['id'],role))

    def revoke(self, pid, username):
        with self.store.db() as db:
            row=self._find(db,username)
            if not row:raise StoreError('Unknown user.')
            db.execute('DELETE FROM project_memberships WHERE project_id=? AND user_id=?',
                       (pid,row['id']))

    def role(self, user, pid):
        if user['admin']:return 'admin'
        with self.store.db() as db:
            row=db.execute('SELECT role FROM project_memberships WHERE project_id=? AND user_id=?',
                           (pid,user['id'])).fetchone()
        return row['role'] if row else None

    def require(self, user, pid, level='viewer'):
        role=self.role(user,pid)
        if role!='admin' and ROLES.get(role,0)<ROLES[level]:
            raise AccessDenied('Project access denied.')
        return role

    def own(self, user, pid):
        with self.store.db() as db:
            db.execute('INSERT INTO project_memberships VALUES(?,?,?)',(pid,user['id'],'owner'))

    def visible_projects(self,user,projects):
        return [p for p in projects if self.role(user,p['id'])]

    def require_meeting(self, user,mid,level='viewer'):
        with self.store.db() as db:
            row=db.execute('SELECT project_id FROM meetings WHERE id=?',(mid,)).fetchone()
        if not row:raise StoreError('Unknown meeting.')
        if row['project_id']:
            self.require(user,row['project_id'],level)
        elif not user['admin']:
            raise AccessDenied('Standalone meetings are administrator-only in multiuser mode.')

    def require_task(self,user,tid,level='editor'):
        with self.store.db() as db:
            row=db.execute('SELECT project_id FROM messages WHERE id=?',(tid,)).fetchone()
        if not row: raise StoreError('Unknown task.')
        return self.require(user,row['project_id'],level)

    def clear_project(self,pid):
        with self.store.db() as db:
            db.execute('DELETE FROM project_memberships WHERE project_id=?',(pid,))
