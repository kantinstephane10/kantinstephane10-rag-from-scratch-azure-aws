"""SQLite-backed project, communication, retry and approval records."""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from roles import ROLES
from intake import validate_intake

MAX_DEPTH = 4
MAX_MESSAGES_PER_PROJECT = 250
MAX_MESSAGE_LENGTH = 6000


class StoreError(ValueError):
    pass


class Store:
    def __init__(self, filename):
        self.filename = str(Path(filename).resolve())
        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        from meetings import init_meetings
        init_meetings(self)

    @contextmanager
    def db(self):
        conn = sqlite3.connect(self.filename, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize(self):
        with self.db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS projects (
                  id TEXT PRIMARY KEY, name TEXT NOT NULL, brief TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'draft', created_at INTEGER NOT NULL,
                  approved_by TEXT, approved_at INTEGER);
                CREATE TABLE IF NOT EXISTS project_intake (
                  project_id TEXT PRIMARY KEY REFERENCES projects(id),
                  product_type TEXT NOT NULL, model_profile TEXT NOT NULL,
                  provider_consent INTEGER NOT NULL DEFAULT 0,
                  estimated_hours REAL NOT NULL DEFAULT 0,
                  hourly_rate REAL NOT NULL DEFAULT 0,
                  one_time_expenses REAL NOT NULL DEFAULT 0,
                  monthly_cost REAL NOT NULL DEFAULT 0,
                  contingency_pct REAL NOT NULL DEFAULT 0,
                  estimated_total REAL NOT NULL DEFAULT 0,
                  currency TEXT NOT NULL DEFAULT 'USD');
                CREATE TABLE IF NOT EXISTS messages (
                  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                  sender TEXT NOT NULL, recipient TEXT NOT NULL, body TEXT NOT NULL,
                  chain_id TEXT NOT NULL, parent_id TEXT, depth INTEGER NOT NULL DEFAULT 0,
                  status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
                  next_attempt_at INTEGER NOT NULL DEFAULT 0,
                  model TEXT, reply TEXT, error TEXT,
                  created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS ix_messages_status ON messages(status,next_attempt_at,created_at);
                CREATE INDEX IF NOT EXISTS ix_messages_project ON messages(project_id,created_at);
                CREATE TABLE IF NOT EXISTS events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL,
                  kind TEXT NOT NULL, actor TEXT NOT NULL, payload TEXT NOT NULL, created_at INTEGER NOT NULL);
            """)

    def create_project(self, name, brief, *, intake=None):
        if not isinstance(name, str) or not name.strip() or len(name) > 160:
            raise StoreError("Project name must contain 1–160 characters.")
        if not isinstance(brief, str) or not brief.strip() or len(brief) > 20000:
            raise StoreError("Project brief must contain 1–20,000 characters.")
        try:
            config = validate_intake(intake or {})
        except ValueError as exc:
            raise StoreError(str(exc)) from exc
        pid = uuid4().hex[:12]
        now = int(time.time())
        with self.db() as db:
            db.execute("INSERT INTO projects(id,name,brief,created_at) VALUES(?,?,?,?)", (pid,name.strip(),brief.strip(),now))
            db.execute("""INSERT INTO project_intake(project_id,product_type,model_profile,provider_consent,
                    estimated_hours,hourly_rate,one_time_expenses,monthly_cost,contingency_pct,
                    estimated_total,currency) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (pid,config['product_type'],config['model_profile'],int(config['provider_consent']),
                     config['estimated_hours'],config['hourly_rate'],config['one_time_expenses'],
                     config['monthly_cost'],config['contingency_pct'],config['estimated_total'],config['currency']))
            self._event(db,pid,"project_created","human",{"name":name.strip()})
        return self.project(pid)

    def project(self,pid):
        with self.db() as db:
            row=db.execute("SELECT * FROM projects WHERE id=?",(pid,)).fetchone()
            if row is None:
                raise StoreError("Unknown project ID.")
            project = dict(row)
            intake = db.execute("SELECT * FROM project_intake WHERE project_id=?",(pid,)).fetchone()
            project['intake'] = dict(intake) if intake else validate_intake({})
            project['intake']['provider_consent'] = bool(project['intake']['provider_consent'])
            return project

    def projects(self):
        with self.db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM projects ORDER BY created_at DESC,id DESC LIMIT 100")]

    def _event(self,db,pid,kind,actor,payload):
        db.execute("INSERT INTO events(project_id,kind,actor,payload,created_at) VALUES(?,?,?,?,?)",
                   (pid,kind,actor,json.dumps(payload,ensure_ascii=False),int(time.time())))

    def send(self,project_id,sender,recipient,body,*,parent_id=None,chain_id=None,depth=0):
        if sender!="human" and sender not in ROLES:
            raise StoreError("Sender must be the human operator or a registered agent.")
        if recipient not in ROLES:
            raise StoreError("Recipient must be a registered AI agent.")
        if not isinstance(body,str) or not body.strip() or len(body)>MAX_MESSAGE_LENGTH:
            raise StoreError(f"Message must contain 1–{MAX_MESSAGE_LENGTH} characters.")
        if not isinstance(depth,int) or not 0 <= depth <= MAX_DEPTH:
            raise StoreError("Agent relay depth exceeded.")
        now=int(time.time()); mid=uuid4().hex; chain=chain_id or mid
        with self.db() as db:
            row=db.execute("SELECT status FROM projects WHERE id=?",(project_id,)).fetchone()
            if not row:
                raise StoreError("Unknown project ID.")
            if row["status"]=="approved":
                raise StoreError("Approved projects are read-only. Create a new revision project.")
            count=db.execute("SELECT COUNT(*) FROM messages WHERE project_id=?",(project_id,)).fetchone()[0]
            if count>=MAX_MESSAGES_PER_PROJECT:
                raise StoreError("Project reached its message budget; request a new revision.")
            if parent_id:
                parent=db.execute("SELECT * FROM messages WHERE id=? AND project_id=?",(parent_id,project_id)).fetchone()
                if not parent or parent["recipient"]!=sender or parent["chain_id"]!=chain or depth!=parent["depth"]+1:
                    raise StoreError("Invalid agent-to-agent relay ancestry.")
                # Stop repeated identical peer-to-peer traffic within one relay chain.
                duplicate=db.execute("SELECT 1 FROM messages WHERE chain_id=? AND sender=? AND recipient=? AND body=? LIMIT 1",
                                     (chain,sender,recipient,body.strip())).fetchone()
                if duplicate:
                    raise StoreError("Duplicate agent relay suppressed.")
            elif sender!="human":
                raise StoreError("Agents must route messages from a recorded parent task.")
            db.execute("""INSERT INTO messages(id,project_id,sender,recipient,body,chain_id,parent_id,depth,created_at,updated_at)
                          VALUES(?,?,?,?,?,?,?,?,?,?)""",(mid,project_id,sender,recipient,body.strip(),chain,parent_id,depth,now,now))
            db.execute("UPDATE projects SET status='in_progress' WHERE id=? AND status='draft'",(project_id,))
            self._event(db,project_id,"message_queued",sender,{"id":mid,"to":recipient,"parent_id":parent_id})
        return self.message(mid)

    def message(self,mid):
        with self.db() as db:
            row=db.execute("SELECT * FROM messages WHERE id=?",(mid,)).fetchone()
            if not row: raise StoreError("Unknown message.")
            return dict(row)

    def activity(self, pid):
        """Live workflow status from actual mailbox states, never inferred progress."""
        project = self.project(pid)
        with self.db() as db:
            all_messages = [dict(row) for row in db.execute(
                "SELECT * FROM messages WHERE project_id=? ORDER BY created_at,rowid",(pid,))]
        counts = {key: 0 for key in ('queued','processing','done','failed')}
        for message in all_messages:
            counts[message['status']] = counts.get(message['status'],0) + 1
        counts['total'] = len(all_messages)
        return {'project':project,'counts':counts,'messages':all_messages}

    def messages(self,pid,limit=200):
        self.project(pid)
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM messages WHERE project_id=? ORDER BY created_at ASC,rowid ASC LIMIT ?",(pid,limit))]

    def events(self,pid,limit=200):
        self.project(pid)
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM events WHERE project_id=? ORDER BY id DESC LIMIT ?",(pid,limit))]

    def claim(self):
        now=int(time.time())
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("""SELECT m.* FROM messages m JOIN projects p ON p.id=m.project_id
                              WHERE m.status='queued' AND m.next_attempt_at<=? AND p.status='in_progress'
                              ORDER BY m.created_at,m.rowid LIMIT 1""",(now,)).fetchone()
            if not row: return None
            db.execute("UPDATE messages SET status='processing',attempts=attempts+1,updated_at=? WHERE id=?",(now,row["id"]))
            self._event(db,row["project_id"],"claimed",row["recipient"],{"id":row["id"],"attempt":row["attempts"]+1})
            return dict(db.execute("SELECT * FROM messages WHERE id=?",(row["id"],)).fetchone())

    def complete(self,mid,model,reply,needs_human=False):
        if not isinstance(reply,str) or not reply.strip():
            raise StoreError("An agent reply cannot be empty.")
        with self.db() as db:
            row=db.execute("SELECT * FROM messages WHERE id=?",(mid,)).fetchone()
            if not row or row["status"]!="processing":
                raise StoreError("Only a processing message can be completed.")
            db.execute("UPDATE messages SET status='done',reply=?,model=?,error=NULL,updated_at=? WHERE id=?",
                       (reply,model,int(time.time()),mid))
            self._event(db,row["project_id"],"agent_replied",row["recipient"],{"id":mid,"model":model,"needs_human":bool(needs_human)})
            if needs_human:
                db.execute("UPDATE projects SET status='awaiting_human' WHERE id=?",(row["project_id"],))

    def failure(self,mid,error,*,retryable,max_attempts=3):
        with self.db() as db:
            row=db.execute("SELECT * FROM messages WHERE id=?",(mid,)).fetchone()
            if not row or row["status"]!="processing":
                raise StoreError("Only a processing message can fail.")
            retry=retryable and row["attempts"]<max_attempts
            state="queued" if retry else "failed"
            delay=min(30,2**row["attempts"]) if retry else 0
            db.execute("UPDATE messages SET status=?,error=?,next_attempt_at=?,updated_at=? WHERE id=?",
                       (state,str(error)[:500],int(time.time())+delay,int(time.time()),mid))
            self._event(db,row["project_id"],"agent_failed",row["recipient"],{"id":mid,"status":state,"error":str(error)[:200]})

    def retry(self,mid):
        with self.db() as db:
            row=db.execute("SELECT * FROM messages WHERE id=?",(mid,)).fetchone()
            if not row or row["status"]!="failed": raise StoreError("Only failed messages can be retried.")
            db.execute("UPDATE messages SET status='queued',attempts=0,error=NULL,next_attempt_at=0 WHERE id=?",(mid,))
            self._event(db,row["project_id"],"manual_retry","human",{"id":mid})

    def resume(self,pid):
        with self.db() as db:
            row=db.execute("SELECT status FROM projects WHERE id=?",(pid,)).fetchone()
            if not row or row["status"]!="awaiting_human": raise StoreError("Only an awaiting-human project can resume.")
            db.execute("UPDATE projects SET status='in_progress' WHERE id=?",(pid,))
            self._event(db,pid,"resumed","human",{})

    def approve(self,pid,confirmation,operator):
        if confirmation != "APPROVE" or not isinstance(operator,str) or not operator.strip():
            raise StoreError("Human approval requires APPROVE and an operator identifier.")
        with self.db() as db:
            row=db.execute("SELECT status FROM projects WHERE id=?",(pid,)).fetchone()
            if not row or row["status"] not in {"in_progress","awaiting_human"}:
                raise StoreError("Project cannot be approved from its current state.")
            pending=db.execute("SELECT COUNT(*) FROM messages WHERE project_id=? AND status IN ('queued','processing','failed')",(pid,)).fetchone()[0]
            if pending: raise StoreError("Unresolved tasks or failures block human approval.")
            db.execute("UPDATE projects SET status='approved',approved_by=?,approved_at=? WHERE id=?",
                       (operator.strip(),int(time.time()),pid))
            self._event(db,pid,"human_approved",operator.strip(),{"confirmation":confirmation})

    def recover_processing(self):
        """On server restart, make interrupted work eligible again without losing messages."""
        with self.db() as db:
            ids=[r[0] for r in db.execute("SELECT id FROM messages WHERE status='processing'")]
            for mid in ids:
                db.execute("UPDATE messages SET status='queued',next_attempt_at=0,error='Recovered after restart' WHERE id=?",(mid,))
            return len(ids)
