"""SQLite-backed project, communication, retry and approval records."""
import json
import shutil
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from roles import ROLES
from intake import validate_intake

# Delegation follows verified handoff ancestry instead of a hard numerical hop limit.
# No fixed project message count; exact duplicate work is suppressed to avoid loops.
MAX_MESSAGE_LENGTH = 6000
MAX_CANCEL_BATCH = 100
CANCELLABLE_STATES = ('queued', 'processing', 'failed', 'blocked')


def validate_cancel_reason(reason):
    if not isinstance(reason, str) or not 5 <= len(reason.strip()) <= 500:
        raise StoreError('A cancellation reason of 5–500 characters is required.')
    return reason.strip()


class StoreError(ValueError):
    pass


class Store:
    def __init__(self, filename):
        self.filename = str(Path(filename).resolve())
        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        from handoffs import init_handoffs
        init_handoffs(self)
        from meetings import init_meetings
        init_meetings(self)
        from execution_workspace import init_workspace
        init_workspace(self)
        from release_management import init_releases
        init_releases(self)

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
            # Add metadata on existing v0.9 SQLite files without dropping history.
            columns = {row['name'] for row in db.execute('PRAGMA table_info(messages)')}
            for name, datatype in [('cancel_reason', 'TEXT'), ('cancelled_at', 'INTEGER'),
                                   ('cancelled_by', 'TEXT'),
                                   ('workflow_stage', "TEXT NOT NULL DEFAULT 'normal'")]:
                if name not in columns:
                    db.execute(f'ALTER TABLE messages ADD COLUMN {name} {datatype}')

    def reserve_llm_call(self, usage_key, max_calls=None, max_tokens=None):
        """Atomically count requests for reporting only; never block on usage totals."""
        if not isinstance(usage_key,str) or len(usage_key)>100:
            raise StoreError('Invalid usage key.')
        with self.db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS llm_usage (
                usage_key TEXT PRIMARY KEY, call_count INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0)''')
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM llm_usage WHERE usage_key=?',(usage_key,)).fetchone()
            calls = row['call_count'] if row else 0
            tokens = row['total_tokens'] if row else 0
            db.execute('''INSERT INTO llm_usage (usage_key, call_count, total_tokens) VALUES (?,1,0)
                       ON CONFLICT(usage_key) DO UPDATE SET call_count=call_count+1''',(usage_key,))
            return {'call_count':calls+1,'total_tokens':tokens}

    def record_llm_usage(self, usage_key, used_tokens):
        if isinstance(used_tokens, bool) or not isinstance(used_tokens, int) or used_tokens < 0:
            return
        with self.db() as db:
            db.execute('UPDATE llm_usage SET total_tokens=total_tokens+? WHERE usage_key=?',(used_tokens,usage_key))

    def llm_usage(self, usage_key):
        with self.db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS llm_usage (
                usage_key TEXT PRIMARY KEY, call_count INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0)''')
            row=db.execute('SELECT * FROM llm_usage WHERE usage_key=?',(usage_key,)).fetchone()
            return dict(row) if row else {'usage_key':usage_key,'call_count':0,'total_tokens':0}

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

    def kickoff_with_business_analysts(self, project_id, manager_prompt):
        """Atomically assign both required analysts before the PM kickoff.

        The PM kickoff may be claimed only once both initial BA tasks finish or
        explicitly fail/cancel. No partial kickoff or duplicate wave is allowed.
        """
        if not isinstance(manager_prompt,str) or not manager_prompt.strip() or len(manager_prompt)>MAX_MESSAGE_LENGTH:
            raise StoreError('Invalid Product Manager kickoff assignment.')
        now=int(time.time())
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            project=db.execute('SELECT name,brief,status FROM projects WHERE id=?',(project_id,)).fetchone()
            if not project or project['status'] not in ('draft','in_progress'):
                raise StoreError('Project cannot start a new kickoff.')
            if db.execute('SELECT 1 FROM messages WHERE project_id=? LIMIT 1',(project_id,)).fetchone():
                raise StoreError('Project has existing messages; use agent messaging to continue.')
            assignments=[
                ('business-analyst-1',
                 'Required initial business analysis: assess the problem, stakeholders, end users, user stories, '
                 'success metrics, scope, acceptance criteria, data ownership and explicit approval needs. '
                 'Report confirmed facts versus assumptions; do not invent authorization or tests. '
                 'Read the FULL stored project brief from your task context.', 'mandatory_business_analysis'),
                ('business-analyst-2',
                 'Required independent business analysis: challenge assumptions, identify exceptions, '
                 'operational workflows, compliance and data-access constraints, cost/benefit and failure cases. '
                 'Define measurable acceptance criteria and open questions. '
                 'Read the FULL stored project brief from your task context.', 'mandatory_business_analysis'),
                ('product-manager', manager_prompt, 'pm_kickoff'),
            ]
            tasks=[]
            for role,body,stage in assignments:
                if len(body)>MAX_MESSAGE_LENGTH:
                    raise StoreError('Project brief is too long for mandatory analyst kickoff. Shorten the brief.')
                mid=uuid4().hex
                db.execute("""INSERT INTO messages(id,project_id,sender,recipient,body,chain_id,
                    parent_id,depth,created_at,updated_at,workflow_stage)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (mid,project_id,'human',role,body,mid,None,0,now,now,stage))
                self._event(db,project_id,'message_queued','human',{'id':mid,'to':role,'workflow_stage':stage})
                tasks.append(mid)
            db.execute("UPDATE projects SET status='in_progress' WHERE id=? AND status='draft'",(project_id,))
            self._event(db,project_id,'mandatory_business_analysis_started','system',
                        {'business_analysts':['business-analyst-1','business-analyst-2'],
                         'product_manager':tasks[-1]})
        return self.message(tasks[-1])

    def initial_business_analysis(self, project_id):
        """Return status and grounded work from mandatory BA kickoff tasks."""
        with self.db() as db:
            return [dict(row) for row in db.execute("""SELECT recipient,status,reply,error,id
                FROM messages WHERE project_id=? AND workflow_stage='mandatory_business_analysis'
                ORDER BY created_at,rowid""",(project_id,))]

    def project(self,pid):
        with self.db() as db:
            row=db.execute("SELECT * FROM projects WHERE id=?",(pid,)).fetchone()
            if row is None or row["status"] == "deleting":
                raise StoreError("Unknown project ID.")
            project = dict(row)
            intake = db.execute("SELECT * FROM project_intake WHERE project_id=?",(pid,)).fetchone()
            project['intake'] = dict(intake) if intake else validate_intake({})
            project['intake']['provider_consent'] = bool(project['intake']['provider_consent'])
            return project

    def projects(self):
        with self.db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM projects ORDER BY created_at DESC,id DESC LIMIT 100")]

    def delete_project(self, pid, confirmation):
        """Explicit destructive operator action: stop work, purge private records and files.

        Stage 1 sets a durable deleting tombstone to block worker claims and uploads.
        Stage 2 removes the project artifact directory; failure leaves the tombstone
        available for retry, never deleting another project's data.
        Stage 3 deletes all associated SQLite rows in one transaction.
        """
        if not isinstance(pid, str) or not re.fullmatch(r'[a-f0-9]{12}', pid):
            raise StoreError('Invalid project ID.')
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM projects WHERE id=?',(pid,)).fetchone()
            if not row: raise StoreError('Unknown project ID.')
            if not isinstance(confirmation,str) or confirmation != row['name']:
                raise StoreError('Type the exact project name to confirm deletion.')
            if row['status']!='deleting':
                now=int(time.time())
                db.execute("UPDATE projects SET status='deleting' WHERE id=?",(pid,))
                db.execute("UPDATE messages SET status='cancelled',updated_at=? WHERE project_id=? AND status IN ('queued','processing','failed','blocked')",(now,pid))
                mids=[r[0] for r in db.execute('SELECT id FROM meetings WHERE project_id=?',(pid,))]
                for mid in mids:
                    db.execute("UPDATE meetings SET status='cancelled',floor='human',epoch=epoch+1,ended_at=? WHERE id=? AND status IN ('live','scheduled')",(now,mid))
                    db.execute("UPDATE meeting_turns SET status='cancelled' WHERE meeting_id=? AND status IN ('queued','processing')",(mid,))
                if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='rag_v2_jobs'").fetchone():
                    db.execute("UPDATE rag_v2_jobs SET cancel_requested=1,status='cancelled' WHERE project_id=? AND status IN ('queued','running')",(pid,))
                self._event(db,pid,'project_deleting','human',{'name':row['name']})
        # Only this project ID under the application's own data directory.
        artifact_parent=(Path(self.filename).parent/'artifacts').resolve()
        target=artifact_parent/pid
        try:
            if target.is_symlink():
                target.unlink()
            elif target.exists():
                shutil.rmtree(target)
        except OSError as exc:
            raise StoreError('Could not remove all project files; retry Delete project. No other projects were affected.') from exc
        # Remove generated website files under the same project ID before removing DB records.
        from website_builder import delete_website
        try:
            delete_website(self, pid)
        except OSError as exc:
            raise StoreError('Could not delete project website; retry deletion.') from exc
        # Purge execution workspaces and staged releases before deleting the project
        # tombstone, so a failed filesystem cleanup remains safely retryable.
        from execution_workspace import delete_workspace
        from release_management import remove_releases
        try:
            delete_workspace(self,pid)
            remove_releases(self,pid)
        except OSError as exc:
            raise StoreError('Project execution files could not be purged; retry deletion.') from exc
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT status FROM projects WHERE id=?',(pid,)).fetchone()
            if not row or row['status']!='deleting':
                raise StoreError('Deletion state changed unexpectedly.')
            mids=[r[0] for r in db.execute('SELECT id FROM meetings WHERE project_id=?',(pid,))]
            for mid in mids:
                db.execute('DELETE FROM meeting_attachments WHERE meeting_id=?',(mid,))
                db.execute('DELETE FROM meeting_messages WHERE meeting_id=?',(mid,))
                db.execute('DELETE FROM meeting_turns WHERE meeting_id=?',(mid,))
            db.execute('DELETE FROM meetings WHERE project_id=?',(pid,))
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='rag_v2_jobs'").fetchone():
                generations=[r[0] for r in db.execute('SELECT generation FROM rag_v2_jobs WHERE project_id=?',(pid,))]
                for gen in generations:
                    db.execute('DELETE FROM rag_v2_sections WHERE generation=?',(gen,))
                    db.execute('DELETE FROM rag_v2_pages WHERE generation=?',(gen,))
                db.execute('DELETE FROM rag_v2_chunks WHERE project_id=?',(pid,))
                db.execute('DELETE FROM rag_v2_indexes WHERE project_id=?',(pid,))
                db.execute('DELETE FROM rag_v2_jobs WHERE project_id=?',(pid,))
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='project_memberships'").fetchone():
                db.execute('DELETE FROM project_memberships WHERE project_id=?',(pid,))
            for table in ('rag_chunks','rag_indexes','rag_jobs','handoffs','artifacts','events','messages','project_intake'):
                if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone():
                    db.execute(f'DELETE FROM {table} WHERE project_id=?',(pid,))
            db.execute('DELETE FROM llm_usage WHERE usage_key=?',(pid,)) if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='llm_usage'").fetchone() else None
            db.execute('DELETE FROM projects WHERE id=?',(pid,))
        return {'deleted':True,'project_id':pid}

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
        if isinstance(depth,bool) or not isinstance(depth,int) or depth < 0:
            raise StoreError('Invalid agent relay depth.')
        now=int(time.time()); mid=uuid4().hex; chain=chain_id or mid
        with self.db() as db:
            row=db.execute("SELECT status FROM projects WHERE id=?",(project_id,)).fetchone()
            if not row:
                raise StoreError("Unknown project ID.")
            if row["status"] in {"approved","deleting"}:
                raise StoreError("This project cannot accept new work.")
            if parent_id:
                parent=db.execute("SELECT * FROM messages WHERE id=? AND project_id=?",(parent_id,project_id)).fetchone()
                if not parent or parent["recipient"]!=sender or parent["chain_id"]!=chain or depth!=parent["depth"]+1:
                    raise StoreError("Invalid agent-to-agent relay ancestry.")
                # Fail closed on repeated assignments in the SAME ancestry chain.
                # Normalized matching also catches capitalization/whitespace-only repeats.
                # New independent human requests may legitimately revisit a topic.
                normalized=' '.join(body.split()).casefold()
                previous=db.execute("SELECT body FROM messages WHERE chain_id=? AND sender=? AND recipient=?",
                                    (chain,sender,recipient))
                if any(' '.join(r['body'].split()).casefold()==normalized for r in previous):
                    raise StoreError('Duplicate agent relay suppressed.')
            elif sender!="human":
                raise StoreError("Agents must route messages from a recorded parent task.")
            handoff_id=None
            if sender != 'human':
                from handoffs import record_handoff
                handoff_id=record_handoff(db,self,project_id,parent,sender,recipient)
            db.execute("""INSERT INTO messages(id,project_id,sender,recipient,body,chain_id,parent_id,depth,created_at,updated_at,handoff_id)
                          VALUES(?,?,?,?,?,?,?,?,?,?,?)""",(mid,project_id,sender,recipient,body.strip(),chain,parent_id,depth,now,now,handoff_id))
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
        counts = {key: 0 for key in ('queued','processing','done','failed','blocked','cancelled')}
        for message in all_messages:
            counts[message['status']] = counts.get(message['status'],0) + 1
        counts['total'] = len(all_messages)
        return {'project':project,'counts':counts,'messages':all_messages,'llm_usage':self.llm_usage(pid)}

    def messages(self,pid,limit=200):
        """Return most RECENT messages in chronological order for long-running projects."""
        self.project(pid)
        if isinstance(limit,bool) or not isinstance(limit,int) or limit<1:
            raise StoreError('Message history limit must be a positive integer.')
        with self.db() as db:
            newest=[dict(r) for r in db.execute(
                'SELECT * FROM messages WHERE project_id=? ORDER BY created_at DESC,rowid DESC LIMIT ?',
                (pid,limit))]
        return list(reversed(newest))

    def events(self,pid,limit=200):
        self.project(pid)
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM events WHERE project_id=? ORDER BY id DESC LIMIT ?",(pid,limit))]

    def claim(self):
        now=int(time.time())
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("""SELECT m.* FROM messages m JOIN projects p ON p.id=m.project_id
                              WHERE m.status='queued' AND m.next_attempt_at<=? AND p.status IN ('in_progress','awaiting_human')
                              AND (m.workflow_stage!='pm_kickoff' OR NOT EXISTS (
                                  SELECT 1 FROM messages ba WHERE ba.project_id=m.project_id
                                  AND ba.workflow_stage='mandatory_business_analysis'
                                  AND ba.status IN ('queued','processing')))
                              ORDER BY m.created_at,m.rowid LIMIT 1""",(now,)).fetchone()
            if not row: return None
            if row['sender']!='human':
                from handoffs import _verified
                try:
                    if not row['handoff_id']:
                        raise StoreError('Agent relay missing mandatory handoff.')
                    package,manifest=_verified(db,self,row['handoff_id'],row['project_id'])
                    if package['source_message_id']!=row['parent_id'] or package['recipient']!=row['recipient']:
                        raise StoreError('Handoff route does not match recipient.')
                except StoreError as exc:
                    db.execute("UPDATE messages SET status='blocked', error=?, updated_at=? WHERE id=?",
                               (str(exc)[:500],now,row['id']))
                    self._event(db,row['project_id'],'handoff_blocked',row['recipient'],{'id':row['id'],'error':str(exc)[:200]})
                    return None
            db.execute("UPDATE messages SET status='processing',attempts=attempts+1,updated_at=? WHERE id=?",(now,row["id"]))
            self._event(db,row["project_id"],"claimed",row["recipient"],{"id":row["id"],"attempt":row["attempts"]+1})
            return dict(db.execute("SELECT * FROM messages WHERE id=?",(row["id"],)).fetchone())

    def complete(self,mid,model,reply,needs_human=False):
        if not isinstance(reply,str) or not reply.strip():
            raise StoreError("An agent reply cannot be empty.")
        with self.db() as db:
            row=db.execute("SELECT * FROM messages WHERE id=?",(mid,)).fetchone()
            if not row:
                return False  # Project deleted while an OpenAI response was in flight.
            if row["status"]=="cancelled":
                return False
            if row["status"]!="processing":
                raise StoreError("Only a processing message can be completed.")
            db.execute("UPDATE messages SET status='done',reply=?,model=?,error=NULL,updated_at=? WHERE id=?",
                       (reply,model,int(time.time()),mid))
            self._event(db,row["project_id"],"agent_replied",row["recipient"],{"id":mid,"model":model,"needs_human":bool(needs_human)})
            if needs_human:
                db.execute("UPDATE projects SET status='awaiting_human' WHERE id=?",(row["project_id"],))
            return True

    def append_worker_receipt(self, mid, pid, role, notes):
        """Append server-verified source receipts after a completed GPT turn."""
        if not notes:return
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT status FROM messages WHERE id=? AND project_id=?',(mid,pid)).fetchone()
            if not row or row['status']!='done':
                raise StoreError('Only completed tasks can receive execution receipts.')
            message='\n\n[Workspace receipt: '+' '.join(notes)[:5000]+']'
            db.execute('UPDATE messages SET reply=reply||? WHERE id=?',(message,mid))
            self._event(db,pid,'workspace_receipt',role,{'task_id':mid,'notes':notes[:8]})

    def failure(self,mid,error,*,retryable,max_attempts=3):
        with self.db() as db:
            row=db.execute("SELECT * FROM messages WHERE id=?",(mid,)).fetchone()
            if not row:
                return False  # Project deleted while an OpenAI response was in flight.
            if row["status"]=="cancelled":
                return False
            if row["status"]!="processing":
                raise StoreError("Only a processing message can fail.")
            retry=retryable and row["attempts"]<max_attempts
            state="queued" if retry else "failed"
            delay=min(30,2**row["attempts"]) if retry else 0
            db.execute("UPDATE messages SET status=?,error=?,next_attempt_at=?,updated_at=? WHERE id=?",
                       (state,str(error)[:500],int(time.time())+delay,int(time.time()),mid))
            self._event(db,row["project_id"],"agent_failed",row["recipient"],{"id":mid,"status":state,"error":str(error)[:200]})
            return True

    def retry(self,mid,project_id=None):
        with self.db() as db:
            row=db.execute("SELECT * FROM messages WHERE id=?",(mid,)).fetchone()
            if not row or (project_id is not None and row['project_id']!=project_id):
                raise StoreError('Task does not belong to the selected project.')
            if row["status"]!="failed": raise StoreError("Only failed messages can be retried.")
            project=db.execute('SELECT status FROM projects WHERE id=?',(row['project_id'],)).fetchone()
            if not project or project['status'] not in ('in_progress','awaiting_human'):
                raise StoreError('This project does not accept retried work.')
            db.execute("UPDATE messages SET status='queued',attempts=0,error=NULL,next_attempt_at=0 WHERE id=?",(mid,))
            self._event(db,row["project_id"],"manual_retry","human",{"id":mid})

    def cancel(self, mid, reason, operator='human'):
        """Cancel one pending assignment, preserving reason and receipt atomically."""
        reason = validate_cancel_reason(reason)
        if operator != 'human':
            raise StoreError('Only the human operator can cancel tasks from this endpoint.')
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM messages WHERE id=?', (mid,)).fetchone()
            if not row:
                raise StoreError('Unknown message.')
            if row['status'] not in CANCELLABLE_STATES:
                raise StoreError('Only unfinished tasks can be cancelled.')
            now = int(time.time())
            db.execute("""UPDATE messages SET status='cancelled', error='Cancelled by operator',
                cancel_reason=?, cancelled_by=?, cancelled_at=?, next_attempt_at=0,
                updated_at=? WHERE id=?""", (reason, operator, now, now, mid))
            self._event(db, row['project_id'], 'task_cancelled', operator,
                        {'id':mid, 'recipient':row['recipient'], 'reason':reason})
            return dict(db.execute('SELECT * FROM messages WHERE id=?', (mid,)).fetchone())

    def cancel_tasks(self, project_id, task_ids, reason, operator='human'):
        """All-or-nothing cancellation of explicitly selected tasks in one project."""
        reason = validate_cancel_reason(reason)
        if operator != 'human':
            raise StoreError('Only the human operator can cancel tasks from this endpoint.')
        if (not isinstance(task_ids, list) or not 1 <= len(task_ids) <= MAX_CANCEL_BATCH or
            any(not isinstance(mid, str) or len(mid) != 32 or
                any(c not in '0123456789abcdef' for c in mid) for mid in task_ids) or
            len(set(task_ids)) != len(task_ids)):
            raise StoreError('Select 1–100 distinct task IDs.')
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if not db.execute('SELECT 1 FROM projects WHERE id=?', (project_id,)).fetchone():
                raise StoreError('Unknown project ID.')
            markers = ','.join('?' for _ in task_ids)
            rows = list(db.execute('SELECT * FROM messages WHERE id IN ('+markers+')', tuple(task_ids)))
            if (len(rows) != len(task_ids) or
                any(row['project_id'] != project_id or row['status'] not in CANCELLABLE_STATES
                    for row in rows)):
                raise StoreError('Every selected task must belong to this project and be unfinished; nothing was cancelled.')
            now = int(time.time())
            db.execute("UPDATE messages SET status='cancelled', error='Cancelled by operator', "
                       'cancel_reason=?, cancelled_by=?, cancelled_at=?, next_attempt_at=0, updated_at=? '
                       'WHERE id IN ('+markers+')',
                       (reason, operator, now, now, *task_ids))
            for row in rows:
                self._event(db, project_id, 'task_cancelled', operator,
                            {'id':row['id'], 'recipient':row['recipient'], 'reason':reason})
            self._event(db, project_id, 'tasks_cancelled', operator,
                        {'count':len(rows), 'ids':task_ids, 'reason':reason})
            return {'cancelled':len(rows), 'task_ids':task_ids, 'reason':reason}

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
            pending=db.execute("SELECT COUNT(*) FROM messages WHERE project_id=? AND status IN ('queued','processing','failed','blocked','cancelled')",(pid,)).fetchone()[0]
            if pending: raise StoreError("Unresolved tasks or failures block human approval.")
            unresolved=db.execute("SELECT COUNT(*) FROM handoffs WHERE project_id=? AND status!='acknowledged'",(pid,)).fetchone()[0]
            if unresolved: raise StoreError('Unacknowledged agent handoffs block approval.')
            db.execute("UPDATE projects SET status='approved' ,approved_by=?,approved_at=? WHERE id=?",
                       (operator.strip(),int(time.time()),pid))
            self._event(db,pid,"human_approved",operator.strip(),{"confirmation":confirmation})

    def recover_processing(self):
        """Recover interrupted work and the legacy (pre-V1.3.1) relay-limit bug.

        Only the exact known deterministic failure is requeued. Other failures
        remain available for operator review. Replaying a GPT call may be billed.
        """
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            ids=[r[0] for r in db.execute("SELECT id FROM messages WHERE status='processing'")]
            for mid in ids:
                db.execute("UPDATE messages SET status='queued',next_attempt_at=0,error='Recovered after restart' WHERE id=?",(mid,))
            legacy=list(db.execute("""SELECT id,project_id,recipient FROM messages
                WHERE status='failed' AND error='Agent relay limit reached; no further peer messages allowed.'
                  AND project_id IN (SELECT id FROM projects WHERE status IN ('in_progress','awaiting_human'))"""))
            for row in legacy:
                db.execute("UPDATE messages SET status='queued',next_attempt_at=0,attempts=0,error='Recovered legacy relay-limit failure' WHERE id=?",(row['id'],))
                self._event(db,row['project_id'],'legacy_relay_recovered','system',{'id':row['id'],'recipient':row['recipient']})
            return len(ids)+len(legacy)

    def record_worker_notice(self, pid, role, message_id, messages):
        """Audit suppressed model requests without mistaking them for work done."""
        with self.db() as db:
            self._event(db,pid,'agent_delegation_suppressed',role,
                        {'message_id':message_id,'notices':[str(item)[:150] for item in messages[:6]]})
