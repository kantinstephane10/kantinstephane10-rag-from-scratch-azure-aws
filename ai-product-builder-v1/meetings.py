"""Durable, bounded meeting room queue. PM is always chair; human floor wins.

All methods below work through Store.db (SQLite transactions). In-flight GPT answers
are discarded after an interruption by comparing the claim generation/epoch.
"""
import json
import time
from uuid import uuid4
from roles import ROLES
from store import StoreError, validate_cancel_reason


def init_meetings(store):
    with store.db() as db:
        db.executescript('''
            CREATE TABLE IF NOT EXISTS meetings (
                id TEXT PRIMARY KEY, project_id TEXT, topic TEXT NOT NULL,
                requested_by TEXT NOT NULL, participants TEXT NOT NULL,
                chair TEXT NOT NULL DEFAULT 'product-manager',
                status TEXT NOT NULL DEFAULT 'scheduled',
                floor TEXT NOT NULL DEFAULT 'agents', epoch INTEGER NOT NULL DEFAULT 0,
                starts_at INTEGER NOT NULL, created_at INTEGER NOT NULL, ended_at INTEGER);
            CREATE TABLE IF NOT EXISTS meeting_turns (
                id TEXT PRIMARY KEY, meeting_id TEXT NOT NULL REFERENCES meetings(id),
                agent TEXT NOT NULL, phase TEXT NOT NULL, prompt TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 10,
                status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
                claimed_epoch INTEGER, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                error TEXT, reply_to TEXT NOT NULL DEFAULT 'all',
                depth INTEGER NOT NULL DEFAULT 0, question_message_id INTEGER);
            CREATE INDEX IF NOT EXISTS ix_meeting_turns ON meeting_turns(status,priority,created_at);
            CREATE TABLE IF NOT EXISTS meeting_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT NOT NULL REFERENCES meetings(id), actor TEXT NOT NULL,
                target TEXT NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL,
                model TEXT, created_at INTEGER NOT NULL);
        ''')
        # Migrate existing meeting databases without discarding their history.
        columns={r['name'] for r in db.execute('PRAGMA table_info(meeting_turns)')}
        for col,definition in [('reply_to',"TEXT NOT NULL DEFAULT 'all'"),
                               ('depth','INTEGER NOT NULL DEFAULT 0'),
                               ('question_message_id','INTEGER')]:
            if col not in columns:
                db.execute(f'ALTER TABLE meeting_turns ADD COLUMN {col} {definition}')
        columns={r['name'] for r in db.execute('PRAGMA table_info(meetings)')}
        for col,datatype in [('cancel_reason', 'TEXT'), ('cancelled_by','TEXT')]:
            if col not in columns:
                db.execute(f'ALTER TABLE meetings ADD COLUMN {col} {datatype}')


def _row(db, mid):
    row = db.execute('SELECT * FROM meetings WHERE id=?', (mid,)).fetchone()
    if not row:
        raise StoreError('Unknown meeting.')
    return row


def _message(db, mid, actor, target, body, kind='message', model=None):
    cursor=db.execute('''INSERT INTO meeting_messages(meeting_id,actor,target,kind,body,model,created_at)
               VALUES(?,?,?,?,?,?,?)''', (mid, actor, target, kind, body, model, int(time.time())))
    return cursor.lastrowid


def _turn(db, mid, agent, phase, prompt='', priority=10, reply_to='all', depth=0, question_message_id=None):
    now=int(time.time())
    db.execute('''INSERT INTO meeting_turns(id,meeting_id,agent,phase,prompt,priority,created_at,updated_at,
                                            reply_to,depth,question_message_id)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
               (uuid4().hex, mid, agent, phase, prompt, priority, now, now, reply_to, depth, question_message_id))


def create_meeting(store, *, topic, participants, requested_by='human', project_id=None,
                   starts_at=None, start_now=True):
    if requested_by != 'human' and requested_by not in ROLES:
        raise StoreError('Requester must be the operator or a registered agent.')
    if not isinstance(topic,str) or not (1 <= len(topic.strip()) <= 280):
        raise StoreError('Meeting topic must contain 1–280 characters.')
    if not isinstance(participants,list) or not participants or len(participants) > 49:
        raise StoreError('Choose 1–49 participants.')
    if any(not isinstance(p,str) or p not in ROLES for p in participants):
        raise StoreError('Unknown meeting participant.')
    participants = list(dict.fromkeys(['product-manager', *participants]))
    if starts_at is None:
        starts_at=int(time.time())
    elif not isinstance(starts_at,int) or starts_at < int(time.time())-120:
        raise StoreError('Meeting start must be a current or future Unix timestamp.')
    if project_id:
        store.project(project_id)
    mid=uuid4().hex[:12]; now=int(time.time())
    with store.db() as db:
        if requested_by!='human':
            # Prevent runaway model-initiated meetings in one project.
            if not project_id:
                raise StoreError('Agent meetings must belong to a project.')
            n=db.execute('SELECT COUNT(*) FROM meetings WHERE project_id=? AND requested_by!=?',
                         (project_id,'human')).fetchone()[0]
            if n>=12:
                raise StoreError('Automatic meeting budget reached; human review required.')
        db.execute('''INSERT INTO meetings(id,project_id,topic,requested_by,participants,starts_at,created_at)
                   VALUES(?,?,?,?,?,?,?)''', (mid,project_id,topic.strip(),requested_by,
                                             json.dumps(participants),starts_at,now))
        _message(db,mid,'system','all',f'Meeting scheduled by {requested_by}; Product Manager chairs.','system')
    if start_now and starts_at<=now:
        start_meeting(store,mid)
    return meeting(store,mid)


def start_meeting(store,mid):
    with store.db() as db:
        r=_row(db,mid)
        if r['status']!='scheduled':
            raise StoreError('Only a scheduled meeting can start.')
        db.execute("UPDATE meetings SET status='live',floor='agents' WHERE id=?",(mid,))
        participants=json.loads(r['participants'])
        _turn(db,mid,'product-manager','opening',priority=0)
        for a in participants:
            if a!='product-manager':
                _turn(db,mid,a,'contribution')
        _turn(db,mid,'product-manager','summary',priority=20)
        _message(db,mid,'system','all','Meeting opened. All agent comments pause when the human takes the floor.','system')


def activate_due(store):
    with store.db() as db:
        mids=[r[0] for r in db.execute('SELECT id FROM meetings WHERE status=? AND starts_at<=?',
                                      ('scheduled',int(time.time())))]
    for mid in mids:
        try: start_meeting(store,mid)
        except StoreError: pass
    return len(mids)


def meeting(store,mid):
    with store.db() as db:
        out=dict(_row(db,mid)); out['participants']=json.loads(out['participants'])
        out['messages']=[dict(r) for r in db.execute('SELECT * FROM meeting_messages WHERE meeting_id=? ORDER BY id ASC',(mid,))]
        out['turns']=[dict(r) for r in db.execute('SELECT * FROM meeting_turns WHERE meeting_id=? ORDER BY priority,created_at,rowid',(mid,))]
        return out


def list_meetings(store,project_id=None):
    with store.db() as db:
        if project_id:
            result=db.execute('SELECT * FROM meetings WHERE project_id=? ORDER BY created_at DESC,rowid DESC',(project_id,))
        else:
            result=db.execute('SELECT * FROM meetings ORDER BY created_at DESC,rowid DESC LIMIT 100')
        return [{**dict(r), 'participants':json.loads(r['participants'])} for r in result]


def claim_turn(store):
    with store.db() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('''SELECT t.* FROM meeting_turns t JOIN meetings m ON m.id=t.meeting_id
                         WHERE t.status='queued' AND m.status='live' AND m.floor='agents'
                         ORDER BY t.priority ASC,t.created_at ASC,t.rowid ASC LIMIT 1''').fetchone()
        if not row:
            return None
        m=_row(db,row['meeting_id'])
        db.execute('''UPDATE meeting_turns SET status='processing',claimed_epoch=?,attempts=attempts+1,updated_at=? WHERE id=?''',
                   (m['epoch'],int(time.time()),row['id']))
        return dict(db.execute('SELECT * FROM meeting_turns WHERE id=?',(row['id'],)).fetchone())


MAX_AGENT_QUESTIONS = 32
MAX_QUESTION_DEPTH = 2


def finish_turn(store,tid,body,model,questions=None):
    """Publish a contribution and route any explicit questions atomically.

    Any role may address any other role; a previously absent specialist is invited
    automatically. The respondent gets highest priority, ahead of generic turns.
    Limits prevent ping-pong conversations and runaway API spend.
    """
    if not isinstance(body,str) or not body.strip() or len(body)>8000:
        raise StoreError('Meeting reply must contain 1–8000 characters.')
    questions = questions if questions is not None else []
    if not isinstance(questions,list) or len(questions)>1:
        raise StoreError('One directed agent question is allowed per turn.')
    with store.db() as db:
        db.execute('BEGIN IMMEDIATE')
        r=db.execute('SELECT * FROM meeting_turns WHERE id=?',(tid,)).fetchone()
        if not r or r['status']!='processing': return False
        m=_row(db,r['meeting_id'])
        if m['status']!='live':
            db.execute("UPDATE meeting_turns SET status='cancelled' WHERE id=?",(tid,))
            return False
        if m['floor']!='agents' or m['epoch']!=r['claimed_epoch']:
            # Stale response after human interruption: never publish it.
            db.execute("UPDATE meeting_turns SET status='queued',claimed_epoch=NULL,priority=5 WHERE id=?",(tid,))
            return False
        if r['depth']>=MAX_QUESTION_DEPTH and questions:
            raise StoreError('Meeting question depth limit reached.')
        count=db.execute("SELECT COUNT(*) FROM meeting_messages WHERE meeting_id=? AND kind='question' AND actor!='human'",
                         (r['meeting_id'],)).fetchone()[0]
        participants=json.loads(m['participants'])
        for q in questions:
            if not isinstance(q,dict) or set(q)!={'recipient','message'}:
                raise StoreError('Directed question must have recipient and message.')
            dest=q['recipient']; question=q['message']
            if dest not in ROLES or dest==r['agent']:
                raise StoreError('A meeting question must target a different registered agent.')
            if not isinstance(question,str) or not question.strip() or len(question)>2000 or count>=MAX_AGENT_QUESTIONS:
                raise StoreError('Meeting question is invalid or question budget has been exhausted.')
        db.execute("UPDATE meeting_turns SET status='done',error=NULL,updated_at=? WHERE id=?",(int(time.time()),tid))
        audience=r['reply_to'] if r['phase'] in ('answer','response') else 'all'
        _message(db,r['meeting_id'],r['agent'],audience,body.strip(),r['phase'],model)
        for q in questions:
            dest=q['recipient']; question=q['message'].strip()
            if dest not in participants:
                participants.append(dest)
            qid=_message(db,r['meeting_id'],r['agent'],dest,question,'question',model)
            _turn(db,r['meeting_id'],dest,'answer',question,priority=-15,
                  reply_to=r['agent'],depth=r['depth']+1,question_message_id=qid)
            if r['agent']!='product-manager' and dest!='product-manager':
                _turn(db,r['meeting_id'],'product-manager','moderation',
                      'A question was directed to '+dest+': '+question,priority=-8,depth=MAX_QUESTION_DEPTH)
        if participants!=json.loads(m['participants']):
            db.execute('UPDATE meetings SET participants=? WHERE id=?',
                       (json.dumps(participants),r['meeting_id']))
        return True


def fail_turn(store,tid,error,retryable=False):
    with store.db() as db:
        r=db.execute('SELECT * FROM meeting_turns WHERE id=?',(tid,)).fetchone()
        if not r or r['status']!='processing': return
        status='queued' if retryable and r['attempts']<2 else 'failed'
        db.execute('UPDATE meeting_turns SET status=?,error=?,updated_at=? WHERE id=?',
                   (status,str(error)[:300],int(time.time()),tid))
        if status=='failed':
            _message(db,r['meeting_id'],'system','all',f'{r["agent"]}: response failed ({type(error).__name__}).','error')


def take_floor(store,mid,*,human_message=None,target='all'):
    if target!='all' and target not in ROLES:
        raise StoreError('Unknown audience.')
    if human_message is not None and (not isinstance(human_message,str) or not human_message.strip()
                                      or len(human_message)>4000):
        raise StoreError('Human contribution must contain 1–4000 characters.')
    with store.db() as db:
        r=_row(db,mid)
        if r['status']!='live': raise StoreError('Meeting is not live.')
        db.execute("UPDATE meetings SET floor='human',epoch=epoch+1 WHERE id=?",(mid,))
        if human_message:
            who=target if target!='all' else 'product-manager'
            participants=json.loads(r['participants'])
            if who not in participants:
                participants.append(who)
                db.execute('UPDATE meetings SET participants=? WHERE id=?',
                           (json.dumps(participants),mid))
            qid=_message(db,mid,'human',who,human_message.strip(),kind='question')
            # Directed specialist answers before general agenda resumes.
            _turn(db,mid,who,'response',human_message.strip(),priority=-15,
                  reply_to='human',question_message_id=qid)
            if who!='product-manager':
                _turn(db,mid,'product-manager','moderation',human_message.strip(),priority=-9,
                      depth=MAX_QUESTION_DEPTH)
        else:
            _message(db,mid,'system','all','Operator took the floor. Agent turns are paused.','system')
    return meeting(store,mid)


def yield_floor(store,mid):
    with store.db() as db:
        r=_row(db,mid)
        if r['status']!='live': raise StoreError('Meeting is not live.')
        db.execute("UPDATE meetings SET floor='agents' WHERE id=?",(mid,))
        _message(db,mid,'system','all','Operator returned the floor to agents.','system')
    return meeting(store,mid)


def end_meeting(store,mid):
    with store.db() as db:
        r=_row(db,mid)
        if r['status'] in ('ended','cancelled'): raise StoreError('Meeting already closed.')
        db.execute("UPDATE meetings SET status='ended',floor='human',epoch=epoch+1,ended_at=? WHERE id=?",
                   (int(time.time()),mid))
        db.execute("UPDATE meeting_turns SET status='cancelled' WHERE meeting_id=? AND status IN ('queued','processing')",(mid,))
        _message(db,mid,'system','all','Meeting ended by the operator.','system')
    return meeting(store,mid)


def cancel_meeting(store, mid, reason, operator='human'):
    """Cancel a scheduled/live room, never publishing a stale in-flight turn."""
    reason = validate_cancel_reason(reason)
    if operator != 'human':
        raise StoreError('Meeting cancellation requires the human operator.')
    with store.db() as db:
        db.execute('BEGIN IMMEDIATE')
        r=_row(db,mid)
        if r['status'] not in ('scheduled','live'):
            raise StoreError('Only scheduled or live meetings can be cancelled.')
        now = int(time.time())
        db.execute("""UPDATE meetings SET status='cancelled',floor='human',epoch=epoch+1,
                   ended_at=?,cancel_reason=?,cancelled_by=? WHERE id=?""",
                   (now,reason,operator,mid))
        db.execute("""UPDATE meeting_turns SET status='cancelled',claimed_epoch=NULL,
                   updated_at=? WHERE meeting_id=? AND status IN ('queued','processing')""",
                   (now,mid))
        _message(db,mid,'system','all',f'Meeting cancelled by {operator}. Reason: {reason}','system')
    return meeting(store,mid)


def recover_meetings(store):
    with store.db() as db:
        n=db.execute("SELECT COUNT(*) FROM meeting_turns WHERE status='processing'").fetchone()[0]
        db.execute("UPDATE meeting_turns SET status='queued' WHERE status='processing'")
    return n
