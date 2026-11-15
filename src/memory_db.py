from pathlib import Path
import json
import re
import sqlite3
import uuid


# ============================================================
# RAG V10 - SQLITE PERSISTENT CONVERSATIONAL MEMORY
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

MEMORY_DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "memory"
    / "rag_v10_memory.db"
)

MEMORY_MAX_MESSAGES = 12


# ============================================================
# DATABASE
# ============================================================

def get_connection(
    db_path=MEMORY_DB_PATH,
):
    db_path = Path(db_path)

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        db_path
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


def initialize_database(
    db_path=MEMORY_DB_PATH,
):
    connection = get_connection(
        db_path
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            query_type TEXT,
            standalone_question TEXT,
            sources_json TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (conversation_id)
            REFERENCES conversations(id)
            ON DELETE CASCADE
        )
        """
    )

    connection.commit()
    connection.close()


# ============================================================
# CONVERSATIONS
# ============================================================

def create_conversation(
    title="New conversation",
    db_path=MEMORY_DB_PATH,
):
    initialize_database(
        db_path
    )

    conversation_id = uuid.uuid4().hex

    connection = get_connection(
        db_path
    )

    connection.execute(
        """
        INSERT INTO conversations (
            id,
            title
        )
        VALUES (?, ?)
        """,
        (
            conversation_id,
            title,
        ),
    )

    connection.commit()
    connection.close()

    return conversation_id


def create_new_chat(
    db_path=MEMORY_DB_PATH,
):
    return create_conversation(
        title="New conversation",
        db_path=db_path,
    )


def list_conversations(
    db_path=MEMORY_DB_PATH,
):
    initialize_database(
        db_path
    )

    connection = get_connection(
        db_path
    )

    rows = connection.execute(
        """
        SELECT
            c.id,
            c.title,
            c.created_at,
            c.updated_at,
            COUNT(m.id) AS message_count

        FROM conversations AS c

        LEFT JOIN messages AS m
            ON c.id = m.conversation_id

        GROUP BY
            c.id,
            c.title,
            c.created_at,
            c.updated_at

        ORDER BY
            c.updated_at DESC,
            c.rowid DESC
        """
    ).fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


def get_conversation(
    conversation_id,
    db_path=MEMORY_DB_PATH,
):
    initialize_database(
        db_path
    )

    connection = get_connection(
        db_path
    )

    row = connection.execute(
        """
        SELECT
            id,
            title,
            created_at,
            updated_at

        FROM conversations

        WHERE id = ?
        """,
        (
            conversation_id,
        ),
    ).fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)


def conversation_exists(
    conversation_id,
    db_path=MEMORY_DB_PATH,
):
    return (
        get_conversation(
            conversation_id,
            db_path,
        )
        is not None
    )


def get_latest_conversation_id(
    db_path=MEMORY_DB_PATH,
):
    initialize_database(
        db_path
    )

    connection = get_connection(
        db_path
    )

    row = connection.execute(
        """
        SELECT id

        FROM conversations

        ORDER BY
            updated_at DESC,
            rowid DESC

        LIMIT 1
        """
    ).fetchone()

    connection.close()

    if row is None:
        return None

    return row["id"]


def get_or_create_latest_conversation(
    db_path=MEMORY_DB_PATH,
):
    conversation_id = get_latest_conversation_id(
        db_path
    )

    if conversation_id is None:
        conversation_id = create_new_chat(
            db_path
        )

    return conversation_id


def create_title_from_question(
    question,
    max_length=55,
):
    title = re.sub(
        r"\s+",
        " ",
        question,
    ).strip()

    if len(title) <= max_length:
        return title

    return (
        title[:max_length].rstrip()
        + "..."
    )


def rename_conversation(
    conversation_id,
    title,
    db_path=MEMORY_DB_PATH,
):
    title = title.strip()

    if not title:
        return

    connection = get_connection(
        db_path
    )

    connection.execute(
        """
        UPDATE conversations

        SET
            title = ?,
            updated_at = CURRENT_TIMESTAMP

        WHERE id = ?
        """,
        (
            title,
            conversation_id,
        ),
    )

    connection.commit()
    connection.close()


def auto_title_conversation(
    conversation_id,
    first_question,
    db_path=MEMORY_DB_PATH,
):
    conversation = get_conversation(
        conversation_id,
        db_path,
    )

    if conversation is None:
        return

    if conversation["title"] == "New conversation":
        rename_conversation(
            conversation_id,
            create_title_from_question(
                first_question
            ),
            db_path,
        )


def clear_conversation(
    conversation_id,
    db_path=MEMORY_DB_PATH,
):
    connection = get_connection(
        db_path
    )

    connection.execute(
        """
        DELETE FROM messages

        WHERE conversation_id = ?
        """,
        (
            conversation_id,
        ),
    )

    connection.execute(
        """
        UPDATE conversations

        SET updated_at = CURRENT_TIMESTAMP

        WHERE id = ?
        """,
        (
            conversation_id,
        ),
    )

    connection.commit()
    connection.close()


def delete_conversation(
    conversation_id,
    db_path=MEMORY_DB_PATH,
):
    connection = get_connection(
        db_path
    )

    connection.execute(
        """
        DELETE FROM conversations

        WHERE id = ?
        """,
        (
            conversation_id,
        ),
    )

    connection.commit()
    connection.close()


# ============================================================
# MESSAGES
# ============================================================

def save_message(
    conversation_id,
    role,
    content,
    query_type=None,
    standalone_question=None,
    sources=None,
    db_path=MEMORY_DB_PATH,
):
    initialize_database(
        db_path
    )

    if sources is None:
        sources = []

    sources_json = json.dumps(
        sources
    )

    connection = get_connection(
        db_path
    )

    connection.execute(
        """
        INSERT INTO messages (
            conversation_id,
            role,
            content,
            query_type,
            standalone_question,
            sources_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            conversation_id,
            role,
            content,
            query_type,
            standalone_question,
            sources_json,
        ),
    )

    connection.execute(
        """
        UPDATE conversations

        SET updated_at = CURRENT_TIMESTAMP

        WHERE id = ?
        """,
        (
            conversation_id,
        ),
    )

    connection.commit()
    connection.close()


def load_messages(
    conversation_id,
    db_path=MEMORY_DB_PATH,
):
    initialize_database(
        db_path
    )

    connection = get_connection(
        db_path
    )

    rows = connection.execute(
        """
        SELECT
            id,
            role,
            content,
            query_type,
            standalone_question,
            sources_json,
            created_at

        FROM messages

        WHERE conversation_id = ?

        ORDER BY id ASC
        """,
        (
            conversation_id,
        ),
    ).fetchall()

    connection.close()

    messages = []

    for row in rows:
        try:
            sources = json.loads(
                row["sources_json"]
                or "[]"
            )
        except json.JSONDecodeError:
            sources = []

        messages.append(
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "query_type": row["query_type"],
                "standalone_question": row["standalone_question"],
                "sources": sources,
                "created_at": row["created_at"],
            }
        )

    return messages


def load_recent_memory(
    conversation_id,
    db_path=MEMORY_DB_PATH,
):
    return load_messages(
        conversation_id,
        db_path,
    )[-MEMORY_MAX_MESSAGES:]


def count_messages(
    conversation_id,
    db_path=MEMORY_DB_PATH,
):
    connection = get_connection(
        db_path
    )

    row = connection.execute(
        """
        SELECT COUNT(*) AS total

        FROM messages

        WHERE conversation_id = ?
        """,
        (
            conversation_id,
        ),
    ).fetchone()

    connection.close()

    return row["total"]
