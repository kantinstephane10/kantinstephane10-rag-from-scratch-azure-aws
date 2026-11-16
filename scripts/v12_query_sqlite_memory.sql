-- RAG V12 - SQLite persistent memory proof
-- Run from the repository root:
--
-- sqlite3 data/memory/rag_v10_memory.db < scripts/v12_query_sqlite_memory.sql
--
-- For the isolated Stage 5A test database:
--
-- sqlite3 data/memory/rag_v12_stage5_test.db < scripts/v12_query_sqlite_memory.sql

.headers on
.mode column

.print '============================================================'
.print 'CONVERSATIONS'
.print '============================================================'

SELECT
    id,
    title,
    created_at,
    updated_at
FROM conversations
ORDER BY updated_at DESC, rowid DESC;

.print ''
.print '============================================================'
.print 'MESSAGE COUNTS BY CONVERSATION'
.print '============================================================'

SELECT
    c.id AS conversation_id,
    c.title,
    COUNT(m.id) AS message_count
FROM conversations AS c
LEFT JOIN messages AS m
    ON m.conversation_id = c.id
GROUP BY
    c.id,
    c.title
ORDER BY
    c.updated_at DESC;

.print ''
.print '============================================================'
.print 'MESSAGES'
.print '============================================================'

SELECT
    id,
    conversation_id,
    role,
    query_type,
    standalone_question,
    created_at
FROM messages
ORDER BY id ASC;

.print ''
.print '============================================================'
.print 'MESSAGES + STORED SOURCE COUNTS'
.print '============================================================'

SELECT
    id,
    role,
    query_type,
    CASE
        WHEN sources_json IS NULL OR sources_json = '' THEN 0
        ELSE json_array_length(sources_json)
    END AS stored_source_count,
    standalone_question
FROM messages
ORDER BY id ASC;

.print ''
.print '============================================================'
.print 'LATEST CONVERSATION - FULL TURN SUMMARY'
.print '============================================================'

WITH latest_conversation AS (
    SELECT id
    FROM conversations
    ORDER BY updated_at DESC, rowid DESC
    LIMIT 1
)
SELECT
    m.id,
    m.role,
    m.query_type,
    CASE
        WHEN m.sources_json IS NULL OR m.sources_json = '' THEN 0
        ELSE json_array_length(m.sources_json)
    END AS stored_source_count,
    m.standalone_question,
    substr(m.content, 1, 180) AS content_preview
FROM messages AS m
JOIN latest_conversation AS lc
    ON lc.id = m.conversation_id
ORDER BY m.id ASC;
