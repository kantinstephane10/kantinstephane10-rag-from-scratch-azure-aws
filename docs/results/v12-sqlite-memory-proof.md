# RAG V12 — SQLite Persistent Memory Proof

## Purpose

This document records the observed SQLite persistence validation from V12 Stage 5A.

The SQLite database itself is intentionally **not committed to Git**. The repository stores the query script and measured proof instead.

Query script:

`scripts/v12_query_sqlite_memory.sql`

## Tables

V12 reuses the persistent memory schema from `src/memory_db.py`.

### conversations

Stores:

- conversation ID
- title
- created timestamp
- updated timestamp

### messages

Stores:

- message ID
- conversation ID
- role
- content
- query type
- standalone rewritten question
- retrieved sources as JSON
- created timestamp

## Observed Stage 5A conversation

~~~text
Conversation ID:
5610b8dfbcc9426ca35a092bbc13bffb

Conversation title:
What does AWS recommend for building reliable cloud wor...

Stored messages:
4
~~~

## Observed message persistence

~~~text
MESSAGE 1
Role: user
Query type: provider:AWS
Standalone question:
What does AWS recommend for building reliable cloud workloads?
Stored sources: 0

MESSAGE 2
Role: assistant
Query type: provider:AWS
Standalone question:
What does AWS recommend for building reliable cloud workloads?
Stored sources: 5

MESSAGE 3
Role: user
Query type: cross_provider
Standalone question:
How does Azure compare with AWS's reliability recommendations for building reliable cloud workloads?
Stored sources: 0

MESSAGE 4
Role: assistant
Query type: cross_provider
Standalone question:
How does Azure compare with AWS's reliability recommendations for building reliable cloud workloads?
Stored sources: 6
~~~

## What this proves

The Stage 5A run demonstrated that:

- a new conversation was created in SQLite;
- the first user turn was persisted;
- the first assistant answer was persisted with five retrieved AWS evidence records;
- the follow-up question was rewritten into a standalone retrieval question;
- the rewritten question was persisted;
- the comparison answer was persisted with six AWS/Azure evidence records;
- the database reported four total messages;
- the recent-memory window respected the configured 12-message limit;
- re-reading the database after a simulated restart returned the same four messages;
- the final assistant answer survived restart;
- the final assistant source evidence survived restart.

Observed validation:

~~~text
Two exchanges persisted as 4 messages: PASS
SQLite reports 4 stored messages: PASS
Recent memory respects 12-message limit: PASS
Standalone follow-up persisted in SQLite: PASS
Retrieved evidence persisted with assistant answer: PASS
Persisted comparison sources include AWS and Azure: PASS
Conversation survives simulated restart: PASS
Assistant answer survives restart: PASS
Assistant source evidence survives restart: PASS
~~~

## How to query the database

From the repository root:

~~~powershell
sqlite3 data/memory/rag_v10_memory.db < scripts/v12_query_sqlite_memory.sql
~~~

For the isolated Stage 5A validation database:

~~~powershell
sqlite3 data/memory/rag_v12_stage5_test.db < scripts/v12_query_sqlite_memory.sql
~~~

If the `sqlite3` CLI is not installed, the same database can be queried through Python's built-in `sqlite3` module.

## Security / repository decision

The live SQLite database is not committed because it can contain conversation content and retrieved evidence. Only schema-aware query scripts and measured validation proof belong in Git.
