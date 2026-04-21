"""
MCP Server — exposes Pinecone (vector store) and MySQL tools to any MCP client.

Start:
    python mcp_server.py

The server speaks the MCP stdio protocol so Claude Code (and agent.py) can
connect to it automatically.
"""
import os
import json
import uuid
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import mysql.connector
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

mcp = FastMCP("db-agent-server")

# ─── Pinecone ────────────────────────────────────────────────────────────────

_pc: Optional[Pinecone] = None
_index = None

INDEX_NAME    = os.getenv("PINECONE_INDEX", "agent-memory")
EMBED_MODEL   = "multilingual-e5-large"
EMBED_DIM     = 1024


def _get_pc() -> Pinecone:
    global _pc
    if _pc is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY is not set")
        _pc = Pinecone(api_key=api_key)
    return _pc


def _get_index():
    global _index
    if _index is None:
        pc = _get_pc()
        existing = [i.name for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1"),
                ),
            )
        _index = pc.Index(INDEX_NAME)
    return _index


def _embed(text: str, input_type: str = "passage") -> list[float]:
    pc = _get_pc()
    resp = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[text],
        parameters={"input_type": input_type, "truncate": "END"},
    )
    return resp[0].values


# ─── MySQL ───────────────────────────────────────────────────────────────────

def _get_conn():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "agent_db"),
        autocommit=False,
    )


# ─── Pinecone tools ──────────────────────────────────────────────────────────

@mcp.tool()
def store_memory(text: str, metadata: str = "{}") -> str:
    """Store text as a vector embedding in Pinecone for semantic retrieval.

    Args:
        text:     The text content to embed and store.
        metadata: Optional JSON string of extra key/value metadata.

    Returns:
        JSON with the generated vector ID on success.
    """
    try:
        meta = json.loads(metadata) if metadata.strip() else {}
        embedding = _embed(text, "passage")
        doc_id = str(uuid.uuid4())
        meta["text"] = text
        meta["created_at"] = datetime.utcnow().isoformat()
        _get_index().upsert(vectors=[{"id": doc_id, "values": embedding, "metadata": meta}])
        return json.dumps({"success": True, "id": doc_id})
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)})


@mcp.tool()
def search_memory(query: str, top_k: int = 5) -> str:
    """Search Pinecone for the most semantically similar stored content.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of results to return (default 5).

    Returns:
        JSON array of matches, each with id, score, text, and metadata.
    """
    try:
        embedding = _embed(query, "query")
        results = _get_index().query(vector=embedding, top_k=top_k, include_metadata=True)
        matches = [
            {
                "id": m.id,
                "score": round(m.score, 4),
                "text": m.metadata.get("text", ""),
                "metadata": {k: v for k, v in m.metadata.items() if k != "text"},
            }
            for m in results.matches
        ]
        return json.dumps(matches, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def delete_memory(vector_id: str) -> str:
    """Delete a vector from Pinecone by its ID.

    Args:
        vector_id: UUID of the vector to delete.

    Returns:
        JSON confirmation.
    """
    try:
        _get_index().delete(ids=[vector_id])
        return json.dumps({"success": True, "deleted_id": vector_id})
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)})


@mcp.tool()
def pinecone_stats() -> str:
    """Return statistics for the Pinecone index (total vectors, dimension, namespaces).

    Returns:
        JSON with index metadata.
    """
    try:
        stats = _get_index().describe_index_stats()
        return json.dumps(
            {
                "index_name": INDEX_NAME,
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "namespaces": {k: v.vector_count for k, v in (stats.namespaces or {}).items()},
            }
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ─── MySQL tools ─────────────────────────────────────────────────────────────

_ALLOWED_READ = ("SELECT", "SHOW", "DESCRIBE", "EXPLAIN")


@mcp.tool()
def mysql_query(sql: str) -> str:
    """Execute a read-only SQL query (SELECT / SHOW / DESCRIBE) and return results.

    Args:
        sql: The SQL query to execute.

    Returns:
        JSON array of result rows.
    """
    first_word = sql.strip().split()[0].upper() if sql.strip() else ""
    if first_word not in _ALLOWED_READ:
        return json.dumps({"error": f"mysql_query only allows {_ALLOWED_READ}. Use mysql_execute for writes."})
    try:
        conn = _get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return json.dumps(rows, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def mysql_execute(sql: str) -> str:
    """Execute a write SQL statement (INSERT / UPDATE / DELETE / CREATE / ALTER).

    Args:
        sql: The SQL statement to execute.

    Returns:
        JSON with affected_rows and last_insert_id.
    """
    first_word = sql.strip().split()[0].upper() if sql.strip() else ""
    if first_word in ("SELECT",):
        return json.dumps({"error": "Use mysql_query for SELECT statements."})
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        result = {"affected_rows": cursor.rowcount, "last_insert_id": cursor.lastrowid}
        cursor.close()
        conn.close()
        return json.dumps(result)
    except Exception as exc:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return json.dumps({"error": str(exc)})


@mcp.tool()
def mysql_list_tables() -> str:
    """List all tables in the configured MySQL database.

    Returns:
        JSON array of table name strings.
    """
    try:
        conn = _get_conn()
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return json.dumps(tables)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def mysql_describe_table(table_name: str) -> str:
    """Get the column definitions for a MySQL table.

    Args:
        table_name: Name of the table to describe.

    Returns:
        JSON array of column definitions (Field, Type, Null, Key, Default, Extra).
    """
    try:
        conn = _get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(f"DESCRIBE `{table_name}`")
        schema = cursor.fetchall()
        cursor.close()
        conn.close()
        return json.dumps(schema, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
