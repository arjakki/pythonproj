"""
Visual dashboard — Flask web app showing the AI agent, Pinecone, and MySQL live.
Run:  python dashboard.py
Open: http://localhost:5001
"""
import os, json, uuid, traceback
from datetime import datetime, timezone
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request
import mysql.connector
from pinecone import Pinecone, ServerlessSpec
import anthropic

load_dotenv()
app = Flask(__name__)

# ── Pinecone ─────────────────────────────────────────────────────────────────
_pc = None
_idx = None
INDEX_NAME = os.getenv("PINECONE_INDEX", "agent-memory")
EMBED_MODEL = "multilingual-e5-large"
EMBED_DIM   = 1024

def get_pc():
    global _pc
    if not _pc:
        _pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return _pc

def get_index():
    global _idx
    if not _idx:
        pc = get_pc()
        if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
            pc.create_index(
                name=INDEX_NAME, dimension=EMBED_DIM, metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1"),
                ),
            )
        _idx = pc.Index(INDEX_NAME)
    return _idx

def embed(text, input_type="passage"):
    resp = get_pc().inference.embed(
        model=EMBED_MODEL, inputs=[text],
        parameters={"input_type": input_type, "truncate": "END"},
    )
    return resp[0].values

# ── MySQL ─────────────────────────────────────────────────────────────────────
def get_conn():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
    )

# ── Tool functions ────────────────────────────────────────────────────────────
def t_store_memory(text, metadata="{}"):
    meta = json.loads(metadata) if isinstance(metadata, str) and metadata.strip() else (metadata or {})
    doc_id = str(uuid.uuid4())
    meta.update({"text": text, "created_at": datetime.now(timezone.utc).isoformat()})
    get_index().upsert(vectors=[{"id": doc_id, "values": embed(text), "metadata": meta}])
    return {"success": True, "id": doc_id}

def t_search_memory(query, top_k=5):
    res = get_index().query(vector=embed(query, "query"), top_k=top_k, include_metadata=True)
    return [{"id": m.id, "score": round(m.score, 4),
             "text": m.metadata.get("text", ""),
             "metadata": {k: v for k, v in m.metadata.items() if k != "text"}}
            for m in res.matches]

def t_mysql_query(sql):
    conn = get_conn(); cur = conn.cursor(dictionary=True)
    cur.execute(sql); rows = cur.fetchall()
    cur.close(); conn.close(); return rows

def t_mysql_execute(sql):
    conn = get_conn(); cur = conn.cursor()
    cur.execute(sql); conn.commit()
    res = {"affected_rows": cur.rowcount, "last_insert_id": cur.lastrowid}
    cur.close(); conn.close(); return res

def t_list_tables():
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SHOW TABLES"); tables = [r[0] for r in cur.fetchall()]
    cur.close(); conn.close(); return tables

def t_describe_table(table_name):
    conn = get_conn(); cur = conn.cursor(dictionary=True)
    cur.execute(f"DESCRIBE `{table_name}`"); schema = cur.fetchall()
    cur.close(); conn.close(); return schema

# ── Claude tool definitions ───────────────────────────────────────────────────
CLAUDE_TOOLS = [
    {"name": "store_memory",
     "description": "Store text as a vector embedding in Pinecone.",
     "input_schema": {"type": "object", "required": ["text"],
                      "properties": {"text": {"type": "string"},
                                     "metadata": {"type": "string"}}}},
    {"name": "search_memory",
     "description": "Semantic search in Pinecone.",
     "input_schema": {"type": "object", "required": ["query"],
                      "properties": {"query": {"type": "string"},
                                     "top_k": {"type": "integer"}}}},
    {"name": "mysql_query",
     "description": "Run a SELECT SQL query.",
     "input_schema": {"type": "object", "required": ["sql"],
                      "properties": {"sql": {"type": "string"}}}},
    {"name": "mysql_execute",
     "description": "Run INSERT/UPDATE/DELETE.",
     "input_schema": {"type": "object", "required": ["sql"],
                      "properties": {"sql": {"type": "string"}}}},
    {"name": "mysql_list_tables",
     "description": "List all MySQL tables.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "mysql_describe_table",
     "description": "Describe a MySQL table schema.",
     "input_schema": {"type": "object", "required": ["table_name"],
                      "properties": {"table_name": {"type": "string"}}}},
]

DISPATCH = {
    "store_memory":        lambda i: t_store_memory(**i),
    "search_memory":       lambda i: t_search_memory(**i),
    "mysql_query":         lambda i: t_mysql_query(**i),
    "mysql_execute":       lambda i: t_mysql_execute(**i),
    "mysql_list_tables":   lambda i: t_list_tables(),
    "mysql_describe_table":lambda i: t_describe_table(**i),
}

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with Pinecone (vector/semantic memory) "
    "and MySQL (relational data) tools. Use tools when relevant to the user's request."
)

def _serialize(obj):
    if isinstance(obj, list):  return [_serialize(i) for i in obj]
    if isinstance(obj, dict):  return {k: _serialize(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"): return {k: _serialize(v) for k, v in obj.__dict__.items()}
    return obj

def run_agent(message: str, history: list):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    messages = list(history) + [{"role": "user", "content": message}]
    tool_log = []

    while True:
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=2048,
            system=SYSTEM_PROMPT, tools=CLAUDE_TOOLS, messages=messages,
        )
        texts     = [b.text for b in resp.content if hasattr(b, "text")]
        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        if resp.stop_reason == "end_turn" or not tool_uses:
            messages.append({"role": "assistant", "content": resp.content})
            return {"reply": " ".join(texts) or "(no text)",
                    "tool_calls": tool_log,
                    "history": [{"role": m["role"],
                                 "content": _serialize(m["content"])} for m in messages]}

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for tu in tool_uses:
            try:
                out = DISPATCH[tu.name](tu.input)
                out_text = json.dumps(out, default=str)
                status = "success"
            except Exception as e:
                out = {"error": str(e)}
                out_text = json.dumps(out)
                status = "error"
            tool_log.append({"tool": tu.name, "input": tu.input,
                              "output": out, "status": status})
            results.append({"type": "tool_result",
                             "tool_use_id": tu.id, "content": out_text})
        messages.append({"role": "user", "content": results})

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/status")
def api_status():
    out = {}
    try:
        idxs = [i.name for i in get_pc().list_indexes()]
        out["pinecone"] = {"ok": True, "indexes": idxs}
    except Exception as e:
        out["pinecone"] = {"ok": False, "error": str(e)}
    try:
        conn = get_conn(); cur = conn.cursor()
        cur.execute("SELECT VERSION()"); ver = cur.fetchone()[0]
        cur.close(); conn.close()
        out["mysql"] = {"ok": True, "version": ver}
    except Exception as e:
        out["mysql"] = {"ok": False, "error": str(e)}
    out["anthropic"] = {"ok": bool(os.getenv("ANTHROPIC_API_KEY"))}
    return jsonify(out)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    try:
        result = run_agent(data.get("message", ""), data.get("history", []))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "detail": traceback.format_exc()}), 500

@app.route("/api/memories/stats")
def api_mem_stats():
    try:
        s = get_index().describe_index_stats()
        return jsonify({"total": s.total_vector_count, "dim": s.dimension, "index": INDEX_NAME})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/memories/search")
def api_mem_search():
    q = request.args.get("q", "")
    if not q: return jsonify([])
    try:
        return jsonify(t_search_memory(q, int(request.args.get("top_k", 5))))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/tables")
def api_tables():
    try: return jsonify(t_list_tables())
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/table/<name>")
def api_table(name):
    try:
        return jsonify({"rows": t_mysql_query(f"SELECT * FROM `{name}` ORDER BY id DESC LIMIT 50"),
                        "schema": t_describe_table(name)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
