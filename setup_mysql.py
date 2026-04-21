"""
One-time script to create the MySQL database and tables used by the AI agent.
Run this before starting the MCP server or agent.

Usage:
    python setup_mysql.py
"""
import os
import sys
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()

DDL = [
    """
    CREATE TABLE IF NOT EXISTS memories (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        content     TEXT NOT NULL,
        source      VARCHAR(255),
        pinecone_id VARCHAR(36),
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        title       VARCHAR(500) NOT NULL,
        content     LONGTEXT NOT NULL,
        category    VARCHAR(100),
        tags        JSON,
        pinecone_id VARCHAR(36),
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_history (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        session_id VARCHAR(36)                    NOT NULL,
        role       ENUM('user', 'assistant')      NOT NULL,
        content    TEXT                           NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_session (session_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


def main() -> None:
    host     = os.getenv("MYSQL_HOST", "localhost")
    port     = int(os.getenv("MYSQL_PORT", "3306"))
    user     = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    db_name  = os.getenv("MYSQL_DATABASE", "agent_db")

    print(f"Connecting to MySQL at {host}:{port} as '{user}' ...")
    try:
        conn = mysql.connector.connect(
            host=host, port=port, user=user, password=password
        )
        cursor = conn.cursor()

        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        cursor.execute(f"USE `{db_name}`")
        print(f"Database '{db_name}' ready.")

        for ddl in DDL:
            cursor.execute(ddl)
        conn.commit()

        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Tables created: {tables}")

        cursor.close()
        conn.close()
        print("\nSetup complete. You can now start mcp_server.py and agent.py.")
    except Error as exc:
        print(f"MySQL error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
