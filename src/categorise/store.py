from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg import errors
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .models import CandidateRanking, Category, DocumentRecord


DEFAULT_DATABASE_URL = os.getenv(
    "DOCUMIND_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/documind",
)


class PostgresStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or DEFAULT_DATABASE_URL
        self.initialize()

    def initialize(self) -> None:
        with self._connect() as connection:
            try:
                connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except errors.FeatureNotSupported as exc:
                raise RuntimeError(
                    "PostgreSQL pgvector extension is not available. Use a pgvector-enabled "
                    "PostgreSQL image, for example pgvector/pgvector:pg16."
                ) from exc
            except errors.UndefinedFile as exc:
                raise RuntimeError(
                    "PostgreSQL pgvector extension is not installed on this server. Use a "
                    "pgvector-enabled PostgreSQL image, for example pgvector/pgvector:pg16."
                ) from exc
            register_vector(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS categories (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source_path TEXT,
                    status TEXT NOT NULL,
                    primary_category_id TEXT,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS category_embeddings (
                    category_id TEXT PRIMARY KEY REFERENCES categories(id) ON DELETE CASCADE,
                    embedding vector NOT NULL,
                    embedding_model TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS documents_status_idx
                ON documents(status)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS category_embeddings_model_idx
                ON category_embeddings(embedding_model)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS documents_primary_category_idx
                ON documents(primary_category_id)
                """
            )

    def load_categories(self) -> list[Category]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM categories ORDER BY name"
            ).fetchall()
        return [Category.model_validate(row["payload"]) for row in rows]

    def save_categories(self, categories: list[Category]) -> None:
        with self._connect() as connection:
            for category in categories:
                self._upsert_category(connection, category)

    def add_category(self, category: Category) -> Category:
        with self._connect() as connection:
            self._upsert_category(connection, category)
        return category

    def load_documents(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [DocumentRecord.model_validate(row["payload"]) for row in rows]

    def save_documents(self, documents: list[DocumentRecord]) -> None:
        with self._connect() as connection:
            for document in documents:
                self._upsert_document(connection, document)

    def add_document(self, document: DocumentRecord) -> DocumentRecord:
        with self._connect() as connection:
            self._upsert_document(connection, document)
        return document

    def search_categories(
        self, embedding: list[float], embedding_model: str, limit: int = 10
    ) -> list[CandidateRanking]:
        with self._connect() as connection:
            register_vector(connection)
            rows = connection.execute(
                """
                SELECT
                    c.id,
                    c.name,
                    e.embedding <=> %s::vector AS distance
                FROM category_embeddings e
                JOIN categories c ON c.id = e.category_id
                WHERE c.status = 'active'
                  AND e.embedding_model = %s
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding_model, embedding, limit),
            ).fetchall()

        return [
            CandidateRanking(
                category_id=row["id"],
                category_name=row["name"],
                similarity=1.0 - float(row["distance"]),
            )
            for row in rows
        ]

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    @staticmethod
    def _upsert_category(connection: psycopg.Connection, category: Category) -> None:
        payload = category.model_dump(mode="json")
        connection.execute(
            """
            INSERT INTO categories (id, name, status, payload, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                status = EXCLUDED.status,
                payload = EXCLUDED.payload,
                updated_at = EXCLUDED.updated_at
            """,
            (
                category.id,
                category.name,
                category.status,
                Jsonb(payload),
                category.created_at,
                category.updated_at,
            ),
        )
        if category.embedding and category.embedding_model:
            connection.execute(
                """
                INSERT INTO category_embeddings (
                    category_id,
                    embedding,
                    embedding_model,
                    updated_at
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (category_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    category.id,
                    category.embedding,
                    category.embedding_model,
                    category.updated_at,
                ),
            )

    @staticmethod
    def _upsert_document(connection: psycopg.Connection, document: DocumentRecord) -> None:
        payload = document.model_dump(mode="json")
        connection.execute(
            """
            INSERT INTO documents (
                id,
                filename,
                source_path,
                status,
                primary_category_id,
                payload,
                created_at,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                filename = EXCLUDED.filename,
                source_path = EXCLUDED.source_path,
                status = EXCLUDED.status,
                primary_category_id = EXCLUDED.primary_category_id,
                payload = EXCLUDED.payload,
                updated_at = EXCLUDED.updated_at
            """,
            (
                document.id,
                document.filename,
                document.source_path,
                document.status,
                document.primary_category_id,
                Jsonb(payload),
                document.created_at,
                document.updated_at,
            ),
        )


class JsonStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.categories_path = self.root / "categories.json"
        self.documents_path = self.root / "documents.json"

    def load_categories(self) -> list[Category]:
        return [Category.model_validate(item) for item in self._read_list(self.categories_path)]

    def save_categories(self, categories: list[Category]) -> None:
        self._write_list(self.categories_path, [item.model_dump(mode="json") for item in categories])

    def add_category(self, category: Category) -> Category:
        categories = self.load_categories()
        categories.append(category)
        self.save_categories(categories)
        return category

    def load_documents(self) -> list[DocumentRecord]:
        return [DocumentRecord.model_validate(item) for item in self._read_list(self.documents_path)]

    def save_documents(self, documents: list[DocumentRecord]) -> None:
        self._write_list(self.documents_path, [item.model_dump(mode="json") for item in documents])

    def add_document(self, document: DocumentRecord) -> DocumentRecord:
        documents = self.load_documents()
        documents.append(document)
        self.save_documents(documents)
        return document

    @staticmethod
    def _read_list(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_list(path: Path, data: list[dict]) -> None:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
