"""CodeS-style per-query schema filtering and value retrieval."""

from __future__ import annotations

import csv
import logging
import re
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from .codes_db_utils import get_db_schema, get_db_schema_sequence, get_matched_content_sequence

logger = logging.getLogger(__name__)


def _norm(text: str) -> str:
    return " ".join(str(text).lower().replace("_", " ").split())


def _load_bird_comments(description_dir: Path) -> dict[str, dict[str, Any]]:
    comments: dict[str, dict[str, Any]] = {}
    if not description_dir.exists():
        return comments

    for csv_path in description_dir.glob("*.csv"):
        table_name = csv_path.stem.lower()
        column_comments: dict[str, str] = {}
        try:
            with csv_path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    original = (row.get("original_column_name") or "").strip().lower()
                    column_name = (row.get("column_name") or "").strip()
                    column_desc = (row.get("column_description") or "").strip()
                    value_desc = (row.get("value_description") or "").strip()
                    pieces = [p for p in (column_name, column_desc, value_desc) if p]
                    if original and pieces:
                        column_comments[original] = " ; ".join(dict.fromkeys(pieces))
        except Exception:
            logger.debug("Failed reading BIRD description %s", csv_path, exc_info=True)
            continue

        comments[table_name] = {
            "table_comment": "",
            "column_comments": column_comments,
        }

    return comments


@lru_cache(maxsize=16)
def _schema_for_db(db_path: str) -> dict[str, Any]:
    path = Path(db_path)
    comments = _load_bird_comments(path.parent / "database_description")
    db_id = path.stem.lower()
    return get_db_schema(db_path, {db_id: comments}, db_id, column_content_limit=2)


class CodeSContextBuilder:
    """Build the dynamic prompt pieces used by the CodeS BIRD pipeline."""

    def __init__(
        self,
        root_dir: str,
        db_path: str,
        sic_path: str | None = None,
        top_k_tables: int = 5,
        top_k_columns: int = 5,
        value_retrieval: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.db_path = str(db_path)
        self.top_k_tables = top_k_tables
        self.top_k_columns = top_k_columns
        self.value_retrieval = value_retrieval
        self.schema = _schema_for_db(self.db_path)
        self.sic = self._load_sic(sic_path)
        self._value_cache: dict[tuple[str, str, tuple[str, ...]], list[str]] = {}

    def _load_sic(self, sic_path: str | None):
        if not sic_path:
            return None
        path = Path(sic_path)
        if not path.exists():
            logger.warning("CodeS SIC path does not exist: %s", path)
            return None

        print(f"Loading CodeS schema item classifier from {path}", flush=True)
        codes_root = self.root_dir / "utils" / "CodeS"
        if str(codes_root) not in sys.path:
            sys.path.insert(0, str(codes_root))
        try:
            from schema_item_filter import SchemaItemClassifierInference

            sic = SchemaItemClassifierInference(str(path))
            print("Loaded CodeS schema item classifier", flush=True)
            return sic
        except Exception:
            logger.warning("Failed to load CodeS schema item classifier: %s", path, exc_info=True)
            return None

    def build(self, question: str, evidence: str | None = None) -> tuple[str, str, str]:
        query_text = question
        if evidence:
            query_text = f"{question}\nevidence: {evidence}"

        matched_contents = self._retrieve_matched_contents(query_text) if self.value_retrieval else {}
        sample = {
            "text": query_text,
            "schema": self.schema,
            "matched_contents": matched_contents,
        }

        if self.sic is not None:
            try:
                codes_root = self.root_dir / "utils" / "CodeS"
                if str(codes_root) not in sys.path:
                    sys.path.insert(0, str(codes_root))
                from schema_item_filter import filter_schema

                sample = filter_schema(
                    [sample],
                    "eval",
                    self.sic,
                    self.top_k_tables,
                    self.top_k_columns,
                )[0]
            except Exception:
                logger.warning("CodeS schema filtering failed; using full schema", exc_info=True)

        return (
            get_db_schema_sequence(sample["schema"]),
            get_matched_content_sequence(sample["matched_contents"]),
            query_text,
        )

    def _retrieve_matched_contents(self, query_text: str) -> dict[str, list[str]]:
        try:
            from rapidfuzz import fuzz
        except Exception:
            fuzz = None

        query_norm = _norm(query_text)
        search_terms = self._search_terms(query_text)
        matched: dict[str, list[str]] = {}

        if not search_terms:
            return matched

        connection = sqlite3.connect(self.db_path)
        connection.text_factory = lambda b: b.decode(errors="ignore")
        try:
            cursor = connection.cursor()
            for table in self.schema["schema_items"]:
                table_name = table["table_name"]
                pairs = zip(table["column_names"], table["column_types"], strict=False)
                for column_name, column_type in pairs:
                    if not self._should_scan_column(column_type, query_norm):
                        continue
                    hits = []
                    for value in self._candidate_values(
                        cursor,
                        table_name,
                        column_name,
                        search_terms,
                    ):
                        value_norm = _norm(value)
                        if not value_norm:
                            continue
                        score = 100 if value_norm in query_norm else 0
                        if score == 0 and fuzz is not None:
                            score = fuzz.partial_ratio(value_norm, query_norm)
                        if score >= 88:
                            hits.append((score, value))
                    if hits:
                        hits.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
                        matched[f"{table_name}.{column_name}"] = [
                            value for _, value in hits[:3]
                        ]
        finally:
            connection.close()

        return matched

    @staticmethod
    def _should_scan_column(column_type: str, query_norm: str) -> bool:
        column_type = (column_type or "").lower()
        if any(t in column_type for t in ("char", "text", "date", "time")):
            return True
        return any(ch.isdigit() for ch in query_norm)

    @staticmethod
    def _quote_ident(name: str) -> str:
        return "`" + name.replace("`", "``") + "`"

    @staticmethod
    def _search_terms(query_text: str) -> tuple[str, ...]:
        raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}|\d{2,}(?:[-/]\d{1,2}){0,2}", query_text)
        terms = []
        stop = {
            "select",
            "from",
            "where",
            "what",
            "which",
            "list",
            "count",
            "many",
            "with",
            "that",
            "have",
            "does",
            "were",
            "after",
            "before",
            "evidence",
        }
        for term in raw_terms:
            cleaned = term.strip("._-").lower()
            if len(cleaned) >= 3 and cleaned not in stop:
                terms.append(cleaned)
        return tuple(list(dict.fromkeys(terms))[:12])

    @staticmethod
    def _escape_like(term: str) -> str:
        return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def _candidate_values(
        self,
        cursor,
        table_name: str,
        column_name: str,
        search_terms: tuple[str, ...],
    ) -> list[str]:
        cache_key = (table_name, column_name, search_terms)
        if cache_key in self._value_cache:
            return self._value_cache[cache_key]

        table = self._quote_ident(table_name)
        column = self._quote_ident(column_name)
        values: list[str] = []
        seen = set()
        try:
            for term in search_terms:
                pattern = f"%{self._escape_like(term)}%"
                sql = (
                    f"SELECT DISTINCT {column} FROM {table} "
                    f"WHERE CAST({column} AS TEXT) LIKE ? ESCAPE '\\' LIMIT 25"
                )
                cursor.execute(sql, (pattern,))
                for row in cursor.fetchall():
                    value = str(row[0]).strip()
                    if 0 < len(value) <= 50 and value not in seen:
                        seen.add(value)
                        values.append(value)
        except Exception:
            logger.debug("Value retrieval failed for %s.%s", table_name, column_name, exc_info=True)
            values = []

        self._value_cache[cache_key] = values
        return values
