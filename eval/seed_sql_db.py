"""Seeds a minimal `sales` schema into DATABASE_URL so SQL
testset generation (eval/generate_testset.py) and SQL eval
(eval/run_ragas_eval.py) have a real, deterministic schema to work
against in CI. Idempotent -- safe to run against an already-seeded
database. Mirrors the schema in data/init.sql.
"""

from __future__ import annotations

import argparse
import logging
import os

from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_sql_db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY,
    region TEXT NOT NULL,
    product TEXT NOT NULL,
    amount NUMERIC NOT NULL,
    sale_date DATE NOT NULL
);
"""

_SALES = [
    (1, "North", "Widget", 100.00, "2026-01-05"),
    (2, "South", "Widget", 25.00, "2026-01-12"),
    (3, "East", "Gadget", 150.00, "2026-02-01"),
    (4, "West", "Gizmo", 50.00, "2026-02-14"),
    (5, "East", "Gadget", 30.00, "2026-03-03"),
]


def seed(database_url: str) -> None:
    engine = create_engine(database_url)
    with engine.begin() as conn:
        for statement in _SCHEMA.strip().split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))

        existing = conn.execute(text("SELECT COUNT(*) FROM sales")).scalar()
        if existing:
            logger.info("sales table already seeded (%d rows); skipping inserts", existing)
            return

        for sale_id, region, product, amount, sale_date in _SALES:
            conn.execute(
                text(
                    "INSERT INTO sales (id, region, product, amount, sale_date) "
                    "VALUES (:id, :region, :product, :amount, :sale_date)"
                ),
                {"id": sale_id, "region": region, "product": product, "amount": amount, "sale_date": sale_date},
            )

    logger.info("seeded %d sales", len(_SALES))


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the eval SQL database")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    args = parser.parse_args()

    if not args.database_url:
        raise SystemExit("DATABASE_URL is not set and --database-url was not given")

    seed(args.database_url)


if __name__ == "__main__":
    main()
