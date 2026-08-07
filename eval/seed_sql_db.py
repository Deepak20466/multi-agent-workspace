"""Seeds a minimal `users`/`orders` schema into DATABASE_URL so SQL
testset generation (eval/generate_testset.py) and SQL eval
(eval/run_ragas_eval.py) have a real, deterministic schema to work
against in CI. Idempotent -- safe to run against an already-seeded
database.
"""

from __future__ import annotations

import argparse
import logging
import os

from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_sql_db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    amount NUMERIC NOT NULL
);
"""

_USERS = [
    (1, "Alice", "East"),
    (2, "Bob", "West"),
    (3, "Carol", "East"),
]

_ORDERS = [
    (1, 1, "completed", 100.00),
    (2, 1, "pending", 25.00),
    (3, 2, "completed", 150.00),
    (4, 3, "completed", 50.00),
    (5, 3, "cancelled", 30.00),
]


def seed(database_url: str) -> None:
    engine = create_engine(database_url)
    with engine.begin() as conn:
        for statement in _SCHEMA.strip().split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))

        existing = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
        if existing:
            logger.info("users table already seeded (%d rows); skipping inserts", existing)
            return

        for user_id, name, region in _USERS:
            conn.execute(
                text("INSERT INTO users (id, name, region) VALUES (:id, :name, :region)"),
                {"id": user_id, "name": name, "region": region},
            )
        for order_id, user_id, status, amount in _ORDERS:
            conn.execute(
                text(
                    "INSERT INTO orders (id, user_id, status, amount) "
                    "VALUES (:id, :user_id, :status, :amount)"
                ),
                {"id": order_id, "user_id": user_id, "status": status, "amount": amount},
            )

    logger.info("seeded %d users, %d orders", len(_USERS), len(_ORDERS))


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the eval SQL database")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    args = parser.parse_args()

    if not args.database_url:
        raise SystemExit("DATABASE_URL is not set and --database-url was not given")

    seed(args.database_url)


if __name__ == "__main__":
    main()
