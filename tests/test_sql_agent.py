import pytest

from src.agents.sql_agent import UnsafeSQLError, enforce_row_limit, validate_sql


def test_validate_sql_accepts_plain_select():
    tree = validate_sql("SELECT id, name FROM users WHERE active = true")
    assert tree is not None


def test_validate_sql_rejects_delete():
    with pytest.raises(UnsafeSQLError):
        validate_sql("DELETE FROM users WHERE id = 1")


def test_validate_sql_rejects_drop():
    with pytest.raises(UnsafeSQLError):
        validate_sql("DROP TABLE users")


def test_validate_sql_rejects_multiple_statements():
    with pytest.raises(UnsafeSQLError):
        validate_sql("SELECT 1; DROP TABLE users;")


def test_validate_sql_rejects_insert_disguised_in_cte():
    with pytest.raises(UnsafeSQLError):
        validate_sql("WITH x AS (INSERT INTO users(id) VALUES (1) RETURNING id) SELECT * FROM x")


def test_enforce_row_limit_adds_limit_when_missing():
    tree = validate_sql("SELECT * FROM users")
    sql = enforce_row_limit(tree, max_rows=100)
    assert "LIMIT 100" in sql.upper()


def test_enforce_row_limit_caps_existing_larger_limit():
    tree = validate_sql("SELECT * FROM users LIMIT 100000")
    sql = enforce_row_limit(tree, max_rows=500)
    assert "LIMIT 500" in sql.upper()


def test_enforce_row_limit_keeps_smaller_existing_limit():
    tree = validate_sql("SELECT * FROM users LIMIT 10")
    sql = enforce_row_limit(tree, max_rows=500)
    assert "LIMIT 10" in sql.upper()
