-- Sample schema + seed data for the SQL agent (src/agents/sql_agent.py)
-- to query against. Read-only SELECTs only reach this DB; see
-- validate_sql() for the AST-level enforcement.

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    signed_up_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    amount NUMERIC(10, 2) NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

INSERT INTO users (name, email) VALUES
    ('Alice Johnson', 'alice@example.com'),
    ('Bob Martinez', 'bob@example.com'),
    ('Carla Nguyen', 'carla@example.com')
ON CONFLICT (email) DO NOTHING;

INSERT INTO orders (user_id, amount, status) VALUES
    (1, 49.99, 'completed'),
    (1, 12.50, 'completed'),
    (2, 100.00, 'pending'),
    (3, 75.25, 'completed')
ON CONFLICT DO NOTHING;
