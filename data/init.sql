-- Sample schema + seed data for the SQL agent (src/agents/sql_agent.py)
-- to query against. Read-only SELECTs only reach this DB; see
-- validate_sql() for the AST-level enforcement.

CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    region VARCHAR(50) NOT NULL,
    product VARCHAR(50) NOT NULL,
    amount FLOAT NOT NULL,
    sale_date DATE NOT NULL
);

INSERT INTO sales (region, product, amount, sale_date) VALUES
    ('North', 'Widget', 1200.50, '2026-01-05'),
    ('North', 'Gadget', 850.00, '2026-02-14'),
    ('South', 'Widget', 640.75, '2026-01-20'),
    ('South', 'Gizmo', 990.25, '2026-03-02'),
    ('East', 'Gadget', 430.00, '2026-02-28'),
    ('West', 'Widget', 1575.00, '2026-03-15');
