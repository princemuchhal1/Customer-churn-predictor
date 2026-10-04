from database import get_connection


def init_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    churn_probability DOUBLE PRECISION NOT NULL,
                    churn_prediction VARCHAR(10) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

        conn.commit()


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")