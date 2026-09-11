"""Create all database tables. Run once before starting the API/workers
against a fresh Postgres instance."""

from app.models.db import init_db

if __name__ == "__main__":
    init_db()
    print("Database tables created.")
