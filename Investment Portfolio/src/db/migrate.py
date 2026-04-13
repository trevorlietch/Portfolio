from src.db.connection import connect
from src.db.schema import SQL_SCHEMA

def migrate():
    connection = connect()
    try:
        connection.executescript(SQL_SCHEMA)
        connection.commit()
    finally:
        connection.close()

if __name__ == "__main__":
    migrate()
    print("Database tables created")