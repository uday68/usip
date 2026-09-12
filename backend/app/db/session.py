import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from configs.settings import config

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./usip_sonar.db")

# SQLite connection args for threading
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL, 
    connect_args=connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

