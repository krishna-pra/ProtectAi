# backend/db.py
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import IntegrityError

# -----------------------------
# Database setup (SQLite local file)
# -----------------------------
DATABASE_URL = "sqlite:///protectai.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# -----------------------------
# Database Model
# -----------------------------
class Fingerprint(Base):
    __tablename__ = "fingerprints"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    hash_value = Column(String, index=True)  # index added for faster lookup

# -----------------------------
# DB Utilities
# -----------------------------
def init_db(drop: bool = False):
    """Initialize database tables. Use drop=True to reset DB."""
    if drop:
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

def save_fingerprint(filename: str, hash_value: str):
    """Insert or update fingerprint entry."""
    session = SessionLocal()
    try:
        fp = Fingerprint(filename=filename, hash_value=hash_value)
        session.add(fp)
        session.commit()
    except IntegrityError:
        session.rollback()
        # Update existing record instead of failing
        existing_fp = session.query(Fingerprint).filter_by(filename=filename).first()
        if existing_fp:
            existing_fp.hash_value = hash_value
            session.commit()
    finally:
        session.close()

def get_fingerprints():
    """Fetch all fingerprints."""
    session = SessionLocal()
    try:
        return session.query(Fingerprint).all()
    finally:
        session.close()

def get_fingerprint_by_filename(filename: str):
    """Fetch single fingerprint by filename."""
    session = SessionLocal()
    try:
        return session.query(Fingerprint).filter_by(filename=filename).first()
    finally:
        session.close()

def delete_fingerprint(filename: str) -> bool:
    """Delete a fingerprint by filename."""
    session = SessionLocal()
    try:
        fp = session.query(Fingerprint).filter_by(filename=filename).first()
        if fp:
            session.delete(fp)
            session.commit()
            return True
        return False
    finally:
        session.close()

def count_fingerprints() -> int:
    """Return total number of fingerprints stored."""
    session = SessionLocal()
    try:
        return session.query(Fingerprint).count()
    finally:
        session.close()
