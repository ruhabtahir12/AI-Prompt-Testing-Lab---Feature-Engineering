from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./prompt_lab.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Test(Base):
    __tablename__ = "tests"
    id = Column(Integer, primary_key=True)
    task_description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    prompts = relationship("Prompt", back_populates="test", cascade="all, delete-orphan")

class Prompt(Base):
    __tablename__ = "prompts"
    id = Column(Integer, primary_key=True)
    test_id = Column(Integer, ForeignKey("tests.id"))
    prompt_text = Column(String)
    test = relationship("Test", back_populates="prompts")
    response = relationship("Response", back_populates="prompt", uselist=False, cascade="all, delete-orphan")

class Response(Base):
    __tablename__ = "responses"
    id = Column(Integer, primary_key=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id"))
    response_text = Column(String)
    accuracy_score = Column(Integer)
    relevance_score = Column(Integer)
    completeness_score = Column(Integer)
    clarity_score = Column(Integer)
    creativity_score = Column(Integer)
    conciseness_score = Column(Integer)
    instruction_following_score = Column(Integer)
    overall_score = Column(Float)
    prompt = relationship("Prompt", back_populates="response")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()