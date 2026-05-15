from sqlalchemy import create_engine, Column, Integer, String, Date, Float
from sqlalchemy.orm import declarative_base, sessionmaker

engine = create_engine("sqlite:///diet.db")

Base = declarative_base()

class FoodHistory(Base):
    __tablename__ = "food_history"

    id = Column(Integer, primary_key=True)
    date = Column(Date)
    food = Column(String)
    calories = Column(Float)
    protein = Column(Float)
    carbs = Column(Float)
    fat = Column(Float)

Base.metadata.create_all(engine)

SessionLocal = sessionmaker(bind=engine)