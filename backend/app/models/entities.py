import datetime
import uuid
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.app.db.session import Base

def generate_uuid() -> str:
    return str(uuid.uuid4())

class SurveyRecord(Base):
    __tablename__ = "surveys"

    id = Column(String, primary_key=True, default=generate_uuid)
    survey_name = Column(String, nullable=False, index=True)
    vessel_name = Column(String, default="AUV Explorer-1")
    mission_code = Column(String, default="SIH-SSS-SURVEY")
    status = Column(String, default="COMPLETED")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    notes = Column(Text, nullable=True)

    images = relationship("ImageRecord", back_populates="survey", cascade="all, delete-orphan")

class ImageRecord(Base):
    __tablename__ = "sonar_images"

    id = Column(String, primary_key=True, default=generate_uuid)
    survey_id = Column(String, ForeignKey("surveys.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    processed_path = Column(String, nullable=True)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    channels = Column(Integer, default=1)
    estimated_snr_db = Column(Float, nullable=True)
    dynamic_range = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    survey = relationship("SurveyRecord", back_populates="images")
    telemetry = relationship("TelemetryRecord", back_populates="image", uselist=False, cascade="all, delete-orphan")
    candidates = relationship("CandidateRecord", back_populates="image", cascade="all, delete-orphan")

class TelemetryRecord(Base):
    __tablename__ = "telemetry_records"

    id = Column(String, primary_key=True, default=generate_uuid)
    image_id = Column(String, ForeignKey("sonar_images.id"), nullable=False, unique=True)
    timestamp = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    x_local_m = Column(Float, nullable=True)
    y_local_m = Column(Float, nullable=True)
    depth_m = Column(Float, nullable=True)
    altitude_m = Column(Float, nullable=True)
    heading_deg = Column(Float, nullable=True)
    metadata_available = Column(Boolean, default=False)
    status_note = Column(String, default="Geolocation unavailable — no valid navigation metadata associated with candidate.")

    image = relationship("ImageRecord", back_populates="telemetry")

class CandidateRecord(Base):
    __tablename__ = "candidates"

    id = Column(String, primary_key=True, default=generate_uuid)
    image_id = Column(String, ForeignKey("sonar_images.id"), nullable=False, index=True)
    candidate_id = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False)  # 'detector' | 'anomaly'
    target_class = Column(String, nullable=False)
    bbox_x = Column(Float, nullable=False)
    bbox_y = Column(Float, nullable=False)
    bbox_w = Column(Float, nullable=False)
    bbox_h = Column(Float, nullable=False)
    model_confidence = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    shadow_evidence = Column(Float, default=1.0)
    shadow_evidence_level = Column(String, default="Moderate")
    context_score = Column(Float, default=0.5)
    final_priority = Column(String, nullable=False)  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    priority_score = Column(Float, nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    metadata_available = Column(Boolean, default=False)
    geolocation_note = Column(String, default="Geolocation unavailable — no valid navigation metadata associated with candidate.")
    explanation = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    image = relationship("ImageRecord", back_populates="candidates")
    reviews = relationship("ReviewRecord", back_populates="candidate", cascade="all, delete-orphan")
    feedbacks = relationship("FeedbackRecord", back_populates="candidate", cascade="all, delete-orphan")

class ReviewRecord(Base):
    __tablename__ = "analyst_reviews"

    id = Column(String, primary_key=True, default=generate_uuid)
    candidate_id = Column(String, ForeignKey("candidates.id"), nullable=False, index=True)
    analyst_id = Column(String, default="ANALYST-01")
    review_status = Column(String, nullable=False)  # 'confirmed', 'rejected', 'uncertain'
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    candidate = relationship("CandidateRecord", back_populates="reviews")

class FeedbackRecord(Base):
    __tablename__ = "feedbacks"

    id = Column(String, primary_key=True, default=generate_uuid)
    candidate_id = Column(String, ForeignKey("candidates.id"), nullable=False, index=True)
    rating = Column(Integer, default=5)
    corrected_class = Column(String, nullable=True)
    feedback_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    candidate = relationship("CandidateRecord", back_populates="feedbacks")

