"""
Neo4j relationship models for RAGShelf graph database.
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class Contains(BaseModel):
    """CONTAINS relationship: Document contains Concept."""
    
    frequency: int = Field(..., ge=1, description="Number of occurrences")
    positions: Optional[List[int]] = Field(
        default_factory=list,
        description="Character positions in document"
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score"
    )
    extraction_method: Optional[str] = Field(None, description="Method used for extraction")

    class Config:
        schema_extra = {
            "example": {
                "frequency": 5,
                "positions": [100, 250, 300, 450, 600],
                "relevance_score": 0.85,
                "extraction_method": "NER+TF-IDF"
            }
        }


class RelatedTo(BaseModel):
    """RELATED_TO relationship: Concept is related to another Concept."""
    
    relationship_type: str = Field(
        ...,
        pattern="^(synonym|antonym|hypernym|hyponym|meronym|holonym|semantic)$",
        description="Type of relationship"
    )
    strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relationship strength"
    )
    co_occurrence_count: Optional[int] = Field(
        0, ge=0, description="Number of co-occurrences"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in relationship"
    )

    class Config:
        schema_extra = {
            "example": {
                "relationship_type": "synonym",
                "strength": 0.9,
                "co_occurrence_count": 15,
                "confidence": 0.85
            }
        }


class Searched(BaseModel):
    """SEARCHED relationship: User performed search Query."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Search timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Optional[str] = Field(None, description="Search context")

    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_123",
                "context": "academic research"
            }
        }


class Retrieved(BaseModel):
    """RETRIEVED relationship: Query retrieved Document."""
    
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score"
    )
    rank: int = Field(..., ge=1, description="Result ranking")
    click_through: Optional[bool] = Field(
        False, description="Whether user clicked result"
    )
    dwell_time: Optional[float] = Field(
        None, description="Time spent on result in seconds"
    )

    class Config:
        schema_extra = {
            "example": {
                "relevance_score": 0.92,
                "rank": 1,
                "click_through": True,
                "dwell_time": 45.5
            }
        }