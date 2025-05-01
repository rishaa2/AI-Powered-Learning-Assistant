from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QAPair(BaseModel):
    question: str
    answer: str

class DocumentInfo(BaseModel):
    filename: str
    summary: str
    qa_pairs: List[QAPair]

class ProcessingStatus(BaseModel):
    task_id: str
    status: str  # "processing", "completed", "failed"
    message: str
    documents: Optional[List[DocumentInfo]] = None
    
class SummaryResponse(BaseModel):
    summaries: List[Dict[str, Any]]