"""
models.py — Pydantic schemas for Agent answers, citations, and execution results.
"""

from typing import Literal, Union
from pydantic import BaseModel, Field


class PdfCitation(BaseModel):
    id: int
    type: Literal["pdf"] = "pdf"
    chunk_id: str = ""
    page: int = 0
    section: str = ""
    quote: str = ""


class FdaLabelCitation(BaseModel):
    id: int
    type: Literal["fda_label"] = "fda_label"
    drug: str = ""
    section: str = ""
    set_id: str = ""
    effective_time: str = ""
    url: str = ""
    quote: str = ""


class WebCitation(BaseModel):
    id: int
    type: Literal["web"] = "web"
    title: str = ""
    url: str = ""


Citation = Union[PdfCitation, FdaLabelCitation, WebCitation]


class AgentAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    source_summary: Literal["pdf_only", "fda_label_only", "pdf_and_fda", "with_web", "none"]


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict
    result_count: int | None = None
    error: str | None = None


class AgentResult(BaseModel):
    answer: AgentAnswer
    tool_calls: list[ToolCall] = Field(default_factory=list)
    verification_warning: str | None = None
