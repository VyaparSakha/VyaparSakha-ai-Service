"""
app/agent/subgraphs/document_generation/state.py
==================================================
State definition for the Document Generation Subgraph.
"""

from typing import Any, Literal
from typing_extensions import TypedDict


class DocumentGenerationState(TypedDict, total=False):
    """
    Isolated state for the Document Generation Subgraph.
    Contains only what is needed for document specification, generation,
    validation, repair/retry, and R2 storage upload.
    """
    # Scope & Request context
    user_id: str
    store_id: str
    request: dict[str, Any]
    document_type: Literal["excel", "word"]
    document_format: Literal["xlsx", "docx"]
    source_data: Any

    # Specification & Generation outputs
    document_spec: dict[str, Any]
    generated_code: str
    preview_data: dict[str, Any]
    generated_file_path: str
    generated_filename: str
    mime_type: str

    # Validation & Bounded Repair
    validation_result: dict[str, Any]
    validation_errors: list[str]
    repair_instructions: dict[str, Any]
    retry_count: int
    max_retries: int

    # R2 Storage & Result Metadata
    file_id: str
    storage_key: str
    file_metadata: dict[str, Any]

    # Execution status
    status: Literal["pending", "in_progress", "success", "failed"]
    error: str | None
