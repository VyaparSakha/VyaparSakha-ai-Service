"""
Visual Spec Pydantic v2 schemas for VyaparSathi AI assistant.

The LLM NEVER invents numbers — it only selects a tool.
The backend runs the real DB query and builds a validated VisualSpec here.
The frontend receives this exact JSON and renders it via VisualRenderer.jsx.

Keep this file in sync with client/src/lib/visuals/schema.js (Zod).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────
# Shared primitives
# ─────────────────────────────────────────────────────────

ValueFormat = Literal["number", "currency_inr", "percent"]
Trend = Literal["up", "down", "flat"]


class SeriesItem(BaseModel):
    key: str
    label: str
    color: Optional[str] = None


# ─────────────────────────────────────────────────────────
# Chart specs (bar / line / area)
# ─────────────────────────────────────────────────────────

class BarChartSpec(BaseModel):
    type: Literal["bar_chart"] = "bar_chart"
    title: str
    description: Optional[str] = None
    xKey: str
    series: list[SeriesItem] = Field(min_length=1)
    data: list[dict[str, Any]]
    valueFormat: ValueFormat = "number"


class LineChartSpec(BaseModel):
    type: Literal["line_chart"] = "line_chart"
    title: str
    description: Optional[str] = None
    xKey: str
    series: list[SeriesItem] = Field(min_length=1)
    data: list[dict[str, Any]]
    valueFormat: ValueFormat = "number"


class AreaChartSpec(BaseModel):
    type: Literal["area_chart"] = "area_chart"
    title: str
    description: Optional[str] = None
    xKey: str
    series: list[SeriesItem] = Field(min_length=1)
    data: list[dict[str, Any]]
    valueFormat: ValueFormat = "number"


# ─────────────────────────────────────────────────────────
# Pie chart
# ─────────────────────────────────────────────────────────

class PieDataItem(BaseModel):
    label: str
    value: float
    color: Optional[str] = None


class PieChartSpec(BaseModel):
    type: Literal["pie_chart"] = "pie_chart"
    title: str
    description: Optional[str] = None
    data: list[PieDataItem]
    valueFormat: ValueFormat = "number"


# ─────────────────────────────────────────────────────────
# Table
# ─────────────────────────────────────────────────────────

ColumnDataType = Literal["string", "number", "currency_inr", "date", "badge"]


class TableColumn(BaseModel):
    key: str
    label: str
    dataType: ColumnDataType = "string"
    sortable: bool = True


class TableSpec(BaseModel):
    type: Literal["table"] = "table"
    title: str
    columns: list[TableColumn] = Field(min_length=1)
    rows: list[dict[str, Any]]
    pageSize: int = Field(default=10, ge=1)


# ─────────────────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────────────────

class KpiCard(BaseModel):
    label: str
    value: Union[str, float, int]
    change: Optional[str] = None
    trend: Trend = "flat"
    valueFormat: ValueFormat = "number"


class KpiCardsSpec(BaseModel):
    type: Literal["kpi_cards"] = "kpi_cards"
    cards: list[KpiCard] = Field(min_length=1)


# ─────────────────────────────────────────────────────────
# Discriminated union — the main VisualSpec
# ─────────────────────────────────────────────────────────

VisualSpec = Annotated[
    Union[
        BarChartSpec,
        LineChartSpec,
        AreaChartSpec,
        PieChartSpec,
        TableSpec,
        KpiCardsSpec,
    ],
    Field(discriminator="type"),
]


# ─────────────────────────────────────────────────────────
# Full assistant message (text + visuals list)
# ─────────────────────────────────────────────────────────

class AssistantMessage(BaseModel):
    text: str
    visuals: list[VisualSpec] = []

