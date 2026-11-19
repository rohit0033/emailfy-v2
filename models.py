# models.py — the CONTRACT between the LLM and our code.
#
# pydantic = Python's version of zod. You describe the shape of data once,
# and pydantic checks real data against it and throws a ValidationError if it doesn't match.

# Literal["a", "b"] = only these exact string values are allowed. Like TS: "a" | "b"
from typing import Literal

# BaseModel: every schema class inherits from it (like z.object({...}) in zod).
# Field: lets you add extra rules to a field, e.g. a max length.
from pydantic import BaseModel, Field


# Contract designed by Rohit (W1 build, 2026-11-11).
# Every category maps to a downstream action in Kirana's routing code.
#
# "class X(BaseModel):" = "X is a pydantic model". The lines inside are its fields:
#     field_name: type
# zod equivalent:
#     const TicketClassification = z.object({
#       category: z.enum(["delivery", "refund", "change_request", "spam", "other"]),
#       needs_human: z.boolean(),
#       reason: z.string().max(200),
#     })
class TicketClassification(BaseModel):
    # Must be EXACTLY one of these 5 strings. "payments" -> ValidationError.
    # "other" is the escape hatch, so the model is never forced to pick a wrong category.
    category: Literal["delivery", "refund", "change_request", "spam", "other"]

    # True/False. (Python booleans are capitalized; in JSON they're true/false.)
    needs_human: bool

    # Why the model chose this category, so a human can debug misroutes.
    # Field(max_length=200): longer than 200 characters -> ValidationError.
    reason: str = Field(max_length=200)
