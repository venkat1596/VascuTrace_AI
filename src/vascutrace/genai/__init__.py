"""VascuTrace GenAI layer: gpt-5-mini LLM client + RAG evidence pipeline.

Research prototype. The language model orchestrates, retrieves grounded evidence,
and explains deterministic measurements; it never invents numbers (spec sec. 8-9,
"Separation of responsibility"). All numeric values in any report come from
deterministic imaging/quant code; the LLM only arranges retrieved text + verified
measurements into a schema-constrained narrative.
"""

from src.vascutrace.genai.llm import (
    LLMConfig,
    LLMUnavailableError,
    VascuTraceLLM,
    load_openai_key,
)

__all__ = [
    "LLMConfig",
    "LLMUnavailableError",
    "VascuTraceLLM",
    "load_openai_key",
]
