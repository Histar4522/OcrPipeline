"""Configuration and prompts for the OCR pipeline."""

from dataclasses import dataclass

DEFAULT_SYSTEM_PROMPT = (
    "You are a verbatim archival OCR transcription engine. "
    "Transcribe every word of printed and handwritten text from the image into clean Markdown.\n"
    "Rules:\n"
    "1. Preserve original wording, spelling, numbers, and punctuation verbatim. "
    "Do NOT fix spelling errors, paraphrase, or summarize text.\n"
    "2. Use Markdown headings (#, ##), bullet/numbered lists, and Markdown tables where appropriate.\n"
    "3. For non-text graphics, diagrams, charts, sketches, or stamps, insert a concise inline "
    "placeholder in the format `[Figure: brief description of visual content]`.\n"
    "4. Output ONLY the transcribed Markdown content—no preamble, commentary, or closing remarks."
)

DEFAULT_USER_PROMPT = "Transcribe all text and describe any figures on this page in Markdown."


@dataclass(slots=True)
class OcrConfig:
    """Runtime configuration for the OCR pipeline."""

    model: str = "glm-ocr"
    ollama_url: str = "http://localhost:11434"
    dpi: int = 220
    max_image_dim: int = 1536
    num_ctx: int = 4096
    timeout_seconds: float = 180.0
    max_retries: int = 2
    force: bool = False
    include_page_headers: bool = True
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt: str = DEFAULT_USER_PROMPT
