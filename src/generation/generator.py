# =============================================================================
# src/generation/generator.py
#
# PURPOSE:
#   Takes the user's query and the top-N reranked chunks, builds a grounded
#   prompt, calls the OpenAI chat API, and returns a structured response that
#   includes the answer text and source citations.
#
#   Key design decisions:
#   - temperature=0 for deterministic, factual answers
#   - Context is injected as numbered chunks so the model can cite sources
#   - If no relevant context exists the model is instructed to say so
#     (prevents hallucination / "I'll make something up" behavior)
#
# INPUT:
#   - query:   str          the user's question
#   - chunks:  list[dict]   top-N reranked chunks, each with fields:
#                           text, title, source_url, section_title,
#                           rerank_score, provider
#
# OUTPUT:
#   A dict with the following shape:
#     {
#       "answer":  str,
#       "sources": [
#         {
#           "chunk_id":      str,
#           "title":         str,
#           "section_title": str,
#           "source_url":    str,
#           "rerank_score":  float
#         }, ...
#       ]
#     }
# =============================================================================

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def _build_context_block(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] Source: {chunk['title']} — {chunk['section_title']}")
        lines.append(chunk["text"])
        lines.append("")
    return "\n".join(lines)


def _build_user_message(query: str, chunks: list[dict]) -> str:
    context = _build_context_block(chunks)
    return (
        f"Context:\n{context}\n"
        f"Question: {query}"
    )


class Generator:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 512,
        temperature: float = 0.0,
        system_prompt: str = "",
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, query: str, chunks: list[dict]) -> dict:
        user_message = _build_user_message(query, chunks)

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )

        answer = response.choices[0].message.content.strip()

        sources = [
            {
                "chunk_id":      chunk.get("chunk_id", ""),
                "title":         chunk.get("title", ""),
                "section_title": chunk.get("section_title", ""),
                "source_url":    chunk.get("source_url", ""),
                "rerank_score":  chunk.get("rerank_score", 0.0),
            }
            for chunk in chunks
        ]

        return {"answer": answer, "sources": sources}
