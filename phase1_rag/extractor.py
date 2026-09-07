import logging
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from pydantic import BaseModel, Field

from phase1_rag.embedder import DocumentEmbedder

load_dotenv()


class FunctionSpec(BaseModel):
    signature: str = Field(
        description="Python-style function signature with parameter types"
    )
    purpose: str = Field(
        description="Brief explanation of what the function accomplishes"
    )


class FeatureRequirement(BaseModel):
    feature_name: str = Field(
        description="Name of the functional domain or feature module"
    )
    target_module: str = Field(
        description="Suggested snake_case target python filename, e.g., projects.py or work_items.py"
    )
    business_rules: list[str] = Field(
        description="Key operational and validation rules governing this feature"
    )
    functions: list[FunctionSpec] = Field(
        description="Required function signatures belonging to this module"
    )
    exceptions: list[str] = Field(
        description="List of custom exception names explicitly related to this module"
    )


class ExtractedSpecs(BaseModel):
    project_title: str
    features: list[FeatureRequirement]


class RequirementExtractor:

    def __init__(self, api_key: str = None):
        self.embedder = DocumentEmbedder()
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not found. Ensure it is set in your .env file."
            )
        self.client = genai.Client(api_key=key)

    def _call_gemini_with_retry(self, prompt: str, max_retries: int = 2) -> str:
        models_to_try = [
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-flash-latest",
        ]

        for model in models_to_try:
            for attempt in range(max_retries):
                try:
                    logging.info(
                        f"Dispatching request to {model} (Attempt {attempt + 1})..."
                    )
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=ExtractedSpecs,
                        ),
                    )
                    return response.text
                except ServerError as e:
                    logging.warning(
                        f"503 load spike on {model} (Attempt {attempt + 1}): {e}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                except Exception as e:
                    logging.warning(f"Skipping {model} due to error: {e}")
                    break

        raise RuntimeError("All active Gemini endpoints are currently busy.")

    def synthesize_requirements(
        self, task_description: str, top_k: int = 20
    ) -> str:
        """SLOW PATH: Retrieves context chunks from ChromaDB vector store before synthesis."""
        logging.info(f"Retrieving top {top_k} relevant chunks from vector store...")
        chunks = self.embedder.retrieve_relevant_chunks(
            task_description, top_k=top_k
        )

        if not chunks:
            logging.warning("No context chunks were retrieved for the query.")
            return "No matching document context found."

        logging.info(
            f"Successfully retrieved {len(chunks)} chunks. Formatting context for Gemini..."
        )

        raw_context = "\n\n".join([
            f"[Source: {c.get('source', 'Unknown')} | Page: {c.get('page', 1)}]\n{c['text']}"
            for c in chunks
        ])

        prompt = f"""
        You are a Principal Software Architect. Analyze the provided operational documentation across all retrieved pages.
        Extract complete operational specifications and map them cleanly into distinct feature modules based strictly on the text.

        RETRIEVED DOCUMENTATION:
        {raw_context}

        TARGET SCOPE TASK:
        {task_description}

        Extract all distinct feature domains (e.g., Accounts & Identity, Organizations, Projects, Work Items, Comments, Attachments, Notifications, Search, Dashboard, Activity History).
        For each domain, define business rules, function signatures, and explicit custom exceptions. Do not fabricate features outside the document.
        """

        return self._call_gemini_with_retry(prompt)