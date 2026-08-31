import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from phase1_rag.embedder import DocumentEmbedder

# Automatically load key from .env file into os.getenv
load_dotenv()

# --- Feature-Driven Schema Definition ---
class FunctionSpec(BaseModel):
    signature: str
    purpose: str

class FeatureRequirement(BaseModel):
    feature_name: str
    target_module: str = Field(description="e.g., discount_service.py or auth.py")
    business_rules: list[str]
    functions: list[FunctionSpec]
    exceptions: list[str]

class ExtractedSpecs(BaseModel):
    project_title: str
    features: list[FeatureRequirement]


class RequirementExtractor:
    def __init__(self, api_key: str = None):
        self.embedder = DocumentEmbedder()
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not found. Ensure it is set in your .env file.")
        self.client = genai.Client(api_key=key)

    def synthesize_requirements(self, task_description: str) -> str:
        chunks = self.embedder.retrieve_relevant_chunks(task_description, top_k=5)
        if not chunks:
            return "No matching document context found."

        # Format context with document source and exact page numbers
        raw_context = "\n\n".join([
            f"[Source: {c.get('source', 'Unknown')} | Page: {c.get('page', 1)}]\n{c['text']}" 
            for c in chunks
        ])

        prompt = f"""
        You are a Technical Specification Synthesizer. Analyze the retrieved documentation across all pages and extract exact context grouped by distinct FEATURE modules.

        RETRIEVED DOCUMENTATION:
        {raw_context}

        TARGET FEATURE TASK:
        {task_description}

        Group all function signatures, business rules, exceptions, and input constraints under their target feature module.
        """

        # Uses google-genai SDK config for structured output matching ExtractedSpecs
        response = self.client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractedSpecs
            )
        )
        return response.text