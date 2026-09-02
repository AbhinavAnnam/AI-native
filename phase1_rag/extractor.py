import os
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from phase1_rag.embedder import DocumentEmbedder

# Automatically load key from .env file into os.getenv
load_dotenv()

# --- Feature-Driven Schema Definition ---
class FunctionSpec(BaseModel):
    signature: str = Field(description="Python-style function signature with parameter types")
    purpose: str = Field(description="Brief explanation of what the function accomplishes")

class FeatureRequirement(BaseModel):
    feature_name: str = Field(description="Name of the functional domain or feature module")
    target_module: str = Field(description="Suggested snake_case target python filename, e.g., projects.py or work_items.py")
    business_rules: list[str] = Field(description="Key operational and validation rules governing this feature")
    functions: list[FunctionSpec] = Field(description="Required function signatures belonging to this module")
    exceptions: list[str] = Field(description="List of custom exception names explicitly related to this module")

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

    def synthesize_requirements(self, task_description: str, top_k: int = 20) -> str:
        logging.info(f"Retrieving top {top_k} relevant chunks from vector store...")
        chunks = self.embedder.retrieve_relevant_chunks(task_description, top_k=top_k)
        
        if not chunks:
            logging.warning("No context chunks were retrieved for the query.")
            return "No matching document context found."

        logging.info(f"Successfully retrieved {len(chunks)} chunks. Formatting context for Gemini...")

        # Format context with document source and exact page numbers
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

        logging.info("Sending request to Gemini 3.6 Flash for structured specification extraction...")
        
        # Uses google-genai SDK config for structured output matching ExtractedSpecs
        response = self.client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractedSpecs
            )
        )
        
        logging.info("Gemini specification synthesis completed successfully.")
        return response.text