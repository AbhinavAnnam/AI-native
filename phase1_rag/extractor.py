import os
from dotenv import load_dotenv
from google import genai
from phase1_rag.embedder import DocumentEmbedder

# Automatically load key from .env file into os.getenv
load_dotenv()

class RequirementExtractor:
    def __init__(self, api_key: str = None):
        self.embedder = DocumentEmbedder()
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not found. Ensure it is set in your .env file.")
        self.client = genai.Client(api_key=key)

    def synthesize_requirements(self, task_description: str) -> str:
        chunks = self.embedder.retrieve_relevant_chunks(task_description, top_k=4)
        if not chunks:
            return "No matching document context found."

        raw_context = "\n\n".join([f"[Source: {c['source']}]\n{c['text']}" for c in chunks])

        prompt = f"""
        You are a Technical Specification Synthesizer. Analyze the retrieved documentation and extract exact programming context.

        RETRIEVED DOCUMENTATION:
        {raw_context}

        TARGET FEATURE TASK:
        {task_description}

        Format output with clear sections:
        1. Function/Class signatures to construct
        2. Expected input arguments, types, and defaults
        3. Exceptions or error codes to raise
        4. Key business logic constraints
        """

        response = self.client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        return response.text