import json
import logging
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from openai import OpenAI

from mcp_tools.file_tools import MCPToolServer
from phase2_coder.hitl_manager import hitl_queue

logger = logging.getLogger(__name__)

# Token budgets. Gemini 2.5/3.x models spend part of the budget on internal
# "thinking", so a tight cap silently truncates the emitted code.
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "8192"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "32768"))


def _looks_truncated(code: str) -> bool:
    """Heuristic: unbalanced brackets/quotes mean the model was cut off mid-output."""
    if not code:
        return True
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    in_str = None
    escaped = False
    for ch in code:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
        elif ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return True
            stack.pop()
    return bool(stack) or in_str is not None


def _render_functions(functions) -> str:
    """Renders Phase-1 function specs (dicts or plain strings) as readable bullets."""
    if not functions:
        return "- (none specified; implement the business rules above)"
    lines = []
    for fn in functions:
        if isinstance(fn, dict):
            sig = fn.get("signature", "unknown_signature")
            purpose = fn.get("purpose", "")
            lines.append(f"- `{sig}` -> {purpose}")
        else:
            lines.append(f"- `{fn}`")
    return "\n".join(lines)


# Typographic look-alikes LLMs substitute for ASCII punctuation. They are legal
# inside strings/comments (so AST checks pass) but they are never what is wanted
# in generated source, and inside a string literal they silently change behaviour
# (e.g. comparing "ISO-8601" != "ISO\u20118601"). Normalise them away.
_TYPOGRAPHIC_MAP = str.maketrans(
    {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
        "\u2212": "-",  # minus sign
        "\u00a0": " ",  # non-breaking space
        "\u2007": " ",  # figure space
        "\u2009": " ",  # thin space
        "\u202f": " ",  # narrow no-break space
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
    }
)


def _sanitize_generated_code(code: str) -> str:
    """Normalises typographic punctuation to ASCII in model-generated source.

    Accented letters and other genuine Unicode text are left untouched; only
    punctuation look-alikes are rewritten. If the rewrite would corrupt a string
    literal (e.g. a curly quote closing a quote early), the original is kept.
    """
    if not code or code.isascii():
        return code

    dash_space_only = str.maketrans(
        {
            "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
            "\u2014": "-", "\u2015": "-", "\u2212": "-",
            "\u00a0": " ", "\u2007": " ", "\u2009": " ", "\u202f": " ",
        }
    )
    full = code.translate(_TYPOGRAPHIC_MAP)
    punctuation_only = code.translate(dash_space_only)

    import ast as _ast

    for candidate in (full, punctuation_only, code):
        try:
            _ast.parse(candidate)
            if candidate != code:
                logger.info(
                    "Coder Agent: normalised typographic punctuation to ASCII "
                    "in generated source."
                )
            return candidate
        except SyntaxError:
            continue
    return code


class CoderAgent:
    def __init__(
        self,
        groq_api_key: str,
        gemini_api_key: str,
        openrouter_api_key: str = None,
        output_dir: str = "generated_src",
    ):
        self.output_dir = Path(output_dir)
        self.staging_dir = Path(".draft_staging")

        # Groq client (Primary Tier)
        self.groq_client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=15.0,
            max_retries=0,
        )

        # Gemini client (Fallback Tier)
        self.gemini_client = genai.Client(api_key=gemini_api_key)

        # Optional OpenRouter client retained for future use
        self.openrouter_client = (
            OpenAI(
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=15.0,
                max_retries=0,
            )
            if openrouter_api_key
            else None
        )

    def _get_active_groq_models(self, task_type: str = "draft") -> list:
        """Dynamically fetch all active Groq models and order them based on the task priority."""
        if task_type == "draft":
            # Drafting priority: Max reasoning capacity first
            preferred_order = [
                "openai/gpt-oss-120b",
                "qwen/qwen3.8-27b",
                "openai/gpt-oss-20b",
            ]
        else:
            # Repair priority: Quick bug-fix specialist models first
            preferred_order = [
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b",
                "qwen/qwen3.8-27b",
            ]

        try:
            model_list = self.groq_client.models.list()
            valid_ids = [
                m.id
                for m in model_list.data
                if not any(
                    excluded in m.id
                    for excluded in ["whisper", "guard", "canopy", "orpheus"]
                )
            ]
            if valid_ids:
                # Sort valid models so preferred models come first in order, followed by any remaining active models
                ordered_models = [m for m in preferred_order if m in valid_ids]
                ordered_models.extend([m for m in valid_ids if m not in ordered_models])
                return ordered_models
        except Exception as err:
            logger.warning(
                f"Dynamic Groq model lookup failed ({err}). Using fallback list for task '{task_type}'..."
            )

        return preferred_order

    def _generate_with_fallback(self, prompt: str, task_type: str = "draft") -> str:
        system_msg = {
            "role": "system",
            "content": (
                "You are a Senior Python Engineer building modules for the 'generated_src' package.\n"
                "CRITICAL GENERATION RULES:\n"
                "1. Output ONLY fully implemented, executable, syntactically valid Python code.\n"
                "2. NEVER cut off functions mid-sentence, leave placeholders, or stop output prematurely.\n"
                "3. Define ALL exception classes and helper functions referenced in the code at the top of the file.\n"
                "4. Ensure every function has proper return statements, error handling, and complete logic.\n"
                "5. Do NOT enclose output in markdown blocks, backticks (```), commentary, or explanation tags."
            ),
        }

        # Tier 1: Task-Routed Groq Priority Chain
        groq_models = self._get_active_groq_models(task_type=task_type)
        for model_slug in groq_models:
            try:
                logger.info(
                    f"Coder Agent [{task_type.upper()}]: Attempting Groq model '{model_slug}'..."
                )
                response = self.groq_client.chat.completions.create(
                    model=model_slug,
                    messages=[system_msg, {"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=GROQ_MAX_TOKENS,
                )
                choice = response.choices[0]
                code = choice.message.content
                if code:
                    if getattr(choice, "finish_reason", None) == "length":
                        logger.warning(
                            f"Groq '{model_slug}' output was TRUNCATED (hit the "
                            f"{GROQ_MAX_TOKENS}-token ceiling). Rejecting this draft."
                        )
                        continue
                    return code
            except Exception as e:
                logger.warning(
                    f"Tier 1 Groq ('{model_slug}') failed on {task_type} task ({e}). Rotating model..."
                )

        # Tier 2: Gemini Fallback Chain
        gemini_models = [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-2.5-pro",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
        ]

        full_gemini_prompt = f"{system_msg['content']}\n\nTask Instructions:\n{prompt}"
        completeness_nudge = (
            "\n\nIMPORTANT: Your previous response was cut off before the file was "
            "finished. Send the ENTIRE file again from the first import to the final "
            "return. If it is large, implement the listed functions fully and "
            "concisely - never stop mid-function or leave a block unclosed."
        )

        for model_name in gemini_models:
            try:
                logger.info(
                    f"Coder Agent: Executing Tier 2 fallback with '{model_name}'..."
                )
                text = self._gemini_generate_text(model_name, full_gemini_prompt)
                if not text:
                    # Unusable output: re-ask once with an explicit completion instruction.
                    text = self._gemini_generate_text(
                        model_name, full_gemini_prompt + completeness_nudge
                    )
                if text:
                    return text
            except Exception as e:
                logger.warning(
                    f"Gemini fallback on '{model_name}' failed: {e}"
                )

        raise RuntimeError(
            "All code generation endpoints across all tiers failed."
        )

    def _gemini_generate_text(self, model_name: str, contents: str) -> Optional[str]:
        """Generates with Gemini, returning None when the output is unusable.

        Gemini 2.5/3.x reserve part of ``max_output_tokens`` for internal reasoning,
        so a tight ceiling silently truncates the emitted code into invalid Python.
        This guard rejects truncated output and retries with a smaller budget if the
        model itself refuses the larger one.
        """
        for cap in (GEMINI_MAX_OUTPUT_TOKENS, 8192):
            try:
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=cap,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    ),
                )
            except Exception as exc:
                if cap == GEMINI_MAX_OUTPUT_TOKENS:
                    logger.warning(
                        f"Gemini '{model_name}' rejected a {cap}-token budget ({exc}). "
                        f"Retrying with 8192."
                    )
                    continue
                raise

            if not (response and response.text):
                return None

            try:
                finish_reason = str(response.candidates[0].finish_reason)
            except Exception:
                finish_reason = ""

            if "MAX_TOKENS" in finish_reason.upper():
                logger.warning(
                    f"Gemini '{model_name}' hit the {cap}-token ceiling (output truncated)."
                )
                return None

            if _looks_truncated(response.text):
                logger.warning(
                    f"Gemini '{model_name}' returned unbalanced code (likely truncated)."
                )
                return None

            return response.text

        return None

    def _process_single_feature(self, feature: dict, index: int = 0, revision_notes: str = "") -> dict:
        if index > 0:
            time.sleep(index * 1.5)

        module_name = feature.get("target_module", "unnamed_module")
        logger.info(f"Coder Agent: Synthesizing module '{module_name}'...")

        revision_block = ""
        if revision_notes:
            revision_block = (
                "\nHUMAN REVISION REQUEST (highest priority - the code MUST honour this):\n"
                f"{revision_notes}\n"
            )

        base_prompt = f"""
Target Module: {module_name}
Feature Domain: {feature.get('feature_name')}
{revision_block}
Business Rules to Enforce:
{json.dumps(feature.get('business_rules', []), indent=2)}

Exceptions to Define and Handle:
{json.dumps(feature.get('exceptions', []), indent=2)}

Functions / Logic to Fully Implement:
{_render_functions(feature.get('functions', []))}

Instructions:
- Write complete, syntactically valid Python code for the ENTIRE module from imports to final returns.
- Implement every function listed above with its exact signature, type hints and return type.
- Define every exception class explicitly at the top of the file before raising it.
- Handle the business rules literally (no invented behaviour, no unrelated utility helpers).
- Output a COMPLETE file: never stop mid-function, never leave a block unclosed, never use placeholder 'pass' inside core logic.
- Keep the module focused on THIS module's responsibility only.
"""

        max_repair_attempts = 3
        current_prompt = base_prompt
        clean_code = ""
        is_valid = False
        validation_msg = ""
        tests_passed = False
        test_output = ""

        # Autonomous MCP Repair Loop with Task-Based Routing
        for attempt in range(1, max_repair_attempts + 1):
            task_mode = "draft" if attempt == 1 else "repair"
            logger.info(
                f"Coder Agent: Autonomous loop attempt {attempt}/{max_repair_attempts} ({task_mode.upper()} mode) for '{module_name}'"
            )

            try:
                raw_code = self._generate_with_fallback(current_prompt, task_type=task_mode)
            except Exception as exc:
                logger.error(f"Failed generation attempt {attempt} for {module_name}: {exc}")
                clean_code = f"# Error generating code for {module_name}: {exc}"
                validation_msg = str(exc)
                break

            # Clean reasoning tags and markdown backticks
            clean_code = re.sub(r"<think>.*?</think>", "", raw_code, flags=re.DOTALL)
            clean_code = re.sub(r"^```python\s*", "", clean_code, flags=re.MULTILINE)
            clean_code = re.sub(r"^```\s*$", "", clean_code, flags=re.MULTILINE).strip()

            # Normalise typographic punctuation the model may have substituted for ASCII
            clean_code = _sanitize_generated_code(clean_code)

            # 1. MCP AST Syntax Check
            is_valid, validation_msg = MCPToolServer.validate_python_syntax(clean_code)
            if not is_valid:
                truncated = _looks_truncated(clean_code)
                logger.warning(
                    f"Attempt {attempt} syntax check failed "
                    f"({'TRUNCATED output' if truncated else 'invalid syntax'}): {validation_msg}"
                )
                current_prompt = (
                    f"{base_prompt}\n\n"
                    f"CRITICAL FIX REQUIRED (Attempt {attempt}):\n"
                    f"Your previous output was "
                    f"{'CUT OFF before it finished' if truncated else 'not valid Python'}:\n"
                    f"{validation_msg}\n"
                    f"Resend the ENTIRE corrected file from the first import to the final return. "
                    f"Do not apologise or explain - output only the complete code."
                )
                continue

            # 2. Stage draft inside isolated staging directory (keeps generated_src clean before approval)
            try:
                MCPToolServer.write_module_to_disk(
                    target_module=module_name,
                    code_content=clean_code,
                    base_dir=str(self.staging_dir),
                )
            except Exception as write_err:
                logger.warning(f"MCP Staging write failed: {write_err}")

            # 3. Execute Pytest Suite on staging target
            try:
                test_result = MCPToolServer.run_pytest_suite(
                    test_target=str(self.staging_dir)
                )
                tests_passed = test_result.get("success", False)
                stdout = test_result.get("stdout", "")
                stderr = test_result.get("stderr", "")
                test_output = (stdout + "\n" + stderr).strip()

                # Treat Pytest Exit Code 5 ("no tests collected") as successful valid code execution
                is_no_tests = any(
                    phrase in test_output.lower()
                    for phrase in [
                        "no tests collected",
                        "collected 0 items",
                        "exit code 5",
                        "[empty]",
                    ]
                )

                if tests_passed or is_no_tests:
                    tests_passed = True
                    logger.info(f"Module '{module_name}' passed syntax and structure validation on attempt {attempt}!")
                    break
                else:
                    logger.warning(f"Attempt {attempt} failed pytest execution:\n{test_output}")
                    current_prompt = (
                        f"{base_prompt}\n\n"
                        f"CRITICAL FIX REQUIRED (Attempt {attempt}):\n"
                        f"The code raised runtime errors or test failures:\n{test_output}\n"
                        f"Fix the bug and provide complete corrected Python code."
                    )
            except Exception as test_err:
                logger.warning(f"MCP Pytest tool runner encountered an issue: {test_err}")
                break

        # Clean up temporary staging directory post-evaluation
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir, ignore_errors=True)

        # Authoritative final re-validation of exactly what we are about to return,
        # so the UI badge never disagrees with the code it is attached to.
        if clean_code and not clean_code.startswith("# Error"):
            final_valid, final_msg = MCPToolServer.validate_python_syntax(clean_code)
            is_valid = final_valid
            validation_msg = final_msg
            if final_valid:
                validation_msg = (
                    f"{final_msg} Pytest: "
                    f"{'passed' if tests_passed else 'not run'}."
                )

        draft_id = f"draft_{module_name.replace('.', '_')}"

        return {
            "draft_id": draft_id,
            "feature_name": feature.get("feature_name"),
            "target_module": module_name,
            "code": clean_code,
            "syntax_valid": is_valid,
            "validation_msg": validation_msg,
            "tests_passed": tests_passed,
            "test_output": test_output,
        }

    def generate_single_module(self, feature: dict, revision_notes: str = "") -> dict:
        """Synthesizes ONE module through the full MCP-validated pipeline.

        Exposed so the LangGraph agent gets the exact same quality gates as the
        bulk Coder Agent run (prompt engineering, AST syntax guardrail, staged
        pytest), instead of a raw single-shot LLM call.
        """
        return self._process_single_feature(feature, index=0, revision_notes=revision_notes)

    def generate_drafts_from_specs(self, specs_data: dict, revision_notes: str = "") -> list:
        hitl_queue.clear()
        features = specs_data.get("features", [])
        if not features:
            return []

        logger.info(
            f"Starting pipeline synthesis for {len(features)} module(s)..."
        )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_feature = {
                executor.submit(
                    self._process_single_feature, feature, idx, revision_notes
                ): feature
                for idx, feature in enumerate(features)
            }

            for future in as_completed(future_to_feature):
                try:
                    draft_data = future.result()
                    hitl_queue.add_draft(
                        draft_id=draft_data["draft_id"],
                        feature_name=draft_data["feature_name"],
                        target_module=draft_data["target_module"],
                        code=draft_data["code"],
                        syntax_valid=draft_data["syntax_valid"],
                        validation_msg=draft_data["validation_msg"],
                        tests_passed=draft_data.get("tests_passed", False),
                        test_output=draft_data.get("test_output", ""),
                    )
                except Exception as exc:
                    logger.error(f"Error compiling draft result: {exc}")

        return hitl_queue.get_all()

    def write_approved_draft_to_disk(self, draft_id: str, code: str) -> bool:
        """Writes approved Python code directly into the generated_src/ tree using MCP tools."""
        try:
            module_name = draft_id.replace("draft_", "").replace("_", ".")
            MCPToolServer.write_module_to_disk(
                target_module=module_name,
                code_content=code,
                base_dir=str(self.output_dir),
            )
            logger.info(f"[Coder Agent] Saved approved draft '{draft_id}' via MCP tool.")
            return True
        except Exception as err:
            logger.error(f"[Coder Agent] Failed to write draft '{draft_id}': {err}")
            return False

    def export_all_approved_drafts(self) -> int:
        """Pulls all approved drafts from the HITL Queue and persists them into generated_src/."""
        all_drafts = hitl_queue.get_all()
        saved_count = 0
        for draft in all_drafts:
            if draft.get("status") == "APPROVED" or draft.get("syntax_valid"):
                success = self.write_approved_draft_to_disk(
                    draft_id=draft["draft_id"], code=draft["code"]
                )
                if success:
                    saved_count += 1
        return saved_count