import ast
import logging
import os
import subprocess
import sys
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


class MCPToolServer:
    # ------------------------------------------------------------------
    # 1. READ & INSPECT TOOLS (Workspace Visibility)
    # ------------------------------------------------------------------
    @staticmethod
    def read_module_content(target_module: str, base_dir: str = "generated_src") -> str:
        """MCP Tool: Reads existing python module content for inspection or refactoring."""
        module_path = target_module.replace(".", "/") + ".py"
        full_path = os.path.join(os.path.abspath(base_dir), module_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Module '{target_module}' does not exist at {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def list_workspace_files(base_dir: str = "generated_src") -> list[str]:
        """MCP Tool: Returns a list of all generated module paths in the workspace."""
        abs_base = os.path.abspath(base_dir)
        if not os.path.exists(abs_base):
            return []

        file_list = []
        for root, _, files in os.walk(abs_base):
            for file in files:
                if file.endswith(".py"):
                    rel_path = os.path.relpath(os.path.join(root, file), abs_base)
                    file_list.append(rel_path)
        return file_list

    # ------------------------------------------------------------------
    # 2. VALIDATION & WRITE TOOLS (Safety Guardrails)
    # ------------------------------------------------------------------
    @staticmethod
    def validate_python_syntax(code: str) -> Tuple[bool, str]:
        """MCP Tool: Validates Python abstract syntax tree before execution."""
        try:
            ast.parse(code)
            return True, "Syntax validation successful."
        except SyntaxError as err:
            return False, f"Syntax Error at line {err.lineno}: {err.msg}"

    @staticmethod
    def write_module_to_disk(target_module: str, code_content: str, base_dir: str = "generated_src") -> str:
        """MCP Tool: Safe File System Writer for approved production modules."""
        is_valid, err_msg = MCPToolServer.validate_python_syntax(code_content)
        if not is_valid:
            raise ValueError(f"MCP Security Guardrail Rejected Write: {err_msg}")

        module_path = target_module.replace(".", "/") + ".py"
        full_path = os.path.join(os.path.abspath(base_dir), module_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(code_content)

        logger.info(f"MCP Tool: Successfully written module to {full_path}")
        return full_path

    # ------------------------------------------------------------------
    # 3. EXECUTION & FEEDBACK TOOLS (Autonomous Loop)
    # ------------------------------------------------------------------
    @staticmethod
    def run_pytest_suite(test_target: str = "generated_src") -> Dict[str, Any]:
        """MCP Tool: Runs pytest on generated files and returns stdout/stderr feedback to the agent."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_target, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Pytest execution timed out after 30 seconds.",
            }
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e)}