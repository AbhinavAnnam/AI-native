"""Reproduces the approve->commit flow offline and checks the resume semantics."""
import json
import os

from dotenv import load_dotenv

load_dotenv()

import asyncio

from backend.agent import graph as g

SPEC = {
    "project_title": "Demo",
    "features": [
        {"feature_name": "Accounts", "target_module": "accounts_and_identity",
         "business_rules": ["a"], "exceptions": [], "functions": []},
        {"feature_name": "Work", "target_module": "projects_and_work_execution",
         "business_rules": ["b"], "exceptions": [], "functions": []},
    ],
    "cross_cutting_constraints": [],
}

CODE = "def handler() -> int:\n    return 1\n"


class FakeCoder:
    def generate_drafts_from_specs(self, specs, revision_notes=""):
        return [
            {"draft_id": f"draft_{f['target_module']}", "target_module": f["target_module"],
             "feature_name": f["feature_name"], "code": CODE, "syntax_valid": True,
             "validation_msg": "Syntax validation successful.", "tests_passed": True,
             "test_output": ""}
            for f in specs["features"]
        ]

    def generate_single_module(self, feature, revision_notes=""):
        return {"draft_id": f"draft_{feature['target_module']}",
                "target_module": feature["target_module"],
                "feature_name": feature["feature_name"], "code": CODE,
                "syntax_valid": True, "validation_msg": "ok", "tests_passed": True,
                "test_output": ""}


g._build_coder = lambda: FakeCoder()

written = []
import mcp_tools.file_tools as ft
ft.MCPToolServer.write_module_to_disk = staticmethod(
    lambda target_module, code_content, base_dir="generated_src": written.append(target_module)
    or f"{base_dir}\\{target_module}.py"
)

config = {"configurable": {"thread_id": "t-approve-flow"}}


async def main():
    async for _ in g.agent_app.astream(
        {"task_spec": json.dumps(SPEC), "messages": [], "user_feedback": None,
         "modified_spec": None, "status": "started"},
        config,
    ):
        pass
    st = g.agent_app.get_state(config)
    print("after start   next =", st.next, "| status =", st.values.get("status"),
          "| drafts =", len(st.values.get("drafts") or []))

    g.agent_app.update_state(
        config,
        {"user_feedback": "approve", "modified_spec": None},
        as_node="generate_code",
    )
    st2 = g.agent_app.get_state(config)
    print("after update  next =", st2.next)

    async for _ in g.agent_app.astream(None, config):
        pass
    st3 = g.agent_app.get_state(config)
    print("after resume  next =", st3.next, "| status =", st3.values.get("status"))
    print("written files =", written)
    print("RESULT:", "COMMIT OK" if written == ["accounts_and_identity",
                                                 "projects_and_work_execution"] else "COMMIT FAILED")


asyncio.run(main())
