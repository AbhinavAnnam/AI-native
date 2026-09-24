import threading
from typing import Dict, Any, List


class HITLManager:
    def __init__(self):
        self._pending_queue: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def clear(self):
        with self._lock:
            self._pending_queue.clear()

    def add_draft(
        self,
        draft_id: str,
        feature_name: str,
        target_module: str,
        code: str,
        syntax_valid: bool,
        validation_msg: str,
        tests_passed: bool = False,
        test_output: str = "",
    ):
        with self._lock:
            self._pending_queue[draft_id] = {
                "draft_id": draft_id,
                "feature_name": feature_name,
                "target_module": target_module,
                "code": code,
                "syntax_valid": syntax_valid,
                "validation_msg": validation_msg,
                "tests_passed": tests_passed,
                "test_output": test_output,
                "status": "AWAITING_HUMAN_REVIEW",
            }

    def get_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._pending_queue.values())

    def update_status(self, draft_id: str, status: str, updated_code: str = None):
        with self._lock:
            if draft_id in self._pending_queue:
                self._pending_queue[draft_id]["status"] = status
                if updated_code is not None:
                    self._pending_queue[draft_id]["code"] = updated_code


hitl_queue = HITLManager()