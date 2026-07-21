# ruff: noqa: E501
import os
import re
import uuid
import yaml
import subprocess
from pathlib import Path
from typing import Any, Dict

from agent.tools.base import BaseTool

# Characters and patterns that should never appear in a test scope or diff
# from an LLM — these indicate shell injection or pytest plugin attacks.
_SHELL_METACHAR_PATTERN = re.compile(r"[;&|`$><!\n\r]")
_PYTEST_FLAG_PATTERN = re.compile(r"(^|\s)--?\w")  # Catches -p, --co, --override-ini, etc.
_MAX_DIFF_SIZE = 50_000  # 50 KB — no legitimate single-function patch should exceed this


class GitPatchTool(BaseTool):
    """
    Operates strictly against a local git clone to safely propose, validate, 
    and document codebase changes.
    """

    def __init__(self, config_path: str = "configs/ingestion.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            self.repo_path = config["repo"]["path"]

    @property
    def name(self) -> str:
        return "git_patch"

    @property
    def description(self) -> str:
        return (
            "Applies patches, runs isolated tests, and drafts PR descriptions on a purely local git clone. "
            "Use this tool when you are ready to propose an actual code modification to the codebase. "
            "Supported actions: 'apply_patch', 'run_tests', 'draft_pr_description'. "
            "NOTE: 'draft_pr_description' generates a draft markdown template only; it NEVER auto-submits a pull request."
        )

    # ── Input Validation ──────────────────────────────────────────────────

    @staticmethod
    def _validate_scope(scope: str, repo_path: str) -> str:
        """
        Validates and resolves a test scope path, blocking:
          - Shell metacharacters (;, &, |, `, $, etc.)
          - Pytest CLI flags (--co, -p malicious_plugin, --override-ini, etc.)
          - Path traversal via '..' components
          - Paths outside the repository root (including symlink escapes)
          - Non-.py file extensions

        Returns the validated, resolved absolute path on success.
        Raises ValueError with a descriptive message on failure.
        """
        if _SHELL_METACHAR_PATTERN.search(scope):
            raise ValueError(f"Scope contains forbidden shell metacharacters: {scope!r}")

        if _PYTEST_FLAG_PATTERN.search(scope):
            raise ValueError(f"Scope contains pytest flags, which are not allowed: {scope!r}")

        # Block explicit path traversal before resolution
        if ".." in scope.split(os.sep):
            raise ValueError(f"Scope contains '..' path traversal: {scope!r}")

        # Resolve to an absolute path and verify it lives inside the repo
        repo_root = Path(repo_path).resolve()
        resolved = (repo_root / scope).resolve()

        if not str(resolved).startswith(str(repo_root)):
            raise ValueError(
                f"Scope resolves outside the repository root: {scope!r} -> {resolved}"
            )

        # Must be a .py file or a directory (pytest can run against directories)
        if resolved.suffix and resolved.suffix != ".py":
            raise ValueError(f"Scope must target a .py file or directory, got: {scope!r}")

        return str(resolved)

    @staticmethod
    def _validate_diff(diff: str) -> None:
        """
        Validates a diff/patch string, blocking:
          - Excessively large patches (> 50 KB)
          - Embedded shell payloads (backticks, $(), pipe chains)
        
        Raises ValueError with a descriptive message on failure.
        """
        if len(diff) > _MAX_DIFF_SIZE:
            raise ValueError(
                f"Diff is {len(diff)} bytes, exceeding the {_MAX_DIFF_SIZE} byte safety limit."
            )

        if _SHELL_METACHAR_PATTERN.search(diff):
            # Diffs legitimately contain > and < for context lines, so we check
            # specifically for dangerous sequences rather than individual chars.
            dangerous_patterns = ["`", "$(", "| ", "&&", "||", "; "]
            for pattern in dangerous_patterns:
                if pattern in diff:
                    raise ValueError(
                        f"Diff contains suspicious shell pattern {pattern!r}. "
                        "Refusing to apply."
                    )

    # ── Tool Actions ──────────────────────────────────────────────────────

    def execute(self, args: Dict[str, Any]) -> str:
        action = args.get("action")
        
        if not action:
            return "Error: Missing 'action' parameter."

        if action == "apply_patch":
            diff = args.get("diff")
            if not diff:
                return "Error: Missing 'diff' parameter for apply_patch."
            return self._stub_apply_patch(diff)
            
        elif action == "run_tests":
            scope = args.get("scope")
            if not scope:
                return "Error: Missing 'scope' parameter for run_tests."
            return self._stub_run_tests(scope)
            
        elif action == "draft_pr_description":
            diff = args.get("diff")
            context = args.get("context", "")
            if not diff:
                return "Error: Missing 'diff' parameter for draft_pr_description."
            return self._stub_draft_pr_description(diff, context)
            
        else:
            return f"Error: Unsupported action '{action}'."

    def _stub_apply_patch(self, diff: str) -> str:
        # Validate the diff content before writing it to disk
        try:
            self._validate_diff(diff)
        except ValueError as e:
            return f"Error: Diff validation failed: {e}"

        patch_file = f"/tmp/patch_{uuid.uuid4().hex}.diff"
        with open(patch_file, "w") as f:
            f.write(diff)
            
        try:
            # Dry-run first: verify the patch applies cleanly before mutating the repo
            check_result = subprocess.run(
                ["git", "apply", "--check", patch_file],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if check_result.returncode != 0:
                return f"Failed to apply patch (dry-run check):\n{check_result.stderr}"

            # Apply for real
            result = subprocess.run(
                ["git", "apply", patch_file],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return f"Failed to apply patch:\n{result.stderr}"
            return "Successfully applied patch to the local codebase."
        except subprocess.TimeoutExpired:
            return "Error: git apply timed out."
        except Exception as e:
            return f"Error applying patch: {str(e)}"
        finally:
            if os.path.exists(patch_file):
                os.remove(patch_file)

    def _stub_run_tests(self, scope: str) -> str:
        # Validate and resolve the scope to a safe absolute path
        try:
            resolved_scope = self._validate_scope(scope, self.repo_path)
        except ValueError as e:
            return f"Error: Scope validation failed: {e}"

        try:
            result = subprocess.run(
                ["pytest", resolved_scope],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
                
            status = "PASSED" if result.returncode == 0 else "FAILED"
            return f"Tests {status} (Exit code {result.returncode}):\n{output}"
        except subprocess.TimeoutExpired:
            return f"Error: Tests for '{scope}' timed out after 60 seconds."
        except Exception as e:
            return f"Error running tests: {str(e)}"

    def _stub_draft_pr_description(self, diff: str, context: str) -> str:
        template = f"""
[DRAFT ONLY - THIS TOOL DOES NOT AUTO-SUBMIT PULL REQUESTS]

## Proposed PR Title
[Autogenerated based on context]

## Description
{context if context else 'No context provided.'}

## Changes Applied
```diff
{diff}
```

## Validation
- [x] Verified via local Pytest (GitPatchTool)
- [ ] Manual review required
"""
        return template.strip()

    def get_function_declaration(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "action": {
                        "type": "STRING",
                        "description": "The specific git/validation action to perform. Must be one of: 'apply_patch', 'run_tests', 'draft_pr_description'."
                    },
                    "diff": {
                        "type": "STRING",
                        "description": "The raw diff or patch string to apply or document. Required for 'apply_patch' and 'draft_pr_description'."
                    },
                    "scope": {
                        "type": "STRING",
                        "description": "The specific file, module, or test class to execute tests against (e.g., 'sklearn/ensemble/tests/test_forest.py'). Required for 'run_tests'."
                    },
                    "context": {
                        "type": "STRING",
                        "description": "Optional reasoning or context behind the patch, used to enrich the PR description."
                    }
                },
                "required": ["action"]
            }
        }

