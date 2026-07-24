from unittest.mock import MagicMock, patch

import pytest

from agent.tools.ast_search import ASTSearchTool
from agent.tools.git_patch import GitPatchTool
from agent.tools.sandbox import CodeSandboxTool

# --- FIXTURES ---

@pytest.fixture
def valid_sandbox_snippet() -> str:
    """A known-safe snippet for the CodeSandboxTool that fits the environment constraints."""
    return (
        "import numpy as np\n"
        "from sklearn.linear_model import LinearRegression\n\n"
        "X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])\n"
        "y = np.dot(X, np.array([1, 2])) + 3\n"
        "reg = LinearRegression().fit(X, y)\n"
        "print(f'Score: {reg.score(X, y)}')\n"
    )

@pytest.fixture
def ast_target_name() -> str:
    """A known class name that we expect the AST parser to find in scikit-learn."""
    return "RandomForestClassifier"

@pytest.fixture
def valid_git_diff() -> str:
    """A valid, structurally sound diff for the GitPatchTool to apply."""
    return (
        "--- a/sklearn/ensemble/_forest.py\n"
        "+++ b/sklearn/ensemble/_forest.py\n"
        "@@ -100,6 +100,7 @@\n"
        "     def __init__(self, n_estimators=100):\n"
        "         self.n_estimators = n_estimators\n"
        "+        self.is_patched = True\n"
    )


# --- TESTS ---

def test_sandbox_tool_execute(valid_sandbox_snippet: str) -> None:
    tool = CodeSandboxTool()
    
    # Test missing param
    res_missing = tool.execute({})
    assert "Error: Missing 'code' parameter." in res_missing
    
    # Test execution routing via mock to avoid requiring Docker daemon during tests
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = "Score: 1.0\n"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        res_success = tool.execute({"code": valid_sandbox_snippet})
        
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert "docker" in args[0]
        assert "run" in args[0]
        assert "Score: 1.0" in res_success


def test_ast_search_tool_execute(ast_target_name: str) -> None:
    tool = ASTSearchTool()
    
    # Test missing param
    assert "Error: Missing" in tool.execute({"query_type": "find_definition"})
    assert "Error: Missing" in tool.execute({"target_name": ast_target_name})
    
    # Test invalid query type
    assert "Unsupported query_type" in tool.execute({
        "query_type": "find_magic_method", 
        "target_name": ast_target_name
    })
    
    # Test valid implementations
    res_def = tool.execute({"query_type": "find_definition", "target_name": ast_target_name})
    assert "sklearn/ensemble/_forest.py" in res_def
    
    res_sub = tool.execute({"query_type": "find_subclasses", "target_name": "BaseForest"})
    assert "RandomTreesEmbedding" in res_sub
    assert "sklearn/ensemble/_forest.py" in res_sub


def test_git_patch_tool_execute(valid_git_diff: str) -> None:
    tool = GitPatchTool()
    
    # Test missing action
    assert "Error: Missing 'action' parameter." in tool.execute({"diff": valid_git_diff})
    
    with patch("subprocess.run") as mock_run:
        # 1. Test apply_patch
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        res_apply = tool.execute({"action": "apply_patch", "diff": valid_git_diff})
        assert "Successfully applied patch" in res_apply
        args, kwargs = mock_run.call_args
        assert "git" in args[0]
        assert "apply" in args[0]
        
        # 2. Test run_tests
        mock_result_test = MagicMock()
        mock_result_test.returncode = 0
        mock_result_test.stdout = "3 passed in 0.05s"
        mock_result_test.stderr = ""
        mock_run.return_value = mock_result_test
        
        res_tests = tool.execute({"action": "run_tests", "scope": "sklearn/ensemble/tests/test_forest.py"})
        assert "Tests PASSED" in res_tests
        assert "3 passed" in res_tests
        args, kwargs = mock_run.call_args
        assert "pytest" in args[0]
        assert "sklearn/ensemble/tests/test_forest.py" in args[0]
        
    # 3. Test PR description stub
    res_pr = tool.execute({"action": "draft_pr_description", "diff": valid_git_diff, "context": "Fixing a bug"})
    assert "[DRAFT ONLY" in res_pr
    assert "Fixing a bug" in res_pr
