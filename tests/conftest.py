import pytest
from pathlib import Path
from typing import List, Dict, Any


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """
    Creates a temporary directory simulating a tiny repository.
    Useful for testing the crawler and chunker without touching a real repo.
    """
    # Create the root of the fake repo
    repo_root = tmp_path / "fake_repo"
    repo_root.mkdir()

    # 1. Standard python file
    main_py = repo_root / "main.py"
    main_py.write_text(
        "def hello_world():\n"
        "    print('Hello, world!')\n"
    )

    # 2. Nested python file
    utils_dir = repo_root / "utils"
    utils_dir.mkdir()
    helpers_py = utils_dir / "helpers.py"
    helpers_py.write_text(
        "class Helper:\n"
        "    def do_work(self):\n"
        "        pass\n"
    )

    # 3. Markdown file (to test inclusion of other types)
    readme_md = repo_root / "README.md"
    readme_md.write_text(
        "# Fake Repo\n"
        "This is a test repository.\n"
    )

    # 4. Cython file (to test exclusion rules)
    setup_pyx = repo_root / "setup.pyx"
    setup_pyx.write_text("# Cython code here\n")

    # 5. Excluded directory (e.g., .git)
    git_dir = repo_root / ".git"
    git_dir.mkdir()
    git_config = git_dir / "config"
    git_config.write_text("[core]\n    repositoryformatversion = 0\n")

    return repo_root

@pytest.fixture
def sample_chunks() -> List[Dict[str, Any]]:
    """
    Provides a tiny, fake set of 5 code chunks mimicking scikit-learn structure.
    Use this fixture to unit-test the training data generation pipeline (mining, 
    synthetic generation, filtering) without making real LLM API calls or 
    querying the actual database.
    """
    return [
        {
            "chunk_id": "chunk_001",
            "name": "calculate_loss",
            "type": "function",
            "file_path": "sklearn/metrics/_loss.py",
            "content": 'def calculate_loss(y_true, y_pred):\n    """Calculate the mean squared error loss."""\n    return sum((y_t - y_p)**2 for y_t, y_p in zip(y_true, y_pred)) / len(y_true)'
        },
        {
            "chunk_id": "chunk_002",
            "name": "BaseEstimator",
            "type": "class",
            "file_path": "sklearn/base.py",
            "content": 'class BaseEstimator:\n    """Base class for all estimators in scikit-learn."""\n    def get_params(self, deep=True):\n        return {}'
        },
        {
            "chunk_id": "chunk_003",
            "name": "fit",
            "type": "method",
            "file_path": "sklearn/linear_model/_logistic.py",
            "content": 'def fit(self, X, y, sample_weight=None):\n    """Fit the model according to the given training data."""\n    self.coef_ = [0.5]\n    return self'
        },
        {
            "chunk_id": "chunk_004",
            "name": "predict",
            "type": "method",
            "file_path": "sklearn/ensemble/_forest.py",
            "content": 'def predict(self, X):\n    """Predict class for X."""\n    return [1 for _ in X]'
        },
        {
            "chunk_id": "chunk_005",
            "name": "_check_is_fitted",
            "type": "function",
            "file_path": "sklearn/utils/validation.py",
            "content": 'def _check_is_fitted(estimator, attributes=None):\n    """Internal check to ensure estimator is fitted."""\n    if not hasattr(estimator, "coef_"):\n        raise NotFittedError("Not fitted!")'
        }
    ]

class MockResponse:
    def __init__(self, text: str = "", function_call: Any = None):
        self._text = text
        self.function_call = function_call
        
    @property
    def text(self) -> str:
        return self._text
        
    @property
    def parts(self) -> List[Any]:
        class MockPart:
            def __init__(self, text: str, function_call: Any):
                self.text = text
                self.function_call = function_call
        return [MockPart(self._text, self.function_call)]

class MockLLMClient:
    """
    A programmable LLM mock.
    Allows tests to queue up a sequence of native-function-call responses
    perfectly simulating a multi-hop reasoning chain without burning API tokens.
    """
    def __init__(self) -> None:
        self.response_queue: List[Any] = []
        self.call_history: List[str] = []
        
    def set_responses(self, responses: List[Any]) -> None:
        """Queue up a list of MockResponse objects that the mock should yield in order."""
        self.response_queue = responses
        
    def generate_content(self, prompt: str) -> Any:
        """Simulates the LLM API call."""
        self.call_history.append(prompt)
        
        if not self.response_queue:
            raise RuntimeError("MockLLMClient ran out of queued responses!")
            
        # Pop the next programmed response
        return self.response_queue.pop(0)

@pytest.fixture
def mock_llm() -> MockLLMClient:
    """Fixture providing a fresh mock LLM instance for a test."""
    return MockLLMClient()
