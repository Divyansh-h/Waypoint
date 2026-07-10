import pytest
from pathlib import Path


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
