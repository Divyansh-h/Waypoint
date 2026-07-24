from pathlib import Path

from ingestion.crawler import RepoCrawler


def test_crawler_finds_correct_files(fake_repo: Path):
    """
    Test that the crawler finds .py and .md files,
    but skips .pyx files and files inside .git/
    """
    crawler = RepoCrawler(
        repo_path=fake_repo,
        include_extensions={".py", ".md"},
        exclude_extensions={".pyx"},
        exclude_directories={".git"}
    )
    
    found_files = list(crawler.walk())
    
    # We expect main.py, utils/helpers.py, and README.md
    assert len(found_files) == 3
    
    # Extract just the filenames for easier asserting
    file_names = {f.name for f in found_files}
    assert "main.py" in file_names
    assert "helpers.py" in file_names
    assert "README.md" in file_names
    
    # Ensure excluded things are NOT there
    assert "setup.pyx" not in file_names
    assert "config" not in file_names  # The file inside .git
