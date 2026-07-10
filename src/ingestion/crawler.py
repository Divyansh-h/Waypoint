import os
from collections.abc import Iterator
from pathlib import Path
from typing import Optional, Union, Set

class RepoCrawler:
    """
    Crawls a local repository directory, yielding file paths that match
    configured inclusion criteria while skipping excluded directories and extensions.
    """

    def __init__(
        self,
        repo_path: Union[str, Path],
        include_extensions: Set[str],
        exclude_extensions: Optional[Set[str]] = None,
        exclude_directories: Optional[Set[str]] = None,
    ) -> None:
        """
        Initialize the RepoCrawler with filtering rules.

        Args:
            repo_path: The root directory path of the repository to crawl.
            include_extensions: A set of file extensions to include (e.g., {".py", ".md"}).
            exclude_extensions: A set of file extensions to explicitly ignore.
            exclude_directories: A set of directory names to skip (e.g., {".git", "venv"}).
        """
        self.repo_path = Path(repo_path).resolve()
        
        # Ensure extensions start with a dot
        self.include_extensions = {ext if ext.startswith(".") else f".{ext}" for ext in include_extensions}
        
        self.exclude_extensions = set()
        if exclude_extensions:
            self.exclude_extensions = {ext if ext.startswith(".") else f".{ext}" for ext in exclude_extensions}
            
        self.exclude_directories = exclude_directories or {".git", "__pycache__", "node_modules", "venv", ".venv"}

    def walk(self) -> Iterator[Path]:
        """
        Walks the repository directory based on the configured rules.

        Yields:
            Path: The absolute path to a file that matches the inclusion criteria.
        """
        if not self.repo_path.exists() or not self.repo_path.is_dir():
            raise NotADirectoryError(f"Repository path does not exist or is not a directory: {self.repo_path}")

        for root, dirs, files in os.walk(self.repo_path):
            # Modify dirs in-place to prevent os.walk from traversing excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_directories]

            for file in files:
                file_path = Path(root) / file
                
                # Fast extension check
                ext = file_path.suffix
                
                if ext in self.exclude_extensions:
                    continue
                    
                if ext in self.include_extensions:
                    yield file_path
