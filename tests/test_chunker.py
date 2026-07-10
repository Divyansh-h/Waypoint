import pytest
from pathlib import Path
from ingestion.chunker import ASTChunker

@pytest.fixture
def chunker() -> ASTChunker:
    """Provides a fresh ASTChunker instance for each test."""
    return ASTChunker(max_lines_per_chunk=100)

def test_simple_function(tmp_path: Path, chunker: ASTChunker):
    file_path = tmp_path / "simple.py"
    file_path.write_text(
        "def hello_world():\n"
        "    print('Hello!')\n"
    )
    chunks = chunker.chunk_file(file_path)
    
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "function"
    assert chunks[0].function_name == "hello_world"
    assert "print('Hello!')" in chunks[0].content
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 2

def test_nested_class(tmp_path: Path, chunker: ASTChunker):
    file_path = tmp_path / "nested.py"
    file_path.write_text(
        "class Outer:\n"
        "    def method_one(self):\n"
        "        pass\n"
        "    class Inner:\n"
        "        def method_two(self):\n"
        "            pass\n"
    )
    chunks = chunker.chunk_file(file_path)
    
    # We expect 4 chunks: Outer, Outer.method_one, Outer.Inner, Outer.Inner.method_two
    assert len(chunks) == 4
    names = [c.function_name for c in chunks]
    
    assert "Outer" in names
    assert "Outer.method_one" in names
    assert "Outer.Inner" in names
    assert "Outer.Inner.method_two" in names
    
    # Validate the nested inner class and its nested method
    inner_class_chunk = next(c for c in chunks if c.function_name == "Outer.Inner")
    assert inner_class_chunk.chunk_type == "class"
    
    inner_method_chunk = next(c for c in chunks if c.function_name == "Outer.Inner.method_two")
    assert inner_method_chunk.chunk_type == "method"

def test_decorated_function(tmp_path: Path, chunker: ASTChunker):
    file_path = tmp_path / "decorated.py"
    file_path.write_text(
        "@pytest.fixture\n"
        "@lru_cache(maxsize=32)\n"
        "def expensive_op():\n"
        "    return 42\n"
    )
    chunks = chunker.chunk_file(file_path)
    
    assert len(chunks) == 1
    chunk = chunks[0]
    
    # The name should still be extracted accurately from the inner node
    assert chunk.function_name == "expensive_op"
    # But the content MUST include the outer decorators
    assert "@pytest.fixture" in chunk.content
    assert "@lru_cache(maxsize=32)" in chunk.content
    assert "def expensive_op():" in chunk.content

def test_syntax_error_file_skips_gracefully(tmp_path: Path, chunker: ASTChunker, caplog):
    file_path = tmp_path / "broken.py"
    # Deliberately malformed python code
    file_path.write_text(
        "def broken_function(\n"
        "    return 1 +\n"
    )
    
    chunks = chunker.chunk_file(file_path)
    
    # Should safely return an empty list instead of raising an exception
    assert len(chunks) == 0
    # Should trigger the logger warning
    assert "Skipping malformed file (syntax errors detected)" in caplog.text
