import json
import logging
import sys
from pathlib import Path
from typing import List, Union

from pydantic import ValidationError

# Ensure src/ is in the python path for absolute imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ingestion.models import EvalExample  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("eval_loader")


def load_eval_set(filepath: Union[str, Path]) -> List[EvalExample]:
    """
    Loads and validates a JSONL file of evaluation examples.
    Errors clearly on malformed lines, explaining exactly which fields are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Eval set not found at: {filepath}")
        
    examples: List[EvalExample] = []
    errors = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                example = EvalExample(**data)
                examples.append(example)
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num} | Invalid JSON: {e}")
                errors += 1
            except ValidationError as e:
                logger.error(f"Line {line_num} | Validation failed:")
                # Pydantic e.errors() returns a list of dictionaries with 'loc', 'msg', 'type'
                for error in e.errors():
                    # 'loc' is a tuple of the field path, e.g., ('ground_truth', 'acceptable_paths')
                    field = " -> ".join([str(loc) for loc in error['loc']])
                    logger.error(f"  ❌ Field '{field}': {error['msg']}")
                errors += 1
                
    if errors > 0:
        raise ValueError(
            f"Failed to load eval set. Found {errors} malformed line(s). "
            "Please fix the JSONL file."
        )
        
    logger.info(f"✅ Successfully loaded and validated {len(examples)} evaluation examples.")
    return examples


if __name__ == "__main__":
    # Test the loader on the placeholder file we just created
    target_path = Path(__file__).parent.parent.parent / "data" / "eval" / "eval_set.jsonl"
    
    try:
        load_eval_set(target_path)
    except Exception as exc:
        logger.error(exc)
        sys.exit(1)
