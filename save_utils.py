"""
Utilities to save model checkpoints and metadata
"""
import os
import json
from datetime import datetime
from dataclasses import is_dataclass, asdict


def save_full_checkpoint(trainer, tokenizer, output_dir, config=None):
    """Save trainer model (adapters), tokenizer and metadata to output_dir.

    Args:
        trainer: trainer object (should implement save_model(output_dir))
        tokenizer: tokenizer object with save_pretrained
        output_dir: path to save
        config: optional dataclass or dict with configuration to write to metadata
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save adapter / model via trainer if available
    try:
        if hasattr(trainer, "save_model"):
            trainer.save_model(output_dir)
        elif hasattr(trainer, "model") and hasattr(trainer.model, "save_pretrained"):
            trainer.model.save_pretrained(output_dir)
        else:
            # last resort: try to call save_pretrained on trainer (some wrappers)
            if hasattr(trainer, "save_pretrained"):
                trainer.save_pretrained(output_dir)
            else:
                raise AttributeError("Trainer object has no save_model/save_pretrained method")
    except Exception as e:
        # If saving via trainer fails, still attempt to save tokenizer and raise
        try:
            tokenizer.save_pretrained(output_dir)
        except Exception:
            pass
        raise

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Write metadata
    meta = {
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "output_dir": os.path.abspath(output_dir)
    }

    if config is not None:
        try:
            if is_dataclass(config):
                meta["config"] = asdict(config)
            elif isinstance(config, dict):
                meta["config"] = config
            else:
                # best-effort conversion
                meta["config"] = str(config)
        except Exception:
            meta["config"] = str(config)

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return output_dir
