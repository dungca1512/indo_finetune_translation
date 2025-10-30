"""
Inference utilities: load fine-tuned model (adapter or merged) and run a sample translation
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def _build_bnb_config(model_config):
    return BitsAndBytesConfig(
        load_in_4bit=getattr(model_config, "load_in_4bit", True),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, model_config.compute_dtype, torch.float16),
        bnb_4bit_use_double_quant=True,
    )


def load_and_merge(base_model_name, peft_checkpoint_dir, model_config, device_map="auto"):
    """Load base model (quantized) and merge adapter from peft_checkpoint_dir.

    Returns the merged model (HF model ready for inference).
    """
    print(f"Loading base model {base_model_name} with quantization...")
    bnb_config = _build_bnb_config(model_config)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, model_config.compute_dtype, torch.float16),
        device_map=device_map,
        trust_remote_code=getattr(model_config, "trust_remote_code", True),
    )
    base_model.config.use_cache = True

    print("Applying PEFT adapter from checkpoint...")
    model_finetuned = PeftModel.from_pretrained(base_model, peft_checkpoint_dir)

    # Merge adapters into base weights for faster inference and unload adapter
    try:
        merged = model_finetuned.merge_and_unload()
        print("Merged adapter into base model.")
        return merged
    except Exception:
        # If merge_and_unload not available, return PeftModel wrapper
        print("merge_and_unload() failed; returning PeftModel wrapper.")
        return model_finetuned


def run_demo(peft_checkpoint_dir, base_model_name, model_config, test_sentence=None, max_new_tokens=100, temperature=0.7):
    """Run a single-sentence translation demo using the fine-tuned adapter.

    If `peft_checkpoint_dir` points to a merged model (full model saved), `base_model_name`
    can be None and `peft_checkpoint_dir` will be used directly to load the model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If the checkpoint dir contains full model files (config.json, pytorch_model.bin...), load directly
    load_dir = peft_checkpoint_dir
    is_full_model = os.path.exists(os.path.join(peft_checkpoint_dir, "config.json")) and os.path.exists(os.path.join(peft_checkpoint_dir, "pytorch_model.bin"))

    if is_full_model:
        print("Detected full/merged model in checkpoint dir; loading directly.")
        model = AutoModelForCausalLM.from_pretrained(load_dir, device_map="auto", trust_remote_code=True)
    else:
        if base_model_name is None:
            raise ValueError("base_model_name is required when loading adapter-style checkpoint")
        model = load_and_merge(base_model_name, peft_checkpoint_dir, model_config)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name if base_model_name is not None else peft_checkpoint_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare prompt
    if test_sentence is None:
        test_sentence = "This new small language model is very powerful and efficient."

    prompt_template = [
        {"role": "user", "content": f"Dịch câu sau từ tiếng Anh sang tiếng Indonesia: '{test_sentence}'"}
    ]

    # Prefer model-specific apply_chat_template if available
    try:
        prompt_text = tokenizer.apply_chat_template(prompt_template, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback to a simple textual prompt
        prompt_text = f"Dịch câu sau từ tiếng Anh sang tiếng Indonesia: '{test_sentence}'"

    print("Prompt:\n", prompt_text)

    inputs = tokenizer(prompt_text, return_tensors="pt")
    # move tensors to model device if model placed on GPU
    try:
        device_of_model = next(model.parameters()).device
    except StopIteration:
        device_of_model = torch.device(device)

    inputs = {k: v.to(device_of_model) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Try to extract assistant portion (model chat markers vary)
    assistant_response = response
    if "<|im_start|>assistant\n" in response:
        assistant_response = response.split("<|im_start|>assistant\n")[-1]
        assistant_response = assistant_response.replace("<|im_end|>", "").strip()

    print("\n--- Câu gốc (EN):\n", test_sentence)
    print("\n--- Câu dịch (ID):\n", assistant_response)

    return assistant_response
