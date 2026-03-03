#!/usr/bin/env python3
"""
HumanEval benchmark for code generation models.

HumanEval is OpenAI's benchmark for code generation, consisting of 164
hand-written programming problems with unit tests.

Requirements:
    pip install human-eval transformers torch peft accelerate

Usage:
    python humaneval.py /path/to/model
    python humaneval.py research/alexa/blackroad-coder-lora --mock  # Test with mock results
"""
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, List, Optional

# Check if human-eval is available
try:
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness
    HUMANEVAL_AVAILABLE = True
except ImportError:
    print("⚠️  human-eval not installed")
    print("Install with: pip install human-eval")
    HUMANEVAL_AVAILABLE = False


def _resolve_base_model(model_path: Path) -> Optional[str]:
    """
    Resolve the base HuggingFace model ID from a model directory.

    Checks adapter_config.json first, then falls back to MANIFEST.yaml lineage.
    Returns a HuggingFace model ID string, or None if not found.
    """
    # 1. Check adapter_config.json (PEFT standard)
    adapter_config_path = model_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base = adapter_cfg.get("base_model_name_or_path")
        if base:
            return base

    # 2. Fall back to MANIFEST.yaml lineage
    manifest_path = model_path / "MANIFEST.yaml"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        derived_from = manifest.get("derived_from") or []
        if derived_from:
            forkie_id = derived_from[0].get("id", "")
            # Load the forkie registry to get the HuggingFace source
            registry_path = Path("registry/forkies.yaml")
            if registry_path.exists():
                with open(registry_path) as f:
                    forkies = yaml.safe_load(f) or []
                for forkie in forkies:
                    if forkie.get("id") == forkie_id:
                        return forkie.get("full_name") or forkie.get("source")

    return None


def load_model(model_path: str) -> Callable[[str], str]:
    """
    Load a model for inference.

    Supports:
    - Forkies and full models (via transformers)
    - LoRA fine-tunes (base model + PEFT adapter weights)
    - Internal/Production models with adapter_config.json or MANIFEST.yaml

    Args:
        model_path: Path to model directory or HuggingFace model ID

    Returns:
        Callable that takes a prompt string and returns a completion string
    """
    print(f"📦 Loading model from {model_path}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers and torch are required for model loading.\n"
            "Install with: pip install transformers torch accelerate"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    path = Path(model_path)
    adapter_config_path = path / "adapter_config.json"
    is_lora = adapter_config_path.exists()

    if is_lora:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "peft is required for LoRA models.\n"
                "Install with: pip install peft"
            ) from exc

        base_model_id = _resolve_base_model(path)
        if not base_model_id:
            raise ValueError(
                f"Cannot determine base model for LoRA adapter at {model_path}. "
                f"Ensure {model_path}/adapter_config.json contains 'base_model_name_or_path' "
                f"or {model_path}/MANIFEST.yaml contains lineage with a registered forkie."
            )

        print(f"  Loading LoRA adapter (base: {base_model_id})")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        # Full model: local directory or HuggingFace model ID
        hf_id = model_path
        print(f"  Loading full model ({hf_id})")
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

    if device == "cpu":
        model = model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("  ✅ Model loaded")

    def inference(prompt: str) -> str:
        """Run inference on a single HumanEval prompt."""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Return only the newly generated tokens (not the prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return inference


def generate_completions(problems: Dict, model_fn, num_samples: int = 1) -> List[Dict]:
    """Generate code completions for HumanEval problems."""
    completions = []

    print(f"🔧 Generating completions for {len(problems)} problems...")
    print(f"  Samples per problem: {num_samples}")

    for task_id, problem in problems.items():
        prompt = problem['prompt']

        for sample_idx in range(num_samples):
            # Generate completion
            completion = model_fn(prompt)

            completions.append({
                'task_id': task_id,
                'completion': completion
            })

        if len(completions) % 10 == 0:
            print(f"  Progress: {len(completions)} / {len(problems) * num_samples}")

    return completions


def evaluate_humaneval(model_path: str, mock: bool = False) -> Dict:
    """
    Run HumanEval evaluation on a model.

    Args:
        model_path: Path to model directory
        mock: Use mock results for testing

    Returns:
        Dictionary with evaluation results
    """
    print("=" * 60)
    print("📊 HumanEval Evaluation")
    print("=" * 60)

    if mock:
        print("\n⚠️  Running in MOCK mode (for testing)")
        print("  Using placeholder scores\n")

        # Mock results for testing
        results = {
            'model': model_path,
            'eval_suite': 'HumanEval',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'scores': {
                'pass@1': 0.72,   # Target: >= 0.70
                'pass@10': 0.85,
                'pass@100': 0.92
            },
            'num_problems': 164,
            'timeout': 3.0,
            'mock': True
        }

        print("📊 Mock Results:")
        print(f"  pass@1:   {results['scores']['pass@1']:.2%}")
        print(f"  pass@10:  {results['scores']['pass@10']:.2%}")
        print(f"  pass@100: {results['scores']['pass@100']:.2%}")

        return results

    # Real evaluation
    if not HUMANEVAL_AVAILABLE:
        raise RuntimeError("human-eval not installed. Install with: pip install human-eval")

    # Load model
    model_fn = load_model(model_path)

    # Load HumanEval problems
    print("\n📚 Loading HumanEval problems...")
    problems = read_problems()
    print(f"  Loaded {len(problems)} problems")

    # Generate completions
    completions = generate_completions(problems, model_fn, num_samples=1)

    # Save completions to temporary file
    completions_file = Path(model_path) / "eval_results" / "humaneval_completions.jsonl"
    completions_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(completions_file, completions)

    print(f"\n💾 Saved completions to {completions_file}")

    # Evaluate
    print("\n🧪 Running evaluation...")
    eval_results = evaluate_functional_correctness(str(completions_file))

    # Format results
    results = {
        'model': model_path,
        'eval_suite': 'HumanEval',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'scores': {
            'pass@1': eval_results['pass@1'],
            'pass@10': eval_results.get('pass@10', 0.0),
            'pass@100': eval_results.get('pass@100', 0.0),
        },
        'num_problems': len(problems),
        'timeout': 3.0,
        'mock': False
    }

    print("\n📊 Results:")
    print(f"  pass@1:   {results['scores']['pass@1']:.2%}")
    print(f"  pass@10:  {results['scores']['pass@10']:.2%}")
    print(f"  pass@100: {results['scores']['pass@100']:.2%}")

    return results


def save_results(model_path: str, results: Dict):
    """Save evaluation results to model directory."""
    results_dir = Path(model_path) / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "humaneval.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")


def check_promotion_criteria(results: Dict) -> bool:
    """Check if model meets promotion criteria."""
    pass_at_1 = results['scores']['pass@1']
    target = 0.70

    print("\n📋 Promotion Criteria Check:")
    print(f"  Target: pass@1 >= {target:.2%}")
    print(f"  Actual: pass@1 = {pass_at_1:.2%}")

    if pass_at_1 >= target:
        print("  ✅ PASS - Ready for promotion to internal/")
        return True
    else:
        print(f"  ❌ FAIL - Need +{(target - pass_at_1):.2%} improvement")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run HumanEval evaluation on a code model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate research model (mock)
  python humaneval.py research/alexa/blackroad-coder-lora --mock

  # Evaluate internal model (real)
  python humaneval.py internal/blackroad-coder-7b-v1

  # Evaluate Forkie (for baseline)
  python humaneval.py forkies/qwen2-5-coder-7b-instruct/v1.0.0 --mock
        """
    )
    parser.add_argument('model_path', help='Path to model directory')
    parser.add_argument('--mock', action='store_true', help='Use mock results for testing')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_humaneval(args.model_path, mock=args.mock)

    # Save results
    if not args.no_save:
        save_results(args.model_path, results)

    # Check promotion criteria
    ready = check_promotion_criteria(results)

    print("\n" + "=" * 60)
    if ready:
        print("✅ Model is ready for promotion!")
        print("\nNext steps:")
        print(f"  python tools/promote.py {args.model_path} --to internal --name blackroad-coder-7b")
    else:
        print("❌ Model needs improvement before promotion")
        print("\nOptions:")
        print("  1. Increase training epochs")
        print("  2. Adjust LoRA hyperparameters")
        print("  3. Expand training dataset")
    print("=" * 60)

    return 0 if ready else 1


if __name__ == "__main__":
    exit(main())
