"""
Quick diagnostic: Can the model actually do the task we're testing?
If the model can't distinguish correct from incorrect, Φ cannot exist.

Tests:
1. Arithmetic: Does the model assign higher probability to correct answers?
2. Factual: Does the model know basic facts?
3. Sentiment: Can the model distinguish positive/negative?
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer

def test_arithmetic_capability(model, tokenizer, n_samples=50, seed=42):
    """Check if model assigns higher logprob to correct vs incorrect completions."""
    rng = np.random.RandomState(seed)
    
    correct_wins = 0
    total = 0
    details = []
    
    for _ in range(n_samples):
        a = rng.randint(2, 50)  # smaller numbers — give model a chance
        b = rng.randint(2, 50)
        op = rng.choice(['+', '-'])
        
        if op == '+':
            correct = a + b
        else:
            correct = a - b
        
        wrong = correct + rng.choice([-3, -2, -1, 1, 2, 3])
        
        prompt = f"{a} {op} {b} ="
        
        # Get logprobs for correct vs wrong answer tokens
        correct_str = f" {correct}"
        wrong_str = f" {wrong}"
        
        correct_tokens = tokenizer.encode(correct_str)
        wrong_tokens = tokenizer.encode(wrong_str)
        
        prompt_tokens = tokenizer.encode(prompt)
        
        with torch.no_grad():
            input_ids = torch.tensor([prompt_tokens], device=model.cfg.device)
            logits = model(input_ids)[:, -1, :]  # logits for next token
            log_probs = torch.log_softmax(logits[0], dim=-1)
            
            # Compare first token of answer
            if len(correct_tokens) > 0 and len(wrong_tokens) > 0:
                correct_logp = log_probs[correct_tokens[0]].item()
                wrong_logp = log_probs[wrong_tokens[0]].item()
                
                if correct_logp > wrong_logp:
                    correct_wins += 1
                total += 1
                details.append({
                    'prompt': prompt,
                    'correct': correct,
                    'wrong': wrong,
                    'correct_logp': correct_logp,
                    'wrong_logp': wrong_logp,
                    'model_right': correct_logp > wrong_logp
                })
    
    acc = correct_wins / max(total, 1)
    return acc, details


def test_factual_capability(model, tokenizer):
    """Check if model has basic factual knowledge."""
    pairs = [
        # (prompt, correct_completion, wrong_completion)
        ("The capital of France is", " Paris", " Madrid"),
        ("The capital of Germany is", " Berlin", " Rome"),
        ("Water freezes at", " 0", " 50"),
        ("The Sun is a", " star", " planet"),
        ("Humans have", " two", " six"),
        ("The Earth orbits the", " Sun", " Moon"),
        ("Dogs are", " animals", " plants"),
        ("Ice is", " cold", " hot"),
        ("Fire is", " hot", " cold"),
        ("The sky is", " blue", " green"),
        ("Shakespeare wrote", " plays", " equations"),
        ("Oxygen is a", " gas", " liquid"),
        ("The largest ocean is the", " Pacific", " Arctic"),
        ("A triangle has", " three", " five"),
        ("The speed of light is", " fast", " slow"),
        ("Cats are", " animals", " minerals"),
        ("The Moon orbits", " Earth", " Mars"),
        ("Blood is", " red", " blue"),
        ("Grass is", " green", " purple"),
        ("Einstein developed the theory of", " relativity", " evolution"),
    ]
    
    correct_wins = 0
    details = []
    
    for prompt, correct, wrong in pairs:
        prompt_tokens = tokenizer.encode(prompt)
        correct_tokens = tokenizer.encode(correct)
        wrong_tokens = tokenizer.encode(wrong)
        
        with torch.no_grad():
            input_ids = torch.tensor([prompt_tokens], device=model.cfg.device)
            logits = model(input_ids)[:, -1, :]
            log_probs = torch.log_softmax(logits[0], dim=-1)
            
            correct_logp = log_probs[correct_tokens[0]].item()
            wrong_logp = log_probs[wrong_tokens[0]].item()
            
            won = correct_logp > wrong_logp
            if won:
                correct_wins += 1
            
            details.append({
                'prompt': prompt,
                'correct': correct.strip(),
                'wrong': wrong.strip(),
                'correct_logp': f"{correct_logp:.3f}",
                'wrong_logp': f"{wrong_logp:.3f}",
                'model_right': won
            })
    
    acc = correct_wins / len(pairs)
    return acc, details


def test_sentiment_capability(model, tokenizer):
    """Check if model can distinguish positive/negative sentiment."""
    pairs = [
        ("This movie was absolutely", " wonderful", " terrible"),
        ("The food tasted", " great", " awful"),
        ("I love this", " product", " disaster"),
        ("The experience was", " amazing", " horrible"),
        ("This is the best", " thing", " worst"),
        ("I really enjoyed", " the", " nothing"),
        ("What a beautiful", " day", " mess"),
        ("This is truly", " great", " bad"),
        ("I feel so", " happy", " sad"),
        ("The weather is", " nice", " terrible"),
    ]
    
    # Also test: does model distinguish positive review continuations?
    pos_prompts = [
        "This restaurant is fantastic. The food was",
        "I absolutely loved this movie. The acting was", 
        "Best purchase I ever made. The quality is",
        "Amazing experience! Everything was",
        "Five stars! The service was",
    ]
    neg_prompts = [
        "This restaurant is terrible. The food was",
        "I absolutely hated this movie. The acting was",
        "Worst purchase I ever made. The quality is",
        "Horrible experience! Everything was",
        "One star! The service was",
    ]
    
    # For sentiment, check if positive context → positive next word
    # and negative context → negative next word
    pos_words = tokenizer.encode(" great")[0]
    neg_words = tokenizer.encode(" terrible")[0]
    
    sentiment_correct = 0
    total = 0
    
    for prompt in pos_prompts:
        tokens = tokenizer.encode(prompt)
        with torch.no_grad():
            logits = model(torch.tensor([tokens], device=model.cfg.device))[:, -1, :]
            lp = torch.log_softmax(logits[0], dim=-1)
            if lp[pos_words] > lp[neg_words]:
                sentiment_correct += 1
            total += 1
    
    for prompt in neg_prompts:
        tokens = tokenizer.encode(prompt)
        with torch.no_grad():
            logits = model(torch.tensor([tokens], device=model.cfg.device))[:, -1, :]
            lp = torch.log_softmax(logits[0], dim=-1)
            if lp[neg_words] > lp[pos_words]:
                sentiment_correct += 1
            total += 1
    
    return sentiment_correct / total, []


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "pythia-410m"
    
    print(f"Loading {model_name}...")
    model = HookedTransformer.from_pretrained(model_name, device="mps")
    tokenizer = model.tokenizer
    
    print(f"\n{'='*60}")
    print(f"MODEL CAPABILITY DIAGNOSTIC: {model_name}")
    print(f"{'='*60}")
    
    # Test 1: Arithmetic
    print(f"\n--- Test 1: Arithmetic (a op b = ?) ---")
    arith_acc, arith_details = test_arithmetic_capability(model, tokenizer)
    print(f"  Accuracy: {arith_acc:.1%}")
    print(f"  (chance = 50%)")
    if arith_acc < 0.6:
        print(f"  ⚠ Model CANNOT do arithmetic — Φ_arithmetic cannot exist")
    elif arith_acc < 0.75:
        print(f"  ~ Model has WEAK arithmetic ability — signal will be noisy")
    else:
        print(f"  ✓ Model CAN do arithmetic — Φ search is meaningful")
    
    # Show some examples
    for d in arith_details[:8]:
        mark = "✓" if d['model_right'] else "✗"
        print(f"    {mark} {d['prompt']} correct={d['correct']} "
              f"(logp={d['correct_logp']:.2f}) vs wrong={d['wrong']} "
              f"(logp={d['wrong_logp']:.2f})")
    
    # Test 2: Factual
    print(f"\n--- Test 2: Factual Knowledge ---")
    fact_acc, fact_details = test_factual_capability(model, tokenizer)
    print(f"  Accuracy: {fact_acc:.1%}")
    if fact_acc < 0.6:
        print(f"  ⚠ Model lacks basic factual knowledge")
    elif fact_acc < 0.8:
        print(f"  ~ Model has partial factual knowledge")
    else:
        print(f"  ✓ Model has solid factual knowledge — Φ_factual search viable")
    
    for d in fact_details:
        mark = "✓" if d['model_right'] else "✗"
        print(f"    {mark} {d['prompt']}... → {d['correct']} "
              f"(logp={d['correct_logp']}) vs {d['wrong']} "
              f"(logp={d['wrong_logp']})")
    
    # Test 3: Sentiment
    print(f"\n--- Test 3: Sentiment Discrimination ---")
    sent_acc, _ = test_sentiment_capability(model, tokenizer)
    print(f"  Accuracy: {sent_acc:.1%}")
    if sent_acc > 0.75:
        print(f"  ✓ Model understands sentiment — good candidate for Φ_sentiment")
    else:
        print(f"  ~ Weak sentiment discrimination")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION")
    print(f"{'='*60}")
    best_task = max([
        ('arithmetic', arith_acc),
        ('factual', fact_acc), 
        ('sentiment', sent_acc)
    ], key=lambda x: x[1])
    
    print(f"  Best capability: {best_task[0]} ({best_task[1]:.1%})")
    
    if best_task[1] < 0.65:
        print(f"  → Model too weak for ANY Φ search. Try larger model.")
    else:
        print(f"  → Run e00 with {best_task[0]} task on this model")
        if best_task[0] == 'factual':
            print(f"  → Need factual correct/incorrect prompt pairs")
        elif best_task[0] == 'sentiment':
            print(f"  → Need sentiment-based Φ definition")
    
    # Also suggest model sizes
    print(f"\n  Model size guidance for 8GB M2:")
    print(f"    pythia-410m  (~1.6GB) ← current")
    print(f"    pythia-1b    (~4.0GB) ← should fit")
    print(f"    pythia-1.4b  (~5.6GB) ← tight but possible")
    print(f"    gpt2-xl      (~6.0GB) ← very tight")
