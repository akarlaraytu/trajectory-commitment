# Experiment 0: Does Φ Exist?

## The Question
Is there a clean, separable direction in LLM activation space that
corresponds to "hallucination" vs "correct reasoning"?

If YES → TLoT has a foundation. Proceed to projection.
If NO  → TLoT is dead. Rethink everything.

## Method
1. Take a model (Llama-3-8B or Pythia-1.4B for speed)
2. Feed it correct arithmetic: "2+3=5", "7*8=56", etc.
3. Feed it wrong arithmetic: "2+3=7", "7*8=43", etc.
4. Record residual stream activations at each layer
5. Compute contrastive direction: mean(wrong) - mean(correct)
6. Measure: is this direction consistent? Does it generalize?

## Success Criteria
- Cosine similarity between contrastive directions from different
  data splits > 0.7 (the direction is stable, not noise)
- Linear probe accuracy on held-out data > 80% (the direction
  actually separates correct from hallucinated)
- Direction transfers across problem types (addition → multiplication)

## What We Learn
- Which layers contain the strongest signal
- How many dimensions we need (rank of Φ)
- Whether "hallucination" is one direction or many
