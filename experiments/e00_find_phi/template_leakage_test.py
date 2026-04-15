"""
Template Leakage Test: Is LDA finding truthfulness or template identity?

If we split by TEMPLATE (not by sample), does LDA still work?
If yes -> real truthfulness direction
If no -> template memorization (fake win)
"""
import numpy as np
import torch
import os
import random

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'

from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

rng = random.Random(42)

fact_pool = {
    "geography": [
        ("The capital of France is Paris.", "The capital of France is Lyon."),
        ("The capital of Germany is Berlin.", "The capital of Germany is Munich."),
        ("The capital of Japan is Tokyo.", "The capital of Japan is Osaka."),
        ("The capital of Italy is Rome.", "The capital of Italy is Milan."),
        ("The capital of Spain is Madrid.", "The capital of Spain is Barcelona."),
        ("The capital of Australia is Canberra.", "The capital of Australia is Sydney."),
        ("The capital of Canada is Ottawa.", "The capital of Canada is Toronto."),
        ("The capital of Brazil is Brasilia.", "The capital of Brazil is Rio de Janeiro."),
        ("The capital of Egypt is Cairo.", "The capital of Egypt is Alexandria."),
        ("The capital of India is New Delhi.", "The capital of India is Mumbai."),
        ("The capital of Turkey is Ankara.", "The capital of Turkey is Istanbul."),
        ("The capital of Russia is Moscow.", "The capital of Russia is St. Petersburg."),
        ("The capital of China is Beijing.", "The capital of China is Shanghai."),
        ("The capital of South Korea is Seoul.", "The capital of South Korea is Busan."),
        ("The capital of Mexico is Mexico City.", "The capital of Mexico is Guadalajara."),
        ("The longest river in the world is the Nile.", "The longest river in the world is the Amazon."),
        ("The largest ocean is the Pacific Ocean.", "The largest ocean is the Atlantic Ocean."),
        ("The smallest continent is Australia.", "The smallest continent is Europe."),
        ("Mount Everest is the tallest mountain.", "Mount Kilimanjaro is the tallest mountain."),
        ("The Sahara is the largest hot desert.", "The Gobi is the largest hot desert."),
        ("The Amazon rainforest is in South America.", "The Amazon rainforest is in Africa."),
        ("The Great Barrier Reef is in Australia.", "The Great Barrier Reef is in Indonesia."),
        ("The Dead Sea is the lowest point on land.", "The Caspian Sea is the lowest point on land."),
        ("Iceland is in the North Atlantic Ocean.", "Iceland is in the Arctic Ocean."),
        ("The Andes are in South America.", "The Andes are in Asia."),
    ],
    "science": [
        ("Water boils at 100 degrees Celsius.", "Water boils at 90 degrees Celsius."),
        ("The speed of light is approximately 300,000 km/s.", "The speed of light is approximately 500,000 km/s."),
        ("Oxygen has atomic number 8.", "Oxygen has atomic number 12."),
        ("Gold has the chemical symbol Au.", "Gold has the chemical symbol Gd."),
        ("DNA has a double helix structure.", "DNA has a triple helix structure."),
        ("The Earth orbits the Sun.", "The Sun orbits the Earth."),
        ("Light travels faster than sound.", "Sound travels faster than light."),
        ("Water freezes at 0 degrees Celsius.", "Water freezes at 10 degrees Celsius."),
        ("The chemical formula for water is H2O.", "The chemical formula for water is H3O."),
        ("Diamonds are made of carbon.", "Diamonds are made of silicon."),
        ("Iron has the chemical symbol Fe.", "Iron has the chemical symbol Ir."),
        ("Helium is the second lightest element.", "Helium is the third lightest element."),
        ("The speed of sound in air is about 343 m/s.", "The speed of sound in air is about 1200 m/s."),
        ("Absolute zero is minus 273 degrees Celsius.", "Absolute zero is minus 200 degrees Celsius."),
        ("Pi is approximately 3.14159.", "Pi is approximately 3.17320."),
        ("Gravity on Earth is about 9.8 m/s squared.", "Gravity on Earth is about 12.5 m/s squared."),
        ("The pH of pure water is 7.", "The pH of pure water is 5."),
        ("The Planck constant is fundamental to quantum mechanics.", "The Planck constant is fundamental to thermodynamics."),
        ("Copper is a good conductor of electricity.", "Glass is a good conductor of electricity."),
        ("Photosynthesis produces oxygen.", "Photosynthesis produces nitrogen."),
        ("Einstein developed the theory of relativity.", "Einstein developed the theory of evolution."),
        ("Newton discovered the law of gravity.", "Darwin discovered the law of gravity."),
        ("The periodic table was created by Mendeleev.", "The periodic table was created by Bohr."),
        ("Sound cannot travel through a vacuum.", "Sound can travel through a vacuum."),
        ("Electrons have a negative charge.", "Electrons have a positive charge."),
    ],
}

# Generate data tracking template IDs
prompts = []
labels = []
template_ids = []
n_per_cat = 200

for cat in ["geography", "science"]:
    pool = fact_pool[cat]
    for _ in range(n_per_cat):
        idx = rng.randint(0, len(pool) - 1)
        real, fake = pool[idx]
        prompts.append(real)
        labels.append(0)
        template_ids.append(f"{cat}_{idx}")
        prompts.append(fake)
        labels.append(1)
        template_ids.append(f"{cat}_{idx}")

labels = np.array(labels)
template_ids = np.array(template_ids)

print(f"Loading model...")
import sys
model_name = sys.argv[1] if len(sys.argv) > 1 else 'pythia-410m'
best_layer = int(sys.argv[2]) if len(sys.argv) > 2 else 23
print(f"Model: {model_name}, Layer: {best_layer}")
model = HookedTransformer.from_pretrained(model_name, device='mps',
    dtype=torch.float32 if 'cpu' in str('mps') else torch.float16)

print(f"Extracting activations from {len(prompts)} prompts...")
activations = []
# Smaller batch for larger models to avoid OOM
batch_size = 8 if '1b' in model_name or '1.4b' in model_name else 32
for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i+batch_size]
    _, cache = model.run_with_cache(
        batch,
        names_filter=lambda name: f"blocks.{best_layer}.hook_resid_post" in name,
    )
    act = cache[f"blocks.{best_layer}.hook_resid_post"][:, -1, :].float().cpu()
    activations.append(act)
    del cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if i % 64 == 0:
        print(f"  {i}/{len(prompts)}")

X = torch.cat(activations, dim=0).numpy()
print(f"Data: {X.shape[0]} samples, {X.shape[1]} dims")
print(f"Unique templates: {len(set(template_ids))}")

# Test 1: Standard random split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
scaler = StandardScaler()
clf = LogisticRegression(max_iter=5000, C=1.0)
clf.fit(scaler.fit_transform(X_train), y_train)
standard_acc = clf.score(scaler.transform(X_test), y_test)
print(f"\n1. Standard split (leakage possible):  {standard_acc:.3f}")

# Test 2: Template-aware split
unique_templates = sorted(set(template_ids))
rng_np = np.random.RandomState(42)
rng_np.shuffle(unique_templates)
split = len(unique_templates) // 2
train_templates = set(unique_templates[:split])
test_templates = set(unique_templates[split:])

train_mask = np.array([t in train_templates for t in template_ids])
test_mask = np.array([t in test_templates for t in template_ids])

scaler2 = StandardScaler()
clf2 = LogisticRegression(max_iter=5000, C=1.0)
clf2.fit(scaler2.fit_transform(X[train_mask]), labels[train_mask])
template_acc = clf2.score(scaler2.transform(X[test_mask]), labels[test_mask])
print(f"2. Template-aware split (NO leakage):  {template_acc:.3f}")

# Test 3: Cross-category
geo_mask = np.array([t.startswith("geography") for t in template_ids])
sci_mask = np.array([t.startswith("science") for t in template_ids])

scaler3 = StandardScaler()
clf3 = LogisticRegression(max_iter=5000, C=1.0)
clf3.fit(scaler3.fit_transform(X[geo_mask]), labels[geo_mask])
geo_to_sci = clf3.score(scaler3.transform(X[sci_mask]), labels[sci_mask])
print(f"3. Train geography -> test science:    {geo_to_sci:.3f}")

scaler4 = StandardScaler()
clf4 = LogisticRegression(max_iter=5000, C=1.0)
clf4.fit(scaler4.fit_transform(X[sci_mask]), labels[sci_mask])
sci_to_geo = clf4.score(scaler4.transform(X[geo_mask]), labels[geo_mask])
print(f"4. Train science -> test geography:    {sci_to_geo:.3f}")

# Test 4: Random baseline
rng_np2 = np.random.RandomState(42)
random_accs = []
for _ in range(50):
    y_shuf = labels.copy()
    rng_np2.shuffle(y_shuf)
    Xtr, Xte, ytr, yte = train_test_split(X, y_shuf, test_size=0.3, random_state=rng_np2.randint(10000))
    sc = StandardScaler()
    cr = LogisticRegression(max_iter=2000, C=1.0)
    cr.fit(sc.fit_transform(Xtr), ytr)
    random_accs.append(cr.score(sc.transform(Xte), yte))
print(f"5. Random baseline (shuffled labels):  {np.mean(random_accs):.3f} +/- {np.std(random_accs):.3f}")

print(f"\n{'='*60}")
print("DIAGNOSIS")
print(f"{'='*60}")
if template_acc > 0.7 and geo_to_sci > 0.6 and sci_to_geo > 0.6:
    print("STRONG: Direction is REAL - survives template split AND cross-category")
    print("-> Phi is a genuine truthfulness direction")
elif template_acc > 0.7:
    print("MODERATE: Direction survives template split but not cross-category")
    print("-> Phi captures truthfulness within categories, not universally")
elif standard_acc > 0.8 and template_acc < 0.6:
    print("FAKE WIN: Template leakage confirmed!")
    print("-> LDA memorized template identity, not truthfulness")
else:
    print(f"MIXED: standard={standard_acc:.3f}, template={template_acc:.3f}")
