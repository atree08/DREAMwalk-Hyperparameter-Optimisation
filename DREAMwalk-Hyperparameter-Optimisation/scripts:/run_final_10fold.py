"""
Final 10-Fold Cross-Validation with Optimized DREAMwalk Parameters
"""
import os
import pickle
import random
import subprocess
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier

# Optimized parameters from Bayesian optimization
OPTIMIZED_DREAMWALK_PARAMS = {
    'num_walks': 200,
    'walk_length': 10,
    'tp_factor': 0.007547319316293211,
    'p': 0.5,
    'q': 0.5,
    'dimension': 256,
    'window_size': 8
}

# Fixed XGBoost configuration (from baseline)
FIXED_XGB_PARAMS = {
    'max_depth': 2,
    'learning_rate': 0.007,
    'n_estimators': 80,
    'min_child_weight': 12,
    'gamma': 6,
    'subsample': 0.57,
    'colsample_bytree': 0.57,
    'reg_alpha': 1.5,
    'reg_lambda': 6.0,
    'seed': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss'
}

def create_balanced_dataset(pairs_file, output_file, seed=42):
    """Create balanced dataset"""
    random.seed(seed)
    
    positive_pairs = []
    all_drugs = set()
    all_diseases = set()
    
    with open(pairs_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                all_drugs.add(parts[0])
                all_diseases.add(parts[1])
                if parts[2] == '1':
                    positive_pairs.append((parts[0], parts[1]))
    
    print(f"Total positive pairs: {len(positive_pairs)}")
    
    # Generate harder negatives
    all_positives = set(positive_pairs)
    negative_pairs = set()
    
    connected_drugs = set([d for d, _ in positive_pairs])
    connected_diseases = set([dis for _, dis in positive_pairs])
    
    attempts = 0
    while len(negative_pairs) < len(positive_pairs) and attempts < len(positive_pairs) * 100:
        drug = random.choice(list(connected_drugs))
        disease = random.choice(list(connected_diseases))
        if (drug, disease) not in all_positives:
            negative_pairs.add((drug, disease))
        attempts += 1
    
    if len(negative_pairs) < len(positive_pairs):
        while len(negative_pairs) < len(positive_pairs):
            drug = random.choice(list(all_drugs))
            disease = random.choice(list(all_diseases))
            if (drug, disease) not in all_positives:
                negative_pairs.add((drug, disease))
    
    # Write balanced dataset
    all_pairs = []
    for drug, disease in positive_pairs:
        all_pairs.append((drug, disease, 1))
    for drug, disease in negative_pairs:
        all_pairs.append((drug, disease, 0))
    
    random.shuffle(all_pairs)
    
    with open(output_file, 'w') as f:
        for drug, disease, label in all_pairs:
            f.write(f"{drug}\t{disease}\t{label}\n")
    
    print(f"✅ Created {output_file}: {len(positive_pairs)} pos + {len(negative_pairs)} neg")
    return all_pairs

def create_fold_network(backbone_file, test_pairs, output_file):
    """Remove test pairs from backbone"""
    test_edges = set()
    for drug, disease, label in test_pairs:
        if label == 1:
            test_edges.add(tuple(sorted((drug, disease))))
    
    removed = 0
    with open(backbone_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            edge = tuple(sorted((parts[0], parts[1])))
            if edge in test_edges:
                removed += 1
                continue
            fout.write(line)
    
    print(f"  Removed {removed} test edges")
    return removed

def clean_similarity_network(backbone_file, sim_file, output_file):
    """Clean similarity network"""
    valid_nodes = set()
    with open(backbone_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                valid_nodes.add(parts[0])
                valid_nodes.add(parts[1])
    
    kept = 0
    with open(sim_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] in valid_nodes and parts[1] in valid_nodes:
                fout.write(line)
                kept += 1
    
    print(f"  Kept {kept} similarity edges")

def generate_embeddings(network_file, sim_file, output_file, params, seed=42):
    """Generate embeddings with optimized parameters"""
    cmd = [
        'python3', 'DREAMwalk/generate_embeddings.py',
        '--network_file', network_file,
        '--sim_network_file', sim_file,
        '--output_file', output_file,
        '--num_walks', str(params['num_walks']),
        '--walk_length', str(params['walk_length']),
        '--tp_factor', str(params['tp_factor']),
        '--dimension', str(params['dimension']),
        '--window_size', str(params['window_size']),
        '--p', str(params['p']),
        '--q', str(params['q']),
        '--seed', str(seed)
    ]
    
    print(f"  Generating embeddings...")
    result = subprocess.run(cmd, capture_output=True, text=True,
                          env={**os.environ, 'PYTHONPATH': os.getcwd()})
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    return True

def evaluate_fold(embedding_file, train_pairs, test_pairs, seed=42):
    """Evaluate on one fold"""
    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    X_train, y_train = [], []
    for drug, disease, label in train_pairs:
        if drug in embeddings and disease in embeddings:
            X_train.append(embeddings[drug] - embeddings[disease])
            y_train.append(label)
    
    X_test, y_test = [], []
    for drug, disease, label in test_pairs:
        if drug in embeddings and disease in embeddings:
            X_test.append(embeddings[drug] - embeddings[disease])
            y_test.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"  Train: {len(X_train)} pairs, Test: {len(X_test)} pairs")
    
    clf = XGBClassifier(**FIXED_XGB_PARAMS)
    clf.fit(X_train, y_train, verbose=False)
    
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_proba)
    aupr = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return acc, auroc, aupr, f1

def run_final_10fold_cv(
    backbone_file='combined_backbone_04.txt',
    sim_file='combined_sim_04.txt',
    pairs_file='final_pairs.txt',
    seed=42
):
    """Run final 10-fold CV with optimized parameters"""
    print("="*70)
    print("FINAL 10-FOLD CROSS-VALIDATION WITH OPTIMIZED PARAMETERS")
    print("="*70)
    print("\nOptimized DREAMwalk Parameters:")
    for key, val in OPTIMIZED_DREAMWALK_PARAMS.items():
        print(f"  {key}: {val}")
    print()
    
    # Create balanced dataset
    if not os.path.exists('balanced_pairs.txt'):
        print("Creating balanced dataset...")
        all_pairs = create_balanced_dataset(pairs_file, 'balanced_pairs.txt', seed)
    else:
        print("Loading existing balanced dataset...")
        all_pairs = []
        with open('balanced_pairs.txt', 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    all_pairs.append((parts[0], parts[1], int(parts[2])))
    
    pairs_data = [(d, dis) for d, dis, _ in all_pairs]
    labels = [l for _, _, l in all_pairs]
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    
    results = {
        'accuracy': [],
        'auroc': [],
        'aupr': [],
        'f1': []
    }
    
    print("\n" + "="*70)
    print("RUNNING 10-FOLD CROSS-VALIDATION")
    print("="*70)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(pairs_data, labels), 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx}/10")
        print(f"{'='*70}")
        
        train_pairs = [all_pairs[i] for i in train_idx]
        test_pairs = [all_pairs[i] for i in test_idx]
        
        fold_backbone = f'final_fold_{fold_idx}_backbone.txt'
        fold_sim = f'final_fold_{fold_idx}_sim.txt'
        fold_embeddings = f'final_fold_{fold_idx}_embeddings.vec'
        
        print(f"Creating fold {fold_idx} network...")
        create_fold_network(backbone_file, test_pairs, fold_backbone)
        clean_similarity_network(fold_backbone, sim_file, fold_sim)
        
        if not generate_embeddings(fold_backbone, fold_sim, fold_embeddings, 
                                   OPTIMIZED_DREAMWALK_PARAMS, seed):
            print(f"FAILED fold {fold_idx}")
            continue
        
        print(f"Evaluating fold {fold_idx}...")
        acc, auroc, aupr, f1 = evaluate_fold(fold_embeddings, train_pairs, test_pairs, seed)
        
        results['accuracy'].append(acc)
        results['auroc'].append(auroc)
        results['aupr'].append(aupr)
        results['f1'].append(f1)
        
        print(f"Fold {fold_idx:2d} | Acc: {acc*100:5.2f}% | AUROC: {auroc:.4f} | AUPR: {aupr:.4f} | F1: {f1:.4f}")
        
        # Cleanup
        for f in [fold_backbone, fold_sim, fold_embeddings]:
            if os.path.exists(f):
                os.remove(f)
    
    # Final results
    print("\n" + "="*70)
    print("FINAL RESULTS - OPTIMIZED CONFIGURATION")
    print("="*70)
    print(f"Accuracy:  {np.mean(results['accuracy'])*100:.2f}% ± {np.std(results['accuracy'])*100:.2f}%")
    print(f"AUROC:     {np.mean(results['auroc']):.4f} ± {np.std(results['auroc']):.4f}")
    print(f"AUPR:      {np.mean(results['aupr']):.4f} ± {np.std(results['aupr']):.4f}")
    print(f"F1-score:  {np.mean(results['f1']):.4f} ± {np.std(results['f1']):.4f}")
    print("="*70)
    
    # Save results
    import json
    with open('final_optimized_results.json', 'w') as f:
        json.dump({
            'parameters': OPTIMIZED_DREAMWALK_PARAMS,
            'results': {k: [float(x) for x in v] for k, v in results.items()},
            'mean': {k: float(np.mean(v)) for k, v in results.items()},
            'std': {k: float(np.std(v)) for k, v in results.items()}
        }, f, indent=2)
    
    print("\n✅ Results saved to: final_optimized_results.json")
    
    return results

if __name__ == '__main__':
    results = run_final_10fold_cv()