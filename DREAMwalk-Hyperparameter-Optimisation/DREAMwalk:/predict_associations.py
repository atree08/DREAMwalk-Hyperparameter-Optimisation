import argparse
import pickle
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


from DREAMwalk.utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', type=str, required=True)
    parser.add_argument('--pair_file', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--model_checkpoint', type=str, default='clf.pkl')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    args = {'embeddingf':args.embedding_file,
     'pairf':args.pair_file,
     'seed':args.seed,
     'patience':args.patience,
     'modelf':args.model_checkpoint,
     'testr':args.test_ratio,
     'validr':args.valid_ratio
     }
    return args
    
def split_dataset(pairf, embeddingf, validr, testr, seed):
    # Load embeddings - handle both pickle and text formats
    embedding_dict = {}
    
    try:
        # Try loading as pickle first
        with open(embeddingf, 'rb') as fin:
            embedding_dict = pickle.load(fin)
    except:
        # If that fails, load as text file
        print("Loading embeddings from text format...")
        with open(embeddingf, 'r') as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) < 2: continue
                node_id = parts[0]
                vector = np.array([float(x) for x in parts[1:]])
                embedding_dict[node_id] = vector
        print(f"Loaded {len(embedding_dict)} node embeddings from text file.")
    
    xs, ys = [], []
    with open(pairf, 'r') as fin:
        lines = fin.readlines()
    
    skipped = 0
    for line in lines[1:]:  # Skip header if present
        line = line.strip().split('\t')
        if len(line) < 3: continue
        
        drug = line[0]
        dis = line[1]
        label = line[2]
        
        # Skip pairs where either node is missing from embeddings
        if drug not in embedding_dict or dis not in embedding_dict:
            skipped += 1
            continue
        
        xs.append(embedding_dict[drug] - embedding_dict[dis])
        ys.append(int(label))
    
    if skipped > 0:
        print(f"Warning: Skipped {skipped} pairs due to missing embeddings")
    
    # dataset split
    x, y = {}, {}
    x['train'], x['test'], y['train'], y['test'] = train_test_split(
        xs, ys, test_size=testr, random_state=seed, stratify=ys)
    
    if validr > 0:
        x['train'], x['valid'], y['train'], y['valid'] = train_test_split(
            x['train'], y['train'], test_size=validr/(1-testr), 
            random_state=seed, stratify=y['train'])
    else:
        x['valid'], y['valid'] = [], []
    
    return x, y
        
    # dataset split
    x,y = {},{}
    x['train'],x['test'],y['train'],y['test'] = train_test_split(
        xs,ys,test_size = testr, random_state = seed, stratify = ys)
    if validr > 0:
        x['train'],x['valid'],y['train'],y['valid'] = train_test_split(
            x['train'],y['train'],test_size = validr/(1-testr), 
            random_state = seed, stratify = y['train'])
    else:
        x['valid'],y['valid'] = [],[]

    return x, y

def return_scores(target_list, pred_list):
    metric_list = [
        accuracy_score, 
        roc_auc_score, 
        average_precision_score, 
        f1_score
    ] 
    scores = []
    for metric in metric_list:
        if metric is roc_auc_score:
            # Use only the probability of the positive class (column 1)
            scores.append(metric(target_list, pred_list[:, 1]))
        elif metric is average_precision_score:
            # Use only the probability of the positive class (column 1)
            scores.append(metric(target_list, pred_list[:, 1]))
        elif metric is f1_score:
            scores.append(metric(target_list, np.argmax(pred_list, axis=1), average='weighted'))
        else: # accuracy_score
            scores.append(metric(target_list, np.argmax(pred_list, axis=1))) 
    return scores


def predict_dda(embeddingf:str, pairf:str, modelf:str='clf.pkl', seed:int=42,
                validr:float=0.1, testr:float=0.1):

    set_seed(seed)
    x,y = split_dataset(pairf, embeddingf, validr, testr, seed)
    
    clf = XGBClassifier(base_score = 0.5, booster = 'gbtree',eval_metric ='error',objective = 'binary:logistic',
        gamma = 0,learning_rate = 0.1, max_depth = 6,n_estimators = 500,
        tree_method = 'auto',min_child_weight = 4,subsample = 0.8, colsample_bytree = 0.9,
        scale_pos_weight = 1,max_delta_step = 1,seed = seed)
    
    clf.fit(x['train'], y['train'])
    
    preds = {}
    scores = {}
    for split in ['train','valid','test']:
        preds[split] = clf.predict_proba(np.array(x[split]))
        scores[split] = return_scores(y[split], preds[split])
        print(f'{split.upper():5} set | Acc: {scores[split][0]*100:.2f}% | AUROC: {scores[split][1]:.4f} | AUPR: {scores[split][2]:.4f} | F1-score: {scores[split][3]:.4f}')
    
    with open(modelf,'wb') as fw:
        pickle.dump(clf, fw)
    print(f'saved XGBoost classifier: {modelf}')
    print('='*50)
def predict_dda_10fold(embeddingf, pairf, seed=42):
    """
    10-fold cross-validation for drug-disease association prediction
    """
    from sklearn.model_selection import StratifiedKFold
    
    set_seed(seed)
    
    # Load embeddings
    try:
        with open(embeddingf, 'rb') as fin:
            embedding_dict = pickle.load(fin)
    except:
        # Load text format
        print("Loading embeddings from text format...")
        embedding_dict = {}
        with open(embeddingf, 'r') as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) < 2: continue
                node_id = parts[0]
                vector = np.array([float(x) for x in parts[1:]])
                embedding_dict[node_id] = vector
        print(f"Loaded {len(embedding_dict)} embeddings")
    
    # Load pairs
    xs, ys = [], []
    skipped = 0
    with open(pairf, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            
            drug = parts[0]
            dis = parts[1]
            label = parts[2]
            
            # Skip if embeddings missing
            if drug not in embedding_dict or dis not in embedding_dict:
                skipped += 1
                continue
            
            # Feature: difference of embeddings
            xs.append(embedding_dict[drug] - embedding_dict[dis])
            ys.append(int(label))
    
    if skipped > 0:
        print(f"Skipped {skipped} pairs due to missing embeddings")
    
    xs = np.array(xs)
    ys = np.array(ys)
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(xs)}")
    print(f"Positive: {sum(ys)} ({sum(ys)/len(ys)*100:.1f}%)")
    print(f"Negative: {len(ys)-sum(ys)} ({(len(ys)-sum(ys))/len(ys)*100:.1f}%)")
    
    # Conservative XGBoost parameters
    clf = XGBClassifier(
        base_score=0.5,
        booster='gbtree',
        eval_metric='logloss',
        objective='binary:logistic',
        
        # Strong regularization to prevent overfitting
        max_depth=3,              # Shallow trees
        learning_rate=0.01,       # Slow learning
        n_estimators=100,         # Fewer trees
        min_child_weight=10,      # Require more samples
        gamma=5,                  # Min loss reduction
        subsample=0.6,            # Use 60% of data
        colsample_bytree=0.6,     # Use 60% of features
        reg_alpha=1.0,            # L1 regularization
        reg_lambda=5.0,           # L2 regularization
        
        scale_pos_weight=1,
        seed=seed,
        n_jobs=-1
    )
    
    # 10-fold stratified cross-validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    
    fold_results = {
        'accuracy': [],
        'auroc': [],
        'aupr': [],
        'f1': []
    }
    
    print("\n" + "="*60)
    print("10-Fold Cross-Validation Results")
    print("="*60)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(xs, ys), 1):
        X_train, X_test = xs[train_idx], xs[test_idx]
        y_train, y_test = ys[train_idx], ys[test_idx]
        
        # Train the model
        clf.fit(X_train, y_train, verbose=False)
        
        # Predictions
        y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Probability of positive class
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_pred_proba)
        aupr = average_precision_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        fold_results['accuracy'].append(acc)
        fold_results['auroc'].append(auroc)
        fold_results['aupr'].append(aupr)
        fold_results['f1'].append(f1)
        
        print(f"Fold {fold_idx:2d} | Acc: {acc*100:5.2f}% | AUROC: {auroc:.4f} | AUPR: {aupr:.4f} | F1: {f1:.4f}")
    
    # Calculate mean and std
    print("="*60)
    print("Mean Results (10-fold):")
    print(f"Accuracy:  {np.mean(fold_results['accuracy'])*100:.2f}% ± {np.std(fold_results['accuracy'])*100:.2f}%")
    print(f"AUROC:     {np.mean(fold_results['auroc']):.4f} ± {np.std(fold_results['auroc']):.4f}")
    print(f"AUPR:      {np.mean(fold_results['aupr']):.4f} ± {np.std(fold_results['aupr']):.4f}")
    print(f"F1-score:  {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}")
    print("="*60)
    
    return fold_results        
if __name__ == '__main__':
    args = parse_args()
    
    print("--- 10-Fold Cross-Validation Mode ---")
    print(f"Embedding file: {args['embeddingf']}")
    print(f"Pair file: {args['pairf']}")
    print(f"Random seed: {args['seed']}")
    print("-" * 40)
    
    results = predict_dda_10fold(
        embeddingf=args['embeddingf'],
        pairf=args['pairf'],
        seed=args['seed']
    )
    predict_dda(**args)
