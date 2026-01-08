"""
Systematic Hyperparameter Optimization for DREAMwalk
Groups parameters conceptually and analyzes sensitivity
"""
import os
import pickle
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import optuna
from datetime import datetime

# Fixed XGBoost configuration (from validated baseline Run 3)
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

class DREAMwalkOptimizer:
    """
    Systematic optimizer for DREAMwalk hyperparameters
    Groups parameters into conceptual categories
    """
    
    def __init__(self, backbone_file, sim_file, pairs_file, 
                 output_dir='optimization_results', seed=42):
        self.backbone_file = backbone_file
        self.sim_file = sim_file
        self.pairs_file = pairs_file
        self.output_dir = output_dir
        self.seed = seed
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare data once
        self.X, self.y = self._load_data()
        
    def _load_data(self):
        """Load balanced pairs data"""
        print("Loading data...")
        X, y = [], []
        with open(self.pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    X.append((parts[0], parts[1]))  # (drug, disease)
                    y.append(int(parts[2]))
        return np.array(X), np.array(y)
    
    def generate_embeddings(self, params, trial_name):
        """Generate embeddings with given parameters"""
        output_file = f"{self.output_dir}/{trial_name}_embeddings.vec"
        
        cmd = [
            'python3', 'DREAMwalk/generate_embeddings.py',
            '--network_file', self.backbone_file,
            '--sim_network_file', self.sim_file,
            '--output_file', output_file,
            '--num_walks', str(params['num_walks']),
            '--walk_length', str(params['walk_length']),
            '--tp_factor', str(params['tp_factor']),
            '--dimension', str(params['dimension']),
            '--window_size', str(params['window_size']),
            '--p', str(params['p']),
            '--q', str(params['q']),
            '--seed', str(self.seed)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True,
                              env={**os.environ, 'PYTHONPATH': os.getcwd()})
        
        if result.returncode != 0:
            raise RuntimeError(f"Embedding generation failed: {result.stderr}")
        
        return output_file
    
    def evaluate_embeddings(self, embedding_file, n_folds=5):
        """Evaluate embeddings using 5-fold CV"""
        # Load embeddings
        with open(embedding_file, 'rb') as f:
            embeddings = pickle.load(f)
        
        # Prepare features
        X_features, y_labels = [], []
        for (drug, disease), label in zip(self.X, self.y):
            if drug in embeddings and disease in embeddings:
                X_features.append(embeddings[drug] - embeddings[disease])
                y_labels.append(label)
        
        X_features = np.array(X_features)
        y_labels = np.array(y_labels)
        
        # 5-fold cross-validation
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        auroc_scores, aupr_scores = [], []
        
        for train_idx, test_idx in kfold.split(X_features, y_labels):
            X_train, X_test = X_features[train_idx], X_features[test_idx]
            y_train, y_test = y_labels[train_idx], y_labels[test_idx]
            
            clf = XGBClassifier(**FIXED_XGB_PARAMS)
            clf.fit(X_train, y_train, verbose=False)
            
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            auroc_scores.append(roc_auc_score(y_test, y_pred_proba))
            aupr_scores.append(average_precision_score(y_test, y_pred_proba))
        
        # Cleanup
        if os.path.exists(embedding_file):
            os.remove(embedding_file)
        
        return {
            'auroc_mean': np.mean(auroc_scores),
            'auroc_std': np.std(auroc_scores),
            'aupr_mean': np.mean(aupr_scores),
            'aupr_std': np.std(aupr_scores),
            'combined': (np.mean(auroc_scores) + np.mean(aupr_scores)) / 2
        }
    
    def optimize_walk_dynamics(self, n_trials=30):
        """
        Phase 1: Optimize walk dynamics (num_walks, walk_length, tp_factor, p, q)
        Keep embedding capacity fixed at baseline (dimension=128, window_size=4)
        """
        print("\n" + "="*70)
        print("PHASE 1: WALK DYNAMICS OPTIMIZATION")
        print("="*70)
        
        study_name = f"walk_dynamics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        
        def objective(trial):
            params = {
                # Walk dynamics (being optimized)
                'num_walks': trial.suggest_categorical('num_walks', [50, 100, 150, 200]),
                'walk_length': trial.suggest_categorical('walk_length', [4, 6, 8, 10]),
                'tp_factor': trial.suggest_float('tp_factor', 0.0, 1.0),
                'p': trial.suggest_categorical('p', [0.25, 0.5, 1.0, 2.0, 4.0]),
                'q': trial.suggest_categorical('q', [0.25, 0.5, 1.0, 2.0, 4.0]),
                
                # Embedding capacity (fixed at baseline)
                'dimension': 128,
                'window_size': 4
            }
            
            trial_name = f"phase1_trial_{trial.number}"
            print(f"\nTrial {trial.number}: {params}")
            
            try:
                embedding_file = self.generate_embeddings(params, trial_name)
                results = self.evaluate_embeddings(embedding_file)
                
                print(f"  AUROC: {results['auroc_mean']:.4f} ± {results['auroc_std']:.4f}")
                print(f"  AUPR:  {results['aupr_mean']:.4f} ± {results['aupr_std']:.4f}")
                
                # Store additional metrics
                trial.set_user_attr('auroc_mean', results['auroc_mean'])
                trial.set_user_attr('aupr_mean', results['aupr_mean'])
                trial.set_user_attr('auroc_std', results['auroc_std'])
                trial.set_user_attr('aupr_std', results['aupr_std'])
                
                return results['combined']
            except Exception as e:
                print(f"  Trial failed: {e}")
                return 0.0
        
        study.optimize(objective, n_trials=n_trials)
        
        # Save results
        df = study.trials_dataframe()
        df.to_csv(f"{self.output_dir}/phase1_walk_dynamics.csv", index=False)
        
        print("\n" + "="*70)
        print("PHASE 1 COMPLETE")
        print("="*70)
        print(f"Best combined score: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        return study
    
    def optimize_embedding_capacity(self, best_walk_params, n_trials=20):
        """
        Phase 2: Optimize embedding capacity (dimension, window_size)
        Use best walk dynamics from Phase 1
        """
        print("\n" + "="*70)
        print("PHASE 2: EMBEDDING CAPACITY OPTIMIZATION")
        print("="*70)
        
        study_name = f"embedding_capacity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        
        def objective(trial):
            params = {
                # Walk dynamics (fixed from Phase 1)
                **best_walk_params,
                
                # Embedding capacity (being optimized)
                'dimension': trial.suggest_categorical('dimension', [64, 128, 256]),
                'window_size': trial.suggest_categorical('window_size', [2, 4, 8])
            }
            
            trial_name = f"phase2_trial_{trial.number}"
            print(f"\nTrial {trial.number}: dimension={params['dimension']}, window={params['window_size']}")
            
            try:
                embedding_file = self.generate_embeddings(params, trial_name)
                results = self.evaluate_embeddings(embedding_file)
                
                print(f"  AUROC: {results['auroc_mean']:.4f} ± {results['auroc_std']:.4f}")
                print(f"  AUPR:  {results['aupr_mean']:.4f} ± {results['aupr_std']:.4f}")
                
                trial.set_user_attr('auroc_mean', results['auroc_mean'])
                trial.set_user_attr('aupr_mean', results['aupr_mean'])
                trial.set_user_attr('auroc_std', results['auroc_std'])
                trial.set_user_attr('aupr_std', results['aupr_std'])
                
                return results['combined']
            except Exception as e:
                print(f"  Trial failed: {e}")
                return 0.0
        
        study.optimize(objective, n_trials=n_trials)
        
        # Save results
        df = study.trials_dataframe()
        df.to_csv(f"{self.output_dir}/phase2_embedding_capacity.csv", index=False)
        
        print("\n" + "="*70)
        print("PHASE 2 COMPLETE")
        print("="*70)
        print(f"Best combined score: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        return study
    
    def analyze_sensitivity(self):
        """
        Analyze parameter sensitivity from optimization results
        """
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS")
        print("="*70)
        
        # Load Phase 1 results
        df1 = pd.read_csv(f"{self.output_dir}/phase1_walk_dynamics.csv")
        
        print("\nPhase 1: Walk Dynamics")
        print("-" * 50)
        
        # Analyze each parameter
        for param in ['num_walks', 'walk_length', 'tp_factor', 'params_p', 'params_q']:
            if param in df1.columns:
                print(f"\n{param}:")
                if df1[param].dtype in [np.float64, np.int64]:
                    grouped = df1.groupby(param)['value'].agg(['mean', 'std', 'count'])
                    print(grouped)
        
        # Load Phase 2 results if exists
        phase2_file = f"{self.output_dir}/phase2_embedding_capacity.csv"
        if os.path.exists(phase2_file):
            df2 = pd.read_csv(phase2_file)
            
            print("\n\nPhase 2: Embedding Capacity")
            print("-" * 50)
            
            for param in ['params_dimension', 'params_window_size']:
                if param in df2.columns:
                    print(f"\n{param}:")
                    grouped = df2.groupby(param)['value'].agg(['mean', 'std', 'count'])
                    print(grouped)

def main():
    """
    Run systematic hyperparameter optimization
    """
    # Initialize optimizer
    optimizer = DREAMwalkOptimizer(
        backbone_file='combined_backbone_04.txt',
        sim_file='combined_sim_04.txt',
        pairs_file='balanced_pairs.txt',
        output_dir='dreamwalk_optimization',
        seed=42
    )
    
    # Phase 1: Optimize walk dynamics (30 trials ≈ 15-30 hours)
    print("Starting Phase 1: Walk Dynamics Optimization")
    print("This will take approximately 15-30 hours...")
    study1 = optimizer.optimize_walk_dynamics(n_trials=30)
    
    # Phase 2: Optimize embedding capacity with best walk params (20 trials ≈ 10-20 hours)
    print("\nStarting Phase 2: Embedding Capacity Optimization")
    print("This will take approximately 10-20 hours...")
    best_walk_params = {k: v for k, v in study1.best_params.items() 
                       if k in ['num_walks', 'walk_length', 'tp_factor', 'p', 'q']}
    study2 = optimizer.optimize_embedding_capacity(best_walk_params, n_trials=20)
    
    # Sensitivity analysis
    optimizer.analyze_sensitivity()
    
    # Final recommendation
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE - FINAL RECOMMENDATION")
    print("="*70)
    
    final_params = {**best_walk_params, **study2.best_params}
    print("\nOptimized DREAMwalk parameters:")
    for param, value in final_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nExpected performance:")
    print(f"  Combined score: {study2.best_value:.4f}")
    print(f"\nNext step: Run 10-fold CV with these parameters for final evaluation")

if __name__ == '__main__':
    main()