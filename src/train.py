"""Main training script for Active Learning with Conformal Prediction."""
import os
import copy
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from data import CIFAR10DataModule
from data import CIFAR100DataModule
from data import CIFAR100NDataModule
from data import CIFAR100CDataModule
from data import CIFAR10CDataModule
from data import STL10DataModule
from data import SVHNDataModule
from data import CUB200DataModule
from models import ResNet18
from strategies import get_strategy
from utils import (
    compute_qhat,
    evaluate_conformal_prediction,
    compute_qhat_aps,
    evaluate_aps,
    compute_qhat_rmcp,
    evaluate_rmcp,
    compute_qhat_rcs,
    evaluate_rcs,
    compute_qhat_classwise,
    evaluate_classwise,
    get_probs,
    get_features_and_probs,
    compute_qhat_feature_cp,
    evaluate_feature_cp,
    train_round,
    eval_acc
)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """Main training function.
    
    Args:
        cfg: Hydra configuration object
    """
    print("=" * 80)
    print(f"Active Learning + Conformal Prediction")
    print(f"Strategy: {cfg.strategy.name.upper()}")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    set_seed(cfg.seed)
    
    # Set device
    device = cfg.trainer.device if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Initialize data module
    print("\nInitializing data...")

    if cfg.data.name == "CIFAR10":
        data_module = CIFAR10DataModule(cfg.data)
    elif cfg.data.name == "CIFAR100":
        data_module = CIFAR100DataModule(cfg.data)
    elif cfg.data.name == "CIFAR100N":
        data_module = CIFAR100NDataModule(cfg.data)
    elif cfg.data.name == "CIFAR100C":
        data_module = CIFAR100CDataModule(cfg.data)
    elif cfg.data.name == "CIFAR10C":
        data_module = CIFAR10CDataModule(cfg.data)
    elif cfg.data.name == "STL10":
        data_module = STL10DataModule(cfg.data)
    elif cfg.data.name == "SVHN":
        data_module = SVHNDataModule(cfg.data)
    elif cfg.data.name == "CUB200":
        data_module = CUB200DataModule(cfg.data)
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.name}")
    
    # Setup active learning splits
    labeled_idx, calib_idx, pool_idx = data_module.setup_active_learning(
        initial_labeled=cfg.data.initial_labeled,
        calibration_size=cfg.data.calibration_size,
        seed=cfg.seed
    )
    
    print(f"Initial labeled: {len(labeled_idx)}")
    print(f"Calibration: {len(calib_idx)}")
    print(f"Pool: {len(pool_idx)}")
    print(f"Total available: {len(labeled_idx) + len(pool_idx)}")
    
    
    # Initialize model
    print(f"\nInitializing {cfg.model.name} model...")
    
    # Use num_classes from data config as source of truth
    # Model config can override only if explicitly specified via command line
    num_classes = cfg.data.num_classes
    
    model = ResNet18(
        num_classes=num_classes,
        pretrained=cfg.model.pretrained
    )
    print(f"Model configured for {num_classes} classes ({cfg.data.name})")


    
    # Initialize acquisition strategy
    print(f"Initializing {cfg.strategy.name} strategy...")
    strategy = get_strategy(cfg.strategy.name)
    
    # Get calibration and test loaders (fixed throughout)
    calib_loader = data_module.get_loader(calib_idx, shuffle=False)
    test_loader = data_module.get_test_loader()
    
    # Results tracking
    results = {
        'strategy': cfg.strategy.name,
        'rounds': [],
        'labeled_sizes': [],
        'accuracies': [],
        'cp_coverage': [],
        'cp_avg_set_size': [],
        'cp_zero_sets': []
    }
    
    samples_trained = 0
    total_available = len(labeled_idx) + len(pool_idx)
    
    print(f"\nStarting active learning loop ({cfg.num_rounds} rounds)...")
    print("=" * 80)
    
    pending_log = None  # Log từ lần select trước, sẽ in sau round hiện tại
    
    for round_idx in range(cfg.num_rounds + 1):
        # Evaluate model
        acc = eval_acc(model, test_loader, device)
        
        # Use strategy-specific methods
        if cfg.strategy.name in ('cp_aps', 'cp_boundary_uncertainty', 'cp_diversity', 'cp_aps_spm', 'cp_wise'):
            qhat = compute_qhat_aps(model, calib_loader, cfg.cp_alpha, device)
            cp_metrics = evaluate_aps(model, test_loader, qhat, device)
        elif cfg.strategy.name == 'cp_rmcp':
            qhat = compute_qhat_rmcp(model, calib_loader, cfg.cp_alpha, device)
            cp_metrics = evaluate_rmcp(model, test_loader, qhat, device)
        elif cfg.strategy.name in ('cp_rel_margin', 'cp_rcs_mi', 'cp_rcs_v2'):
            qhat = compute_qhat_rcs(model, calib_loader, cfg.cp_alpha, device)
            cp_metrics = evaluate_rcs(model, test_loader, qhat, device)
        elif cfg.strategy.name == 'cp_classwise':
            qhat = compute_qhat_classwise(model, calib_loader, cfg.cp_alpha, device)
            cp_metrics = evaluate_classwise(model, test_loader, qhat, device)
        elif cfg.strategy.name == 'cp_feature_size':
            qhat = compute_qhat_feature_cp(model, calib_loader, cfg.cp_alpha, device)
            cp_metrics = evaluate_feature_cp(model, test_loader, qhat, device)
        else:
            qhat = compute_qhat(model, calib_loader, cfg.cp_alpha, device)
            cp_metrics = evaluate_conformal_prediction(model, test_loader, qhat, device)
        
        # Record results
        results['rounds'].append(round_idx)
        results['labeled_sizes'].append(samples_trained)
        results['accuracies'].append(acc)
        results['cp_coverage'].append(cp_metrics['coverage'])
        results['cp_avg_set_size'].append(cp_metrics['avg_set_size'])
        results['cp_zero_sets'].append(cp_metrics['zero_sets'])
        
        # Print status (no AvgSet/Zero to keep log clean)
        pct = 100 * samples_trained / total_available if samples_trained > 0 else 0
        print(f"Round {round_idx:2d}: "
              f"Acc={acc:5.2f}% | "
              f"Trained={samples_trained:5d} ({pct:4.1f}%) | "
              f"Cov={cp_metrics['coverage']:.3f}")
        
        # Print selection log from previous round (if any)
        if pending_log:
            print(pending_log)
            pending_log = None
        
        # Break before last training round
        if round_idx == cfg.num_rounds:
            break
        
        # Train model
        train_loader = data_module.get_loader(labeled_idx, shuffle=True)
        train_round(
            model=model,
            loader=train_loader,
            epochs=cfg.trainer.init_epochs if round_idx == 0 else cfg.trainer.epochs_per_round,
            lr=cfg.trainer.lr,
            momentum=cfg.trainer.momentum,
            weight_decay=cfg.trainer.weight_decay,
            device=device,
            use_amp=cfg.trainer.get('use_amp', True)  # Default to True for 2-3x speedup
        )
        samples_trained = len(labeled_idx)
        
        # Select new samples (skip last round)
        if round_idx < cfg.num_rounds - 1:
            # Compute qhat on calibration set (use strategy-specific method)
            if cfg.strategy.name in ('cp_aps', 'cp_boundary_uncertainty', 'cp_diversity', 'cp_aps_spm', 'cp_wise'):
                qhat = compute_qhat_aps(model, calib_loader, cfg.cp_alpha, device)
            elif cfg.strategy.name == 'cp_rmcp':
                qhat = compute_qhat_rmcp(model, calib_loader, cfg.cp_alpha, device)
            elif cfg.strategy.name in ('cp_rel_margin', 'cp_rcs_mi', 'cp_rcs_v2'):
                qhat = compute_qhat_rcs(model, calib_loader, cfg.cp_alpha, device)
            elif cfg.strategy.name == 'cp_classwise':
                qhat = compute_qhat_classwise(model, calib_loader, cfg.cp_alpha, device)
            elif cfg.strategy.name == 'cp_feature_size':
                qhat = compute_qhat_feature_cp(model, calib_loader, cfg.cp_alpha, device)
            else:
                qhat = compute_qhat(model, calib_loader, cfg.cp_alpha, device)
            
            # Get probabilities for pool and select samples
            pool_loader = data_module.get_loader(pool_idx, shuffle=False)
            
            if cfg.strategy.name == 'cp_feature_size':
                pool_features, pool_probs, _ = get_features_and_probs(model, pool_loader, device)
                if hasattr(model, 'model') and hasattr(model.model, 'fc'):
                    weights = model.model.fc.weight.data.cpu()
                else:
                    raise ValueError("Could not find weights of final fully connected layer.")
                    
                selected_idx = strategy.select(
                    probs=pool_probs,
                    budget=cfg.data.budget_per_round,
                    qhat=qhat,
                    features=pool_features,
                    weights=weights
                )
            else:
                pool_probs, _ = get_probs(model, pool_loader, device)
                selected_idx = strategy.select(
                    probs=pool_probs,
                    budget=cfg.data.budget_per_round,
                    qhat=qhat
                )
            
            # Update labeled and pool sets
            selected_global = [pool_idx[i] for i in selected_idx.tolist()]
            
            # Build pending log for clean vs corrupt breakdown (printed after next round header)
            if hasattr(data_module, 'corrupted_indices') and data_module.corrupted_indices:
                n_corrupt = sum(1 for i in selected_global if i in data_module.corrupted_indices)
                n_clean = len(selected_global) - n_corrupt
                pending_log = (f"  → Selected: {n_clean} clean + {n_corrupt} corrupted "
                               f"({100*n_corrupt/len(selected_global):.1f}% corrupted)")
            
            labeled_idx.extend(selected_global)
            pool_idx = [i for i in pool_idx if i not in set(selected_global)]
    
    print("=" * 80)
    print("Training complete!")
    print(f"Final accuracy: {results['accuracies'][-1]:.2f}%")
    print(f"Final coverage: {results['cp_coverage'][-1]:.3f}")
    print(f"Final avg set size: {results['cp_avg_set_size'][-1]:.2f}")
    
    # Save results
    if cfg.save_results:
        os.makedirs(cfg.output_dir, exist_ok=True)
        output_path = os.path.join(cfg.output_dir, "results.pt")
        torch.save(results, output_path)
        print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
