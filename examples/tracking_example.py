from src.utils.tracking import ExperimentTracker, run_experiment_with_tracking
import jax
import time

# 方法1: コンテキストマネージャを使用
def example_with_context_manager():
    """withステートメントを使った安全な追跡"""
    hparams = {
        'learning_rate': 0.01,
        'bond_dimension': 4,
        'num_layers': 3
    }
    
    with ExperimentTracker('mera_test', hparams, system_tracking_interval=5) as tracker:
        # 訓練ループ
        for step in range(10):
            # 訓練ステップ（仮想）
            train_loss = 1.0 / (step + 1)
            tracker.track_metrics(
                {'loss': train_loss, 'energy': -train_loss}, 
                step=step,
                context={'subset': 'train'}
            )
            
            # 検証ステップ（5ステップごと）
            if step % 5 == 0:
                val_loss = train_loss * 1.1
  tracker.track_metrics(
                    {'loss': val_loss}, 
                    step=step,
                    context={'subset': 'validation'}
                )
            
            time.sleep(0.1)  # システムメトリクスのデモ用
    
    print("実験完了 - データは自動的に保存されました")


# 方法2: 高階関数パターンを使用
def mera_experiment(tracker):
    """実験のメインロジック"""
    for step in range(10):
        # 何か計算
        energy = -1.0 * (step + 1)
        grad_norm = 10.0 / (step + 1)
        
        tracker.track_metrics({
            'energy': energy,
            'gradient_norm': grad_norm
        }, step=step)
        
        # 収束チェック
        if grad_norm < 0.1:
            tracker.run['converged'] = True
            tracker.run['converged_at_step'] = step
            break
            
    return energy


if __name__ == "__main__":
    print("=== 方法1: コンテキストマネージャの例 ===")
   example_with_context_manager()
    
    print("\n=== 方法2: 高階関数パターンの例 ===")
    final_energy = run_experiment_with_tracking(
        'mera_optimization',
        {'chi': 4, 'learning_rate': 0.1},
        mera_experiment,
        system_tracking_interval=5
    )
    print(f"最終エネルギー: {final_energy}")
