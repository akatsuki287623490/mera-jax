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
            
# Pythonバージョン確認（Python 3.11.13が表示されるはず）
            if step % 5 == 0:
                val_loss = train_loss * 1.1
                tracker.track_metrics(
                    {'loss': val_loss},
                    step=step,
                    context={'subset': 'validation'}
                )
            
            time.sleep(0.1)  # システムメトリクスのデモ用

# 方法2: デコレータを使用
@run_experiment_with_tracking(
    experiment_name='mera_decorated',
    hparams={'method': 'decorated', 'version': 1}
)
def decorated_experiment(tracker):
    """デコレータで追跡される実験"""
    for i in range(5):
        tracker.track_metrics({'progress': i/5})
        time.sleep(0.1)
    return {'final_result': 'success'}

if __name__ == '__main__':
    print("Example 1: Context Manager")
    example_with_context_manager()
    
    print("\nExample 2: Decorator")
    result = decorated_experiment()
    print(f"Result: {result}")
