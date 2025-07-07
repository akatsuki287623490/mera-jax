import aim
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable
import functools

class ExperimentTracker:
    """実験追跡を管理するクラス（コンテキストマネージャ対応）"""
    
    def __init__(self, experiment_name: str, hparams: Dict[str, Any], 
                 system_tracking_interval: int = 10):
        """
        Args:
            experiment_name: 実験名
            hparams: ハイパーパラメータ
            system_tracking_interval: システムメトリクス追跡間隔（秒）
        """
        self.experiment_name = experiment_name
        self.hparams = hparams
        self.system_tracking_interval = system_tracking_interval
        self.run = None
        
    def __enter__(self):
        """コンテキストマネージャの開始"""
        self.run = aim.Run(
            experiment=self.experiment_name,
            system_tracking_interval=self.system_tracking_interval
        )
        self.run["hparams"] = self.hparams
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャの終了（自動的にclose）"""
        if self.run:
            self.run.close()
            
    def track_metrics(self, metrics: Dict[str, float], step: int, 
                     context: Optional[Dict[str, Any]] = None):
        """メトリクスの記録（コンテキスト対応）
        
        Args:
            metrics: メトリクス辞書
            step: ステップ数
            context: オプションのコンテキスト（例: {'subset': 'validation'}）
        """
        if not self.run:
            raise RuntimeError("ExperimentTracker must be used within a with statement")
            
        for name, value in metrics.items():
            self.run.track(value, name=name, step=step, context=context)


# 高階関数パターンでの実験実行
def run_experiment_with_tracking(
    experiment_name: str,
    hparams: Dict[str, Any],
    experiment_fn: Callable,
    system_tracking_interval: int = 10
):
    """実験をAim追跡付きで実行する高階関数
    
    Args:
        experiment_name: 実験名
        hparams: ハイパーパラメータ
        experiment_fn: tracker引数を受け取る実験関数
        system_tracking_interval: システムメトリクス追跡間隔
        
    Example:
        def my_experiment(tracker):
            for i in range(100):
                loss = train_step()
                tracker.track_metrics({'loss': loss}, step=i, 
                                    context={'subset': 'train'})
                
        run_experiment_with_tracking('my_exp', {'lr': 0.01}, my_experiment)
    """
    with ExperimentTracker(experiment_name, hparams, system_tracking_interval) as tracker:
        try:
            result = experiment_fn(tracker)
            # 実験成功をメタデータとして記録
            tracker.run['status'] = 'completed'
            return result
        except Exception as e:
            # エラー情報を記録
            tracker.run['status'] = 'failed'
            tracker.run['error'] = str(e)
            raise


# 後方互換性のための関数（非推奨だが移行期間用）
def setup_experiment_tracking(experiment_name: str, hparams: Dict[str, Any]):
    """[非推奨] ExperimentTrackerの使用を推奨"""
    import warnings
    warnings.warn(
        "setup_experiment_tracking is deprecated. Use ExperimentTracker with 'with' statement.",
        DeprecationWarning
    )
    run = aim.Run(experiment=experiment_name)
    run["hparams"] = hparams
    return run


def track_metrics(run, metrics: Dict[str, float], step: int, 
                 context: Optional[Dict[str, Any]] = None):
    """[非推奨] ExperimentTracker.track_metricsの使用を推奨"""
    import warnings
    warnings.warn(
        "track_metrics is deprecated. Use ExperimentTracker.track_metrics.",
        DeprecationWarning
    )
    for name, value in metrics.items():
        run.track(value, name=name, step=step, context=context)
