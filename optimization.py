"""
Riemannian optimization for MERA on Stiefel manifold.
JAXネイティブ実装 - PyTree対応版
"""

from typing import Tuple, Dict, Any, Optional, NamedTuple, Callable, List
import jax
import jax.numpy as jnp
from functools import partial

from .network import MERAParams, MERALayer, TensorType, TensorConstraint

# JAX v0.6.0対応
if hasattr(jax, 'tree'):
    tree_map = jax.tree.map
else:
    tree_map = jax.tree_util.tree_map


class OptimizerState(NamedTuple):
    """オプティマイザの状態"""
    iteration: int
    params: MERAParams
    momentum: Optional[MERAParams] = None  # Adam用
    variance: Optional[MERAParams] = None  # Adam用


class OptimizationConfig(NamedTuple):
    """最適化設定"""
    learning_rate: float = 0.01
    gradient_tolerance: float = 1e-6
    max_iterations: int = 1000
    # Adam specific
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8


class RiemannianOptimizer:
    """Stiefel多様体上のリーマン最適化

    JAXネイティブ実装で、基本的な幾何学的操作を内蔵。
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config

    @staticmethod
    def project_to_stiefel_tangent(X: jnp.ndarray, G: jnp.ndarray) -> jnp.ndarray:
        """Stiefel多様体の接空間への射影

        Args:
            X: Stiefel多様体上の点 (n, p) where X^T X = I_p
            G: ユークリッド勾配 (n, p)

        Returns:
            接空間への射影された勾配
        """
        # Riemannian gradient: G - X(X^T G + G^T X)/2
        XtG = jnp.dot(X.T, G)
        sym_part = (XtG + XtG.T) / 2
        return G - jnp.dot(X, sym_part)

    @staticmethod
    def retract_to_stiefel(X: jnp.ndarray, V: jnp.ndarray, 
                          step_size: float = 1.0) -> jnp.ndarray:
        """Stiefel多様体へのレトラクション（QR分解を使用）

        Args:
            X: 現在の点
            V: 接ベクトル
            step_size: ステップサイズ

        Returns:
            多様体上の新しい点
        """
        Y = X + step_size * V
        Q, R = jnp.linalg.qr(Y)
        # Rの対角要素の符号を保持
        D = jnp.diag(jnp.sign(jnp.diag(R)))
        return jnp.dot(Q, D)

    def _process_tensor_gradient(self, tensor: jnp.ndarray, grad: jnp.ndarray,
                               constraint: TensorConstraint) -> jnp.ndarray:
        """制約に応じて勾配を処理"""

        if constraint.tensor_type == TensorType.ISOMETRY:
            # アイソメトリ制約: 行列形式に変換して処理
            input_size = int(jnp.prod(jnp.array(list(constraint.input_dims))))
            output_size = int(jnp.prod(jnp.array(list(constraint.output_dims))))

            # 3サイトMERAのアイソメトリ (χ,χ,χ) の場合
            if tensor.ndim == 3:
                # 勾配をそのまま返す（正規化は更新時に行う）
                return grad

            # テンソルを行列形式に
            tensor_mat = tensor.reshape(input_size, output_size)
            grad_mat = grad.reshape(input_size, output_size)

            # Stiefel接空間への射影
            riemannian_grad = self.project_to_stiefel_tangent(tensor_mat, grad_mat)

            # 元の形状に戻す
            return riemannian_grad.reshape(tensor.shape)

        elif constraint.tensor_type == TensorType.DISENTANGLER:
            # ディスエンタングラー: ユニタリ制約
            # 簡単のため、現時点では勾配をそのまま返す
            # TODO: 適切なユニタリ制約の実装
            return grad

        else:
            # その他: 制約なし
            return grad

    def compute_riemannian_gradient(self, params: MERAParams, 
                                  euclidean_grads: MERAParams) -> MERAParams:
        """ユークリッド勾配をリーマン勾配に変換

        各テンソルに対して適切な射影を適用。
        PyTree対応版。
        """
        def process_layer(layer: MERALayer, grad_layer: MERALayer) -> MERALayer:
            """層ごとの処理"""
            new_tensors = {}

            for key, tensor in layer.tensors.items():
                # テンソルがJAX配列であることを確認
                if not isinstance(tensor, jnp.ndarray):
                    continue

                if key in grad_layer.tensors and key in layer.constraints:
                    grad = grad_layer.tensors[key]

                    # 勾配もJAX配列であることを確認
                    if not isinstance(grad, jnp.ndarray):
                        new_tensors[key] = jnp.zeros_like(tensor)
                        continue

                    constraint = layer.constraints[key]

                    # 制約に応じた勾配処理
                    riemannian_grad = self._process_tensor_gradient(
                        tensor, grad, constraint
                    )
                    new_tensors[key] = riemannian_grad
                else:
                    # 勾配がない場合はゼロ
                    if isinstance(tensor, jnp.ndarray):
                        new_tensors[key] = jnp.zeros_like(tensor)

            return MERALayer(
                tensors=new_tensors,
                constraints=layer.constraints,
                layer_type=layer.layer_type
            )

        # 全層を処理
        new_layers = [
            process_layer(layer, grad_layer)
            for layer, grad_layer in zip(params.layers, euclidean_grads.layers)
        ]

        return MERAParams(
            layers=new_layers,
            boundary_condition=params.boundary_condition
        )

    def _update_tensor_on_manifold(self, tensor: jnp.ndarray, 
                                 update: jnp.ndarray,
                                 constraint: TensorConstraint,
                                 step_size: float) -> jnp.ndarray:
        """多様体上でテンソルを更新"""

        if constraint.tensor_type == TensorType.ISOMETRY:
            # 3サイトアイソメトリの場合
            if tensor.ndim == 3:
                # 更新して正規化
                new_tensor = tensor - step_size * update
                return new_tensor / jnp.linalg.norm(new_tensor.reshape(-1))

            # 通常のアイソメトリ: Stiefel多様体上のレトラクション
            input_size = int(jnp.prod(jnp.array(list(constraint.input_dims))))
            output_size = int(jnp.prod(jnp.array(list(constraint.output_dims))))

            tensor_mat = tensor.reshape(input_size, output_size)
            update_mat = update.reshape(input_size, output_size)

            # レトラクション
            new_tensor_mat = self.retract_to_stiefel(
                tensor_mat, update_mat, step_size
            )

            return new_tensor_mat.reshape(tensor.shape)

        elif constraint.tensor_type == TensorType.DISENTANGLER:
            # ディスエンタングラー: 簡易更新
            # TODO: 適切なユニタリレトラクション
            new_tensor = tensor - step_size * update
            # 簡易正規化
            return new_tensor / jnp.linalg.norm(new_tensor.reshape(-1))

        else:
            # 制約なし: 通常の更新
            return tensor - step_size * update

    def sgd_step(self, state: OptimizerState, 
                 euclidean_grads: MERAParams) -> OptimizerState:
        """Riemannian SGDステップ"""

        # リーマン勾配を計算
        riemannian_grads = self.compute_riemannian_gradient(
            state.params, euclidean_grads
        )

        # パラメータを更新
        def update_layer(layer: MERALayer, grad_layer: MERALayer) -> MERALayer:
            new_tensors = {}

            for key, tensor in layer.tensors.items():
                if key in grad_layer.tensors:
                    grad = grad_layer.tensors[key]
                    constraint = layer.constraints[key]

                    new_tensor = self._update_tensor_on_manifold(
                        tensor, grad, constraint, self.config.learning_rate
                    )
                    new_tensors[key] = new_tensor
                else:
                    new_tensors[key] = tensor

            return MERALayer(
                tensors=new_tensors,
                constraints=layer.constraints,
                layer_type=layer.layer_type
            )

        new_layers = [
            update_layer(layer, grad_layer)
            for layer, grad_layer in zip(state.params.layers, riemannian_grads.layers)
        ]

        new_params = MERAParams(
            layers=new_layers,
            boundary_condition=state.params.boundary_condition
        )

        return OptimizerState(
            iteration=state.iteration + 1,
            params=new_params
        )

    def adam_step(self, state: OptimizerState,
                  euclidean_grads: MERAParams) -> OptimizerState:
        """Riemannian Adamステップ（PyTree対応版）

        接空間でモーメントを管理し、レトラクションで更新。
        """
        # リーマン勾配を計算
        riemannian_grads = self.compute_riemannian_gradient(
            state.params, euclidean_grads
        )

        # 初回はモーメントを初期化（PyTree対応）
        if state.momentum is None:
            # JAX配列の葉のみにゼロを生成する関数
            def create_zero_like(x):
                if isinstance(x, jnp.ndarray):
                    return jnp.zeros_like(x)
                else:
                    # 非配列要素（BoundaryCondition等）はそのまま返す
                    return x

            # tree_mapを使用
            zero_momentum = tree_map(create_zero_like, state.params)
            zero_variance = tree_map(create_zero_like, state.params)

            state = OptimizerState(
                iteration=state.iteration,
                params=state.params,
                momentum=zero_momentum,
                variance=zero_variance
            )

        beta1, beta2 = self.config.beta1, self.config.beta2

        # モーメントの更新（接空間で）- PyTree対応
        def update_momentum(m, g):
            if isinstance(m, jnp.ndarray) and isinstance(g, jnp.ndarray):
                return beta1 * m + (1 - beta1) * g
            else:
                # 非配列要素はそのまま
                return m

        new_momentum = tree_map(
            update_momentum,
            state.momentum, 
            riemannian_grads
        )

        # 分散の更新（勾配のノルムの二乗）- PyTree対応
        def update_variance(v, g):
            if isinstance(v, jnp.ndarray) and isinstance(g, jnp.ndarray):
                return beta2 * v + (1 - beta2) * g**2
            else:
                # 非配列要素はそのまま
                return v

        new_variance = tree_map(
            update_variance,
            state.variance,
            riemannian_grads
        )

        # バイアス補正
        t = state.iteration + 1
        bias_correction1 = 1 - beta1**t
        bias_correction2 = 1 - beta2**t

        # 適応的学習率
        step_size = self.config.learning_rate * jnp.sqrt(bias_correction2) / bias_correction1

        # パラメータ更新
        def update_layer(layer: MERALayer, m_layer: MERALayer, 
                        v_layer: MERALayer) -> MERALayer:
            new_tensors = {}

            for key, tensor in layer.tensors.items():
                # テンソルがJAX配列であることを確認
                if not isinstance(tensor, jnp.ndarray):
                    new_tensors[key] = tensor  # 非配列はそのまま
                    continue

                if key in m_layer.tensors and key in v_layer.tensors:
                    m = m_layer.tensors[key]
                    v = v_layer.tensors[key]

                    # m, vもJAX配列であることを確認
                    if isinstance(m, jnp.ndarray) and isinstance(v, jnp.ndarray):
                        constraint = layer.constraints[key]

                        # Adamの更新方向
                        update_direction = m / (jnp.sqrt(v) + self.config.epsilon)

                        # 多様体上で更新
                        new_tensor = self._update_tensor_on_manifold(
                            tensor, update_direction, constraint, step_size
                        )
                        new_tensors[key] = new_tensor
                    else:
                        new_tensors[key] = tensor
                else:
                    new_tensors[key] = tensor

            return MERALayer(
                tensors=new_tensors,
                constraints=layer.constraints,
                layer_type=layer.layer_type
            )

        new_layers = [
            update_layer(layer, m_layer, v_layer)
            for layer, m_layer, v_layer in zip(
                state.params.layers, 
                new_momentum.layers,
                new_variance.layers
            )
        ]

        new_params = MERAParams(
            layers=new_layers,
            boundary_condition=state.params.boundary_condition
        )

        return OptimizerState(
            iteration=t,
            params=new_params,
            momentum=new_momentum,
            variance=new_variance
        )

    def compute_gradient_norm(self, grads: MERAParams) -> float:
        """勾配のフロベニウスノルムを計算（PyTree対応）"""
        total_norm_sq = 0.0

        for layer in grads.layers:
            for tensor_name, tensor in layer.tensors.items():
                # JAX配列のみを処理
                if isinstance(tensor, jnp.ndarray):
                    total_norm_sq += jnp.sum(tensor**2)

        return jnp.sqrt(total_norm_sq)


# 便利な関数
def create_optimizer(method: str = "adam", **kwargs) -> RiemannianOptimizer:
    """オプティマイザを作成

    Args:
        method: "sgd" または "adam"
        **kwargs: OptimizationConfigの引数

    Returns:
        設定されたオプティマイザ
    """
    config = OptimizationConfig(**kwargs)
    optimizer = RiemannianOptimizer(config)

    # メソッドに応じた更新関数を設定
    if method == "adam":
        optimizer.step = optimizer.adam_step
    else:  # sgd
        optimizer.step = optimizer.sgd_step

    return optimizer
