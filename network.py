# src/mera/network.py
"""
MERA (Multi-scale Entanglement Renormalization Ansatz) Network
3-site binary MERA implementation with JAX PyTree support
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


class TensorType(Enum):
    """MERAテンソルのタイプ"""
    DISENTANGLER = "disentangler"
    ISOMETRY = "isometry"
    PROJECTOR = "projector"
    TOP = "top"


class BoundaryCondition(Enum):
    """境界条件のタイプ"""
    PERIODIC = "periodic"
    OPEN = "open"


@dataclass
class TensorConstraint:
    """テンソルの制約条件を定義"""
    tensor_type: TensorType
    input_dims: Tuple[int, ...]
    output_dims: Tuple[int, ...]
    
    def check_constraint(self, tensor: jnp.ndarray, tolerance: float = 1e-6) -> float:
        """制約違反量を計算"""
        if self.tensor_type == TensorType.DISENTANGLER:
            # ディスエンタングラー: U†U = I
            shape = tensor.shape
            n_in = shape[0] * shape[1]
            n_out = shape[2] * shape[3]
            
            u_mat = tensor.reshape(n_in, n_out)
            identity = jnp.eye(n_in)
            violation = jnp.linalg.norm(u_mat @ u_mat.T - identity)
            return violation
            
        elif self.tensor_type == TensorType.ISOMETRY:
            # アイソメトリ: W†W = I (3サイトMERAでは W†W = 1)
            # 3次元テンソルを1次元ベクトルとして扱う
            w_vec = tensor.flatten()
            violation = jnp.abs(jnp.dot(w_vec, w_vec) - 1.0)
            return violation
            
        elif self.tensor_type == TensorType.PROJECTOR:
            # プロジェクター: P†P = I
            shape = tensor.shape
            n_in = jnp.prod(jnp.array(shape[:-1]))
            n_out = shape[-1]
            
            p_mat = tensor.reshape(n_in, n_out)
            should_be_identity = p_mat.T @ p_mat
            identity = jnp.eye(n_out)
            violation = jnp.linalg.norm(should_be_identity - identity)
            return violation
            
        else:  # TOP
            return 0.0


@register_pytree_node_class
@dataclass
class MERALayer:
    """MERA層を表現するクラス（JAX PyTree対応）"""
    tensors: Dict[str, jnp.ndarray]
    constraints: Dict[str, TensorConstraint]
    layer_type: str
    
    def tree_flatten(self) -> Tuple[Tuple[Dict[str, jnp.ndarray]], Tuple[Dict[str, TensorConstraint], str]]:
        """PyTreeのflattenメソッド"""
        children = (self.tensors,)  # 動的な部分（JAX配列）を子とする
        aux_data = (self.constraints, self.layer_type)  # 静的な部分を補助データとする
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[Dict[str, TensorConstraint], str], children: Tuple[Dict[str, jnp.ndarray]]) -> MERALayer:
        """PyTreeのunflattenメソッド"""
        tensors, = children
        constraints, layer_type = aux_data
        return cls(tensors=tensors, constraints=constraints, layer_type=layer_type)
    
    def get_isometries(self) -> List[str]:
        """アイソメトリテンソルのキーのリストを返す"""
        return [name for name, constraint in self.constraints.items() 
                if constraint.tensor_type == TensorType.ISOMETRY]


class MERAParams(NamedTuple):
    """MERAパラメータのコンテナ（PyTree互換）"""
    layers: List[MERALayer]
    boundary_condition: BoundaryCondition


@register_pytree_node_class
@dataclass
class MERANetwork:
    """3サイトbinary MERAネットワーク（JAX PyTree対応）"""
    num_sites: int
    bond_dim: int
    phys_dim: int = 2
    boundary_condition: BoundaryCondition = BoundaryCondition.PERIODIC
    translation_invariant: bool = True  # 並進不変性フラグを追加
    layers: List[MERALayer] = field(default_factory=list)
    
    def tree_flatten(self) -> Tuple[Tuple[List[MERALayer]], Tuple[int, int, int, BoundaryCondition, bool]]:
        """PyTreeのflattenメソッド"""
        children = (self.layers,)
        aux_data = (self.num_sites, self.bond_dim, self.phys_dim, 
                    self.boundary_condition, self.translation_invariant)
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[int, int, int, BoundaryCondition, bool], 
                       children: Tuple[List[MERALayer]]) -> MERANetwork:
        """PyTreeのunflattenメソッド"""
        layers, = children
        num_sites, bond_dim, phys_dim, boundary_condition, translation_invariant = aux_data
        return cls(
            num_sites=num_sites,
            bond_dim=bond_dim,
            phys_dim=phys_dim,
            boundary_condition=boundary_condition,
            translation_invariant=translation_invariant,
            layers=layers
        )
    
    def initialize(self, key: jax.random.PRNGKey) -> MERAParams:
        """ネットワークの初期化"""
        layers = []
        chi = self.bond_dim
        d = self.phys_dim
        
        # 層の数を計算
        num_layers = self._calculate_num_layers()
        
        # 各層を初期化
        for layer_idx in range(num_layers):
            key, subkey = jax.random.split(key)
            
            if layer_idx == 0:
                # 物理層
                layer = self._create_physical_layer(subkey, chi, d)
            elif layer_idx == num_layers - 1:
                # トップ層
                layer = self._create_top_layer(subkey, chi)
            else:
                # バルク層
                layer = self._create_bulk_layer(subkey, chi)
            
            layers.append(layer)
        
        # layersをセット
        self.layers = layers
        
        return MERAParams(layers=layers, boundary_condition=self.boundary_condition)
    
    def _calculate_num_layers(self) -> int:
        """必要な層数を計算"""
        import math
        # 3サイトMERAでは、各層でサイト数が1/3になる
        return max(1, int(math.log(self.num_sites) / math.log(3)) + 1)
    
    def _create_physical_layer(self, key: jax.random.PRNGKey, chi: int, d: int) -> MERALayer:
        """物理層の作成"""
        tensors = {}
        constraints = {}
        
        # プロジェクター: (d, d, d) -> chi
        key, subkey = jax.random.split(key)
        p_tensor = self._initialize_tensor(
            subkey,
            TensorConstraint(TensorType.PROJECTOR, (d, d, d), (chi,))
        )
        tensors['p'] = p_tensor
        constraints['p'] = TensorConstraint(TensorType.PROJECTOR, (d, d, d), (chi,))
        
        # ディスエンタングラー（初期化用）
        key, subkey = jax.random.split(key)
        u_init = self._initialize_tensor(
            subkey,
            TensorConstraint(TensorType.DISENTANGLER, (chi, chi), (chi, chi))
        )
        tensors['u_init'] = u_init
        constraints['u_init'] = TensorConstraint(TensorType.DISENTANGLER, (chi, chi), (chi, chi))
        
        return MERALayer(tensors=tensors, constraints=constraints, layer_type='physical')
    
    def _create_bulk_layer(self, key: jax.random.PRNGKey, chi: int) -> MERALayer:
        """バルク層の作成"""
        tensors = {}
        constraints = {}
        
        # ディスエンタングラー
        key, subkey = jax.random.split(key)
        u_tensor = self._initialize_tensor(
            subkey,
            TensorConstraint(TensorType.DISENTANGLER, (chi, chi), (chi, chi))
        )
        tensors['u'] = u_tensor
        constraints['u'] = TensorConstraint(TensorType.DISENTANGLER, (chi, chi), (chi, chi))
        
        # アイソメトリ
        key, subkey = jax.random.split(key)
        w_tensor = self._initialize_tensor(
            subkey,
            TensorConstraint(TensorType.ISOMETRY, (chi, chi, chi), (chi,))
        )
        tensors['w'] = w_tensor
        constraints['w'] = TensorConstraint(TensorType.ISOMETRY, (chi, chi, chi), (chi,))
        
        return MERALayer(tensors=tensors, constraints=constraints, layer_type='bulk')
    
    def _create_top_layer(self, key: jax.random.PRNGKey, chi: int) -> MERALayer:
        """トップ層の作成"""
        tensors = {}
        constraints = {}
        
        # トップテンソル
        top_tensor = jnp.zeros((chi, chi, chi))
        tensors['top'] = top_tensor
        constraints['top'] = TensorConstraint(TensorType.TOP, (chi, chi, chi), ())
        
        return MERALayer(tensors=tensors, constraints=constraints, layer_type='top')
    
    def _initialize_tensor(self, key: jax.random.PRNGKey, constraint: TensorConstraint) -> jnp.ndarray:
        """制約に従ってテンソルを初期化"""
        if constraint.tensor_type == TensorType.DISENTANGLER:
            # ディスエンタングラー: 直交行列として初期化
            chi = constraint.input_dims[0]
            shape = (chi, chi, chi, chi)
            tensor = jax.random.normal(key, shape) * 0.1
            
            # 単位行列に近い初期化
            n = shape[0] * shape[1]
            eye = jnp.eye(n).reshape(shape)
            tensor = eye + tensor * 0.01
            
            # QR分解で直交化
            tensor_mat = tensor.reshape(n, n)
            q, _ = jnp.linalg.qr(tensor_mat)
            return q.reshape(shape)
            
        elif constraint.tensor_type == TensorType.ISOMETRY:
            # アイソメトリ: W†W = 1 を満たすように初期化
            # 3サイトMERA: (chi, chi, chi) の形状
            chi = constraint.input_dims[0]
            
            # ランダムテンソルを生成して正規化
            tensor = jax.random.normal(key, (chi, chi, chi))
            
            # ベクトルとして正規化（||W||_2 = 1となるように）
            tensor_flat = tensor.flatten()
            norm = jnp.linalg.norm(tensor_flat)
            tensor_normalized = tensor_flat / norm
            
            return tensor_normalized.reshape(chi, chi, chi)
            
        elif constraint.tensor_type == TensorType.PROJECTOR:
            # プロジェクター: P†P = I を満たすように初期化
            d = constraint.input_dims[0]
            chi = constraint.output_dims[0]
            n_in = d * d * d
            
            # ランダム行列を生成
            tensor_mat = jax.random.normal(key, (n_in, chi))
            
            # QR分解で正規直交化
            q, _ = jnp.linalg.qr(tensor_mat)
            
            return q[:, :chi].reshape(d, d, d, chi)
            
        else:  # TOP
            chi = constraint.input_dims[0]
            return jnp.zeros((chi, chi, chi))
    
    def get_layer(self, layer_idx: int) -> Optional[MERALayer]:
        """指定インデックスの層を取得"""
        if 0 <= layer_idx < len(self.layers):
            return self.layers[layer_idx]
        return None
    
    def get_layer_count(self) -> int:
        """層の総数を返す"""
        return len(self.layers)
    
    def get_tensors_for_superoperator(
        self, layer_idx: int, position: int = 0
    ) -> Tuple[Optional[jnp.ndarray], ...]:
        """スーパーオペレーター計算用のテンソルを取得"""
        layer = self.get_layer(layer_idx)
        if layer is None:
            return (None, None, None)
        
        if layer.layer_type == 'physical':
            return (None, None, None)
        elif layer.layer_type == 'bulk':
            u = layer.tensors.get('u')
            w = layer.tensors.get('w')
            return (u, u, w)  # 簡略化: 同じuを2回返す
        else:
            return (None, None, None)
    
    def validate_network(self) -> bool:
        """ネットワーク全体の妥当性を検証"""
        if len(self.layers) == 0:
            return False
        
        # 各層の構造を検証
        for i, layer in enumerate(self.layers):
            if i == 0 and layer.layer_type != 'physical':
                return False
            elif i == len(self.layers) - 1 and layer.layer_type != 'top':
                return False
            elif 0 < i < len(self.layers) - 1 and layer.layer_type != 'bulk':
                return False
        
        return True


# ヘルパー関数
def check_stiefel_constraint(params: MERAParams, tolerance: float = 1e-6) -> Dict[str, float]:
    """全テンソルのStiefel制約違反をチェック"""
    violations = {}
    
    for layer_idx, layer in enumerate(params.layers):
        for tensor_name, tensor in layer.tensors.items():
            if tensor_name in layer.constraints:
                constraint = layer.constraints[tensor_name]
                violation = constraint.check_constraint(tensor, tolerance)
                
                if violation > tolerance:
                    key = f"layer_{layer_idx}_{layer.layer_type}/{tensor_name}"
                    violations[key] = violation
    
    return violations


def get_max_constraint_violation(params: MERAParams, tolerance: float = 1e-6) -> float:
    """最大の制約違反量を取得"""
    violations = check_stiefel_constraint(params, tolerance)
    if violations:
        return max(violations.values())
    return 0.0


def validate_all_constraints(params: MERAParams, tolerance: float = 1e-6) -> Tuple[bool, Dict[str, float]]:
    """全ての制約を検証"""
    violations = check_stiefel_constraint(params, tolerance)
    is_valid = len(violations) == 0
    return is_valid, violations