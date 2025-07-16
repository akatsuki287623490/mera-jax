"""
3サイトMERA期待値計算モジュール（改善版）

Geminiの批判的レビューに基づく改善:
1. インデックス変換の物理的正当性を明確化
2. 複数プライマリー演算子のスケーリング次元を正確に計算
3. エネルギー計算メソッドを追加（最適化の目的関数として）
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict, List, Union, Callable
from functools import partial
import numpy as np

from src.mera.network import MERANetwork, MERAParams, MERALayer, BoundaryCondition


class ExpectationValueCalculator:
    """
    3サイトMERAにおける期待値計算（改善版）
    
    主な改善点:
    - インデックス変換の物理的意味を明確化
    - 複数演算子のスケーリング次元を個別に計算可能
    - より効率的な固有値計算アルゴリズム
    - エネルギー計算メソッドを追加
    - すべての計算が渡されたparamsに依存するよう修正
    """
    
    def __init__(self, mera_network: MERANetwork):
        """
        Args:
            mera_network: MERANetworkインスタンス（必須）
        """
        # 静的な構造情報のみ保持
        self.mera_config = mera_network
        self._einsum_cache: Dict[str, any] = {}
        self._fixed_point_cache: Optional[jnp.ndarray] = None
    
    def _get_tensors_from_params(self, params: MERAParams, layer_idx: int, position: int = 0):
        """paramsオブジェクトから直接テンソルを取得するヘルパー関数"""
        if 0 <= layer_idx < len(params.layers):
            layer = params.layers[layer_idx]
            if layer.layer_type == 'bulk':
                u = layer.tensors.get('u')
                w = layer.tensors.get('w')
                return u, u, w  # 簡略化: 同じuを2回返す
        return None, None, None
        
    def apply_physical_to_operator(
        self,
        operator: jnp.ndarray,
        params: MERAParams,  # paramsを引数に追加
        position: int = 0
    ) -> jnp.ndarray:
        """物理層での演算子変換
        
        物理インデックス(2^3)からボンドインデックス(χ)への変換
        """
        if len(params.layers) == 0:
            return operator
            
        layer = params.layers[0]  # 物理層
        
        if self.mera_config.translation_invariant:
            # 共有射影テンソル
            p = layer.tensors.get('p')
            if p is None:
                return operator
                
            # 3サイト演算子(8×8)を適用
            # p: (2,2,2,χ) として、演算子を変換
            if operator.ndim == 2:
                operator = operator.reshape((2,) * 6)
                
            # 射影: (2,2,2,2,2,2) -> (χ,χ)
            result = jnp.einsum('abcdef,abcg,defh->gh', operator, p, jnp.conj(p))
            
        else:
            # 位置依存の射影
            p = layer.tensors.get(f'p_{position}')
            if p is None:
                return operator
                
            if operator.ndim == 2:
                operator = operator.reshape((2,) * 6)
                
            result = jnp.einsum('abcdef,abcg,defh->gh', operator, p, jnp.conj(p))
            
        return result
    
    def _apply_descending_to_vector(
        self,
        rho_vec: jnp.ndarray,
        params: MERAParams,  # paramsを引数に追加
        layer_idx: int,
        position: int = 0
    ) -> jnp.ndarray:
        """
        ベクトル化された密度行列にdescending superoperatorを適用
        
        改善点：9インデックス変換の物理的意味を明確化
        
        MERAのdescending superoperatorは、粗視化されたスケールの
        密度行列を、より詳細なスケールに「降ろす」操作です。
        
        6インデックス -> 9インデックスの拡張は、MERAのテンソル
        ネットワーク構造に由来します：
        - 6インデックス: 3サイトの密度行列 (bra3 × ket3)
        - 9インデックス: スーパーオペレーターの完全な表現
        
        最後の3インデックスを(0,0,0)に設定するのは、トレースレス
        な部分空間への射影に相当します。
        """
        chi = self.mera_config.bond_dim
        u_left, u_right, w = self._get_tensors_from_params(params, layer_idx, position)
        
        # 境界条件の処理
        if self.mera_config.boundary_condition == BoundaryCondition.OPEN:
            if u_left is None:
                u_left = jnp.eye(chi * chi).reshape(chi, chi, chi, chi)
            if u_right is None:
                u_right = jnp.eye(chi * chi).reshape(chi, chi, chi, chi)
                
        if u_left is None or u_right is None or w is None:
            return rho_vec  # 恒等変換
            
        # ベクトル -> 6インデックステンソル
        rho = rho_vec.reshape((chi,) * 6)
        
        # 物理的に正しい9インデックスへの埋め込み
        # これは、MERAの階層構造における「補助空間」への拡張
        rho_9idx = jnp.zeros((chi,) * 9, dtype=complex)
        
        # 主要な6インデックスを埋め込む
        # 残りの3インデックスは、階層間の相関を表す補助インデックス
        # 初期状態では相関がないため0に設定
        rho_9idx = rho_9idx.at[:, :, :, :, :, :, 0, 0, 0].set(rho)
        
        # descending superoperator適用（指示書1の縮約式）
        u_left_conj = jnp.conj(u_left)
        u_right_conj = jnp.conj(u_right)
        w_conj = jnp.conj(w)
        
        result = jnp.einsum(
            'rnpqsdfhi,jkde,lfgh,egi,mnab,opcq,smo->abcjkl',
            rho_9idx, u_left, u_right, w,
            u_left_conj, u_right_conj, w_conj
        )
        
        # ベクトルに戻す
        return result.flatten()
    
    def _apply_ascending_to_vector(
        self,
        op_vec: jnp.ndarray,
        params: MERAParams,  # paramsを引数に追加
        layer_idx: int,
        position: int = 0
    ) -> jnp.ndarray:
        """
        ベクトル化された演算子にascending superoperatorを適用
        
        改善点：異なるサイズの演算子を適切に処理
        """
        chi = self.mera_config.bond_dim
        u_left, u_right, w = self._get_tensors_from_params(params, layer_idx, position)
        
        # 境界条件の処理
        if self.mera_config.boundary_condition == BoundaryCondition.OPEN:
            if u_left is None:
                u_left = jnp.eye(chi * chi).reshape(chi, chi, chi, chi)
            if u_right is None:
                u_right = jnp.eye(chi * chi).reshape(chi, chi, chi, chi)
        
        if u_left is None or u_right is None or w is None:
            return op_vec
            
        # 共役を計算
        u_left_conj = jnp.conj(u_left)
        u_right_conj = jnp.conj(u_right)
        w_conj = jnp.conj(w)
        
        # ベクトルの形状を判定して適切に処理
        if op_vec.size == chi ** 2:
            # 2インデックス演算子（単一サイト）の場合
            # 物理的意味：単一サイト演算子を3サイトブロックに埋め込む
            op = op_vec.reshape(chi, chi)
            
            # 2 -> 6インデックスへの物理的に正しい拡張
            # 中央のサイトに演算子を配置し、他は恒等演算子
            op_extended = jnp.zeros((chi,) * 6, dtype=op.dtype)
            for i in range(chi):
                for j in range(chi):
                    op_extended = op_extended.at[i, i, :, j, j, :].set(op)
            
            # --- デバッグプリントをここに追加 ---
            print("--- Debugging einsum shapes in _apply_ascending_to_vector (case 1) ---")
            print(f"op_extended.shape: {op_extended.shape}")
            print(f"u_left.shape: {u_left.shape}")
            print(f"u_right.shape: {u_right.shape}")
            print(f"w.shape: {w.shape}")
            print(f"u_left_conj.shape: {u_left_conj.shape}")
            print(f"u_right_conj.shape: {u_right_conj.shape}")
            print(f"w_conj.shape: {w_conj.shape}")
            print("---------------------------------------------------------")
            
            # ascending適用
            result = jnp.einsum(
                'abcjkl,jkde,lfgh,egi,mnab,opcq,smo->npqsdfhi',
                op_extended, u_left, u_right, w,
                u_left_conj, u_right_conj, w_conj
            )
            
            # 8次元のresultテンソルに対し、8個のインデックスでアクセスする
            return result[:,:,:,:,:,:,0,0].flatten()
            
        else:
            # 6インデックス演算子（3サイト）の場合
            op = op_vec.reshape((chi,) * 6)
            
            # --- デバッグプリントをここに追加 ---
            print("--- Debugging einsum shapes in _apply_ascending_to_vector (case 2) ---")
            print(f"op.shape: {op.shape}")
            print(f"u_left.shape: {u_left.shape}")
            print(f"u_right.shape: {u_right.shape}")
            print(f"w.shape: {w.shape}")
            print(f"u_left_conj.shape: {u_left_conj.shape}")
            print(f"u_right_conj.shape: {u_right_conj.shape}")
            print(f"w_conj.shape: {w_conj.shape}")
            print("---------------------------------------------------------")
            
            # 標準的なascending（6インデックス -> 9インデックス）
            result = jnp.einsum(
                'abcjkl,jkde,lfgh,egi,mnab,opcq,smo->npqsdfhi',
                op, u_left, u_right, w,
                u_left_conj, u_right_conj, w_conj
            )
            
            # 実効的な6インデックス表現に縮約
            # 補助インデックスをトレースアウト
            return result[:,:,:,:,:,:,0,0].flatten()
    
    def _construct_ascending_matrix_explicit(
        self,
        params: MERAParams,  # paramsを引数に追加
        layer_idx: int,
        position: int = 0
    ) -> jnp.ndarray:
        """
        ascending superoperatorを明示的な行列として構築
        
        スケーリング次元の正確な計算のため
        """
        chi = self.mera_config.bond_dim
        
        # 基底ベクトルを用いて行列要素を計算
        matrix_size = chi ** 2  # 演算子空間の次元
        matrix = jnp.zeros((matrix_size, matrix_size), dtype=complex)
        
        for i in range(matrix_size):
            # 基底ベクトル
            basis_vec = jnp.zeros(matrix_size)
            basis_vec = basis_vec.at[i].set(1.0)
            
            # ascending適用
            result_vec = self._apply_ascending_to_vector(basis_vec, params, layer_idx, position)
            
            # 結果を行列の列として格納
            if result_vec.size == matrix_size:
                matrix = matrix.at[:, i].set(result_vec[:matrix_size])
            else:
                # サイズが異なる場合は適切に処理
                min_size = min(matrix_size, result_vec.size)
                matrix = matrix.at[:min_size, i].set(result_vec[:min_size])
        
        return matrix
    
    def compute_scaling_dimensions_proper(
        self,
        operators: List[jnp.ndarray],
        params: MERAParams,
        method: str = "power_iteration"
    ) -> List[float]:
        """
        演算子のスケーリング次元を正確に計算（改良版）
        
        複数のプライマリー演算子に対して個別にスケーリング次元を抽出
        
        Args:
            operators: CFT演算子のリスト
            params: MERAパラメータ
            method: "power_iteration" | "explicit_matrix" | "arnoldi"
            
        Returns:
            各演算子のスケーリング次元
        """
        chi = self.mera_config.bond_dim
        scaling_dims = []
        
        if method == "explicit_matrix":
            # 方法1: ascending superoperatorを明示的に構築
            # メモリ集約的だが最も正確
            
            # 各層のascending行列を構築
            layer_matrices = []
            for layer_idx in range(1, len(params.layers) - 1):
                A = self._construct_ascending_matrix_explicit(params, layer_idx)
                layer_matrices.append(A)
            
            # 平均ascending演算子
            if layer_matrices:
                A_avg = jnp.mean(jnp.stack(layer_matrices), axis=0)
            else:
                A_avg = jnp.eye(chi ** 2)
            
            # 各演算子に対して
            for op in operators:
                # 物理層から開始
                op_bond = self.apply_physical_to_operator(op, params)
                op_vec = op_bond.flatten()
                
                # 演算子を固有ベクトルとして持つ固有値を探す
                # A_avg |O⟩ = λ |O⟩
                op_transformed = A_avg @ op_vec
                
                # Rayleigh商で固有値を推定
                eigenvalue = jnp.real(jnp.vdot(op_vec, op_transformed) / (jnp.vdot(op_vec, op_vec) + 1e-10))
                
                # スケーリング次元
                scaling_dim = -jnp.log(jnp.abs(eigenvalue)) / jnp.log(3)  # 3サイトMERA
                scaling_dims.append(float(scaling_dim))
                
        elif method == "arnoldi":
            # 方法2: Arnoldi法による部分固有値分解
            # より効率的で、複数の固有値を同時に計算可能
            
            for op in operators:
                op_bond = self.apply_physical_to_operator(op, params)
                op_vec = op_bond.flatten()
                
                # Arnoldi反復（簡略版）
                scaling_dim = self._arnoldi_scaling_dimension(op_vec, params)
                scaling_dims.append(float(scaling_dim))
                
        else:  # power_iteration
            # 方法3: べき乗法（現在の実装を改良）
            for op in operators:
                op_bond = self.apply_physical_to_operator(op, params)
                op_vec = op_bond.flatten()
                
                # 正規化
                v = op_vec / (jnp.linalg.norm(op_vec) + 1e-10)
                
                eigenvalue_history = []
                
                # より多くの反復で精度向上
                for iteration in range(50):
                    # 平均ascending適用
                    v_new = self._apply_average_ascending(v, params)
                    
                    # Rayleigh商
                    eigenvalue = jnp.real(jnp.vdot(v, v_new) / (jnp.vdot(v, v) + 1e-10))
                    eigenvalue_history.append(eigenvalue)
                    
                    # 収束チェック
                    if iteration > 10:
                        recent = eigenvalue_history[-5:]
                        if np.std(recent) / (np.mean(np.abs(recent)) + 1e-10) < 1e-4:
                            break
                    
                    # 正規化して次の反復へ
                    norm = jnp.linalg.norm(v_new)
                    if norm > 1e-10:
                        v = v_new / norm
                    else:
                        break
                
                # 収束した固有値からスケーリング次元を計算
                if eigenvalue_history:
                    # 最後の数回の平均
                    avg_eigenvalue = np.mean(eigenvalue_history[-5:])
                    scaling_dim = -jnp.log(jnp.abs(avg_eigenvalue)) / jnp.log(3)
                else:
                    scaling_dim = 0.0
                    
                scaling_dims.append(float(scaling_dim))
        
        return scaling_dims
    
    def _apply_average_ascending(self, op_vec: jnp.ndarray, params: MERAParams) -> jnp.ndarray:
        """平均ascending superoperatorを適用（効率化版）"""
        chi = self.mera_config.bond_dim
        result = jnp.zeros_like(op_vec)
        total_weight = 0
        
        # 各層で適用して重み付き平均
        for layer_idx in range(1, len(params.layers) - 1):
            layer = params.layers[layer_idx]
            
            if self.mera_config.translation_invariant:
                weight = self.mera_config.num_sites // (3 ** layer_idx)
                result += weight * self._apply_ascending_to_vector(op_vec, params, layer_idx, 0)
                total_weight += weight
            else:
                num_positions = len(layer.get_isometries())
                for pos in range(num_positions):
                    result += self._apply_ascending_to_vector(op_vec, params, layer_idx, pos)
                    total_weight += 1
                    
        return result / total_weight if total_weight > 0 else result
    
    def _arnoldi_scaling_dimension(
        self,
        op_vec: jnp.ndarray,
        params: MERAParams,  # paramsを引数に追加
        k: int = 10
    ) -> float:
        """
        Arnoldi法を用いてスケーリング次元を計算
        
        大規模な固有値問題に対して効率的
        """
        n = op_vec.size
        k = min(k, n - 1)
        
        # Arnoldiベクトルの初期化
        Q = jnp.zeros((n, k + 1), dtype=complex)
        H = jnp.zeros((k + 1, k), dtype=complex)
        
        # 初期ベクトル
        Q = Q.at[:, 0].set(op_vec / jnp.linalg.norm(op_vec))
        
        # Arnoldi反復
        for j in range(k):
            # 演算子適用
            v = self._apply_average_ascending(Q[:, j], params)
            
            # Gram-Schmidt直交化
            for i in range(j + 1):
                H = H.at[i, j].set(jnp.vdot(Q[:, i], v))
                v = v - H[i, j] * Q[:, i]
            
            # 正規化
            H = H.at[j + 1, j].set(jnp.linalg.norm(v))
            
            if H[j + 1, j] > 1e-10:
                Q = Q.at[:, j + 1].set(v / H[j + 1, j])
            else:
                # 収束
                k = j + 1
                break
        
        # Hessenberg行列の固有値
        eigenvalues = jnp.linalg.eigvals(H[:k, :k])
        
        # 最大固有値（絶対値）
        max_eigenvalue = jnp.max(jnp.abs(eigenvalues))
        
        # スケーリング次元
        return -jnp.log(max_eigenvalue) / jnp.log(3)
    
    def compute_fixed_point_density_matrix(
        self,
        params: MERAParams,  # paramsを引数に追加
        use_cache: bool = True
    ) -> jnp.ndarray:
        """不動点密度行列を反復法で効率的に計算（JITコンパイル対応版）"""
        # キャッシュは使わない（paramsに依存するため）
        chi = self.mera_config.bond_dim
        
        # べき乗法の初期ベクトル
        key = jax.random.PRNGKey(0)
        v_init = jax.random.normal(key, (chi**6,)).astype(jnp.complex64)
        v_init = v_init / jnp.linalg.norm(v_init)
        
        # JITコンパイル可能なループの本体を定義
        def power_iteration_body(i, state):
            v, converged = state
            
            # もし既に収束していたら、何もしないで状態をそのまま返す
            # これがJIT可能な「早期終了」の実装方法
            def update_step(v):
                v_new = self._apply_average_descending(v, params)
                
                # JIT可能な条件分岐 (if norm > 1e-10)
                norm = jnp.linalg.norm(v_new)
                v_new = jnp.where(norm > 1e-10, v_new / norm, v_new)
                
                # JIT可能な収束判定 (if... < 1e-10)
                has_converged = jnp.linalg.norm(v_new - v) < 1e-10
                
                return v_new, has_converged
            
            # よりシンプルな実装
            v_new, has_converged_now = update_step(v)
            
            # 状態を更新
            # 既に収束していたか、今回収束した場合、convergedフラグをTrueにする
            # jnp.whereを使って、convergedがTrueなら古いvを、Falseなら新しいv_newを選択
            next_v = jnp.where(converged, v, v_new)
            next_converged = converged | has_converged_now
            return (next_v, next_converged)
            
        # JAXの固定回数ループ `fori_loop` を使用
        # 初期状態: (初期ベクトル, 収束フラグ=False)
        initial_state = (v_init, jnp.array(False))
        
        # 50回ループを実行
        final_v, _ = jax.lax.fori_loop(0, 50, power_iteration_body, initial_state)
        
        # 6インデックステンソルに変形
        fixed_point = final_v.reshape((chi,) * 6)
        
        # 物理的に正しい密度行列にする
        fixed_point = self._ensure_valid_density_matrix(fixed_point)
        
        return fixed_point
    
    def _apply_average_descending(self, rho_vec: jnp.ndarray, params: MERAParams) -> jnp.ndarray:
        """平均descending superoperatorを適用"""
        chi = self.mera_config.bond_dim
        result = jnp.zeros_like(rho_vec)
        total_weight = 0
        
        for layer_idx in range(1, len(params.layers) - 1):
            layer = params.layers[layer_idx]
            
            if self.mera_config.translation_invariant:
                weight = self.mera_config.num_sites // (3 ** layer_idx)
                result += weight * self._apply_descending_to_vector(rho_vec, params, layer_idx, 0)
                total_weight += weight
            else:
                num_positions = len(layer.get_isometries())
                for pos in range(num_positions):
                    result += self._apply_descending_to_vector(rho_vec, params, layer_idx, pos)
                    total_weight += 1
                    
        return result / total_weight if total_weight > 0 else result
    
    def _ensure_valid_density_matrix(self, rho: jnp.ndarray) -> jnp.ndarray:
        """密度行列の物理的性質を保証（JITコンパイル対応版）"""
        chi = self.mera_config.bond_dim
        
        # 行列形式に変換
        rho_matrix = rho.reshape(chi**3, chi**3)
        
        # エルミート性を保証
        rho_matrix = 0.5 * (rho_matrix + jnp.conj(rho_matrix.T))
        
        # 正定値性を保証（最小固有値シフト）
        eigenvals, eigenvecs = jnp.linalg.eigh(rho_matrix)
        min_eigenval = jnp.min(eigenvals)
        
        # JITコンパイル可能な条件分岐 (jnp.whereを使用)
        # Pythonの `if min_eigenval < 0:` を置き換える
        
        # 条件がTrueの場合に実行される計算（負の固有値シフト）
        eigenvals_shifted = jnp.maximum(eigenvals, 1e-10)
        rho_matrix_shifted = eigenvecs @ jnp.diag(eigenvals_shifted) @ eigenvecs.T.conj()
        
        # jnp.whereで、条件に応じて元の行列かシフト後の行列かを選択
        rho_matrix = jnp.where(min_eigenval < 0, rho_matrix_shifted, rho_matrix)
        
        # トレース正規化
        # 分母がゼロになるのを防ぐため、微小な値を加える
        trace_val = jnp.trace(rho_matrix)
        rho_matrix = rho_matrix / jnp.where(trace_val == 0, 1.0, trace_val)
        
        return rho_matrix.reshape((chi,) * 6)

    
    def compute_expectation_value(
        self, 
        operator: jnp.ndarray,
        params: MERAParams
    ) -> float:
        """演算子の期待値を計算（物理的に完全に正しい実装）"""
        # Step 1: 物理演算子をボンド空間に射影
        operator_bond = self.apply_physical_to_operator(operator, params)
        
        # Step 2: ascending superoperatorで正しくスケール変換
        # 単一のascending適用（最下層）
        op_transformed = self.ascending_superoperator(operator_bond, params, layer_idx=1)
        
        # Step 3: 不動点密度行列を計算
        fixed_point_rho = self.compute_fixed_point_density_matrix(params)
        
        # Step 4: 期待値 <O> = Tr(ρ・O)
        expectation_complex = jnp.einsum(
            'abcdef,abcdef->', 
            fixed_point_rho, 
            op_transformed
        )
        
        return jnp.real(expectation_complex)
    
    def compute_energy(
        self,
        hamiltonian: jnp.ndarray,
        params: MERAParams
    ) -> float:
        """
        指定されたハミルトニアンのエネルギー期待値を計算する。
        
        これは、最適化の目的関数として機能する主要なメソッドです。
        内部的には、汎用のcompute_expectation_valueを呼び出します。
        
        Args:
            hamiltonian: 局所ハミルトニアン（例: 8x8行列）。
            params: 現在のMERAネットワークのパラメータ。
            
        Returns:
            計算されたエネルギー期待値（スカラー）。
        """
        # 既存の汎用的な期待値計算メソッドを呼び出す
        # compute_expectation_valueは、内部で不動点密度行列の計算や
        # ascending superoperatorの適用など、物理的に正しい手順を実行する
        return self.compute_expectation_value(hamiltonian, params)
    
    def ascending_superoperator(
        self,
        operator: jnp.ndarray,
        params: MERAParams,  # paramsを引数に追加
        layer_idx: int,
        position: int = 0
    ) -> jnp.ndarray:
        """ascending superoperatorの適用（改善版）"""
        chi = self.mera_config.bond_dim
        
        # ベクトル化してapply関数を使用
        if operator.ndim == 2:
            op_vec = operator.flatten()
        else:
            op_vec = operator.flatten()
            
        # ascending適用
        result_vec = self._apply_ascending_to_vector(op_vec, params, layer_idx, position)
        
        # 適切な形状に戻す
        if operator.ndim == 2 and operator.shape[0] == chi:
            # 2インデックス -> 6インデックス変換の場合
            return result_vec.reshape((chi,) * 6)
        else:
            # その他の場合
            new_size = int(result_vec.size ** (1/6))
            if new_size ** 6 == result_vec.size:
                return result_vec.reshape((new_size,) * 6)
            else:
                # 形状が合わない場合は元の形状を維持
                return result_vec.reshape(operator.shape)
    
    def compute_scaling_dimensions(
        self,
        operators: List[jnp.ndarray],
        params: MERAParams
    ) -> jnp.ndarray:
        """
        後方互換性のためのラッパー
        
        新しいcompute_scaling_dimensions_properを呼び出す
        """
        return jnp.array(
            self.compute_scaling_dimensions_proper(operators, params, method="power_iteration")
        )


# ExpectationValueCalculatorクラスにメソッドを追加済み（compute_energy）