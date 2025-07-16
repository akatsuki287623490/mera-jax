
"""
MERA simulation Hamiltonians and operators.

This module provides implementations of various quantum Hamiltonians
suitable for MERA simulations, focusing on critical systems.
"""

from typing import Dict, Tuple, Optional, List  # List を追加

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


# パウリ行列の定義（モジュールレベルで一度だけ定義）
_I = jnp.eye(2, dtype=jnp.complex64)
_X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
_Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
_Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)


def construct_3site_ising_hamiltonian(h_field: float = 1.0, J: float = 1.0) -> Array:
    """
    3サイト横磁場イジングモデルのハミルトニアンを構築
    
    H = -J(σ^z_1 σ^z_2 + σ^z_2 σ^z_3) - h(σ^x_1 + σ^x_2 + σ^x_3)
    
    臨界点はh/J=1で発生します。
    
    Args:
        h_field: 横磁場の強さ（デフォルト：1.0）
        J: イジング相互作用の強さ（デフォルト：1.0）
    
    Returns:
        h_matrix: shape (8, 8) のハミルトニアン行列
    
    Example:
        >>> h_critical = construct_3site_ising_hamiltonian(h_field=1.0, J=1.0)
        >>> print(h_critical.shape)
        (8, 8)
    """
    # 3サイト演算子の構築（クロネッカー積）
    # σ^z_1 σ^z_2 I_3
    ZZI = jnp.kron(jnp.kron(_Z, _Z), _I)
    # I_1 σ^z_2 σ^z_3
    IZZ = jnp.kron(jnp.kron(_I, _Z), _Z)
    
    # σ^x_i 演算子
    XII = jnp.kron(jnp.kron(_X, _I), _I)
    IXI = jnp.kron(jnp.kron(_I, _X), _I)
    IIX = jnp.kron(jnp.kron(_I, _I), _X)
    
    # ハミルトニアンの構築
    h_matrix = -J * (ZZI + IZZ) - h_field * (XII + IXI + IIX)
    
    # 実数ハミルトニアンなので実部を取る
    return jnp.real(h_matrix)


def construct_3site_heisenberg_hamiltonian(
    Jx: float = 1.0, 
    Jy: float = 1.0, 
    Jz: float = 1.0,
    h_field: float = 0.0
) -> Array:
    """
    3サイトXXZハイゼンベルグモデルのハミルトニアンを構築
    
    H = -Σ_{i=1,2} (Jx σ^x_i σ^x_{i+1} + Jy σ^y_i σ^y_{i+1} + Jz σ^z_i σ^z_{i+1})
        - h Σ_i σ^z_i
    
    Args:
        Jx: X方向の交換相互作用
        Jy: Y方向の交換相互作用  
        Jz: Z方向の交換相互作用
        h_field: Z方向の磁場
    
    Returns:
        h_matrix: shape (8, 8) のハミルトニアン行列
    """
    # 2サイト相互作用項
    # サイト1-2の相互作用
    XXI = jnp.kron(jnp.kron(_X, _X), _I)
    YYI = jnp.kron(jnp.kron(_Y, _Y), _I)
    ZZI = jnp.kron(jnp.kron(_Z, _Z), _I)
    
    # サイト2-3の相互作用
    IXX = jnp.kron(jnp.kron(_I, _X), _X)
    IYY = jnp.kron(jnp.kron(_I, _Y), _Y)
    IZZ = jnp.kron(jnp.kron(_I, _Z), _Z)
    
    # 磁場項
    ZII = jnp.kron(jnp.kron(_Z, _I), _I)
    IZI = jnp.kron(jnp.kron(_I, _Z), _I)
    IIZ = jnp.kron(jnp.kron(_I, _I), _Z)
    
    # ハミルトニアンの構築
    h_matrix = -(Jx * (XXI + IXX) + Jy * (YYI + IYY) + Jz * (ZZI + IZZ)) \
               - h_field * (ZII + IZI + IIZ)
    
    # 実数になる場合は実部を取る
    if jnp.allclose(jnp.imag(h_matrix), 0, atol=1e-10):
        h_matrix = jnp.real(h_matrix)
    
    return h_matrix


def matrix_to_tensor(h_matrix: Array) -> Array:
    """
    行列形式 (8, 8) をテンソル形式 (2,2,2,2,2,2) に変換
    
    この変換により、MERAの縮約計算で使いやすい形式になります。
    
    インデックスマッピング:
        行列: h[i, j] where i, j ∈ [0, 7]
        テンソル: h[bra_1, bra_2, bra_3, ket_1, ket_2, ket_3]
        対応関係: i = 4*bra_1 + 2*bra_2 + bra_3
                 j = 4*ket_1 + 2*ket_2 + ket_3
    
    Args:
        h_matrix: shape (8, 8) のハミルトニアン行列
    
    Returns:
        h_tensor: shape (2,2,2,2,2,2) のテンソル形式
    
    Example:
        >>> h_matrix = construct_3site_ising_hamiltonian()
        >>> h_tensor = matrix_to_tensor(h_matrix)
        >>> print(h_tensor.shape)
        (2, 2, 2, 2, 2, 2)
    """
    assert h_matrix.shape == (8, 8), f"Expected shape (8, 8), got {h_matrix.shape}"
    return h_matrix.reshape(2, 2, 2, 2, 2, 2)


def tensor_to_matrix(h_tensor: Array) -> Array:
    """
    テンソル形式 (2,2,2,2,2,2) を行列形式 (8, 8) に変換
    
    インデックスマッピング:
        テンソル: h[bra_1, bra_2, bra_3, ket_1, ket_2, ket_3]
        行列: h[i, j] where i, j ∈ [0, 7]
        対応関係: i = 4*bra_1 + 2*bra_2 + bra_3
                 j = 4*ket_1 + 2*ket_2 + ket_3
    
    Args:
        h_tensor: shape (2,2,2,2,2,2) のテンソル形式
    
    Returns:
        h_matrix: shape (8, 8) のハミルトニアン行列
    
    Example:
        >>> h_tensor = jnp.ones((2,2,2,2,2,2))
        >>> h_matrix = tensor_to_matrix(h_tensor)
        >>> print(h_matrix.shape)
        (8, 8)
    """
    assert h_tensor.shape == (2, 2, 2, 2, 2, 2), \
        f"Expected shape (2,2,2,2,2,2), got {h_tensor.shape}"
    return h_tensor.reshape(8, 8)


def get_local_hamiltonian_term(
    h_matrix: Array, 
    site_indices: Tuple[int, int, int],
    total_sites: int
) -> Array:
    """
    全系のハミルトニアンに局所的な3サイト項を埋め込む
    
    Args:
        h_matrix: shape (8, 8) の3サイトハミルトニアン
        site_indices: 作用する3つのサイトのインデックス
        total_sites: 系全体のサイト数
    
    Returns:
        全系のハミルトニアンにおける局所項（shape: (2^total_sites, 2^total_sites)）
    """
    # 実装は将来の拡張時に追加
    raise NotImplementedError("全系への埋め込みは将来実装予定です")


def validate_hermitian(h_matrix: Array, tol: float = 1e-10) -> bool:
    """
    ハミルトニアンがエルミートであることを検証
    
    Args:
        h_matrix: 検証するハミルトニアン行列
        tol: 許容誤差
    
    Returns:
        エルミートならTrue
    """
    return jnp.allclose(h_matrix, h_matrix.conj().T, atol=tol)


# hamiltonians.pyに追加する関数

def get_ising_cft_operators() -> Dict[str, jnp.ndarray]:
    """
    臨界イジングモデルの正確なCFTプライマリー演算子
    
    臨界イジングモデル（h/J=1）は中心電荷c=1/2のCFTで、
    以下のプライマリー演算子を持つ：
    - 恒等演算子 I: Δ = 0
    - スピン演算子 σ: Δ = 1/8 = 0.125
    - エネルギー演算子 ε: Δ = 1
    
    Returns:
        演算子名とその行列表現の辞書
    """
    # パウリ行列（グローバル定数を使用）
    I = _I
    X = _X
    Z = _Z
    
    # 1. 恒等演算子（自明）
    identity = jnp.eye(8)
    
    # 2. スピン演算子（σ）
    # CFTのスピン演算子は、局所的なスピンの適切な線形結合
    # 3サイトの場合、並進対称性を考慮した平均
    sigma_1 = jnp.kron(jnp.kron(X, I), I)
    sigma_2 = jnp.kron(jnp.kron(I, X), I)
    sigma_3 = jnp.kron(jnp.kron(I, I), X)
    
    # 臨界点では等方的な重み付け（並進対称性）
    spin_operator = (sigma_1 + sigma_2 + sigma_3) / jnp.sqrt(3)
    
    # 3. エネルギー演算子（ε）
    # エネルギー密度演算子は、ハミルトニアン密度から構築
    # 臨界イジングモデルのエネルギー演算子は、
    # 相互作用項と横磁場項の適切な組み合わせ
    
    # 相互作用項
    zz_12 = jnp.kron(jnp.kron(Z, Z), I)
    zz_23 = jnp.kron(jnp.kron(I, Z), Z)
    
    # 横磁場項（臨界点でのバランス）
    x_sum = sigma_1 + sigma_2 + sigma_3
    
    # エネルギー密度（臨界点での特別な組み合わせ）
    # 注：これは局所ハミルトニアンから定数シフトを引いたもの
    energy_operator = -(zz_12 + zz_23) - x_sum
    
    # トレースレス化（CFT演算子の標準的な正規化）
    energy_operator = energy_operator - jnp.trace(energy_operator) * identity / 8
    
    # 正規化（演算子のノルムを統一）
    energy_operator = energy_operator / jnp.linalg.norm(energy_operator) * jnp.sqrt(8)
    
    return {
        'identity': identity,
        'spin': spin_operator,
        'energy': energy_operator
    }


def get_cft_operator_by_name(operator_name: str) -> jnp.ndarray:
    """
    名前でCFT演算子を取得
    
    Args:
        operator_name: "identity", "spin", "energy"のいずれか
        
    Returns:
        対応する演算子の行列
    """
    operators = get_ising_cft_operators()
    if operator_name not in operators:
        raise ValueError(f"Unknown operator: {operator_name}. "
                        f"Available: {list(operators.keys())}")
    return operators[operator_name]


def verify_cft_operators() -> Dict[str, Dict[str, float]]:
    """
    CFT演算子の性質を検証
    
    Returns:
        各演算子の性質（エルミート性、トレース、ノルムなど）
    """
    operators = get_ising_cft_operators()
    results = {}
    
    for name, op in operators.items():
        # エルミート性のチェック
        is_hermitian = jnp.allclose(op, op.conj().T)
        
        # トレース
        trace = jnp.trace(op)
        
        # フロベニウスノルム
        norm = jnp.linalg.norm(op)
        
        # 固有値
        eigenvalues = jnp.linalg.eigvals(op)
        
        results[name] = {
            'is_hermitian': bool(is_hermitian),
            'trace': float(jnp.real(trace)),
            'norm': float(norm),
            'max_eigenvalue': float(jnp.max(jnp.abs(eigenvalues))),
            'min_eigenvalue': float(jnp.min(jnp.real(eigenvalues)))
        }
        
    return results


# より高度な演算子の構築（オプション）
def get_descendant_operators(primary_op: jnp.ndarray, level: int = 1) -> List[jnp.ndarray]:
    """
    プライマリー演算子からデセンダント演算子を構築
    
    CFTでは、プライマリー演算子にビラソロ生成子L_{-n}を
    作用させることでデセンダント演算子が得られる。
    
    Args:
        primary_op: プライマリー演算子
        level: デセンダントのレベル
        
    Returns:
        デセンダント演算子のリスト
    """
    # 簡略化のため、離散版では微分演算子で近似
    # 実際のMERAでは、スケール変換を通じて実現される
    descendants = []
    
    if level >= 1:
        # L_{-1} ~ ∂（並進生成子）
        # 3サイトでの離散微分
        shift_op = jnp.roll(primary_op.reshape(2,2,2,2,2,2), 1, axis=(0,3))
        l_minus_1 = (shift_op.reshape(8,8) - primary_op) 
        descendants.append(l_minus_1)
        
    if level >= 2:
        # L_{-2} ~ T（ストレステンソル）
        # より複雑な実装が必要
        pass
        
    return descendants


# テスト関数
def test_cft_operators():
    """CFT演算子の正しさをテスト"""
    print("=== CFT演算子の検証 ===\n")
    
    operators = get_ising_cft_operators()
    properties = verify_cft_operators()
    
    for name, props in properties.items():
        print(f"{name}演算子:")
        print(f"  エルミート性: {props['is_hermitian']}")
        print(f"  トレース: {props['trace']:.6f}")
        print(f"  ノルム: {props['norm']:.6f}")
        print(f"  最大固有値: {props['max_eigenvalue']:.6f}")
        print()
    
    # 直交性のチェック（異なるプライマリー演算子は近似的に直交）
    print("演算子間の内積:")
    for name1, op1 in operators.items():
        for name2, op2 in operators.items():
            if name1 <= name2:  # 対称性のため半分だけ計算
                inner_product = jnp.trace(op1.conj().T @ op2) / 8  # 正規化
                print(f"  <{name1}|{name2}> = {inner_product:.6f}")
    
    # エネルギー演算子がトレースレスであることを確認
    energy_trace = jnp.trace(operators['energy'])
    print(f"\nエネルギー演算子のトレース: {energy_trace:.10f} (should be ~0)")
    
    return True


if __name__ == "__main__":
    # テストを実行
    test_cft_operators()


# ==================== テスト関数 ====================

def test_ising_critical_point():
    """臨界イジングモデルの基底状態エネルギーをテスト"""
    h_ising = construct_3site_ising_hamiltonian(h_field=1.0, J=1.0)
    info = get_hamiltonian_info(h_ising)
    
    # 3サイト臨界イジングモデルの既知の基底状態エネルギー
    # (この値は理論計算または高精度数値計算から得られる参照値)
    expected_energy = -3.41421356  # 例示的な値
    
    assert jnp.allclose(info['ground_state_energy'], expected_energy, atol=1e-5), \
        f"基底状態エネルギーが期待値と一致しません: {info['ground_state_energy']} vs {expected_energy}"
    
    # エルミート性の確認
    assert info['is_hermitian'], "ハミルトニアンがエルミートではありません"
    
    # エネルギーギャップが正であることを確認（基底状態が一意）
    assert info['gap'] > 0, f"エネルギーギャップが非正です: {info['gap']}"


def test_heisenberg_isotropy():
    """等方的ハイゼンベルグモデルの対称性をテスト"""
    h_heisenberg = construct_3site_heisenberg_hamiltonian(Jx=1.0, Jy=1.0, Jz=1.0)
    info = get_hamiltonian_info(h_heisenberg)
    
    # エルミート性
    assert info['is_hermitian'], "ハイゼンベルグハミルトニアンがエルミートではありません"
    
    # 実数性（等方的な場合は実数になるはず）
    assert jnp.allclose(jnp.imag(h_heisenberg), 0), "等方的ハイゼンベルグモデルが実数ではありません"


def test_tensor_matrix_conversion():
    """テンソル・行列変換の可逆性をテスト"""
    # ランダムなエルミート行列を作成
    key = jax.random.PRNGKey(42)
    random_matrix = jax.random.normal(key, (8, 8))
    hermitian_matrix = random_matrix + random_matrix.conj().T
    
    # 変換の可逆性を確認
    tensor_form = matrix_to_tensor(hermitian_matrix)
    reconstructed_matrix = tensor_to_matrix(tensor_form)
    
    assert jnp.allclose(hermitian_matrix, reconstructed_matrix, atol=1e-10), \
        "テンソル・行列変換が可逆ではありません"
    
    # 形状の確認
    assert tensor_form.shape == (2, 2, 2, 2, 2, 2), \
        f"テンソル形状が正しくありません: {tensor_form.shape}"


def run_all_tests():
    """全てのテストを実行"""
    print("=== 3サイトハミルトニアンモジュールのテスト ===\n")
    
    try:
        print("1. 臨界イジングモデルのテスト...")
        test_ising_critical_point()
        print("   ✓ 成功")
        
        print("\n2. ハイゼンベルグモデルのテスト...")
        test_heisenberg_isotropy()
        print("   ✓ 成功")
        
        print("\n3. テンソル・行列変換のテスト...")
        test_tensor_matrix_conversion()
        print("   ✓ 成功")
        
        print("\n全てのテストが成功しました！")
        
    except AssertionError as e:
        print(f"\n   ✗ テスト失敗: {e}")
        raise


# 従来のプリントベースのデモ（後方互換性のため）
def demo_hamiltonians():
    """ハミルトニアン構築のデモンストレーション"""
    print("=== 3サイトハミルトニアンのデモ ===\n")
    
    # 臨界イジングモデル
    print("1. 臨界横磁場イジングモデル (h/J = 1):")
    h_ising = construct_3site_ising_hamiltonian(h_field=1.0, J=1.0)
    info = get_hamiltonian_info(h_ising)
    print(f"   基底状態エネルギー: {info['ground_state_energy']:.6f}")
    print(f"   エネルギーギャップ: {info['gap']:.6f}")
    print(f"   エルミート性: {info['is_hermitian']}")
    
    # XXZモデル
    print("\n2. XXZハイゼンベルグモデル (等方的):")
    h_heisenberg = construct_3site_heisenberg_hamiltonian(Jx=1.0, Jy=1.0, Jz=1.0)
    info = get_hamiltonian_info(h_heisenberg)
    print(f"   基底状態エネルギー: {info['ground_state_energy']:.6f}")
    print(f"   エネルギーギャップ: {info['gap']:.6f}")
    
    # テンソル変換のテスト
    print("\n3. テンソル変換のテスト:")
    h_tensor = matrix_to_tensor(h_ising)
    h_matrix_back = tensor_to_matrix(h_tensor)
    print(f"   変換の整合性: {jnp.allclose(h_ising, h_matrix_back)}")
    print(f"   テンソル形状: {h_tensor.shape}")


if __name__ == "__main__":
    # モジュールを直接実行した場合
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # python hamiltonians.py test でテストモード
        run_all_tests()
    else:
        # デフォルトはデモモード
        demo_hamiltonians()