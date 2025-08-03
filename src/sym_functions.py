import multiprocessing
from typing import Callable, Dict, Tuple

import sympy as sp


def symbolic_symmetric_inverse_template(
    n: int,
) -> Tuple[sp.Matrix, Dict[Tuple[int, int], sp.Symbol]]:
    """
    Generates the symbolic inverse of a symmetric n x n matrix with placeholders.

    Args:
        n: Dimension of the matrix

    Returns:
        A tuple:
        - The inverse of the symbolic symmetric matrix (expanded and simplified)
        - A mapping from (i, j) indices to the corresponding symbolic entries
    """
    assert n >= 1, "Matrix size must be positive"

    symbols: Dict[Tuple[int, int], sp.Symbol] = {}
    for i in range(n):
        for j in range(i, n):
            sym = sp.symbols(f"a{i+1}{j+1}")
            symbols[(i, j)] = sym
            symbols[(j, i)] = sym  # symmetry

    A_sym = sp.Matrix(n, n, lambda i, j: symbols[(i, j)])

    print("creating template for symbolic symmetric inverse")
    A_inv_sym = A_sym.inv()

    A_inv_sym = A_inv_sym.applyfunc(sp.expand)
    A_inv_sym = A_inv_sym.applyfunc(sp.expand_trig)
    A_inv_sym = A_inv_sym.applyfunc(sp.factor)
    A_inv_sym = A_inv_sym.applyfunc(sp.cancel)
    A_inv_sym = A_inv_sym.applyfunc(sp.simplify)

    return A_inv_sym, symbols


def inverse_via_symmetric_substitution(mat: sp.Matrix) -> sp.Matrix:
    """
    Computes the inverse of a symmetric matrix using symbolic substitution into
    the generic closed-form inverse of a symbolic symmetric matrix.

    Args:
        mat: A symmetric sympy.Matrix of shape (n, n)

    Returns:
        sympy.Matrix: The inverse of mat, via substitution into a symbolic inverse
    """
    n = mat.shape[0]
    assert mat.shape == (n, n), "Matrix must be square"

    # Precompute symbolic inverse template and symbol mapping
    A_inv_sym, symbols = symbolic_symmetric_inverse_template(n)

    # Build substitution map from symbolic entries to actual matrix elements
    subs_map = {symbols[(i, j)]: mat[i, j] for i in range(n) for j in range(i, n)}
    print("applying substitution to symbolic inverse")
    return A_inv_sym.subs(subs_map)


def simplify_expr(expr: sp.Expr) -> sp.Expr:
    """
    Apply a series of simplifications to a SymPy expression.
    Modify this function as needed to control the order and type of simplification.
    """
    expr = sp.expand(expr)
    expr = sp.expand_trig(expr)
    expr = sp.factor(expr)
    expr = sp.cancel(expr)
    # expr = sp.simplify(expr)
    return expr


def parallel_simplify_matrix(
    mat: sp.Matrix, simplifier: Callable[[sp.Expr], sp.Expr] = simplify_expr
) -> sp.Matrix:
    """
    Simplifies each element of a sympy.Matrix using multiprocessing.

    Args:
        mat: The sympy.Matrix to simplify
        simplifier: A function that takes and returns a sympy.Expr

    Returns:
        A simplified sympy.Matrix
    """
    flat_exprs = list(mat)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        simplified_flat = pool.map(simplifier, flat_exprs)
    return sp.Matrix(mat.rows, mat.cols, simplified_flat)


def count_non_integer_atoms(expr: sp.Expr) -> dict:
    """
    Count how many times each non-integer atomic expression appears in a SymPy expression.

    Returns:
        dict: Mapping from atom to count
    """
    atoms = expr.atoms()
    return {
        atom: expr.count(atom) for atom in atoms if not isinstance(atom, sp.Integer)
    }


def count_non_integer_atoms(expr: sp.Expr) -> dict:
    """
    Count how many times each non-integer atomic expression appears in a SymPy expression.

    Returns:
        dict: Mapping from atom to count
    """
    atoms = expr.atoms()
    return {
        atom: expr.count(atom) for atom in atoms if not isinstance(atom, sp.Integer)
    }


def sym_info(e: sp.Expr):
    """Prints information about a symbolic expression."""
    num_terms = sp.core.function.count_ops(e)
    str_len = len(str(e))
    print(f"symbolic expression has {num_terms:_} terms and length {str_len:_}")
    inbuilt_count = dict()
    inbuilt_count["sp.pow"] = e.count(sp.Pow)
    inbuilt_count["sp.trig"] = e.count(sp.sin)

    print("inbuilt count:")
    for atom, count in sorted(inbuilt_count.items(), key=lambda kv: -kv[1]):
        count = inbuilt_count[atom]
        num = f"{count:_}"
        print(f"  {str(atom):<20} {num:>5}")
    non_integer_atoms = count_non_integer_atoms(e)
    print(f"symbols count:         {len(non_integer_atoms):>5}")
    for atom, count in sorted(non_integer_atoms.items(), key=lambda kv: -kv[1]):
        num = f"{count:_}"
        print(f"  {str(atom):<20} {num:>5}")
