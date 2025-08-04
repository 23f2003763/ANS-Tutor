import sys

def compute_ap_sum(a: float, d: float, n: int):
    # last term
    l = a + (n - 1) * d
    # sum formula (average of first and last term times n)
    S_n = n * (a + l) / 2
    # alternate formula: n/2 * (2a + (n-1)d)
    alternate = n / 2 * (2 * a + (n - 1) * d)
    return l, S_n, alternate

def format_explanation(a: float, d: float, n: int):
    l, S_n, alternate = compute_ap_sum(a, d, n)
    explanation = f"""Problem: Given an arithmetic progression with first term a = {a}, common difference d = {d}, and number of terms n = {n}.

1. Formula: S_n = n/2 * (2a + (n - 1)d)  (which is equivalent to S_n = n * (first + last)/2)
2. Derivation intuition: The sum of the first n terms equals the average of the first and last term multiplied by n. The last term is a + (n-1)d = {a} + ({n}-1)*{d} = {l}.
   So average = ({a} + {l}) / 2.
3. Computation:
   - Last term: l = a + (n - 1)d = {l}
   - Using average method: S_n = n * (a + l) / 2 = {n} * ({a} + {l}) / 2 = {S_n}
   - Using expanded formula: S_n = n/2 * (2a + (n - 1)d) = {n}/2 * (2*{a} + ({n}-1)*{d}) = {alternate}
4. Final answer: S_{n} = {S_n}
"""
    return explanation

def main():
    if len(sys.argv) != 4:
        print("Usage: py -3.10 explain_ap_sum.py <a> <d> <n>")
        sys.exit(1)
    a = float(sys.argv[1])
    d = float(sys.argv[2])
    n = int(sys.argv[3])
    print(format_explanation(a, d, n))

if __name__ == "__main__":
    main()
