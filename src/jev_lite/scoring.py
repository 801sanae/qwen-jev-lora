"""top_logprobs -> P(Y) 계산. Y/N 두 토큰만으로 재정규화(나머지 토큰 질량은 버림)."""
import math

Y_VARIANTS = {"Y", " Y"}
N_VARIANTS = {"N", " N"}


def p_yes(top_logprobs: list[dict]) -> tuple[float, float]:
    """returns (p_yes, coverage). coverage = Y+N 원래 확률 합 (낮으면 형식 붕괴 신호)."""
    py = sum(math.exp(t["logprob"]) for t in top_logprobs if t["token"] in Y_VARIANTS)
    pn = sum(math.exp(t["logprob"]) for t in top_logprobs if t["token"] in N_VARIANTS)
    cov = py + pn
    return (py / cov if cov > 0 else 0.5), cov
