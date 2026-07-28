from math import comb


def hypergeometric_probability(N, D, n, x):
    """
    Probability of finding exactly x defective units.

    N = lot size
    D = defective units in lot
    n = sample size
    x = observed defectives
    """
    if x < 0 or x > n:
        return 0

    if D < x:
        return 0

    if N - D < n - x:
        return 0

    return (
        comb(D, x)
        * comb(N - D, n - x)
        / comb(N, n)
    )


def probability_of_acceptance(N, D, n, c):
    """
    Probability that the lot is accepted.

    c = acceptance number
    """

    return sum(
        hypergeometric_probability(N, D, n, x)
        for x in range(c + 1)
    )

def defectives_from_rate(N, defect_rate):
    """
    defect_rate = fraction (0.02 = 2%)
    """

    return round(N * defect_rate)

def acceptance_probability_from_rate(
    N,
    defect_rate,
    n,
    c,
):
    D = defectives_from_rate(N, defect_rate)

    return probability_of_acceptance(
        N,
        D,
        n,
        c,
    )

def operating_characteristic_curve(
    N,
    n,
    c,
    defect_rates,
):
    """
    Returns list of
    (defect_rate, probability_of_acceptance)
    """

    results = []

    for p in defect_rates:
        pa = acceptance_probability_from_rate(
            N,
            p,
            n,
            c,
        )

        results.append((p, pa))

    return results


def make_curve():
    rates = [0.00,0.01,0.02,0.03,0.05,0.10]

    curve = operating_characteristic_curve(
        N=200,
        n=50,
        c=1,
        defect_rates=rates,
    )

    for r, pa in curve:
        print(f"{r:.1%} defects -> {pa:.3f}")

def producers_risk(
    N,
    good_quality_rate,
    n,
    c,
):
    """
    α = rejecting a good lot
    """

    pa = acceptance_probability_from_rate(
        N,
        good_quality_rate,
        n,
        c,
    )

    return 1 - pa

def consumers_risk(
    N,
    bad_quality_rate,
    n,
    c,
):
    """
    β = accepting a bad lot
    """

    return acceptance_probability_from_rate(
        N,
        bad_quality_rate,
        n,
        c,
    )

def average_outgoing_quality(
    N,
    defect_rate,
    n,
    c,
):
    """
    AOQ

    assumes rejected lots are fully corrected
    """

    pa = acceptance_probability_from_rate(
        N,
        defect_rate,
        n,
        c,
    )

    return pa * defect_rate * (1 - n / N)

def aoq_curve(
    N,
    n,
    c,
    defect_rates,
):
    return [
        (
            p,
            average_outgoing_quality(
                N,
                p,
                n,
                c,
            ),
        )
        for p in defect_rates
    ]

def average_outgoing_quality_limit(
    N,
    n,
    c,
    resolution=1000,
):
    """
    Finds the maximum AOQ numerically.
    """

    best_rate = 0
    best_aoq = 0

    for i in range(resolution + 1):

        p = i / resolution

        aoq = average_outgoing_quality(
            N,
            p,
            n,
            c,
        )

        if aoq > best_aoq:
            best_aoq = aoq
            best_rate = p

    return best_rate, best_aoq


from math import comb

def binomial_probability(
    n,
    x,
    p,
):
    """Binomial approximation

        Useful for very large lots."""
    return (
        comb(n, x)
        * p**x
        * (1 - p)**(n - x)
    )

from math import exp, factorial

def poisson_probability(
    lam,
    x,
):
    """Poisson approximation

    Useful when defects are rare where

    lam = n * defect_rate"""

    return (
        exp(-lam)
        * lam**x
        / factorial(x)
    )




from math import sqrt

def wilson_interval(x, n, z=1.96):
    """
    Confidence interval (Wilson)

    Although not part of ISO 2859-2, Wilson intervals are useful when estimating the underlying defect rate from an inspection sample.
    Wilson score interval for a proportion.

    x = observed defectives
    n = sample size
    """

    if n == 0:
        raise ValueError("n must be > 0")

    p = x / n

    denominator = 1 + z**2 / n

    centre = (
        p + z**2 / (2 * n)
    ) / denominator

    margin = (
        z
        * sqrt(
            p * (1 - p) / n
            + z**2 / (4 * n**2)
        )
        / denominator
    )

    return centre - margin, centre + margin

def main():
    st.write("""These functions cover the core statistical machinery behind ISO 2859-2: hypergeometric probabilities, acceptance probability, Operating Characteristic curves, producer's and consumer's risks, Average Outgoing Quality (AOQ), Average Outgoing Quality Limit (AOQL), plus binomial and Poisson approximations. The only thing not included is the standard's predefined sampling tables, which are based on selected risk levels rather than direct formulas.""")
