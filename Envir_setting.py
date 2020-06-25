import numpy as np

"""
For setting environment.
"""


def generate_arrivals(edges, lower, upper,coef, frame_size, frame_rep):
    np.random.seed(0)
    with open('./arrival_patterns/' + f'Arrivals_rates from {lower:.4f} to {upper:.4f}.txt', 'r') as f:
        lambs = f.read()
    lambs = list(map(lambda x: float(x)*coef,lambs.split()))
    origin_labs = lambs[:]
    lambs = iter(lambs)
    patterns = [np.random.binomial(1, lamb, int(frame_size * frame_rep)) for lamb in lambs]
    arrival_pattern = [''.join(map(str, pattern)) for pattern in patterns]

    with open(
            './arrival_patterns/' + f'Arrivals(edge {edges}, lambda from {lower*coef:.4f} to {upper*coef:.4f}, frame_size {frame_size}, frame_rep {frame_rep}).txt',
            'w') as f:
        for arr_pat in arrival_pattern:
            f.write(arr_pat)
            f.write('\n')

    return arrival_pattern


def generate_transmissions(edges, lower, upper):
    np.random.seed(0)
    pattern = np.random.uniform(lower, upper, edges)
    transmission_pattern = ' '.join(map(lambda x: f'{x:.3f}', pattern))

    with open('./transmission_patterns/' + f'Capacities({edges}, capacity from {lower:.2f} to {upper:.2f}).txt',
              'w') as f:
        f.write(transmission_pattern)

    return transmission_pattern


def generate_random_arrival_rates(edges, lower, upper):
    np.random.seed(0)
    pattern = np.random.uniform(lower, upper, edges)
    arrival_pattern = ' '.join(map(lambda x: f'{x:.4f}', pattern))
    with open('./arrival_patterns/' + f'Arrivals_rates from {lower:.4f} to {upper:.4f}.txt', 'w') as f:
        f.write(arrival_pattern)


