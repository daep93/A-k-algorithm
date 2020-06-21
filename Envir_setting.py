import numpy as np

"""
For setting environment.
"""


def generate_arrivals(edges, lower, upper, frame_size, frame_rep):
    np.random.seed(0)
    with open('./arrival_patterns/' + f'Arrivals_rates from {lower:.4f} to {upper:.4f}.txt', 'r') as f:
        lambs = f.read()
    lambs = lambs.split()
    origin_labs = lambs[:]
    lambs = iter(lambs)
    patterns = [np.random.binomial(1, float(lamb), int(frame_size * frame_rep)) for lamb in lambs]
    arrival_pattern = [''.join(map(str, pattern)) for pattern in patterns]

    with open(
            './arrival_patterns/' + f'Arrivals(edge {edges}, lambda from {lower:.4f} to {upper:.4f}, frame_size {frame_size}, frame_rep {frame_rep}).txt',
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


for i in range(1, 10):
    generate_random_arrival_rates(200, 0.06 * i / 10, 0.18 * i / 10)
    generate_arrivals(200, 0.06 * i / 10, 0.18 * i / 10, 5000, 20)

generate_transmissions(200,0.25,0.75)
