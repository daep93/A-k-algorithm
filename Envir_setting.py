import numpy as np
"""
For setting environment.
"""


def generate_arrivals(title, rnd_seed, edges, lamb, time):
    np.random.seed(rnd_seed)
    pattern = np.random.binomial(1, lamb, edges * time)
    arrival_pattern = ''.join(map(str, pattern))

    with open('./arrival_patterns/' + title + f'_seed({rnd_seed}).txt', 'w') as f:
        f.write(arrival_pattern)

    return


def generate_transmissions(title, rnd_seed, edges, lower, upper):
    np.random.seed(rnd_seed)
    pattern = np.random.uniform(lower, upper, edges)
    transmission_pattern = ' '.join(map(lambda x: f'{x:.3f}', pattern))

    with open('./transmission_patterns/' + title + f'_seed({rnd_seed}).txt', 'w') as f:
        f.write(transmission_pattern)

    return


