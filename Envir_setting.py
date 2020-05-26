import numpy as np

"""
For setting environment.
"""


def generate_arrivals(rnd_seed, edges, lamb, time):
    np.random.seed(int(rnd_seed))
    pattern = np.random.binomial(1, float(lamb), edges * int(time))
    arrival_pattern = ''.join(map(str, pattern))

    with open('./arrival_patterns/' + f'ar_seed({rnd_seed}, {edges}, {lamb}, {time}).txt', 'w') as f:
        f.write(arrival_pattern)

    return arrival_pattern


def generate_transmissions(rnd_seed, edges, lower, upper):
    np.random.seed(int(rnd_seed))
    pattern = np.random.uniform(float(lower), float(upper), edges)
    transmission_pattern = ' '.join(map(lambda x: f'{x:.3f}', pattern))

    with open('./transmission_patterns/' + f'tr_seed({rnd_seed}, {edges}, {lower}, {upper}).txt', 'w') as f:
        f.write(transmission_pattern)

    return transmission_pattern


# generate_arrivals(0, 24, '0.4', 10)
# generate_transmissions(0, 24, '0.25', '0.75')
