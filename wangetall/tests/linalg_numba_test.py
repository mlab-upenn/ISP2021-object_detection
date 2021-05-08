import time
from typing import Tuple

import numpy
import torch
from matplotlib import pyplot


def solve_numpy(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
    parameters = numpy.linalg.solve(a, b)
    return parameters


def get_data(dim: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
    a = numpy.random.random((dim, dim))
    b = numpy.random.random(dim)
    return a, b


def get_data_torch(dim: int) -> Tuple[torch.tensor, torch.tensor]:
    a = torch.rand(dim, dim)
    b = torch.rand(dim, 1)
    return a, b


def solve_torch(a: torch.tensor, b: torch.tensor) -> torch.tensor:
    parameters, _ = torch.solve(b, a)
    return parameters


def experiment_numpy(matrix_size: int, repetitions: int = 100):
    for i in range(repetitions):
        a, b = get_data(matrix_size)
        p = solve_numpy(a, b)


def experiment_pytorch(matrix_size: int, repetitions: int = 100):
    for i in range(repetitions):
        a, b = get_data_torch(matrix_size)
        p = solve_torch(a, b)


def main():
    matrix_size = [x for x in range(5, 505, 5)]
    experiments = experiment_numpy, experiment_pytorch
    results = tuple([] for _ in experiments)

    for i, each_experiment in enumerate(experiments):
        for j, each_matrix_size in enumerate(matrix_size):
            time_start = time.time()
            each_experiment(each_matrix_size, repetitions=100)
            d_t = time.time() - time_start
            print(f"{each_matrix_size:d} {each_experiment.__name__:<30s}: {d_t:f}")
            results[i].append(d_t)

    for each_experiment, each_result in zip(experiments, results):
        pyplot.plot(matrix_size, each_result, label=each_experiment.__name__)

    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    main()
