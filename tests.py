from time import process_time_ns
from statistics import mode
from math import sqrt

import tabulate as tab
import numpy as np

from tin import Tin

class TinTest:
    def __init__(self, name, tin_prg, validator, test_data):
        self.name = name
        self.tin_prg = tin_prg
        self.validator = validator
        self.test_data = test_data

    def execute(self):
        table = {
            'Test': self.name,
            'Code': self.tin_prg,
            'Inputs': self.test_data
        }

        compiled_tin = Tin(self.tin_prg)

        start_time = process_time_ns()

        for i in iter(self.test_data):
            correct = self.validator(i)
            tin_res = compiled_tin.execute([i])[-1]

            if np.any(correct != tin_res):
                table['Result'] = 'Failed'
                table['Reason'] = f'Invalid result for i = {i}: {tin_res} != {correct}'
                break

        table.setdefault('Result', 'Ok')

        time_ms = (process_time_ns() - start_time) / 10**6

        table['Time'] = f'{time_ms:.3} ms (~{time_ms / len(self.test_data):.3} ms/iter.)'

        print(tab.tabulate(table.items(), headers='firstrow', tablefmt='fancy_grid', stralign='center'))


class RandomSequenceGenerator:
    def __init__(self, items, size, min_val, max_val):
        self.items = items
        self.size = size
        self.min_val = min_val
        self.max_val = max_val

    def __iter__(self):
        for i in range(self.items):
            yield np.random.randint(self.min_val, self.max_val, self.size)

    def __len__(self): 
        return self.items

    def __str__(self): 
        return f'{self.items} sequences of size {self.size}, items ∈ [{self.min_val}, {self.max_val})'


TESTS = [
    TinTest('Naive primality test', '→n(.nι``.n%𝔹∀1.n>)∀←n', 
            lambda i: i > 1 and all(i % j for j in range(2, i)), range(1000)),

    TinTest('Identity matrix generation', '→n(.nι{0.nR↶1↶↑})←n', lambda i: np.eye(i), range(1, 100)),

    TinTest('Iterative factorial', 'ι⊳∏', np.math.factorial, range(13)),
    TinTest('Recursive factorial', '|◊⟨!!⊲∇·→n⟩:⟨1→n⟩.n←n|→|F| F', np.math.factorial, range(50)),

    TinTest('Statistical mean', '!⍴↶∑/', np.mean, RandomSequenceGenerator(100, 1000, -100, 100)),
    TinTest('Variance', '!!!⍴↶∑/-2↶^∑↶⍴↶/', np.var, RandomSequenceGenerator(100, 10, -100, 100)),
    TinTest('Mode', '→n(.n{.n↶#})!⌈º0↓.n↶↓←n', mode, RandomSequenceGenerator(100, 100, 0, 10)),

    TinTest('Iterative Fibonacci', '!!→n1<?⟨2ι→r ⊲ι{(.r1↓ .r∑)→r}.r1↓→n⟩.n←n', lambda i: int(((1 + sqrt(5)) / 2) ** i / sqrt(5) + 0.5), range(45)),
    TinTest('Recursive Fibonacci', '!1<?⟨⊲!⊲∇↶∇+⟩', lambda i: int(((1 + sqrt(5)) / 2) ** i / sqrt(5) + 0.5), range(15)),
]

def execute_tests():
    for t in TESTS:
        t.execute()