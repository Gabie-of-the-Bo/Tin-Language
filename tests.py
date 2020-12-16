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

        for i in self.test_data:
            correct = self.validator(i)
            tin_res = compiled_tin.execute([i])[0]

            if np.any(correct != tin_res):
                table['Result'] = 'Failed'
                table['Reason'] = f'Invalid result for i = {i}: {tin_res} != {correct}'
                break

        table.setdefault('Result', 'Ok')

        print(tab.tabulate(table.items(), headers='firstrow', tablefmt='fancy_grid', stralign='center'))

TESTS = [
    TinTest('Naive primality test', '→n(.nι``.n%𝔹∀1.n>)∀←n', 
            lambda i: i > 1 and all(i % j for j in range(2, i)), range(100)),

    TinTest('Identity matrix generation', '→n(.nι{0.nR↶1↶↑})←n', lambda i: np.eye(i), range(1, 20)),

    TinTest('Iterative factorial', 'ι⊳∏', np.math.factorial, range(13)),
    TinTest('Recursive factorial', '|◊⟨!!⊲∇·→n⟩:⟨1→n⟩.n←n|→|F| F', np.math.factorial, range(50))
]

def execute_tests():
    for t in TESTS:
        t.execute()