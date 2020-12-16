# ***** Imports *****

import numpy as np

import re

# ***** Constants *****

class META: 
    def __init__(self, f): self.f = f
    def __call__(self, rep): 
        self.rep = rep
        return self

class BLOCK(META): 
    def __init__(self, f=None, rep=None): 
        self.rep = rep
        self.f = f

    def __call__(self, rep): 
        def execute_wrapper(prg):
            def execute_rec(machine, op, ip, tokens, stack):
                prg.parent = machine
                stack = prg.execute(stack)

                return ip, tokens, stack

            return execute_rec
            
        return BLOCK(execute_wrapper(Tin(rep[1:-1])), rep)

class DEF(META):
    def __call__(self, rep): 
        self.rep = rep
        
        pattern = r'\|(.+)\|→\|(.+)\|'

        prg, name = re.match(pattern, rep).groups()
        self.prg = Tin(prg)
        
        def execute_rec(machine, op, ip, tokens, stack):
            stack = self.prg.execute(stack)

            return ip, tokens, stack

        TOKENS[re.compile(name)] = META(execute_rec)
        
        return self

ID = lambda i: lambda _: i 

# Meta functions

def dup(machine, op, ip, tokens, stack):
    stack.append(stack[-1])

    return ip, tokens, stack

def copy(machine, op, ip, tokens, stack):
    stack[-1] = stack[-(stack[-1] + 1)]

    return ip, tokens, stack

def swap(machine, op, ip, tokens, stack):
    stack[-1], stack[-2] = stack[-2], stack[-1]

    return ip, tokens, stack

def skip(machine, op, ip, tokens, stack):
    if not stack.pop(): 
        ip += 1

    return ip, tokens, stack

def skip_dup(machine, op, ip, tokens, stack):
    if not stack[-1]: 
        ip += 1

    return ip, tokens, stack

def skip_inv(machine, op, ip, tokens, stack):
    if stack.pop():   
        ip += 1

    return ip, tokens, stack

branch_stack = []

def branch_init(machine, op, ip, tokens, stack):
    branch_stack.append(ip)    

    return ip, tokens, stack

def branch_end(machine, op, ip, tokens, stack):
    pos = branch_stack.pop()

    if stack.pop(): 
        ip = pos - 1

    return ip, tokens, stack

loop_stack = []

def foreach_init(machine, op, ip, tokens, stack):
    if loop_stack and loop_stack[-1][0] == ip:
        loop_stack[-1][-1] += 1
        
    else:
        loop_stack.append([ip, stack.pop(), 0])

    stack.append(loop_stack[-1][1][loop_stack[-1][2]])

    return ip, tokens, stack

def foreach_end(machine, op, ip, tokens, stack):
    pos, arr, idx = loop_stack[-1]

    if idx < len(arr) - 1: 
        ip = pos - 1

    else:
        loop_stack.pop()

    return ip, tokens, stack

storer_stack = []

def storer_init(machine, op, ip, tokens, stack):
    storer_stack.append(len(stack))    

    return ip, tokens, stack

def storer_end(machine, op, ip, tokens, stack):
    pos = storer_stack.pop()

    arr = np.array(stack[pos:])
    stack = stack[:pos]
    stack.append(arr)

    return ip, tokens, stack

variables = {}

def define_var(machine, op, ip, tokens, stack):
    name = op[1:]

    variables.setdefault(name, [])
    variables[name].append(stack.pop())

    return ip, tokens, stack

def delete_var(machine, op, ip, tokens, stack):
    name = op[1:]
    variables[name].pop()

    if not variables[name]:
        variables.pop(name)

    return ip, tokens, stack

def get_var(machine, op, ip, tokens, stack):
    name = op[1:]
    stack.append(variables[name][-1])

    return ip, tokens, stack

def self_reference(machine, op, ip, tokens, stack):
    if machine.parent:
        machine.parent.execute(stack)

    else:
        machine.execute(stack)

    return ip, tokens, stack

# Array functions

def assign_to_index(index, elem, arr):
    arr[index] = elem

    return arr

# Token list

TOKENS = {
    # Literals
    r'\d+': int,
    r'\'.+?\'': lambda i: i[1:-1],

    # Meta
    r'!': META(dup),
    r'↷': META(copy),
    r'↶': META(swap),
    
    r'\?': META(skip),
    r'◊': META(skip_dup),
    r'\:': META(skip_inv),
    r'\[': META(branch_init),
    r'\]': META(branch_end),
    r'\{': META(foreach_init),
    r'\}': META(foreach_end),
    r'\(': META(storer_init),
    r'\)': META(storer_end),

    r'→[a-z_]+': META(define_var),
    r'←[a-z_]+': META(delete_var),
    r'\.[a-z_]+': META(get_var),

    r'⟨[^⟨⟩]+⟩': BLOCK(),

    r'\|.+\|→\|.+?\|': DEF(lambda i, j, k, l, m: (k, l, m)),

    r'∇': META(self_reference),

    # Functions
    r'\+': ID(lambda i, j: i + j),
    r'\-': ID(lambda i, j: i - j),
    r'\·': ID(lambda i, j: i * j),
    r'\/': ID(lambda i, j: i / j),
    r'\%': ID(lambda i, j: i % j),

    r'⊳': ID(lambda i: i + 1),
    r'⊲': ID(lambda i: i - 1),

    r'𝔹': ID(lambda i: i.astype(np.bool) if isinstance(i, np.ndarray) else bool(i)),

    r'<': ID(lambda i, j: i < j),
    r'>': ID(lambda i, j: i > j),
    r'∃': ID(lambda i: np.any(i)),
    r'∄': ID(lambda i: not np.any(i)),
    r'∀': ID(lambda i: np.all(i)),

    r'\$': ID(lambda i: print(i)),

    # Array operations
    r'ι': ID(lambda i: np.arange(i)),
    r'□': ID(lambda i: np.array([i])),
    r'R': ID(lambda i, j: np.array([j for _ in range(i)])),
    r'↓': ID(lambda i, j: j[i]),
    r'↑': ID(assign_to_index),

    r'∑': ID(lambda i: np.sum(i)),
    r'∏': ID(lambda i: np.prod(i)),

    # Functional array operations
    r'`': ID(lambda i: i[1:]),
    r'´': ID(lambda i: i[:-1]),
}

TOKENS = {re.compile(i): j for i, j in TOKENS.items()}

# ***** Model *****

class Tin:
    def __init__(self, code):
        self.parent = None
        self.tokens, i = [], 0

        while i < len(code):
            if code[i].isspace(): 
                i += 1
                continue

            for r, f in TOKENS.items():
                match = r.search(code, i)

                if match and match.start() == i:
                    self.tokens.append(f(match.group()))
                    i += match.end() - match.start()
                    break
    
    def execute(self, stack=None):
        if not stack:
            stack = []

        ip = 0

        while ip < len(self.tokens):
            tok = self.tokens[ip]

            if isinstance(tok, META):
                ip, self.tokens, stack = tok.f(self, tok.rep, ip, self.tokens, stack)

            elif callable(tok):
                arity = tok.__code__.co_argcount
                res = tok(*(stack.pop() for _ in range(arity)))

                if not isinstance(res, type(None)):
                    stack.append(res)
            
            else:
                stack.append(tok)

            ip += 1

        return stack