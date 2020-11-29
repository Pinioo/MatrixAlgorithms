from dataclasses import dataclass, asdict
from functools import reduce
import numpy as np

def count(elements, n):
    counted = list()
    for i in range(n):
        counted.append(0)
    for el in elements:
        counted[el] += 1
    return counted

def matrix_dict_to_str(elements: dict, n: int) -> str:
    most_digits = len(str(max(elements.values())))
    tabs = int(most_digits / 4) - 1
    rep = ""
    for i in range(n):
        rep += '['
        for j in range(n):
            rep += tabs * '\t'
            rep += str(elements.get((i, j), 0))
            rep += ', '
        rep += '\b\b]\n'
    return rep


@dataclass
class CscSparse:
    n: int
    nnz: int
    irn: list
    val: list
    rowptr: list
    
    def build_sparse_dict(self) -> dict:
        elements = {}
        for col_number, col_start, col_end in zip(range(self.n), self.rowptr, self.rowptr[1:]):
            elements.update({
                (i, col_number): value for i, value in zip(self.irn[col_start:col_end], self.val[col_start:col_end])
            })
        return elements

    def __repr__(self) -> str:
        return matrix_dict_to_str(
            self.build_sparse_dict(), 
            self.n
        )


@dataclass
class CsrSparse:
    n: int
    nnz: int
    icl: list
    val: list
    colptr: list

    def build_sparse_dict(self) -> dict:
        elements = {}
        for row_number, row_start, row_end in zip(range(self.n), self.colptr, self.colptr[1:]):
            elements.update({
                (row_number, i): value for i, value in zip(self.icl[row_start:row_end], self.val[row_start:row_end])
            })
        return elements

    def __repr__(self) -> str:
        return matrix_dict_to_str(
            self.build_sparse_dict(),
            self.n
        )

    def to_csc(self) -> CscSparse:
        colptr_to_indices = sum(
            map(
                lambda t: (t[2] - t[1]) * [t[0]],                   # add as many index value as (row_end - row_start)
                zip(range(self.n), self.colptr, self.colptr[1:])    # get (index, row_start, row_end)
            ),
            []
        ) # flatMap, example [0, 1, 1, 3, 4] -> [0, 2, 2, 3]
        rowptr_indices, irn, csc_val = zip(
            *sorted(
                zip(
                    self.icl,
                    colptr_to_indices,
                    self.val
                )
            )
        ) # lex sort by (icl, colptr_to_indices, val) to get (rowptr_indices, irn, csc_val)
        rowptr = reduce(
            lambda xs, x: xs + [xs[-1] + x],    # calculate col_end for CSC
            count(rowptr_indices, self.n),      # count every number appearance
            [0]                                 # first col_start is index 0
        ) # example [0, 2, 2, 3] ->  [0, 1, 1, 3, 4]
        return CscSparse(
            self.n,
            self.nnz,
            list(irn),
            list(csc_val),
            rowptr
        )

print("===============")
print("======CSR======")
print("===============")
csr = CsrSparse(3, 5, [0,1,2,0,2], [1,2,3,4,5], [0,1,3,5])
print(csr)
print(asdict(csr))
print()

print("===============")
print("======CSC======")
print("===============")
csc_converted = csr.to_csc()
print(csc_converted)
print(asdict(csc_converted))
print()