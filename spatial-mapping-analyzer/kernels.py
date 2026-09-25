"""Small RV32IM kernels for correctness bring-up, not Tensix FPU kernels.

Only the backend emits instructions. Tensor dimensions specialize loop bounds;
input values and expected outputs never enter the instruction stream.
"""
import struct

A, B, OUTPUT, DONE = 0x1000, 0x2000, 0x3000, 0x4000
MAX_ELEMENTS = 1024


class Assembler:
    def __init__(self):
        self.words, self.labels, self.branches = [], {}, []

    def li(self, rd, value):
        upper = (value + 0x800) >> 12
        self.words.append((upper << 12) | (rd << 7) | 0x37)
        self.addi(rd, rd, value - (upper << 12))

    def addi(self, rd, rs, value):
        if not -2048 <= value < 2048:
            raise ValueError("ADDI immediate out of range")
        self.words.append(((value & 0xFFF) << 20) | (rs << 15) | (rd << 7) | 0x13)

    def alu(self, rd, lhs, rhs, multiply=False):
        self.words.append((int(multiply) << 25) | (rhs << 20) | (lhs << 15) | (rd << 7) | 0x33)

    def load(self, rd, base):
        self.words.append((base << 15) | (2 << 12) | (rd << 7) | 0x03)

    def store(self, rs, base):
        self.words.append((rs << 20) | (base << 15) | (2 << 12) | 0x23)

    def label(self, name):
        self.labels[name] = len(self.words) * 4

    def branch(self, lhs, rhs, condition, target):
        self.branches.append((len(self.words), lhs, rhs, condition, target))
        self.words.append(0)

    def finish(self):
        self.addi(5, 0, 1)
        self.store(5, 4)  # completion flag, after every output store
        self.words.append(0x0000006F)  # jal x0, 0: wait for host to assert reset
        for index, lhs, rhs, condition, target in self.branches:
            offset = self.labels[target] - index * 4
            if offset % 2 or not -4096 <= offset < 4096:
                raise ValueError("Branch offset out of range")
            bits = offset & 0x1FFF
            self.words[index] = (((bits >> 12) << 31) | (((bits >> 5) & 63) << 25)
                                 | (rhs << 20) | (lhs << 15) | (condition << 12)
                                 | (((bits >> 1) & 15) << 8) | (((bits >> 11) & 1) << 7) | 0x63)
        return struct.pack(f"<{len(self.words)}I", *self.words)


def compile_kernel(op, input_shapes, output_elements):
    asm = Assembler()
    for register, address in [(1, A), (2, B), (3, OUTPUT), (4, DONE)]:
        asm.li(register, address)
    if op == "matmul":
        m, k = input_shapes[0]
        _, n = input_shapes[1]
        asm.addi(14, 2, 0)  # B base
        asm.li(5, m)
        asm.label("row")
        asm.addi(2, 14, 0)
        asm.li(6, n)
        asm.label("column")
        asm.addi(9, 1, 0)  # A row cursor
        asm.addi(10, 2, 0)  # B column cursor
        asm.addi(8, 0, 0)  # accumulator
        asm.li(7, k)
        asm.label("dot")
        asm.load(11, 9)
        asm.load(12, 10)
        asm.alu(13, 11, 12, multiply=True)
        asm.alu(8, 8, 13)
        asm.addi(9, 9, 4)
        asm.addi(10, 10, n * 4)
        asm.addi(7, 7, -1)
        asm.branch(7, 0, 1, "dot")  # bne
        asm.store(8, 3)
        asm.addi(3, 3, 4)
        asm.addi(2, 2, 4)
        asm.addi(6, 6, -1)
        asm.branch(6, 0, 1, "column")
        asm.addi(1, 1, k * 4)
        asm.addi(5, 5, -1)
        asm.branch(5, 0, 1, "row")
    else:
        asm.li(5, output_elements)
        asm.label("element")
        asm.load(6, 1)
        if op == "add":
            asm.load(7, 2)
            asm.alu(6, 6, 7)
            asm.addi(2, 2, 4)
        elif op == "relu":
            asm.branch(6, 0, 5, "nonnegative")  # signed bge
            asm.addi(6, 0, 0)
            asm.label("nonnegative")
        else:
            raise ValueError(f"Unsupported kernel: {op}")
        asm.store(6, 3)
        asm.addi(1, 1, 4)
        asm.addi(3, 3, 4)
        asm.addi(5, 5, -1)
        asm.branch(5, 0, 1, "element")
    return asm.finish()
