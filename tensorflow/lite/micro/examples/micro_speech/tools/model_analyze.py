"""Analyze a TFLite flatbuffer model: structure, operators, weight size.

Usage:
    python model_analyze.py [path_to_model.tflite ...]

Requires: pip install tflite flatbuffers
"""

import sys

import tflite


def builtin_name(opcode_index, model):
    """Resolve builtin operator name via opcodes table + BuiltinOperator enum."""
    opcode = model.OperatorCodes(opcode_index)
    builtin = opcode.DeprecatedBuiltinCode() or opcode.BuiltinCode()
    try:
        from tflite.BuiltinOperator import BuiltinOperator

        rev = {v: k for k, v in BuiltinOperator.__dict__.items()
               if isinstance(v, int)}
        return rev.get(builtin, f"UNKNOWN({builtin})")
    except Exception:
        return f"UNKNOWN({builtin})"


def analyze(path):
    buf = open(path, "rb").read()
    m = tflite.Model.GetRootAsModel(buf, 0)
    print(f"== {path}")
    print(f"   file size: {len(buf)} bytes, schema version: {m.Version()}")

    subgraphs = m.SubgraphsLength()
    print(f"   subgraphs: {subgraphs}")
    for g in range(subgraphs):
        sg = m.Subgraphs(g)
        n_t = sg.TensorsLength()
        n_o = sg.OperatorsLength()
        print(f"   subgraph {g}: tensors={n_t}, operators={n_o}")
        print("   operators:")
        for i in range(n_o):
            op = sg.Operators(i)
            name = builtin_name(op.OpcodeIndex(), m)
            ins = [op.Inputs(j) for j in range(op.InputsLength())]
            outs = [op.Outputs(j) for j in range(op.OutputsLength())]
            print(f"     op[{i}] {name:24s} in={ins} out={outs}")

    # sum of non-empty buffers = weight/constant data size
    total = 0
    for i in range(1, m.BuffersLength()):
        b = m.Buffers(i)
        if b.DataLength() > 0:
            total += b.DataLength()
    print(f"   weight buffers total: {total} bytes ({total / 1024:.1f} KB)")
    print()


def main():
    paths = sys.argv[1:] or [
        r"..\models\micro_speech_quantized.tflite",
        r"..\models\audio_preprocessor_int8.tflite",
    ]
    for p in paths:
        analyze(p)


if __name__ == "__main__":
    main()
