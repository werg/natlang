"""The compilers' toolchain service backend: check and run LLVM IR with llvmlite (LLVM's own parser, verifier and
JIT). It checks and runs what the natural-language compiler produced; it never generates code.

    toolchain.py verify < module.ll            -> {"ok": true} | {"ok": false, "error": "..."}
    toolchain.py run MODULE.ll < stdin         -> the program's own stdout/stderr and exit code (main is JIT-run)
    toolchain.py assembly < module.ll          -> LLVM's own AArch64 assembly (validation scaffold only)
"""
import ctypes
import json
import os
import sys


def _llvm():
    import llvmlite.binding as llvm
    try:
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
    except RuntimeError:
        pass
    return llvm


def verify(text):
    llvm = _llvm()
    try:
        module = llvm.parse_assembly(text)
        module.verify()
    except RuntimeError as error:
        return {"ok": False, "error": str(error).strip()}
    return {"ok": True}


def run(path):
    """JIT-run main(): the process's stdin, stdout and exit code are the program's."""
    llvm = _llvm()
    module = llvm.parse_assembly(open(path).read())
    module.verify()
    machine = llvm.Target.from_default_triple().create_target_machine()
    engine = llvm.create_mcjit_compiler(module, machine)
    engine.finalize_object()
    engine.run_static_constructors()
    main = module.get_function("main")
    arity = len(list(main.arguments))
    address = engine.get_function_address("main")
    if arity == 0:
        status = ctypes.CFUNCTYPE(ctypes.c_int)(address)()
    else:
        argv = (ctypes.c_char_p * 2)(b"program", None)
        status = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p)(address)(1, ctypes.addressof(argv))
    libc = ctypes.CDLL(None)
    libc.fflush(None)
    os._exit(status & 0xFF)


def assembly(text):
    """LLVM's own AArch64 assembly for a module: the scaffold that validation splices one generated function into."""
    llvm = _llvm()
    try:
        module = llvm.parse_assembly(text)
        module.verify()
        target = llvm.Target.from_triple("aarch64-unknown-linux-gnu")
        machine = target.create_target_machine(opt=2, reloc="pic", codemodel="small")
        return {"ok": True, "assembly": machine.emit_assembly(module)}
    except RuntimeError as error:
        return {"ok": False, "error": str(error).strip()}


if __name__ == "__main__":
    if sys.argv[1] == "verify":
        print(json.dumps(verify(sys.stdin.read())))
    elif sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "assembly":
        print(json.dumps(assembly(sys.stdin.read())))
