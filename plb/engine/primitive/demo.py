def run(args):
    import taichi as ti
    import numpy as np
    gpu, eps = args
    if gpu:
        ti.init(arch=ti.gpu, print_ir=True, fast_math=False)
    else:
        ti.init(arch=ti.cpu, print_ir=True, fast_math=False)
    a = np.array([1., 1., 1.])
    grid_pos = ti.Vector.field(3, shape=(), dtype=ti.f64)
    grid_pos.from_numpy(a)
    @ti.kernel
    def sdf():
        d = eps
        inc = grid_pos[None]
        dec = grid_pos[None]
        inc[0] += d
        dec[0] -= d
        print(ti.sqrt(inc.dot(inc)), ti.sqrt(dec.dot(dec)))
        diff = ti.sqrt(inc.dot(inc)) - ti.sqrt(dec.dot(dec))
        print(diff, diff * 0.5 / d)
        print("using gpu:", gpu, "eps:", d, "ans:", (ti.sqrt(inc.dot(inc)) - ti.sqrt(dec.dot(dec))) *0.5 / d)
    sdf()
run((True, 1e-9))