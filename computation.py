import math
import numpy as np
import argparse


class Params:
    def __init__(
        self,
        num_parties,
        threshold,
        pack,
        field_size,
        binsup_dim,
        lpn_key_len,
        lpn_ctx_len,
        lpn_noise_param,
        lenmac,
        lencheck,
    ):
        self.n = num_parties
        self.t = threshold
        self.l = pack
        self.F = field_size
        self.binsup_dim = binsup_dim
        self.lpn_key_len = lpn_key_len
        self.lpn_ctx_len = lpn_ctx_len
        self.lpn_noise_param = lpn_noise_param
        self.lenmac = lenmac
        self.lencheck = lencheck


class ProtoCtr:
    def __init__(self, params):
        self.counts = {}
        self.params = params
        self.ncons = 0
        self.nmac = 0
        self.nzero = 0

    def __inc(self, protocol, num):
        self.counts[protocol] = self.counts.get(protocol, 0) + num

    def share(self, num, degree=False):
        if not degree:
            degree = self.params.t + self.params.l - 1

        n = self.params.n
        l = self.params.l
        num_interp = n + l - degree - 1
        return num * np.array([num_interp * degree, num_interp * (degree + 1)])

    def recon(self, num):
        n = self.params.n
        l = self.params.l
        return num * np.array([l * (n - 1), l * n])

    def robustrecon(self, num):
        # Check correctness of shares by taking first d + 1 = t + l shares and
        # reconstructing remaining n - d - 1 shares. If a received share is not equal
        # to computed share, there is an error. If all received shares equal computed
        # shares then reconstruct l secrets.
        d = self.params.t + self.params.l - 1
        n = self.params.n
        l = self.params.l
        num_interp = n - d - 1 + l
        return num * np.array([num_interp * d, num_interp * (d + 1)])

    def select(self, num):
        l = self.params.l
        return num * np.array([l - 1, l])

    def ecc(self, num):
        return (
            num
            * self.params.lpn_ctx_len
            * np.array([self.params.lpn_key_len, self.params.lpn_key_len + 1])
        )

    def rand(self, num):
        self.__inc("rand", num)
        return np.array([0, 0])

    def __rand(self, num):
        batch_size = self.params.n - self.params.t
        batches = int(math.ceil(num / batch_size))
        n = self.params.n
        t = self.params.t
        comp = self.share(batches) + batches * (n - t) * np.array([n - 1, n])
        return comp

    def zero(self, num):
        self.__inc("zero", num)
        return np.array([0, 0])

    def __zero(self, num):
        batch_size = self.params.n - self.params.t
        batches = int(math.ceil(num / batch_size))
        n = self.params.n
        t = self.params.t
        comp = self.share(batches, n - 1) + batches * (n - t) * np.array([n - 1, n])
        return comp

    def bitrand(self, num):
        self.__inc("bitrand", num)
        return np.array([0, 0])

    def __bitrand(self, num):
        batch_size = self.params.binsup_dim
        batches = int(math.ceil(num / batch_size))
        n = self.params.n
        comp = self.share(batches) + batches * self.params.binsup_dim * np.array(
            [n - 1, n]
        )
        return comp

    def degreduce(self, num):
        return self.recon(num) + self.share(num)

    def randshare(self, num):
        batch_size = self.params.l
        batches = int(math.ceil(num / batch_size))
        n = self.params.n
        l = self.params.l
        t = self.params.t
        d = t + l - 1
        comp = self.zero(batches * 2 * n)
        comp += self.rand(batches * (n + t))
        comp += batches * l * np.array([n - 1, n])
        comp += batches * (n + l - d - 1) * np.array([d, d + 1])
        comp += self.recon(2 * batches)
        return comp

    def mackeygen(self, num):
        self.__inc("mackeygen", num)
        return np.array([0, 0])

    def __mackeygen(self, num):
        batch_size = 2 * self.params.l
        batches = int(math.ceil(num / batch_size))
        n = self.params.n
        comp = (
            self.share(batches)
            + batches * n * np.array([n - 1, n])
            + self.robustrecon(batches)
        )
        return comp

    def coin(self, _):
        num = int(math.ceil(128 / self.params.F))
        return self.rand(num) + self.robustrecon(num)

    def mult(self, num):
        return (
            self.rand(num)
            + self.zero(num)
            + num * np.array([3, 1])
            + self.degreduce(num)
        )

    def authmult(self, num):
        self.ncons += num * (self.params.lenmac + 1)
        self.nmac += num
        return self.mult((self.params.lenmac + 1) * num)

    def trans(self, num):
        return self.randshare(num) + num * np.array([2, 0]) + self.degreduce(num)

    def authtrans(self, num):
        self.ncons += num * (self.params.lenmac + 1)
        return self.trans((self.params.lenmac + 1) * num)

    def auth(self, num):
        self.ncons += num * self.params.lenmac
        self.nmac += num
        return self.mult(self.params.lenmac * num)

    def error(self, num):
        return (
            self.bitrand(num * self.params.lpn_noise_param)
            + self.mult(num * (self.params.lpn_noise_param - 1))
            + self.rand(num)
            + self.mult(num)
        )

    def checkbit(self, num):
        self.nzero += num
        return self.authmult(num) + num * np.array([1, 0])

    def autherror(self, num):
        return (
            self.bitrand(num * self.params.lpn_noise_param)
            + self.auth(num * self.params.lpn_noise_param)
            + self.checkbit(num * self.params.lpn_noise_param)
            + self.authmult(num * (self.params.lpn_noise_param - 1))
            + self.rand(num)
            + self.mult(num * (self.params.lenmac + 1))
        )

    def checkcons(self):
        m = self.ncons
        k = self.params.lencheck
        return (
            self.rand(k)
            + self.coin((m + 1) * k)
            + k * np.array([m, m + 1])
            + self.robustrecon(k)
        )

    def checkmac(self):
        m = self.nmac
        k = self.params.lenmac
        return (
            self.coin(k * m)
            + self.robustrecon(k)
            + 2 * k * np.array([m - 1, m])
            + np.array([1, 1])
            + self.robustrecon(k)
        )

    def checkzero(self):
        m = self.nzero
        k = self.params.lencheck
        return self.coin(m * k) + self.robustrecon(k) + k * np.array([m - 1, m])

    def get_circ_ind_preproc(self):
        lookup = {
            "rand": self.__rand,
            "zero": self.__zero,
            "bitrand": self.__bitrand,
            "mackeygen": self.__mackeygen,
        }

        res = {}

        for k, v in self.counts.items():
            if k not in lookup:
                raise Exception(f"Unknown protocol encountered: {k}")

            res[k] = lookup[k](v)

        return res


def fmt_time(num):
    units = ["s", "min", "hours"]
    scale = [60, 60, 24]
    for u, s in zip(units, scale):
        if abs(num) < s:
            return f"{num:.3f} {u}"
        num /= s
    return f"{num:.1f} days"


def print_stat_numops(label, val):
    print(
        f"{label}: {val.item(0):.2e}, {val.item(1):.2e}"
    )  # Additions, Multiplications


# tadd and tmul are time for an addition and multiplication in seconds
def get_print_stat_time(threads, tadd, tmul):
    def print_stat_time(label, val):
        print(
            f"{label}: {fmt_time((val.item(0) * tadd + val.item(1) * tmul) / threads)}"
        )  # 1 thread

    return print_stat_time


def ec23(params, num_inp, num_and, num_xor, print_stat):
    get_blocks = lambda x: int(math.ceil(x / params.l))
    num_and_blocks = get_blocks(num_and)
    num_xor_blocks = get_blocks(num_xor)
    num_wire_blocks = get_blocks(num_inp) + num_and_blocks + num_xor_blocks
    num_gate_blocks = num_and_blocks + num_xor_blocks

    ctr = ProtoCtr(params)
    total = 0
    preproc = {}

    print("--- Circuit Dependent Cost ---")

    # Preprocessing
    comp = ctr.mackeygen(params.lenmac)
    preproc["Generate MAC keys"] = comp

    comp = ctr.rand(num_wire_blocks * 2 * params.lpn_key_len)
    ctr.ncons += num_wire_blocks * 2 * params.lpn_key_len
    comp += ctr.auth(num_wire_blocks * 2 * params.lpn_key_len)
    preproc["Generate keys"] = comp

    comp = ctr.bitrand(num_wire_blocks)
    ctr.ncons += num_wire_blocks
    comp += ctr.auth(num_wire_blocks)
    comp += ctr.checkbit(num_wire_blocks)
    preproc["Generate masks"] = comp

    comp = ctr.autherror(num_gate_blocks * 4 * params.lpn_ctx_len)
    preproc["Generate errors"] = comp

    # Garbling
    comp = ctr.authtrans(num_wire_blocks + num_wire_blocks * 2 * params.lpn_key_len)
    comp += ctr.select(
        2 * num_gate_blocks + 2 * num_gate_blocks * 2 * params.lpn_key_len
    )
    comp += ctr.authtrans(
        2 * num_gate_blocks + 2 * num_gate_blocks * 2 * params.lpn_key_len
    )
    ctr.nmac += 2 * num_gate_blocks + 2 * num_gate_blocks * 2 * params.lpn_key_len
    print_stat("Transform masks and labels", comp)
    total += comp

    comp = ctr.authmult(num_and_blocks + num_gate_blocks * 4 * params.lpn_key_len)
    comp += num_and_blocks * np.array([7, 0])
    comp += num_xor_blocks * np.array([4, 0])
    comp += num_gate_blocks * 4 * params.lpn_key_len * np.array([2, 0])
    print_stat("Compute plaintexts", comp)
    total += comp

    comp = ctr.ecc(4 * num_gate_blocks)
    comp += (
        4
        * num_gate_blocks
        * params.lpn_ctx_len
        * np.array([params.lpn_key_len + 1, params.lpn_key_len])
    )
    comp += ctr.degreduce(4 * num_gate_blocks * params.lpn_ctx_len)
    ctr.nmac += 4 * num_gate_blocks * params.lpn_ctx_len
    print_stat("Compute ciphertexts", comp)
    total += comp

    comp = ctr.checkcons()
    comp += ctr.checkmac()
    comp += ctr.checkzero()
    print_stat("Verification", comp)
    total += comp

    print_stat("Total", total)

    print("\n\n--- Circuit Independent Preprocessing ---")
    total = 0

    for k, v in preproc.items():
        print_stat(f"{k}", v)
        total += v

    for k, v in ctr.get_circ_ind_preproc().items():
        print_stat(f"{k}", v)
        total += v

    print_stat("Total", total)


def ec23semhon(params, num_inp, num_and, num_xor, print_stat):
    params = Params(
        200,  # n
        50,  # t
        50,  # l
        18,  # F
        14,  # binsup_dim
        246,  # lpn_key_len
        721,  # lep_ctx_len
        2,  # lpn_noise_param
        3,  # lenmac
        3,  # lencheck
    )

    get_blocks = lambda x: int(math.ceil(x / params.l))
    num_and_blocks = get_blocks(num_and)
    num_xor_blocks = get_blocks(num_xor)
    num_wire_blocks = get_blocks(num_inp) + num_and_blocks + num_xor_blocks
    num_gate_blocks = num_and_blocks + num_xor_blocks

    ctr = ProtoCtr(params)
    total = 0
    preproc = {}

    print("--- Circuit Dependent Cost ---")

    # Preprocessing
    comp = ctr.rand(num_wire_blocks * 2 * params.lpn_key_len)
    preproc["Generate keys"] = comp

    comp = ctr.bitrand(num_wire_blocks)
    preproc["Generate masks"] = comp

    comp = ctr.error(num_gate_blocks * 4 * params.lpn_ctx_len)
    preproc["Generate errors"] = comp

    # Garbling
    comp = ctr.trans(num_wire_blocks + num_wire_blocks * 2 * params.lpn_key_len)
    comp += ctr.select(
        2 * num_gate_blocks + 2 * num_gate_blocks * 2 * params.lpn_key_len
    )
    comp += ctr.trans(
        2 * num_gate_blocks + 2 * num_gate_blocks * 2 * params.lpn_key_len
    )
    print_stat("Transform masks and labels", comp)
    total += comp

    comp = ctr.mult(num_and_blocks + num_gate_blocks * 4 * params.lpn_key_len)
    comp += num_and_blocks * np.array([7, 0])
    comp += num_xor_blocks * np.array([4, 0])
    comp += num_gate_blocks * 4 * params.lpn_key_len * np.array([2, 0])
    print_stat("Compute plaintexts", comp)
    total += comp

    comp = ctr.ecc(4 * num_gate_blocks)
    comp += (
        4
        * num_gate_blocks
        * params.lpn_ctx_len
        * np.array([params.lpn_key_len + 1, params.lpn_key_len])
    )
    comp += ctr.degreduce(4 * num_gate_blocks * params.lpn_ctx_len)
    print_stat("Compute ciphertexts", comp)
    total += comp

    print_stat("Total", total)

    print("\n\n--- Circuit Independent Preprocessing ---")
    total = 0

    for k, v in preproc.items():
        print_stat(f"{k}", v)
        total += v

    for k, v in ctr.get_circ_ind_preproc().items():
        print_stat(f"{k}", v)
        total += v

    print_stat("Total", total)


def cli_args():
    parser = argparse.ArgumentParser(
        description="Computation costs for each phase of the protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "num_inp",
        type=int,
        help="Total number of INPUT wires for the circuit.",
    )
    parser.add_argument(
        "num_and",
        type=int,
        help="Total number of AND gates in the circuit.",
    )
    parser.add_argument(
        "num_xor",
        type=int,
        help="Total number of XOR gates in the circuit.",
    )
    parser.add_argument("--proto", choices=["mal", "semihon"], default="mal")
    parser.add_argument("--field_size", type=int, default=18, help="log (base 2) of the field size")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--stat_type", choices=["numops", "time"], default="time")
    return parser.parse_args()


if __name__ == "__main__":
    args = cli_args()

    field_size = args.field_size

    params = Params(
        200,  # n
        50,  # t
        50,  # l
        field_size,  # F
        14,  # binsup_dim
        246,  # lpn_key_len
        721,  # lep_ctx_len
        2,  # lpn_noise_param
        3,  # lenmac
        3,  # lencheck
    )

    print_stat = print_stat_numops
    if args.stat_type == "time":
        if field_size == 18:
            print_stat = get_print_stat_time(args.threads, 1.6e-10, 7.68e-9) # GF(2^18)
        elif field_size == 19:
            print_stat = get_print_stat_time(args.threads, 1.6e-10, 8.80e-9) # GF(2^19)
        else:
            raise Exception("Timing statistics supported only for fields of size 18 and 19 bits.")

    if args.proto == "mal":
        ec23(params, args.num_inp, args.num_and, args.num_xor, print_stat)
    else:
        ec23semhon(params, args.num_inp, args.num_and, args.num_xor, print_stat)
