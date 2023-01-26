import math
import numpy as np
import argparse
import json
import os


class Params:
    # To be given as input
    n = 200  # number of parties
    t = 50  # corruption threshold

    stat_sec = 40  # statistical security parameter
    comp_sec = 128  # computational security parameter

    # TODO: set these params
    L_Mac = 0 
    L_Cons = 0    
    L_Zero = 0
    Lpn_Err = 2 # aka tau

    G_and, G_xor = (1000, 1000)  # circuit parameters
    W_in = 0 # TODO: add input wires
    W_out = 0 # TODO: add output wires 
    HashLen = 256  # length of hash (bits)
    F_len = 0  # size of element in field (bits)

    # Computed from above
    l = 0  # number of packed secret sharing slots
    Q, V = (721, 246)  # LPN parameters (Q = ciphertext length, V = length of key)
    binH_k = 1  # binary HIM output dim i.e., Binary HIM order is (k x n)

    @staticmethod
    def initialize():
        Params.l = int(math.floor(Params.n / 2 - Params.t + (1 / 2)))
        if Params.l < 1:
            raise Exception(f"Corruption threshold too low ({Params.t})")
        min_field_size = int(math.ceil(math.log(Params.n + Params.l + Params.G_and + Params.G_xor,2)))
        # if field length was not already set, set it now 
        if Params.F_len == 0:
            Params.F_len = min_field_size
        elif Params.F_len < min_field_size:
            raise Exception("The field size is too small")
        # set the number of reps for L_{Mac,Cons,Zero}
        num_reps = int(math.ceil(Params.stat_sec / Params.F_len))
        Params.L_Mac = num_reps 
        Params.L_Cons = num_reps
        Params.L_Zero = num_reps
        if Params.G_and == 0 and Params.G_xor == 0:
            raise Exception("Need non-empty circuit.")

        params = Params.find_BCH_RS(Params.n)
        if params == None:
            raise Exception("No BCH RS params found")
        _, k_RS, _, _, k_bin, _ = params
        Params.binH_k = int(k_RS * k_bin)
        print("PARAMETER CHECK: \nN = {}, T = {}, L = {}, FIELD_SIZE = {}, bin_mtx_rate = {} L_MAC,CONS,ZERO = {}, GATES_AND = {}, GATES_XOR = {}".format(Params.n, Params.t, Params.l, Params.F_len, Params.binH_k,Params.L_Mac, Params.G_and, Params.G_xor))

    @staticmethod
    def check_RS(N, n_bin, k_bin, d_bin):
        n_RS = (N + n_bin - 1) // n_bin
        if n_RS > 2**k_bin:
            return False

        d_RS = (n_RS * n_bin + 3 * d_bin - 1) // (3 * d_bin)

        if d_RS > n_RS:
            return False

        k_RS = n_RS - d_RS + 1
        return (n_RS, k_RS, d_RS)

    @staticmethod
    def find_BCH_RS(N):
        best_params = None
        best_k = 0
        for m in range(1, 11):
            n_bin = 2**m - 1
            t_max = n_bin // m
            for t in np.arange(t_max, 0, -1):
                k_bin = n_bin - m * t
                if k_bin <= 1:
                    continue
                d_bin = 2 * t + 1
                if d_bin < n_bin // 3:
                    break

                check = Params.check_RS(N, n_bin + 1, k_bin, d_bin + 1)

                if check != False:
                    n_RS, k_RS, d_RS = check
                    k = k_RS * k_bin
                    if k > best_k:
                        best_k = k
                        best_params = (n_RS, k_RS, d_RS, n_bin + 1, k_bin, d_bin + 1)

        return best_params


class State:
    def __init__(self):
        self.num_rand = {}
        self.num_zeros = {}
        self.num_constr = 0
        self.num_macs = 0
        self.check_zero_num = 0

    def new_rand(self, num, field_len):
        self.num_rand[field_len] = self.num_rand.get(field_len, 0) + num

    def new_zeros(self, num, field_len):
        self.num_zeros[field_len] = self.num_zeros.get(field_len, 0) + num

    def add_constrs(self, num):
        self.num_constr += num

    def add_macs(self, num):
        self.num_macs += num

    def add_zeros(self, num):
        self.check_zero_num += num

class Network:
    def __init__(self):
        self.comm = {}

    def __init_round(self, rid):
        if rid not in self.comm:
            self.comm[rid] = [[0 for _ in range(Params.n)] for _ in range(Params.n)]

    def communicate(self, rid, sender_pid, receiver_pid, num_bits):
        if num_bits == 0:
            return rid

        self.__init_round(rid)
        self.comm[rid][sender_pid][receiver_pid] += num_bits
        return rid + 1

    def all_to_one(self, rid, receiver_pid, num_bits):
        self.__init_round(rid)
        for i in range(Params.n):
            if i != receiver_pid:
                self.comm[rid][i][receiver_pid] += num_bits
        return rid + 1

    def one_to_all(self, rid, leader_pid, num_bits):
        if num_bits == 0:
            return rid

        self.__init_round(rid)
        for i in range(Params.n):
            if i != leader_pid:
                self.comm[rid][leader_pid][i] += num_bits
        return rid + 1

    def all_to_all(self, rid, num_bits):
        if num_bits == 0:
            return rid

        self.__init_round(rid)
        for i in range(Params.n):
            for j in range(Params.n):
                if i != j:
                    self.comm[rid][i][j] += num_bits
        return rid + 1

    def all_to_2t_sub(self, rid, num_bits):
        if num_bits == 0:
            return rid
        self.__init_round(rid)
        for i in range(Params.n):
            for j in range(Params.n):
                if j <= (2*Params.t) and i != j:
                    self.comm[rid][i][j] += num_bits
        return rid + 1

    def send_matrix(self, pid):
        matrix = []
        for _, round_comm in sorted(self.comm.items()):
            matrix.append(round_comm[pid])

        return matrix

    def recv_matrix(self, pid):
        matrix = []
        for _, round_comm in sorted(self.comm.items()):
            row = [0 for _ in range(Params.n)]
            for i in range(Params.n):
                row[i] = round_comm[i][pid]

            matrix.append(row)
        return matrix

class Multiplier:
    def __init__(self, network, state, robust=True):
        self.net = network
        self.state = state
        self.robust = robust

    def run(self, rid, num, field_len):
        num_elts = num

        if self.robust:
            num_elts += Params.L_Mac * num

        rid = self.net.all_to_one(rid, 0, field_len * num_elts)
        rid = self.net.one_to_all(rid, 0, field_len * num_elts)

        # should probably be new_prand here but I think it's the same
        # TODO: check
        #print("In Multiplier\nNew Zeros: {}\nNew Rand Elts: {}".format(num_elts, num_elts)) 
        self.state.new_zeros(num_elts, field_len)
        self.state.new_rand(num_elts, field_len)
        
        if self.robust:
            self.state.add_constrs(num*(Params.L_Mac + 1))
            self.state.add_macs(3*num) # (x,y,z)

        return rid

class Auth:
    def __init__(self, network, state):
        self.net = network
        self.state = state
        self.mult = Multiplier(self.net, self.state, False)
        
    def run(self, rid, num, field_len):
        #print("AUTH RUN")
        # multiply MAC key with input using normal mult 
        rid = self.mult.run(rid, num*Params.L_Mac, field_len)
        # add stuff to S_Constr
        self.state.add_constrs(num*Params.L_Mac)
        return rid

class ErrorGen:
    def __init__(self, network, state):
        self.net = network
        self.state = state
        self.auth_mult = Multiplier(self.net, self.state)
        self.bit_rand = BitRand(self.net, self.state)
        self.unauth_mult = Multiplier(self.net, self.state, False)
        self.auth = Auth(self.net, self.state)
        self.check_bit = CheckBit(self.net, self.state)

    def run(self, rid, num, field_len):
        #print("In Error Gen")
        # bit rand call 
        #print("Calling Bit rand with {}".format(num*Params.Lpn_Err))
        rid = self.bit_rand.run(rid, num*Params.Lpn_Err, field_len)
        # authenticate bits 
        rid = self.auth.run(rid, num*Params.Lpn_Err, field_len)
        # ensure they are bits 
        rid = self.check_bit.run(rid, num*Params.Lpn_Err, field_len)

        # run auth_mult on instance, will take log_2 of instances 
        # per error gen 
        num_mults_per_err = int(math.ceil(math.log(Params.Lpn_Err,2)))
        rid = self.auth_mult.run(rid, num_mults_per_err*num, field_len)
        print("ErrorGen\nNew Rand Elts: {}".format(num))
        self.state.new_rand(num, field_len)
        rid = self.unauth_mult.run(rid, (1+Params.L_Mac)*num, field_len)
        return rid

class BitRand:
    def __init__(self, network, state):
        self.net = network
        self.state = state

    def run(self, rid, num, field_len):
        num_shares = (num + Params.binH_k - 1) // Params.binH_k
        rid = self.net.all_to_all(rid, num_shares * field_len)
        return rid


class Rand:
    def __init__(self, network):
        self.net = network

    def run(self, rid, num, field_len):
        num_shares = (num + Params.n - Params.t - 1) // (Params.n - Params.t)
        rid = self.net.all_to_all(rid, num_shares * field_len)
        return rid


class MacKeygen:
    def __init__(self, network):
        self.net = network

    def run(self, rid, field_len):
        # not using rand because generating packed secrets (s_1 ... s_l)
        # s.t. s_i = s_j for all i and j (not uniform vals)
        num_batches = (Params.L_Mac + Params.n - (2*Params.t) - 1) // (Params.n - (2*Params.t))
        # HIM 
        print("In Mac Key Gen, calling generation {} times".format(num_batches))
        rid = self.net.all_to_all(rid, num_batches * field_len)
        # validity stuff
        rid = self.net.all_to_2t_sub(rid, num_batches * field_len)
        return rid

class Zeros:
    def __init__(self, network):
        self.net = network
    
    def run(self, rid, num, field_len):
        num_batches = (num + Params.n - Params.t - 1) // (Params.n - Params.t) 
        rid = self.net.all_to_all(rid, num_batches * field_len)        
        return rid

class RandSharing:
    def __init__(self, network, state):
        self.net = network
        self.state = state

    def run(self, rid, num, field_len):
        num_batches = (num + Params.l-1) // Params.l 
        #print("FROM RandSharing\nAdding Zeros: {}\nAdding Rand Elts: {}".format(num_batches*2*Params.n, num_batches*(Params.n + Params.t)))
        self.state.new_zeros(num_batches * 2 * Params.n, field_len)
        self.state.new_rand(num_batches * (Params.n + Params.t), field_len)
        rid = self.net.all_to_all(rid, num_batches*2*field_len)
        return rid
        
class Trans:
    def __init__(self, network, state, robust=True):
        self.net = network 
        self.state = state
        self.rs = RandSharing(self.net, self.state)
        self.robust = robust

    def run(self, rid, num, field_len):
        #print("In Trans")
        num_elts = num
        if self.robust:
            num_elts += num*Params.L_Mac

        rid = self.rs.run(rid, num_elts, field_len)
        rid = self.net.all_to_one(rid, 0, num_elts*field_len)
        rid = self.net.one_to_all(rid, 0, num_elts*field_len)

        if self.robust:
            self.state.add_constrs(Params.L_Mac + 1)            
        return rid

class CommonCoin:
    def __init__(self, network, state):
        self.net = network
        self.state = state

    def run(self, rid, field_len):
        randbits_per_share = field_len * Params.l
        num_shares = (Params.comp_sec + randbits_per_share - 1) // randbits_per_share

        self.state.new_rand(num_shares, field_len)
        # reconstruct to use as key for creating $ vals 
        rid = self.net.all_to_all(rid, num_shares * field_len)

        return rid

class CheckBit:
    def __init__(self, network, state):
        self.net = network
        self.state = state
        self.mult = Multiplier(self.net, self.state, True)

    def run(self, rid, num, field_len):
        #print("In Check Bit")
        rid = self.mult.run(rid, num, field_len)
        self.state.add_zeros(num)  
        return rid

class CheckZero:
    def __init__(self, network, state):
        self.net = network
        self.state = state
        self.mult = Multiplier(self.net, self.state, False)
 
    def run(self, rid, num_elts, field_len):
        #print("CHECK ZERO\nRand Elts: {}".format(num_elts))
        self.state.new_rand(num_elts, field_len) 
        rid = self.mult.run(rid, num_elts, field_len)
        # reconstruct, check if zero
        rid = self.net.all_to_all(rid, num_elts * field_len)
        return rid

class CheckCons:
    def __init__(self, network, state):
        self.net = network
        self.state = state
        self.common_coin = CommonCoin(network, state)

    def run(self, rid, field_len):
        #print("CHECK CONS\nRand Elts: {}".format(Params.L_Cons))
        self.state.new_rand(Params.L_Cons,field_len)
        rid = self.common_coin.run(rid, field_len)
        rid = self.net.all_to_all(rid, Params.L_Cons*field_len)
        return rid

class CheckMAC:
    def __init__(self, network, state):
        self.net = network
        self.state = state
        self.common_coin = CommonCoin(network, state)

    def run(self, rid, field_len):
        rid = self.common_coin.run(rid, field_len)
        # reconstr. MACs
        rid = self.net.all_to_all(rid, Params.L_Mac*field_len)
        # run check zero with L.Mac different values
        rid = self.net.all_to_all(rid, Params.L_Mac*field_len)
        return rid


class Garble:
    def __init__(self, network):
        self.net = network
        self.state = State()
        self.rid = 2
    
    def prepreproc(self):
        # Call this method at the end
        rand = Rand(self.net)
        rand_rid = rand.run(0, self.state.num_rand.get(Params.F_len, 0), Params.F_len)

        zeros = Zeros(self.net)
        zeros_rid = zeros.run(0, self.state.num_zeros.get(Params.F_len, 0), Params.F_len)

        # We assume prepreproc can be run in 2 rounds
        assert(max(rand_rid, zeros_rid) <= 2)

    def preproc(self):
        #print("Garble-Preproc ///////////////")
        bitrand = BitRand(self.net, self.state)
        errorgen = ErrorGen(self.net, self.state)

        packed_wires = (Params.W_in + Params.G_and + Params.G_xor + Params.l - 1) // Params.l
        packed_gates_AND = (Params.G_and + Params.l - 1) // Params.l
        packed_gates_XOR = (Params.G_xor + Params.l - 1) // Params.l 
        packed_gates = packed_gates_AND + packed_gates_XOR

        rid = self.rid

        # Generate MAC key.
        mgen = MacKeygen(self.net)
        mac_gen_rid = mgen.run(rid, Params.F_len)

        # Generate masks. 
        num_masks = packed_wires 
        #print("Garble-Preproc\nMask Elts: {}".format(num_masks))
        maskgen_rid = bitrand.run(rid, num_masks, Params.F_len)
     
        max_mask_or_mac = max(mac_gen_rid, maskgen_rid)

        # Generate wire keys.
        num_keys = packed_wires*2*Params.V
        #print("Garble-Preproc\nRand Elts: {}".format(num_keys))
        self.state.new_rand(num_keys, Params.F_len)

        # Authenticate wire keys and masks / add to validity checks
        auth = Auth(self.net, self.state)
        auth_rid = auth.run(max_mask_or_mac, num_keys + num_masks, Params.F_len)   
        # add to constr 
        self.state.add_constrs(num_keys+num_masks)      
        # ensure that the masks are bits 
        check_bit = CheckBit(self.net, self.state)
        check_rid = check_bit.run(auth_rid, num_masks, Params.F_len)
        
        # Generate errors.
        #print("Garble-Preproc\nError Gen Elts: {}".format(4*Params.Q*packed_gates))
        egen_rid = errorgen.run(rid, 4 * Params.Q * packed_gates, Params.F_len)

        self.rid = max(check_rid, egen_rid)

    def garble(self):
        #print("Garble ///////////")
        mult = Multiplier(self.net, self.state) # robust by default 

        packed_wires = (Params.W_in + Params.G_and + Params.G_xor + Params.l - 1) // Params.l  
        packed_gates_AND = (Params.G_and + Params.l - 1) // Params.l
        packed_gates_XOR = (Params.G_xor + Params.l - 1) // Params.l 

        rid = self.rid
        # need routing info to move packed batches out to 
        # the right x-coordinates
        # this is for all left, right keys and masks 
        trans = Trans(self.net, self.state)
        rid = trans.run(rid, (1+2*Params.V) * packed_wires, Params.F_len) 

        # need to re-route back to default share pos. for garbling this pack of gates
        # b.c. of new gen phase this will be out masks and keys will be in the correct spot
        # 4 <- pack all {0,1} wire labels together for {Left, Right}
        # 2 <- pack all masks together for {Left, Right}
        rid = trans.run(rid, (2+4*Params.V) * (packed_gates_AND + packed_gates_XOR), Params.F_len)

        # pick the plaintexts 
        # compute masks for AND (only need one to get all 4), masks for XOR can be done locally 
        rid = mult.run(rid, packed_gates_AND, Params.F_len)

        # compute the messages to encrypt for each row 
        rid = mult.run(rid, 4*(packed_gates_AND + packed_gates_XOR), Params.F_len)
        # compute c.t. constructs -- just deg red
        #print("Garble\nNew Zeros: {}\nNew Rand Elts: {}".format(4*(packed_gates_AND + packed_gates_XOR)*Params.Q, 4*(packed_gates_AND + packed_gates_XOR)*Params.Q))
        self.state.new_zeros(4*(packed_gates_AND + packed_gates_XOR)*Params.Q, Params.F_len)        
        self.state.new_rand(4*(packed_gates_AND + packed_gates_XOR)*Params.Q, Params.F_len)
        self.state.add_macs(4*(packed_gates_AND + packed_gates_XOR)*Params.Q)

        # transform shares for input wire masks and keys 
        input_output = Params.W_in + Params.W_out + (2 * Params.V * Params.W_in)
        #print("The number of input wires: {}, output wires: {}, total input values: {}".format(Params.W_in, Params.W_out, input_output))
        rid = trans.run(rid, input_output, Params.F_len)
        self.state.add_macs(input_output)
   
        self.rid = rid

    def malicious_check(self):
        rid = self.rid
        cons = CheckCons(self.net, self.state)
        rid = cons.run(rid, Params.F_len)
        mac = CheckMAC(self.net, self.state)
        rid = mac.run(rid, Params.F_len)
        # for CHECKING elements in S_ZERO protocol
        common_coin = CommonCoin(self.net, self.state)
        rid = common_coin.run(rid, Params.F_len)
        rid = self.net.all_to_all(rid, Params.L_Zero*Params.F_len)
 
        self.rid = rid


def cli_args():
    parser = argparse.ArgumentParser(
        description="Compute communication statistics for each round of the garbling protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("num_parties", type=int, help="Number of parties.")
    parser.add_argument(
        "threshold",
        type=int,
        help="Corruption threshold i.e., number of parties adversary can corrupt.",
    )
    parser.add_argument(
        "gates_and",
        type=int,
        help="Total number of AND gates in the circuit.",
    )
    parser.add_argument(
        "gates_xor",
        type=int,
        help="Total number of XOR gates in the circuit.",
    )
    parser.add_argument(
        "--in_wires",
        type=int,
        help="Number of input wires in the circuit."
    )
    parser.add_argument(
        "--out_wires",
        type=int,
        help="Number of output wires in the circuit."
    )
    parser.add_argument(
        "-d",
        "--field_degree",
        type=int,
        default=Params.F_len,
        help="Degree of polynomial modulus of extension field.",
    )
    parser.add_argument(
        "--stat_sec",
        default=Params.stat_sec,
        type=int,
        help="Statistical security parameter.",
    )
    parser.add_argument(
        "--comp_sec",
        default=Params.comp_sec,
        type=int,
        help="Computational security parameter.",
    )
    parser.add_argument(
        "--hashlen",
        default=Params.HashLen,
        type=int,
        help="Length of output of hash function in bits.",
    )
    parser.add_argument("-o", "--output", help="Directory to save the output.")

    return parser.parse_args()


def fmt_size(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1000.0:
            return f"{num:.3f} {unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f} Y{suffix}"


if __name__ == "__main__":
    args = cli_args()

    Params.n = args.num_parties
    Params.t = args.threshold
    Params.stat_sec = args.stat_sec
    Params.comp_sec = args.comp_sec
    # because the field degree must depend on the circuit size
    # packing parameter and number of parties we will set it 
    # programatically 
    Params.F_len = args.field_degree
    Params.G_and = args.gates_and
    Params.G_xor = args.gates_xor
    Params.W_in = args.in_wires
    Params.W_out = args.out_wires
    Params.HashLen = args.hashlen

    try:
        Params.initialize()
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(0)

    class SaveNetState:
        def __init__(self, net):
            self.rounds = len(net.comm.keys())
            self.total_comm = sum(map(lambda x: sum(map(sum, x)), net.comm.values()))

        @staticmethod
        def print_diff(start, end):
            rounds = end.rounds - start.rounds
            total_comm = (end.total_comm - start.total_comm) / 8

            print(f"Rounds: {rounds}")
            print(f"Total communication: {fmt_size(total_comm)}")
            print(f"Average communication: {fmt_size(total_comm / Params.n)}/party")

        def __str__(self):
            return f"Rounds: {self.rounds}, communication: {self.total_comm} bits"

    net = Network()
    garble = Garble(net)
    preproc_st = SaveNetState(net)
    garble.preproc()
    preproc_ed = SaveNetState(net)
    garble_st = SaveNetState(net)
    garble.garble()
    garble_ed = SaveNetState(net)

    malchk_st = SaveNetState(net)
    garble.malicious_check()
    malchk_ed = SaveNetState(net)

    prepreproc_st = SaveNetState(net)
    garble.prepreproc()
    prepreproc_ed = SaveNetState(net)

    preproc_rounds = preproc_ed.rounds - preproc_st.rounds + prepreproc_ed.rounds - prepreproc_st.rounds
    preproc_comm = (
        preproc_ed.total_comm
        - preproc_st.total_comm
        + prepreproc_ed.total_comm
        - prepreproc_st.total_comm
    ) / 8

    print("--- Preprocess --")
    print(f"Rounds: {preproc_rounds}")
    print(f"Total communication: {fmt_size(preproc_comm)}")
    print(f"Average communication: {fmt_size(preproc_comm / Params.n)}/party")
    print("")

    print("--- Garble ---")
    SaveNetState.print_diff(garble_st, garble_ed)

    print("\n--- Malicious Check ---")
    SaveNetState.print_diff(malchk_st, malchk_ed)

    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)

        details = {
            "n": Params.n,
            "t": Params.t,
            "field_degree": Params.F_len,
            "stat_sec": Params.stat_sec,
            "comp_sec": Params.comp_sec,
            "Reps": Params.L_Mac,
            "G_and": Params.G_and,
            "G_xor": Params.G_xor,
            "HashLen": Params.HashLen,
        }

        with open(os.path.join(args.output, "details.json"), "w") as f:
            json.dump(details, f)

        for p in range(Params.n):
            save_data = {"send": net.send_matrix(p), "receive": net.recv_matrix(p)}

            with open(os.path.join(args.output, f"party_{p}.json"), "w") as f:
                json.dump(save_data, f)

        print("")
        print(f"Saved output in {args.output}.")
