import numpy as np
import argparse
import struct
import sys

def read_head_size_from_bin(file_path):
    """Reads the llama2.c v2 checkpoint header to extract head_size."""
    try:
        with open(file_path, 'rb') as f:
            # 1. Magic number (uint32)
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != 0x616b3432:
                print(f"Error: {file_path} is not a valid llama2.c model (Bad magic).")
                return None
            
            # 2. Version (int)
            version = struct.unpack('<i', f.read(4))[0]
            if version != 2:
                print(f"Error: Unsupported version {version}, need version 2.")
                return None
            
            # 3. Config (7 ints)
            # dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
            config = struct.unpack('<7i', f.read(28))
            dim = config[0]
            n_heads = config[3]
            
            head_size = dim // n_heads
            print(f"Auto-detected from {file_path}: dim={dim}, n_heads={n_heads} -> head_size={head_size}")
            return head_size
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def build_lloyd_max(d, k):
    N = 20001
    x = np.linspace(-1.0, 1.0, N)
    t = 1.0 - x**2
    t[t < 0] = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.exp(0.5 * (d - 3) * np.log(t))
    w[t <= 0] = 0
    w /= np.sum(w)
    
    c = np.zeros(k)
    acc, i = 0.0, 0
    for j in range(N):
        acc += w[j]
        while i < k and acc >= (i + 0.5) / k:
            c[i] = x[j]
            i += 1
    while i < k:
        c[i] = x[-1]
        i += 1

    for _ in range(300):
        bnd = 0.5 * (c[:-1] + c[1:])
        indices = np.digitize(x, bnd)
        new_c = np.zeros(k)
        for idx in range(k):
            mask = (indices == idx)
            sw = np.sum(w[mask])
            if sw > 0:
                new_c[idx] = np.sum(w[mask] * x[mask]) / sw
        if np.max(np.abs(new_c - c)) < 1e-13:
            break
        c = new_c
        
    thresh = 0.5 * (c[:-1] + c[1:])
    return c, thresh

def format_c_array(name, arr):
    flat = arr.flatten()
    res = f"static const float {name}[{len(flat)}] = {{\n"
    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        res += "    " + ", ".join([f"{val:.8f}f" for val in chunk]) + ",\n"
    res += "};\n"
    return res

def generate_tie_centroid_rom(c, K, mse_bits, out_filename):
    """Generates the tq_cent_table.tie hardware ROM file."""
    # Convert floats to Q15 format (clamped between -32768 and 32767)
    q15_c = np.round(c * 32768.0).clip(-32768, 32767).astype(np.int16)
    
    lines = []
    lines.append("// Auto-generated tq_cent_table.tie")
    lines.append("// Hardware ROM for TurboQuant Centroids\n")
    
    # Determine bit width for the index parameter in TIE
    idx_width = max(0, mse_bits - 1)
    lines.append(f"function [15:0] tq_get_cent ( [{idx_width}:0] idx ) {{")
    
    # 1. Wire declarations
    for i in range(K):
        # Extract unsigned 16-bit hex representation (cast to Python int first!)
        hex_val = f"{int(q15_c[i]) & 0xFFFF:04X}"
        lines.append(f"    wire [15:0] c{i} = 16'h{hex_val}; // float: {c[i]:.6f}")
        
    lines.append("\n    // Single-cycle combinatorial MUX")
    lines.append("    assign tq_get_cent = ")
    
    # 2. Mux assignments
    for i in range(K - 1):
        lines.append(f"        (idx == {mse_bits}'d{i}) ? c{i} :")
    
    lines.append(f"        c{K-1};")
    lines.append("}")
    
    with open(out_filename, "w") as f:
        f.write("\n".join(lines) + "\n")

#WE CHANGED TO Q12 INSTEAD TO HAVE HEADROOM - FUNCTION NAME IS STILL Q15
def format_c_array_q15(name, arr):
    # Multiply by 32768 and cast to 16-bit signed integer
    #q15_arr = np.round(arr * 32768.0).clip(-32768, 32767).astype(np.int16)

	# Multiply by 4096 for Q12 scaling (leaves 8 bits of headroom for 32-bit accumulators)
    q15_arr = np.round(arr * 4096.0).clip(-4096, 4095).astype(np.int16)
    flat = q15_arr.flatten()
    res = f"static const int16_t {name}[{len(flat)}] = {{\n"
    for i in range(0, len(flat), 16):
        chunk = flat[i:i+16]
        res += "    " + ", ".join([f"{val}" for val in chunk]) + ",\n"
    res += "};\n"
    return res

def main():
    parser = argparse.ArgumentParser(description="Generate TurboQuant C Header Constants and TIE Hardware ROM")
    parser.add_argument('-m', '--model', type=str, help="Path to .bin checkpoint to auto-detect head size")
    parser.add_argument('-d', '--dim', type=int, default=64, help="Manual head size (ignored if -m is used)")
    parser.add_argument('-b', '--bits', type=int, default=4, help="TurboQuant bits (default: 4)")
    parser.add_argument('-o', '--out', type=str, default="tq_constants.h", help="Output C header file name")
    parser.add_argument('-t', '--tie_out', type=str, default="tq_cent_table.tie", help="Output TIE file name")
    args = parser.parse_args()

    # Determine Head Size (D)
    D = args.dim
    if args.model:
        detected_d = read_head_size_from_bin(args.model)
        if detected_d is not None:
            D = detected_d
        else:
            sys.exit(1)

    BITS = args.bits
    PROD_MODE = True
    MSE_BITS = BITS - 1 if PROD_MODE else BITS
    K = 1 << MSE_BITS

    print(f"Generating constants for Head Size (D)={D}, Bits={BITS}...")

    # Math Generation
    np.random.seed(0x54517545)
    H = np.random.randn(D, D)
    Q, R = np.linalg.qr(H)
    Pi = Q @ np.diag(np.sign(np.diag(R)))
    M = np.random.randn(D, D)
    c, thresh = build_lloyd_max(D, K)

    # 1. Write the C Header
    with open(args.out, "w") as f:
        f.write(f"// TurboQuant Precomputed Constants\n")
        f.write(f"// Auto-generated for Head Size: {D}, Bits: {BITS}\n\n")
        
        # Inject Macros for the C code to read!
        f.write(f"#define TQ_EXPECTED_D {D}\n")
        f.write(f"#define TQ_EXPECTED_B {BITS}\n\n")
        
        f.write(format_c_array("TQ_CENTROIDS", c))
        f.write(format_c_array("TQ_THRESHOLDS", thresh))
        f.write(format_c_array("TQ_PI_MATRIX", Pi))
        f.write(format_c_array("TQ_M_MATRIX", M))
		
		#Q15 version for the DEDICATED TIE
        f.write(format_c_array_q15("TQ_PI_Q15", Pi))
        f.write(format_c_array_q15("TQ_M_Q15", M))
        # ---> TRANSPOSE <---
        f.write(format_c_array_q15("TQ_PI_T_Q15", Pi.T))
        f.write(format_c_array_q15("TQ_M_T_Q15", M.T))

    print(f"Successfully saved C Header to {args.out}")

    # 2. Write the TIE ROM File
    generate_tie_centroid_rom(c, K, MSE_BITS, args.tie_out)
    print(f"Successfully saved TIE ROM to {args.tie_out}")

if __name__ == "__main__":
    main()
