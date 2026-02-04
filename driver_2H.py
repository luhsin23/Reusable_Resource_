from gen_data import generate_data
from sim import run_simulation
import numpy as np

if __name__ == "__main__":
    network_type = "2H"

    d_vec = [3, 10]
    Nvec = np.arange(2000, 2100, 100)
    epochN = 100

    print(">>> Generating 2H data")
    generate_data(d_vec, Nvec, network_type=network_type, epochN=epochN, base_seed=111)

    print(">>> Running 2H simulation")
    run_simulation(d_vec, Nvec, network_type=network_type, epochN=epochN, max_workers=40, log_headcount=True)
