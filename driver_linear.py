from gen_data import generate_data
from sim import run_simulation
import numpy as np

if __name__ == "__main__":
    network_type = "linear"

    d_vec = [3, 10]
    Nvec = np.arange(500, 2100, 100)
    epochN = 400

    print(">>> Generating linear data")
    generate_data(d_vec, Nvec, network_type=network_type, epochN=epochN, base_seed=38)

    print(">>> Running linear simulation")
    run_simulation(d_vec, Nvec, network_type=network_type, epochN=epochN)
