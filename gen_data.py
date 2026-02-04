# Generate data for 2H and linear
import numpy as np
import random
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from parameters import parameters_linear, parameters_2H, parameters_non_hier
 

start_time = time.time()   


def parallel_gen_data(d=0, N=500, network_type="2H", epochN=200, base_seed=38):

    # np.random.seed(int(time.time()) + os.getpid())  
    # random.seed(int(time.time()) + os.getpid())
    worker_seed = base_seed + hash((d, N)) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    if network_type == "linear":
        n, Lambda, mu_inv, q, r, LP = parameters_linear(d)
    elif network_type == "2H":
        n, Lambda, mu_inv, q, r, LP = parameters_2H(d)
    elif network_type == "non_hier":
        n, Lambda, mu_inv, q, r, LP = parameters_non_hier()
    else:
        raise ValueError("network_type must be 'linear' or '2H' or 'non_hier'")


    K = 5 * N
    ArrivalTimes = np.empty((n, K))
    ServiceTimes = np.empty((n, K))
    customers = range(n)
    Time_vec = np.zeros(epochN)

    # Create folder if not exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, f"{network_type}_Data_hc/N={N}/d={d}")
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Generate data
    for k in range(epochN):
        for i in customers:
            InterArrivalTimes = np.random.exponential(1 / N / Lambda[i], K)
            ArrivalTimes[i] = np.cumsum(InterArrivalTimes)
            ServiceTimes[i] = np.random.exponential(mu_inv[i], K)
        T = np.amin(ArrivalTimes[:, -1])
        if k == 0:
            Time_vec[k] = T
        else:
            Time_vec[k] = Time_vec[k-1] + T
        ArrivalTimes = np.where(ArrivalTimes<T, ArrivalTimes, 0)
        # Combine arrival times and service times
        TimeChain = np.array([])
        for i in customers:
            Chain = ArrivalTimes[i, :]
            Chain = Chain[Chain != 0]
            # first row arriving time, second row departure time, third row type
            Chain = np.vstack((Chain, Chain+ServiceTimes[i, 0:Chain.size], (i)*np.ones((1, Chain.size))))
            TimeChain = np.hstack((TimeChain, Chain)) if TimeChain.size else Chain
        TimeChain = TimeChain[:, TimeChain[0, :].argsort()]   # sort by arrival time

        # Save data of this epoch
        file_name = os.path.join(folder, f"Epoch-{k}.npz")
        np.savez(file_name, TimeChain=TimeChain, T=T)
        print("File saved:", file_name)



def generate_data(d_vec, Nvec, network_type="2H", epochN=200, max_workers=40, base_seed=42):

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(parallel_gen_data, d, N, network_type, epochN, base_seed)
            for N in Nvec for d in d_vec
        ]
        for future in as_completed(futures):
            future.result()
    print(f"Data generation finished in {time.time() - start_time:.2f} seconds")



if __name__ == '__main__':
    d_vec = [3]           
    Nvec = [500, 600] 
    network_type = "2H"     
    generate_data(d_vec, Nvec, network_type=network_type, max_workers=40)


