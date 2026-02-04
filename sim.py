# simulation for linear network with headcounts
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from parameters import parameters_linear, parameters_2H, parameters_non_hier
from policies_linear import perturbation_prob as perturb_linear, ptp_decision as ptp_linear, chp_decision as chp_linear, sp_decision as sp_linear
from policies_2H import perturbation_prob as perturb_2H, ptp_decision as ptp_2H, chp_decision as chp_2H, sp_decision as sp_2H
from policies_non_hier import perturbation_prob as perturb_non_hier, ptp_decision_2a3b4c as ptp_non_hier1, chp_decision_2a3b4c as chp_non_hier1, sp_decision_2a3b4c as sp_non_hier1, ptp_decision_2b3c4a as ptp_non_hier2, chp_decision_2b3c4a as chp_non_hier2, sp_decision_2b3c4a as sp_non_hier2

import heapq

base_dir = os.path.dirname(os.path.abspath(__file__))



def simulate(d, N, c_vec, epochN=200, network_type="linear",log_headcount=False):
    if network_type == "linear":
        p_vec_upper, p_vec_lower = perturb_linear(d)
        n, Lambda, mu_inv, q, r, LP = parameters_linear(d)
        ptp_decision, chp_decision, sp_decision = ptp_linear, chp_linear, sp_linear
    elif network_type == "2H":
        p_vec_upper, p_vec_lower = perturb_2H(d)
        n, Lambda, mu_inv, q, r, LP = parameters_2H(d)
        ptp_decision, chp_decision, sp_decision = ptp_2H, chp_2H, sp_2H
    elif network_type == "non_hier":
        p_vec_upper, p_vec_lower = perturb_non_hier(d)
        n, Lambda, mu_inv, q, r, LP = parameters_non_hier()
        ptp_decision, chp_decision, sp_decision = ptp_non_hier1, chp_non_hier1, sp_non_hier1
    else:
        raise ValueError("network_type must be 'linear' or '2H'")
    
    np.random.seed(130)  
    random.seed(130) 

    AverageReward_PTP, AverageReward_CHP, AverageReward_SP = np.zeros(epochN), np.zeros(epochN), np.zeros(epochN)
    Time_vec = np.zeros(epochN)
    start_time = time.time()

    Q_N = N * q
    C_logN = np.log(N) * c_vec

    Headcount_PTP, Headcount_CHP, Headcount_SP, Headcount_time = [], [], [], []

    for k in range(epochN):
        # Load data
        file_name = os.path.join(base_dir, f"{network_type}_Data_hc/N={N}/d={d}/Epoch-{k}.npz")
        data = np.load(file_name)
        TimeChain = data['TimeChain']
        T = data['T']

        if k == 0:
            Time_vec[k] = T
        else:
            Time_vec[k] = Time_vec[k - 1] + T

        arrivalN = TimeChain.shape[1]  # total number of arrival batches

        # Initialization
        if k == 0:
            X_PTP, X_CHP, X_SP = np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n, dtype=int)
            departures_PTP, departures_CHP, departures_SP = [], [], []

        else:
            X_PTP,X_CHP, X_SP = postState_PTP.copy(), postState_CHP.copy(), postState_SP.copy()
            departures_PTP, departures_CHP, departures_SP = postDepartures_PTP.copy(), postDepartures_CHP.copy(), postDepartures_SP.copy()

        # Simulations
        TotalReward_PTP, TotalReward_CHP, TotalReward_SP = 0, 0, 0

        Decisions_PTP, Decisions_CHP, Decisions_SP = np.zeros(arrivalN), np.zeros(arrivalN), np.zeros(arrivalN)

        for i in range(arrivalN):
            arriving_time = TimeChain[0, i]
            arriving_type = int(TimeChain[2, i])
            departure_time = TimeChain[1, i]

            decision_SP, decision_CHP, decision_PTP = 0, 0, 0
            dedicated_resource = arriving_type // 2
            
            # Update headcounts
            while departures_PTP and departures_PTP[0][0] <= arriving_time:
                dep_time, job_type = heapq.heappop(departures_PTP)
                X_PTP[job_type] -= 1

            while departures_CHP and departures_CHP[0][0] <= arriving_time:
                dep_time, job_type = heapq.heappop(departures_CHP)
                X_CHP[job_type] -= 1
       
            while departures_SP and departures_SP[0][0] <= arriving_time:
                dep_time, job_type = heapq.heappop(departures_SP)
                X_SP[job_type] -= 1

            # Make decisions
            decision_PTP = ptp_decision(arriving_type, X_PTP, d, n, N, Q_N, C_logN,
                            dedicated_resource, p_vec_upper, p_vec_lower)

            decision_CHP = chp_decision(arriving_type, X_CHP, d, n, N, Q_N, C_logN,
                                        dedicated_resource)

            decision_SP = sp_decision(arriving_type, X_SP, d, n, N, Q_N, C_logN, dedicated_resource)

            # Update departure heaps if accepted
            if decision_PTP == 1:
                X_PTP[arriving_type] += 1
                heapq.heappush(departures_PTP, (departure_time, arriving_type))
            
            if decision_CHP == 1:
                X_CHP[arriving_type] += 1
                heapq.heappush(departures_CHP, (departure_time, arriving_type))
            
            if decision_SP == 1:
                X_SP[arriving_type] += 1
                heapq.heappush(departures_SP, (departure_time, arriving_type))

            
            # Decision
            Decisions_PTP[i] = decision_PTP
            Decisions_CHP[i] = decision_CHP
            Decisions_SP[i] = decision_SP

            # Reward
            TotalReward_PTP += r[arriving_type] * decision_PTP
            TotalReward_CHP += r[arriving_type] * decision_CHP
            TotalReward_SP += r[arriving_type] * decision_SP

            # Log headcounts and time
            Headcount_PTP.append(X_PTP.copy())
            Headcount_CHP.append(X_CHP.copy())
            Headcount_SP.append(X_SP.copy())
            Headcount_time.append(TimeChain[0, i] + Time_vec[k])


        # Update reward vector
        if k == 0:
            AverageReward_PTP[k] = TotalReward_PTP / T
            AverageReward_CHP[k] = TotalReward_CHP / T
            AverageReward_SP[k] = TotalReward_SP / T

        else:
            AverageReward_PTP[k] = (AverageReward_PTP[k - 1] * Time_vec[k - 1] + TotalReward_PTP) / Time_vec[k]
            AverageReward_CHP[k] = (AverageReward_CHP[k - 1] * Time_vec[k - 1] + TotalReward_CHP) / Time_vec[k]
            AverageReward_SP[k] = (AverageReward_SP[k - 1] * Time_vec[k - 1] + TotalReward_SP) / Time_vec[k]
        

        # Update remaining chain for next epoch
        postDepartures_PTP, postDepartures_CHP, postDepartures_SP = [], [], []
        postState_PTP, postState_CHP, postState_SP = np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n, dtype=int)

        while departures_PTP:
            dep_time, job_type = heapq.heappop(departures_PTP)
            new_dep_time = dep_time - T   # shift into next epoch’s clock
            if new_dep_time > 0:          # only keep jobs that survive
                heapq.heappush(postDepartures_PTP, (new_dep_time, job_type))
                postState_PTP[job_type] += 1

        while departures_CHP:
            dep_time, job_type = heapq.heappop(departures_CHP)
            new_dep_time = dep_time - T
            if new_dep_time > 0:
                heapq.heappush(postDepartures_CHP, (new_dep_time, job_type))
                postState_CHP[job_type] += 1

        while departures_SP:
            dep_time, job_type = heapq.heappop(departures_SP)
            new_dep_time = dep_time - T
            if new_dep_time > 0:
                heapq.heappush(postDepartures_SP, (new_dep_time, job_type))
                postState_SP[job_type] += 1

        print(f"d={d} N={N} Epoch {k+1}/{epochN} completed. X_PTP: {X_PTP}, X_CHP: {X_CHP}, X_SP: {X_SP}, time spent: {time.time() - start_time:.2f} seconds")

    # print correlation between less preferred types
    if log_headcount:
        headcounts_PTP, headcounts_CHP, headcounts_SP = np.array(Headcount_PTP), np.array(Headcount_CHP), np.array(Headcount_SP)


    # save data
    folder = os.path.join(base_dir, f"{network_type}_sim_result_hc/N={N}/d={d}")
    os.makedirs(folder, exist_ok=True)
    if log_headcount:
        file_name = os.path.join(folder, f"headcounts_epoch={epochN}.npz")
        np.savez(file_name, headcounts_PTP=headcounts_PTP, headcounts_CHP=headcounts_CHP, headcounts_SP=headcounts_SP)
    file_name = os.path.join(folder, f"reward_epoch={epochN}.npz")
    np.savez(file_name, Time_vec=Time_vec, AverageReward_PTP=AverageReward_PTP, AverageReward_CHP=AverageReward_CHP, AverageReward_SP=AverageReward_SP, LP=LP, N=N, c_vec=c_vec, epochN=epochN, d=d, postState_PTP=postState_PTP, postState_CHP=postState_CHP, postState_SP=postState_SP)
    print(f"Data saved for d={d}")

    end_time = time.time()
    total_time = "{:.0f}".format(end_time - start_time)
    print(f"PROCESS COMPLETED FOR d={d} N={N}")
    print(f'd={d} Total time: {total_time} seconds')
    
    

def run_simulation(d_vec, Nvec, epochN=200, network_type="linear", max_workers=40, log_headcount=False):
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for N in Nvec:
            for d in d_vec:
                c_vec = np.ones(d) * 10
                futures.append(executor.submit(simulate, d, N, c_vec, epochN, network_type, log_headcount))
        for future in as_completed(futures):
            future.result()
    print(f"Simulation finished in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    d_vec = [10]
    Nvec = [3000]
    run_simulation(d_vec, Nvec, network_type="linear")
