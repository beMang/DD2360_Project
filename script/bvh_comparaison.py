import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

exec = "./bin/cudart"
ref_exec = "./bin/cudart_ref"

img_dir = "graphs/"
data_dir = "tmp/"

n_obj = [500, 1000, 2000, 4000, 8000, 16000]
n_obj_accurate = []
times = []
ref_times = []

def get_time_from_output(output):
    for line in output.strip().split('\n'):
        if "took" in line:
            return float(line.split(" ")[1])
    return None

def get_num_of_object_from_output(output):
    for line in output.strip().split('\n'):
        if "objects" in line:
            return int(line.split(" ")[-2])
    return None

# check for data file
if not os.path.exists(f'{data_dir}bvh_comparison_times.csv'):
    for n in n_obj:
        #lanch subprocess to run the executable
        n_run = 5
        ref_time = 0
        opti_time = 0

        for _ in range(n_run):
            opt_result = subprocess.run([exec, str(n)], stdout=subprocess.PIPE, text=True)
            opti_time += get_time_from_output(opt_result.stdout)
            
            n_obj_actual = get_num_of_object_from_output(opt_result.stdout)
            if n_obj_actual not in n_obj_accurate:
                n_obj_accurate.append(n_obj_actual)

            #lanch subprocess to run the reference executable
            ref_result = subprocess.run([ref_exec, str(n)], stdout=subprocess.PIPE, text=True)
            ref_time += get_time_from_output(ref_result.stdout)

        print(f'Completed for n_obj={n}')

        times.append(opti_time / n_run)
        ref_times.append(ref_time / n_run)
    #save data to file
    print(n_obj_accurate)
    np.savetxt(f'{data_dir}bvh_comparison_times.csv', np.array([n_obj, n_obj_accurate, times, ref_times]).T, header='n_obj,n_obj_accurate,opti_time,ref_time', delimiter=',')
else:
    data = np.loadtxt(f'{data_dir}bvh_comparison_times.csv', skiprows=1, delimiter=',')
    n_obj = data[:,0].astype(int).tolist()
    n_obj_accurate = data[:,1].astype(int).tolist()
    times = data[:,2].tolist()
    ref_times = data[:,3].tolist()

plt.plot(n_obj_accurate, times, '--x', label='Optimized', color='green', alpha=0.8)
plt.plot(n_obj_accurate, ref_times, '--x', label='Reference', color='blue', alpha=0.8)
plt.xlabel('Number of objects')
plt.ylabel('Time (s)')
plt.title('Execution time comparison between\n reference and BVH optimized version')
plt.legend()
plt.yscale('log')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'{img_dir}bvh_comparison.png')
plt.close()
