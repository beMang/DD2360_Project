import matplotlib.pyplot as plt
import subprocess
import os

exec = "./bin/cudart_ref"

img_dir = "graphs/"
data_dir = "tmp/"

n_obj = [500, 1000, 2000, 4000, 8000, 16000]
n_obj_accurate = []

rand_init_time = []
create_world_time = []
render_time = []
save_image_time = []
free_cuda_time = []

def get_time_from_output(output, label):
    times = []
    for line in output.strip().split('\n'):
        if label in line:
            times.append(float(line.split(" ")[1]))
    return times

def get_n_obj_from_output(output, label):
    for line in output.strip().split('\n'):
        if label in line:
            return int(line.split(" ")[-2])
    return None

# check for data file
if not os.path.exists(f'{data_dir}profiling_ref_times.csv'):
    for n in n_obj:
        result = subprocess.run([exec, str(n)], stdout=subprocess.PIPE, text=True)
        n_obj_accurate.append(get_n_obj_from_output(result.stdout, "took"))
        record = get_time_from_output(result.stdout, ":")
        rand_init_time.append(record[0])
        create_world_time.append(record[1])
        render_time.append(record[2])
        save_image_time.append(record[3])
        free_cuda_time.append(record[4])
    # save to csv
    with open(f'{data_dir}profiling_ref_times.csv', 'w') as f:
        f.write("n_obj,rand_init,create_world,render,save_image,free_cuda\n")
        for i in range(len(n_obj_accurate)):
            f.write(f"{n_obj_accurate[i]},{rand_init_time[i]},{create_world_time[i]},{render_time[i]},{save_image_time[i]},{free_cuda_time[i]}\n")
else:
    with open(f'{data_dir}profiling_ref_times.csv', 'r') as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            parts = line.strip().split(',')
            n_obj_accurate.append(int(parts[0]))
            rand_init_time.append(float(parts[1]))
            create_world_time.append(float(parts[2]))
            render_time.append(float(parts[3]))
            save_image_time.append(float(parts[4]))
            free_cuda_time.append(float(parts[5]))

# plot bar chart with stacked bars for each number of objects
labels = [str(n) for n in n_obj_accurate]
x = range(len(labels))
plt.bar(x, rand_init_time, label='Memory allocation')
plt.bar(x, create_world_time, bottom=rand_init_time, label='World creation')
bottoms = [i+j for i,j in zip(rand_init_time, create_world_time)]
plt.bar(x, render_time, bottom=bottoms, label='Rendering')
bottoms = [i+j for i,j in zip(bottoms, render_time)]
plt.bar(x, save_image_time, bottom=bottoms, label='Image saving')
bottoms = [i+j for i,j in zip(bottoms, save_image_time)]
plt.bar(x, free_cuda_time, bottom=bottoms, label='Ressource freeing')
plt.xticks(x, labels)
plt.xlabel('Number of Objects')
plt.ylabel('Time (s)')
#plt.title('Profiling of Reference CUDA Ray Tracer')
plt.legend()
plt.tight_layout()
plt.savefig(f'{img_dir}profiling_ref.pdf')
plt.clf()
