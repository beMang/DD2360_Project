import matplotlib.pyplot as plt
import subprocess
import os

exec = "./bin/cudart"

img_dir = "graphs/"
data_dir = "tmp/"

n_obj = [500, 1000, 2000, 4000, 8000, 16000]
n_obj_accurate = []

fb_alloc = []
scene_gen = []
bvh_build = []
render_init = []
render_time = []
image_save = []
cleanup = []

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
if not os.path.exists(f'{data_dir}profiling_opti_times.csv'):
    for n in n_obj:
        result = subprocess.run([exec, str(n)], stdout=subprocess.PIPE, text=True)
        n_obj_accurate.append(get_n_obj_from_output(result.stdout, "took"))
        record = get_time_from_output(result.stdout, "for")
        fb_alloc.append(record[0])
        scene_gen.append(record[1])
        bvh_build.append(record[2])
        render_init.append(record[3])
        render_time.append(record[4])
        image_save.append(record[5])
        cleanup.append(record[6])
    # save to csv
    with open(f'{data_dir}profiling_opti_times.csv', 'w') as f:
        f.write("n_obj,fb_alloc,scene_gen,bvh_build,render_init,render_time,image_save,cleanup\n")
        for i in range(len(n_obj_accurate)):
            f.write(f"{n_obj_accurate[i]},{fb_alloc[i]},{scene_gen[i]},{bvh_build[i]},{render_init[i]},{render_time[i]},{image_save[i]},{cleanup[i]}\n")
else:
    with open(f'{data_dir}profiling_opti_times.csv', 'r') as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            parts = line.strip().split(',')
            n_obj_accurate.append(int(parts[0]))
            fb_alloc.append(float(parts[1]))
            scene_gen.append(float(parts[2]))
            bvh_build.append(float(parts[3]))
            render_init.append(float(parts[4]))
            render_time.append(float(parts[5]))
            image_save.append(float(parts[6]))
            cleanup.append(float(parts[7]))

# plot bar chart with stacked bars for each number of objects
labels = [str(n) for n in n_obj_accurate]
x = range(len(labels))
plt.bar(x, fb_alloc, label='Framebuffer allocation')
plt.bar(x, scene_gen, bottom=fb_alloc, label='World creation')
bottoms = [i+j for i,j in zip(fb_alloc, scene_gen)]
plt.bar(x, bvh_build, bottom=bottoms, label='BVH building')
bottoms = [i+j for i,j in zip(bottoms, bvh_build)]
plt.bar(x, render_init, bottom=bottoms, label='Rendering initialization')
bottoms = [i+j for i,j in zip(bottoms, render_init)]
plt.bar(x, render_time, bottom=bottoms, label='Rendering')
bottoms = [i+j for i,j in zip(bottoms, render_time)]
plt.bar(x, image_save, bottom=bottoms, label='Image saving')
bottoms = [i+j for i,j in zip(bottoms, image_save)]
plt.bar(x, cleanup, bottom=bottoms, label='Ressource freeing')
plt.xticks(x, labels)
plt.xlabel('Number of Objects')
plt.ylabel('Time (s)')
#plt.title('Profiling of Optimized CUDA Ray Tracer')
plt.legend()
plt.tight_layout()
plt.savefig(f'{img_dir}profiling_opti.pdf')
plt.clf()
