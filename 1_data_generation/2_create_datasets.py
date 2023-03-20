from slurm import queue
from config import n_tasks 

queue = queue()
queue.verbose = True
queue.create(label='casiglo', nodes=1, ppn=n_tasks, walltime='72:00:00', alloc='brownstein', partition="notchpeak")

for task_number in range(n_tasks):
    script = f"python ~/casiglo/final_code/1_data_generation/slurm_queue.py {task_number}"
    queue.append(script)  

queue.commit(hard=True, submit=True)
