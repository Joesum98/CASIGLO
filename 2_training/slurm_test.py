from slurm import queue

queue = queue()
queue.verbose = True
queue.create(
    label="casiglo",
    nodes=1,
    ppn=1,
    walltime="1:00:00",
    alloc="notchpeak-gpu",
    gres="gpu",
    mem=24000
)

# script = f"python ~/casiglo/final_code/2_training/slurm_call.py"
script = f"nvidia-smi"
queue.append(script)
queue.commit(hard=True, submit=True)
