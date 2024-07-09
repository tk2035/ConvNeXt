import argparse
import os
import uuid
from pathlib import Path

import main as classification
from custom_executor import CustomLocalExecutor  # Import the custom executor

import submitit

# setup = os.environ["SUBMITIT_LOCAL_NTASKS"]


def parse_args():
    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for ConvNeXt", parents=[classification_parser])
    parser.add_argument("--ngpus", default=4, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=72, type=int, help="Duration of the job, in hours")
    parser.add_argument("--job_name", default="convnext", type=str, help="Job name")
    parser.add_argument("--job_dir", default="convnext_job", type=str, help="Job directory; leave empty for default")
    parser.add_argument("--partition", default="learnlab", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', default=True, help="Big models? Use this")
    parser.add_argument('--comment', default="", type=str, help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()
    

def get_shared_folder() -> Path:
    shared_folder = Path("C:/TEJA/shared_folder").resolve()
    shared_folder.mkdir(parents=True, exist_ok=True)
    return shared_folder

def get_init_file():
    shared_folder = get_shared_folder()
    init_file = shared_folder / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file.absolute()

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main as classification
        self._setup_gpu_args()
        classification.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        self.args.auto_resume = True
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(self.args.job_dir).resolve()
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def get_executor(args):
    
    try:
        
        executor = submitit.AutoExecutor(folder=args.job_dir, cluster="slurm", slurm_max_num_timeout=30)
        executor.update_parameters(
            mem_gb=40 * args.ngpus,
            gpus_per_node=args.ngpus,
            tasks_per_node=args.ngpus,
            cpus_per_task=10,
            nodes=args.nodes,
            timeout_min=args.timeout * 60,
            slurm_partition=args.partition,
            slurm_signal_delay_s=120,
            name=args.job_name,
            **({'slurm_constraint': 'volta32gb'} if args.use_volta32 else {}),
            **({'slurm_comment': args.comment} if args.comment else {})
        )
    except RuntimeError:
        print("SLURM not detected, switching to CustomLocalExecutor")
        os.environ["SUBMITIT_LOCAL_NTASKS"] = str(args.ngpus * args.nodes)  # This is the line you need to add
        os.environ["SUBMITIT_LOCAL_COMMAND"] = "python" 
        os.environ["SUBMITIT_LOCAL_TIMEOUT_S"] = str(args.timeout * 3600)
        os.environ["SUBMITIT_LOCAL_SIGNAL_DELAY_S"] = "120"
        os.environ["SUBMITIT_STDERR_TO_STDOUT"] = "1" 
        os.environ["SUBMITIT_LOCAL_WITH_SHELL"] = "0"
        

        # Debug prints to check environment variables
        print("Environment variables set:")
        print("SUBMITIT_LOCAL_NTASKS =", os.environ["SUBMITIT_LOCAL_NTASKS"])
        print("SUBMITIT_LOCAL_COMMAND =", os.environ["SUBMITIT_LOCAL_COMMAND"])
        print("SUBMITIT_LOCAL_TIMEOUT_S =", os.environ["SUBMITIT_LOCAL_TIMEOUT_S"])
        print("SUBMITIT_LOCAL_SIGNAL_DELAY_S =", os.environ["SUBMITIT_LOCAL_SIGNAL_DELAY_S"])
        print("SUBMITIT_STDERR_TO_STDOUT =", os.environ["SUBMITIT_STDERR_TO_STDOUT"])
        print("SUBMITIT_LOCAL_WITH_SHELL =", os.environ["SUBMITIT_LOCAL_WITH_SHELL"])

        executor = CustomLocalExecutor(folder=args.job_dir)
        executor.update_parameters(
            mem_gb=40 * args.ngpus,
            gpus_per_node=args.ngpus,
            tasks_per_node=args.ngpus,
            cpus_per_task=10,
            # setup = str(args.ngpus * args.nodes),
            nodes=1,  # Use only one node for local execution
            timeout_min=args.timeout * 60,
            name=args.job_name
        )
    return executor

def main():
    args = parse_args()
    
    if args.job_dir == "":
        args.job_dir = str(get_shared_folder() / "%j")

    executor = get_executor(args)

    args.dist_url = get_init_file().as_uri()
    args.output_dir = str(Path(args.job_dir).resolve())

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)

if __name__ == "__main__":
    main()
