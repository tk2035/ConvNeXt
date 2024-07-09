import os
import signal
import submitit
import subprocess

class CustomController:
    def __init__(self, folder: str):
        self.folder = folder
        self.ntasks = int(os.environ.get("SUBMITIT_LOCAL_NTASKS", "1"))
        self.command = os.environ.get("SUBMITIT_LOCAL_COMMAND", "python")
        self.timeout_s = int(os.environ.get("SUBMITIT_LOCAL_TIMEOUT_S", "3600"))
        self.signal_delay_s = int(os.environ.get("SUBMITIT_LOCAL_SIGNAL_DELAY_S", "120"))
        self.stderr_to_stdout = bool(os.environ.get("SUBMITIT_STDERR_TO_STDOUT", "1"))
        self.with_shell = bool(os.environ.get("SUBMITIT_LOCAL_WITH_SHELL", "0"))
        self.pids = []

        print("CustomController initialized with the following parameters:")
        print("folder =", self.folder)
        print("ntasks =", self.ntasks)
        print("command =", self.command)
        print("timeout_s =", self.timeout_s)
        print("signal_delay_s =", self.signal_delay_s)
        print("stderr_to_stdout =", self.stderr_to_stdout)
        print("with_shell =", self.with_shell)

    def run(self):
        print("Running custom controller with folder:", self.folder)
        process = subprocess.Popen(
            self.command,
            shell=self.with_shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if self.stderr_to_stdout else subprocess.PIPE
        )
        self.pids.append(process.pid)
        output, _ = process.communicate()
        print(f"Process {process.pid} finished with return code {process.returncode}")
        print(f"Process output: {output.decode('utf-8')}")

    def kill_tasks(self):
        signals = [signal.SIGINT, signal.SIGTERM]  # Use signals available on Windows
        for sig in signals:
            for pid in self.pids:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass

class CustomLocalExecutor(submitit.LocalExecutor):
    def __init__(self, folder: str):
        super().__init__(folder)
        self._controller = CustomController(folder)
        print("CustomLocalExecutor initialized")



