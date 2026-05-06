"""
Experiment Runner Coordinator
"""
import subprocess

def run_all_algorithms():
    scripts = [
        "experiment_dl.py",
        "experiment_nn.py",
        "experiment_rf.py",
        "experiment_dt.py",
        "experiment_svm.py",
        "experiment_hmm.py"
    ]
    for script in scripts:
        print(f"Menjalankan {script} ...")
        subprocess.run(["python", f"experiments/{script}"])

if __name__ == "__main__":
    run_all_algorithms()
