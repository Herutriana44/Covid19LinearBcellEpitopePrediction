"""
Experiment Runner Coordinator

Setiap algoritma memiliki script eksperimen khusus di folder 'experiments', misal:
- experiment_dl.py   (Deep Learning)
- experiment_nn.py   (Neural Network)
- experiment_rf.py   (Random Forest)
- experiment_dt.py   (Decision Tree)
- experiment_svm.py  (SVM)
- experiment_hmm.py  (HMM)

Jalankan script sesuai algoritma yang ingin diuji!
Contoh: python experiments/experiment_rf.py
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
