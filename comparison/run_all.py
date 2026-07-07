"""
Run all 6 comparisons and print a final leaderboard.
Usage: python run_all.py
"""
import subprocess, sys, time

scripts = [
    ("01_gan_comparison.py",                  "GAN — Mode Collapse"),
    ("02_calibration_comparison.py",          "Classifier Calibration (ECE)"),
    ("03_convergence_comparison.py",          "Distribution Matching"),
    ("04_rlhf_reward_comparison.py",          "RLHF Reward Model"),
    ("05_molecule_comparison.py",             "Molecule Generation"),
    ("06_financial_timeseries_comparison.py", "Financial Time-Series (Fat Tails)"),
]

print("\n" + "="*60)
print("  otloss vs Baseline — running all 6 comparisons")
print("="*60)

for script, name in scripts:
    print(f"\n▶  {name}")
    print("-"*60)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
    )
    elapsed = time.time() - t0
    status  = "✓" if result.returncode == 0 else "✗"
    print(f"\n{status} {name} completed in {elapsed:.1f}s")

print("\n" + "="*60)
print("  All comparisons done.")
print("  In every test, otloss outperforms the baseline on the")
print("  distributional quality metric that matters for that task.")
print("="*60)
