"""Reproduce the pre-registered draw of four additional PR #990 commits (results/blind_commits_plan.json).

Rule: numpy default_rng(20261116); 1 + 2 + 1 commits drawn uniformly without replacement, in this order, from three
ranges of the PR history (A-D excluded): [aec34a6, f8329d0) exact-match after the state fix; [f8329d0, 2322412)
probabilistic acceptance before the lp_accept fix; [2322412, c4963b1) after the fix.
Usage: python -I draw_blind_commits.py PATH_TO_MLX_LM_PR_CLONE
"""
import subprocess
import sys

import numpy as np


def main(repo):
    log = subprocess.run(["git", "-C", repo, "log", "--reverse", "--format=%h", "1ff33d5~1..c4963b1"],
                         capture_output=True, text=True, check=True).stdout.split()
    idx = {h: i for i, h in enumerate(log)}
    used = {"1ff33d5", "673b304", "96093cf", "c4963b1"}
    rng_ = lambda a, b: [h for h in log[idx[a]:idx[b]] if h not in used]
    ranges = [rng_("aec34a6", "f8329d0"), rng_("f8329d0", "2322412"), rng_("2322412", "c4963b1")]
    rng = np.random.default_rng(20261116)
    picks = [list(rng.choice(ranges[0], 1, replace=False)), list(rng.choice(ranges[1], 2, replace=False)),
             list(rng.choice(ranges[2], 1, replace=False))]
    print([[str(x) for x in p] for p in picks])


if __name__ == "__main__":
    main(sys.argv[1])
