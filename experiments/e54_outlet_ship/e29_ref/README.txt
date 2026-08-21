The e29_swa_ep15 recipe rerun in the mainline tree on the outlet-carrying
pickles, 5 seeds. This is two things at once:

  * the IDENTITY GATE. Every seed*.json here is md5-identical to
    experiments/e29_swa_ep15/seed*.json, and the checkpoints these runs wrote
    were md5-identical to experiments/e29_swa_ep15/seed*.pt (seed42
    22785b60360b7c21edc1ca8b7261167e / 1d84e9529c77db0ef5b9c639d2dff96f).
    Appending the outlet block to the pickles moved no existing column.
  * the PAIRED BASELINE for tools/e54_aggregate.py. The frozen e29 directory
    has no seed*.metrics2.json -- it predates the Phase-0 metric suite -- so
    the reference had to be re-scored by the same code that scores the arm.

The checkpoints are not kept: they are byte-identical to
experiments/e29_swa_ep15/seed*.pt, which is what the withheld-outlet
evaluation points at.
