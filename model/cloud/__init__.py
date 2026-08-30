"""Azure ML submission, shaped by what this subscription actually allows.

The constraints below were established the expensive way in the prior study and
are encoded here so they do not have to be rediscovered:

* **6 dedicated regional vCPUs. Zero low-priority quota.** Spot instances are
  not merely a cost saving that is unavailable — requesting them fails outright.
* **No usable GPU.** The VM families that have quota on this subscription are
  retired hardware; the families that exist have zero quota. Every model here
  therefore has to be CPU-viable, which is why the compartments favour
  closed-form and sparse methods.
* **Use a prebuilt curated image.** Supplying a conda file makes Azure ML spin
  up a hidden 4-vCPU cluster to build it, which consumes most of the 6-core
  budget and blocks the actual job with no obvious explanation.
* **Outputs must use `UPLOAD`, not `RW_MOUNT`.** The FUSE write-mount fails with
  a 409 that surfaces as an authentication error, which sends you looking in
  entirely the wrong place.

Budget is not the binding constraint — roughly $4 of the $150 covers 40
node-hours on this hardware. Wall-clock is.
"""
from .config import AzureConfig, load_config
from .submit import submit

__all__ = ["AzureConfig", "load_config", "submit"]
