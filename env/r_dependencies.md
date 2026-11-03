# R dependencies

The focused validation uses R 4.5.1 on Apple silicon. Installed versions of all
packages referenced by retained R code are listed in `tested_r_packages.tsv`.
The table is an environment observation, not proof that each historical output
used those versions.

For the pointwise synthetic demo, install:

```r
install.packages(c("fs", "lme4", "lmerTest", "robustlmm", "readr", "dplyr", "rstudioapi"))
```

For the full selected R recipe set, read `tested_r_packages.tsv`, exclude base
packages (`parallel`, `splines`, `stats`, `utils`), and install the named packages
from CRAN or the appropriate archive. Consult the recorded versions when
reconstructing an analysis environment. This command documents package names;
it does not guarantee historical-version installation:

```r
packages <- read.delim("env/tested_r_packages.tsv")$package
install.packages(setdiff(packages, c("parallel", "splines", "stats", "utils")))
```

The manuscript workflows use both `lmer` and `rlmer` and retain their existing
inference implementations. `robustlmm` 3.3-3 and `emmeans` 1.11.2-8 were installed
for this extraction. Robust emmeans inference uses its covariance adapter and
asymptotic degrees of freedom; Kenward-Roger settings must not be assumed to
apply to `rlmer` simply because an R recipe sets a global lmer option.

Set `MYLFP_PROJECT_ROOT` and `MYLFP_REPO_ROOT` before running scripts. The R
recipes source the included `src/stats/emm/emm.R`,
`plugin/default/tabular_default.R` and `src/tabular/df_schema.R` as applicable.
The pointwise runner has its own documented CLI and does not run these scalar
recipes. Fresh package installation time was not measured.
