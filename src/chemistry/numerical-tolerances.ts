/** Cauchy–Schwarz ERI screening threshold — drop shell pairs whose upper bound is below this. */
export const SCHWARZ_SCREEN_TOL = 1e-10;

/** Density-fitting metric / Cholesky decomposition regularization threshold. */
export const DF_CHOLESKY_TOL = 1e-10;

/** Default SCF DIIS residual (||e||_∞) tolerance. */
export const SCF_RESIDUAL_TOL = 1e-5;

/** Relative finite-difference step scale for numerical XC-kernel derivatives (h ~ ρ · FD_REL_STEP). */
export const FD_REL_STEP = 1e-5;

/** Absolute floor for FD steps, guarding against division by ~zero density. */
export const FD_STEP_FLOOR = 1e-9;
