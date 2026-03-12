pkgpath <- "~/Documents/ecod024/bw05"

devtools::clean_dll(pkgpath)
cpp4r::register(pkgpath)
devtools::document(pkgpath)
devtools::install(pkgpath)

library(bw05)

# Setup ----
cat("\n=== bw05 parameter check ===\n")
print(bw05_params(list()))          # baseline (log utility, psi=0)
print(bw05_params(list(psi = -8)))  # high risk aversion
cat("==============================\n\n")

N_K     <- 100L
T_TOTAL <- 500000L   # BW footnote 31: 500,000 periods
T_BURN  <- 60000L    # BW footnote 31: discard first 60,000
SEED    <- 42L

scenarios <- list(
  "baseline" = list(),
  "high_ra"  = list(psi = -8.0, tau_h = 0.2069),
  "only_z"   = list(sigma_g = 1e-8, rho_g = 0.0),
  "only_g"   = list(sigma_z = 1e-8, rho_z = 0.0),
  "iid"      = list(rho_z = 0.0, rho_g = 0.0)
)

var_labels  <- names(bw5)
stat_labels <- c("E", "sd", "autocorr", "corr_g", "corr_z")


# Helper: run one simulation table ----
run_sim <- function(ref, n_vars = 3) {
  rows <- list()
  for (sc_name in names(scenarios)) {
    res <- bw05_stats(scenarios[[sc_name]],
                       n_k = N_K, T_total = T_TOTAL, T_burn = T_BURN, seed = SEED)

    for (vi in seq_len(n_vars)) {
      vname <- var_labels[vi]
      for (si in seq_along(stat_labels)) {
        # Look up published value from bw5
        pub <- NA_real_
        if (!is.null(ref)) {
          ref_mat <- ref[[ vname ]]
          ref_col <- sc_name
          if (!is.null(ref_mat) && ref_col %in% colnames(ref_mat)) {
            pub <- ref_mat[si, ref_col]
          }
        }
        rows[[length(rows) + 1L]] <- data.frame(
          Scenario   = sc_name,
          Variable   = vname,
          Statistic  = stat_labels[si],
          Published  = pub,
          Replicated = round(res[si, vi], 4),
          DiffPct    = round(100 * (res[si, vi] - pub) / pub, 4),
          stringsAsFactors = FALSE
        )
      }
    }
  }
  ord <- do.call(rbind, rows)
  ord[order(match(ord$Scenario, names(scenarios)),
            match(ord$Variable, var_labels),
            match(ord$Statistic, stat_labels)), ]
}

# Run ----
sink("dev/replicate-table5.txt")

cat("=== BW Table 5 - Statistics on optimal tax rates from Monte Carlo simulation of log-linearized optimal policy rules ===\n")
t5 <- run_sim(bw5, n_vars = 3)
print(t5, row.names = FALSE)

sink()
cat("Results written to dev/replicate-table5.txt\n")
