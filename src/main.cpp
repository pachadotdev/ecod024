// bw05/src/main.cpp
//
// Replication of Benigno & Woodford (2005), NBER WP 11029,
// "Optimal Taxation in an RBC Model: A Linear-Quadratic Approach"
//
// Uses the BW Linear-Quadratic (LQ) method directly:
//   1. Quadratic loss function (BW eq 2.13)
//   2. Log-linear resource constraint (BW eq 2.4)
//   3. Discrete Riccati equation -> optimal decision rules
//   4. Tax formulas (BW eqs 2.14–2.17)
//
// Exported:
//   bw05_stats  -> BW Table 5 (1st-order Monte Carlo)
//   bw05_params -> NSSS + LQ diagnostics

#include <cpp4r.hpp>
#include <armadillo4r.hpp>

using namespace cpp4r;
using namespace arma;

#include "01_parameters.h"
#include "02_solver.h"


// -----------------------------------------------------------------------
//  bw05_stats
//
//  Monte Carlo of first-order LQ optimal policy rules.
//  BW footnote 31: T_total = 500,000; T_burn = 60,000.
//
//  Returns (5 × 3) matrix (percentage points):
//    rows: E | s.d. | autocorr | corr_g | corr_z
//    cols: τ^h | θ^e | τ^k

/* roxygen
@title BW Table 5: 1st-order Monte Carlo statistics
@description Simulates log-linearized Ramsey policy rules using BW's
  LQ approach and returns the same statistics as Table 5 of Benigno &
  Woodford (2005).
@export
*/
[[cpp4r::register]]
writable::doubles_matrix<> bw05_stats(list args, int n_k, int T_total,
                                       int T_burn, int seed)
{
  Params p(args);
  DecisionRules rules = compute_decision_rules(p);

  mat sim = simulate(p, rules, T_total, T_burn, seed);
  sim.cols(0, 3) *= 100.0;   // fraction -> percentage points

  const mat stats = simulation_stats(sim);

  writable::doubles_matrix<> out(5, 3);
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 3; ++c)
      out(r, c) = stats(r, c);
  return out;
}


// -----------------------------------------------------------------------
//  bw05_params — NSSS + LQ diagnostics
//
//  Returns (1 × 10):
//    sk | sc | sb | h_ss | phi_bw | qc | qh | qk | theta_lq | theta_z

/* roxygen
@title NSSS and LQ diagnostics
@description Returns steady-state shares, BW utility coefficients, and
  LQ loss function coefficients for diagnostic purposes.
@export
*/
[[cpp4r::register]]
writable::doubles_matrix<> bw05_params(list args)
{
  Params p(args);

  writable::doubles_matrix<> out(1, 10);
  out(0, 0) = p.sk;
  out(0, 1) = p.sc;
  out(0, 2) = p.sb;
  out(0, 3) = p.h_ss;
  out(0, 4) = p.phi_bw;
  out(0, 5) = p.qc;
  out(0, 6) = p.qh;
  out(0, 7) = p.qk;
  out(0, 8) = p.theta_lq;
  out(0, 9) = p.theta_z;
  return out;
}


// -----------------------------------------------------------------------
//  bw05_diag — Full decision rule diagnostics
//
//  Returns a flat matrix with all coefficients for debugging.
//  Row layout (29 rows × 3 cols, padded with zeros):
//    rows 0-1:   gx (2×3)
//    rows 2-4:   hx (3×3)
//    rows 5:     gamma_w (1×3)
//    rows 6:     gamma_b (1×3)
//    rows 7:     gamma_tau_h (1×3)
//    rows 8:     Gamma_k_0 (1×3)
//    rows 9:     Gamma_k_1 (1×3)
//    row 10:     [sb, bc, bh]
//    row 11:     [bk, btau, 0]
//    row 12:     [dc, dh, phi_bw]
//    row 13:     [sigma_inv, nu, psi_bw]

/* roxygen
@title Full decision rule diagnostics
@description Returns gx, hx, and all BW tax formula coefficients.
@export
*/
[[cpp4r::register]]
writable::doubles_matrix<> bw05_diag(list args)
{
  Params p(args);
  DecisionRules rules = compute_decision_rules(p);

  writable::doubles_matrix<> out(14, 3);
  for (int r = 0; r < 14; ++r)
    for (int c = 0; c < 3; ++c)
      out(r, c) = 0.0;

  // gx (2×3)
  for (int r = 0; r < 2; ++r)
    for (int c = 0; c < 3; ++c)
      out(r, c) = rules.gx(r, c);

  // hx (3×3)
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      out(r + 2, c) = rules.hx(r, c);

  // gamma_w
  for (int c = 0; c < 3; ++c) out(5, c) = rules.gamma_w(c);
  // gamma_b
  for (int c = 0; c < 3; ++c) out(6, c) = rules.gamma_b(c);
  // gamma_tau_h
  for (int c = 0; c < 3; ++c) out(7, c) = rules.gamma_tau_h(c);
  // Gamma_k_0
  for (int c = 0; c < 3; ++c) out(8, c) = rules.Gamma_k_0(c);
  // Gamma_k_1
  for (int c = 0; c < 3; ++c) out(9, c) = rules.Gamma_k_1(c);

  // leverage
  out(10, 0) = p.sb; out(10, 1) = p.bc_bw; out(10, 2) = p.bh_bw;
  out(11, 0) = p.bk_bw; out(11, 1) = p.btau_bw; out(11, 2) = 0.0;
  // dc, dh, phi
  out(12, 0) = p.dc; out(12, 1) = p.dh; out(12, 2) = p.phi_bw;
  // utility coeffs
  out(13, 0) = p.sigma_inv; out(13, 1) = p.nu; out(13, 2) = p.psi_bw;

  return out;
}
