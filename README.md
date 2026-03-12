# Replication of 'Optimal Fiscal Policy in a Business Cycle Model' by Chari, Christiano and Kehoe (1994)

I used the Monte Carlo approach described in "Optimal Taxation in an RBC Model: A Linear-Quadratic Approach" by Benigno and
Woodford (2005), and hence the package's name "bw05".

I do not have a MATLAB license, so I looked at Grohe-Uribe solvers to adapt a bit an write a small program using Armadillo.

Replication result:

```
=== BW Table 5 - Statistics on optimal tax rates from Monte Carlo simulation of log-linearized optimal policy rules ===
 Scenario                 Variable Statistic Published Replicated   DiffPct
 baseline   tax_rate_labour_income         E    23.870    23.8701    0.0003
 baseline   tax_rate_labour_income        sd     0.095     0.0947   -0.2850
 baseline   tax_rate_labour_income  autocorr     0.776     0.7665   -1.2282
 baseline   tax_rate_labour_income    corr_g     0.620     0.6228    0.4548
 baseline   tax_rate_labour_income    corr_z     0.496     0.4927   -0.6583
 baseline ex_ante_tax_rate_capital         E     0.000     0.0000       Inf
 baseline ex_ante_tax_rate_capital        sd     0.000     0.0000       Inf
 baseline ex_ante_tax_rate_capital  autocorr        NA     0.8790        NA
 baseline ex_ante_tax_rate_capital    corr_g        NA     0.9713        NA
 baseline ex_ante_tax_rate_capital    corr_z        NA     0.1495        NA
 baseline ex_post_tax_rate_capital         E     0.000     0.0100       Inf
 baseline ex_post_tax_rate_capital        sd    36.155    36.1640    0.0248
 baseline ex_post_tax_rate_capital  autocorr     0.000     0.0011       Inf
 baseline ex_post_tax_rate_capital    corr_g     0.410     0.4095   -0.1232
 baseline ex_post_tax_rate_capital    corr_z    -0.255    -0.2589    1.5117
  high_ra   tax_rate_labour_income         E    20.690    20.6900   -0.0001
  high_ra   tax_rate_labour_income        sd     0.034     0.0342    0.5147
  high_ra   tax_rate_labour_income  autocorr     0.811     0.8125    0.1809
  high_ra   tax_rate_labour_income    corr_g    -0.550    -0.5637    2.4991
  high_ra   tax_rate_labour_income    corr_z    -0.802    -0.7905   -1.4359
  high_ra ex_ante_tax_rate_capital         E     0.002    -0.0008 -139.5311
  high_ra ex_ante_tax_rate_capital        sd     3.289     3.2901    0.0346
  high_ra ex_ante_tax_rate_capital  autocorr     0.804     0.8042    0.0229
  high_ra ex_ante_tax_rate_capital    corr_g     0.252     0.2564    1.7401
  high_ra ex_ante_tax_rate_capital    corr_z     0.965     0.9637   -0.1372
  high_ra ex_post_tax_rate_capital         E     0.000     0.0069       Inf
  high_ra ex_post_tax_rate_capital        sd    30.581    29.4181   -3.8027
  high_ra ex_post_tax_rate_capital  autocorr    -0.003     0.0030 -199.3927
  high_ra ex_post_tax_rate_capital    corr_g     0.444     0.4567    2.8569
  high_ra ex_post_tax_rate_capital    corr_z    -0.132    -0.0928  -29.6865
   only_z   tax_rate_labour_income         E    23.870    23.8700    0.0001
   only_z   tax_rate_labour_income        sd     0.074     0.0741    0.1350
   only_z   tax_rate_labour_income  autocorr     0.685     0.6854    0.0566
   only_z   tax_rate_labour_income    corr_g        NA    -0.0004        NA
   only_z   tax_rate_labour_income    corr_z     0.632     0.6313   -0.1184
   only_z ex_ante_tax_rate_capital         E     0.000     0.0000      -Inf
   only_z ex_ante_tax_rate_capital        sd     0.000     0.0000       Inf
   only_z ex_ante_tax_rate_capital  autocorr        NA     0.6765        NA
   only_z ex_ante_tax_rate_capital    corr_g        NA     0.0004        NA
   only_z ex_ante_tax_rate_capital    corr_z        NA    -0.7499        NA
   only_z ex_post_tax_rate_capital         E     0.000     0.0030       Inf
   only_z ex_post_tax_rate_capital        sd    15.769    15.9500    1.1478
   only_z ex_post_tax_rate_capital  autocorr    -0.002    -0.0015  -24.4298
   only_z ex_post_tax_rate_capital    corr_g        NA    -0.0001        NA
   only_z ex_post_tax_rate_capital    corr_z    -0.586    -0.5864    0.0727
   only_g   tax_rate_labour_income         E    23.870    23.8701    0.0002
   only_g   tax_rate_labour_income        sd     0.059     0.0591    0.1438
   only_g   tax_rate_labour_income  autocorr     0.895     0.8947   -0.0384
   only_g   tax_rate_labour_income    corr_g     0.999     0.9998    0.0781
   only_g   tax_rate_labour_income    corr_z        NA    -0.0006        NA
   only_g ex_ante_tax_rate_capital         E     0.000     0.0000       Inf
   only_g ex_ante_tax_rate_capital        sd     0.000     0.0000       Inf
   only_g ex_ante_tax_rate_capital  autocorr        NA     0.8907        NA
   only_g ex_ante_tax_rate_capital    corr_g        NA     1.0000        NA
   only_g ex_ante_tax_rate_capital    corr_z        NA    -0.0006        NA
   only_g ex_post_tax_rate_capital         E     0.000     0.0069       Inf
   only_g ex_post_tax_rate_capital        sd    32.512    32.4575   -0.1675
   only_g ex_post_tax_rate_capital  autocorr     0.000     0.0009       Inf
   only_g ex_post_tax_rate_capital    corr_g     0.456     0.4560   -0.0080
   only_g ex_post_tax_rate_capital    corr_z        NA     0.0001        NA
      iid   tax_rate_labour_income         E    23.870    23.8700    0.0001
      iid   tax_rate_labour_income        sd     0.147     0.1473    0.2193
      iid   tax_rate_labour_income  autocorr    -0.068    -0.0680   -0.0145
      iid   tax_rate_labour_income    corr_g     0.099     0.1008    1.8259
      iid   tax_rate_labour_income    corr_z     0.954     0.9536   -0.0419
      iid ex_ante_tax_rate_capital         E     0.000     0.0000       Inf
      iid ex_ante_tax_rate_capital        sd     0.000     0.0000       Inf
      iid ex_ante_tax_rate_capital  autocorr        NA     0.8956        NA
      iid ex_ante_tax_rate_capital    corr_g        NA     0.1055        NA
      iid ex_ante_tax_rate_capital    corr_z        NA    -0.4242        NA
      iid ex_post_tax_rate_capital         E     0.000     0.0030       Inf
      iid ex_post_tax_rate_capital        sd    10.818    10.8806    0.5790
      iid ex_post_tax_rate_capital  autocorr     0.000     0.0011       Inf
      iid ex_post_tax_rate_capital    corr_g     0.913     0.9074   -0.6182
      iid ex_post_tax_rate_capital    corr_z    -0.409    -0.4203    2.7650
```
