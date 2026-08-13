# Rho Continuity Observations

Source PF bank: `/home/linhanjiang/pfdelta-main/data_generation_rho/case30/rho_pf_test_branch7_commonload_curve_heldout2000/raw`

This folder studies branch7 held-out load curves with rho grid `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`.
Complete held-out load curves: `2000` loads, `37992` raw samples.

Main outputs:

- `true_pf_vmin_curves.png`: true PF minimum voltage curves vs rho.
- `true_pf_distance_from_rho1.png`: true state distance from normal PF solution.
- `true_pf_endpoint_linear_interp_error.png`: how far the true curve is from linear interpolation between rho=0 and rho=1 endpoints.
- `true_pf_relative_nonlinearity.png`: endpoint interpolation error normalized by endpoint gap.
- `true_pf_finite_difference_sensitivity.png`: finite-difference sensitivity along the curve.
- `model_t7_curve_pbl_mean.png`: existing trained models evaluated along the same held-out rho curve.
- `model_endpoint_mid_overall_pbl.png`: endpoint/mid/overall curve comparison.

Interpretation angle:
The first group is about the physical PF solution manifold, independent of neural-network performance.
The model group checks whether extra intermediate-rho supervision improves physical residual along unseen-load rho curves.
