# IEEE 9-Bus Line 8 Nose Curve

This folder collects the nose-curve figures used to explain the physical meaning
of continuous rho parameterization.

## What Is A Nose Curve?

Here the nose curve means the trajectory of the minimum bus voltage as one line
is continuously weakened from normal operation to outage:

```text
x-axis: rho, where rho=1 means the original line and rho=0 means line outage
y-axis: min Vm over all buses after solving AC power flow
```

For each rho value, the line parameters are scaled as:

```text
r(rho) = r0 / rho
x(rho) = x0 / rho
c(rho) = c0 * rho
g(rho) = g0 * rho
```

At rho=0, the line is removed. Then Newton-Raphson AC power flow is solved.
If it converges, the point is plotted. If it does not converge, the curve stops.
The stopping point is interpreted as a saddle-node bifurcation / voltage-collapse
boundary for that loading condition.

## Main Figure

- `case9_line8_nose_multiload.png`

This is IEEE 9-Bus Line 8, Bus 8 -> Bus 3, under different load scales.

Key reading:

- load=1.0x: the system converges over the full rho range, but the outage endpoint
  is already below the 0.95 p.u. voltage safety bound.
- load=1.5x: the curve stops around rho=0.14, showing a genuine PF non-convergence
  boundary before the line reaches full outage.
- load=2.0x: the curve stops around rho=0.53, so heavier load pushes the collapse
  boundary closer to normal operation.
- load=2.5x: the base case at rho=1 already fails to converge, so no curve is shown.

This figure is more physically interpretable than just saying the PF solution is
smooth: it shows how continuous topology degradation consumes voltage margin and
eventually leads to loss of PF solvability.

## Source

The figure was copied from the existing experiment folder:

`General_Meeting/.../连续化rho对潮流方程解的影响/case9/case9_nose_multiload_line8.png`

The generation logic is in:

`General_Meeting/.../连续化rho对潮流方程解的影响/rho_sweep_general.py`
