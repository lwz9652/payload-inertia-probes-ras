# Recovering payload induced joint inertia from directional torque probes

Reproduction code and numerical evidence for the 27 November 2026 RAS manuscript. The package includes simulated torque readings, held-out reference matrices, fitted chains and the case-level summaries behind the three figures and validation tables. The public Panda recordings are referenced for the coverage checks.

For the archived-score checks use Python 3.12 and `requirements.txt`. For MuJoCo replays use a separate Python 3.11 environment with `requirements-mujoco.txt`. The repository includes the third-party notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Verification from bundled evidence
-----------------------------------
Use Python 3.12 with NumPy 1.26.4, SciPy 1.13.0, SymPy 1.12 and
Matplotlib 3.8.3 for the perturbation, exact-rank and sampling studies.
Use a new output directory for each replay.

    mkdir replay
    python evidence/directional/exact_certificate.py --output replay/exact.json
    python evidence/directional/audit_two_direction_ambiguity.py --output replay/ambiguity.json
    python evidence/verify_revision.py
    python evidence/verify_review.py
    python evidence/followup/verify_controls.py
    python evidence/solver_controls/verify_solver_controls.py
    python evidence/sampling/verify_sampling.py
    python figure_sources/generate_validation_artifacts.py --check

The sampling verifier recomputes all per-geometry matrix and action metrics for
the two noise draws, from archived fits and the frozen inputs. It checks the
probe geometry, per-fit budgets, held-out action counts, primary failure
comparisons and success counts.

Regenerate the three manuscript figures as PDF and SVG in figures/.
Use --preview-dir to save optional PNG previews.

    python figure_sources/draw_figure1.py
    python figure_sources/draw_transfer_figures.py

Figure 1(a) uses offscreen MuJoCo renderings at the two archived joint
configurations from split 1. The bundled views and marker positions are in
figure_sources/panda_mujoco_views.npz. You can regenerate them with the
MuJoCo replay environment and the Panda directory from
[the pinned CaDeLaC revision](https://github.com/Schulze18/cadelac/tree/a9b889c638b749bd58b337ed66ba38896ad01477/cadelac/robots/panda).

    python figure_sources/render_panda_scene.py /path/to/cadelac/robots/panda

The renderer adds the CaDeLaC display meshes to the archived model and
checks the inverse-dynamics reads at both poses. Run draw_figure1.py in
the numerical environment above to assemble all three panels.

Panel (b) uses the archived signed reads. Panel (c) compares two chains
recovered at the first configuration on the second, before refinement.
Figure 2 uses the paired metric archives and the unprobed action errors
from draw 1. Figure 3 plots each geometry's error from the fixed-budget
case records. The main-text tables describe the calibration loads, the
probe-design comparison, and fixture-only versus full-chain calibration.

To regenerate the matrix-error, fixture and acquisition tables and diagnostic plots
from the archived results, run

    python figure_sources/generate_sampling_artifacts.py
    python figure_sources/generate_validation_artifacts.py

To generate additional diagnostic plots for the original geometry studies, run

    python figure_sources/draw_directional_figures.py

Full replay of the paired probe comparison (Sections 5-6 and Appendix C)
----------------------------------------------------------------
The following commands reconstruct the canonical directional contrasts,
choose the fourth direction on the nominal model, refit 240 chains per
noise draw, and compare the fresh per-geometry results with the archive.
The replay is CPU-based. Adjust the worker count for the available cores.

    python evidence/sampling/freeze_inputs.py --design replay/probe_design.json --inputs replay/inputs
    python evidence/sampling/fit_sampling.py --design replay/probe_design.json --inputs replay/inputs --output replay/fits.json --workers 4
    python evidence/sampling/fit_sampling.py --design replay/probe_design.json --inputs replay/inputs --output replay/replicate_fits.json --workers 4 --replicate-seed 2026111820
    python evidence/sampling/verify_sampling.py --replay-directory replay

To export the result of a fresh replay in the same format as the bundled
case-level metrics, run

    python evidence/sampling/evaluate_sampling.py --design replay/probe_design.json --inputs replay/inputs --fits replay/fits.json --output replay/metrics.json

Within a geometry both initializers and every design share the relevant
torque contrasts. In inputs/Panda_*.npz and inputs/UR10e_*.npz under
evidence/sampling, the calibration arrays and the held-out arrays are
named separately. The fit script reads only calibration fields.
The fits JSON includes the final screw axes and both stages' loss,
termination and residual-evaluation counts. The metrics JSON includes
R_M and R_A as fractions, paired changes in percentage points, the median,
95th percentile and maximum of E_tau and the reference torque in N m,
and the two selected actions.

Ideal MuJoCo replays (Sections 5-6 and Appendix C)
----------------------------------------
The archived signed torque vectors, held-out input grid, and independent
mass-matrix increments are included. This command re-estimates all eight
chains from those signed torque vectors and scores the same held-out grid.

    python evidence/directional/check_mujoco_probes.py --output replay/ideal_from_reads

To regenerate the signed vectors and all held-out reference increments
with the simulator, use Python 3.11, MuJoCo 3.2.6, NumPy 2.4.6 and SciPy
1.17.1. The two ideal replays start from the archived calibration states
and load inertias. The original numerical errors are near machine
precision, so their last digits depend on the numerical libraries.

    python evidence/directional/check_mujoco_probes.py --regenerate --output replay/ideal_mujoco

The constraint-free CaDeLaC Panda model is pinned to upstream commit
a9b889c638b749bd58b337ed66ba38896ad01477. The derivative XML and upstream
license are in evidence/followup. Use the same MuJoCo environment for
sampled-window and state-mismatch replays.

Additional controls (not required to render the figures)
---------------------------------------------------------
The full candidate pools and all geometry, acquisition and load-description
controls are provided as case-level JSON. Example commands for trial 1 of
each applicable robot follow. Increase --trials to 12 for the full cohorts.

    python evidence/solver_controls/candidate_moving.py --trials 1 --workers 2 --output replay/moving.json
    python evidence/solver_controls/candidate_moving.py --fixed-chart --trials 1 --workers 2 --output replay/fixed.json
    python evidence/solver_controls/prior_design.py --trials 1 --workers 2 --output replay/prior.json
    python evidence/solver_controls/descriptor_bias.py --trials 1 --workers 2 --output replay/descriptor.json
    python evidence/solver_controls/sampled_probes.py --trials 1 --workers 1 --output replay/windows.json

Additional control records
-----------------------------------
The fixture baseline and acquisition tables are generated from the
case-level JSON files in evidence/validation/. The following commands reproduce those
records using the same Python 3.11 and MuJoCo environment described above.
Set OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and OMP_NUM_THREADS to 1 before
running several worker processes.

    python evidence/solver_controls/attachment_baseline.py --output replay/attachment_baseline.json
    python evidence/solver_controls/random_starts.py --output replay/random_starts.json
    python evidence/solver_controls/random_starts.py --seed 2026112512 --output replay/random_starts_repeat.json
    python evidence/solver_controls/pose_budget.py --output replay/pose_budget.json
    python evidence/solver_controls/pose_budget.py --seed 2026112524 --output replay/pose_budget_repeat.json
    python evidence/review/geometry_review.py --study terminal --moving-chart --terminal-angles 0.5 --terminal-noise 0.001 --output replay/terminal_moving_low.json
    python evidence/review/geometry_review.py --study terminal --moving-chart --terminal-angles 0.5 --terminal-noise 0.01 --output replay/terminal_moving_high.json
    python evidence/review/geometry_review.py --study terminal --moving-chart --terminal-angles 0.5 --terminal-noise 0.01 --repetitions 25 --output replay/terminal_moving_repeat.json
    python evidence/followup/alignment_study.py --passing-levels-only --torque-noise 0.01 --output replay/alignment_high.json
    python evidence/followup/alignment_study.py --joint --joint-fraction 0.5 1 --torque-noise 0.01 --output replay/alignment_joint_high.json
    python evidence/solver_controls/sampled_probes.py --torque-noise 0.01 --output replay/sampled_high.json
    python evidence/solver_controls/descriptor_bias.py --torque-noise 0.01 --output replay/descriptor_high.json
    python evidence/solver_controls/friction_filter.py --output replay/friction_filter.json

Public records checked for coverage
-----------------------------------
The alignment screen for Yang and Ko's Panda recordings is in
evidence/aligned_val_report.json. The NeuralActuator records checked for
loaded and unloaded coverage are documented in
evidence/validation/recorded_panda_coverage.json. The source files are
available from the cited data repositories.
