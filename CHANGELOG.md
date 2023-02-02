# Changelog

## [Unreleased](https://github.com/ParaConUK/advtraj/tree/HEAD)

[Full Changelog](https://github.com/ParaConUK/advtraj/compare/v0.5.1...HEAD)

*maintenance*

- update version of `isort` used in CI to resolve bug with `poetry` version
  used by Github Actions
  [\#26](https://github.com/ParaConUK/advtraj/pull/26) @leifdenby

- add `scipy` to requirements
  [\#26](https://github.com/ParaConUK/advtraj/pull/26) @leifdenby


## [v0.5.1](https://github.com/ParaConUK/advtraj/tree/v0.5.1)

[Full Changelog](https://github.com/ParaConUK/advtraj/compare/v0.5.0...v0.5.1)

*new features*

- Ability to select how many backward and forward timesteps to use.

- Utility `mask_to_positions` to aid conversion of a logical mask on a 3D data
  field to a set of initial trajectory points.

- Utility `data_to_traj` to interpolate 3D data to trajectory points.

(Pull https://github.com/ParaConUK/advtraj/pull/22) @ReadingClouds

## [v0.5.0](https://github.com/ParaConUK/advtraj/tree/v0.5.0)

[Full Changelog](https://github.com/ParaConUK/advtraj/compare/v0.4.0...v0.5.0)

*new features*

- Additional, generally faster, solvers for forward step, "fixed_point_iterator"
  and "hybrid_fixed_point_iterator".
 [\#11](https://github.com/ParaConUK/advtraj/pull/19) @ReadingClouds

- Improve handing of surface boundary, and ensure trajetories do not leave top.

## [v0.4.0](https://github.com/ParaConUK/advtraj/tree/v0.4.0)

[Full Changelog](https://github.com/ParaConUK/advtraj/compare/v0.3.0...v0.4.0)

*new features*

- Use of 'time' as DimCoord. Make 'time' the coordinate of trajectory points
  instead of a variable. Requires addition of 'ref_time' as a coord,
  indicating the starting point of forward and back trajectories, and
  introduces 'trajectory_no' as DimCoord for trajectories.
  Also removes need for .stack.apply to loop over trajectories.
  Code now works for arbitrary number of start points. [\#8]

- Add support for MONC grid by implementing its c-grid
  configuration for where the position scalars are stored.
  [\#10](https://github.com/ParaConUK/advtraj/pull/10) @ReadingClouds

*maintenance*

- update version of `black` we use for code-cleanup to deal with issue within
  `black` due to changes in its upstream dependency `click`
  (https://github.com/psf/black/issues/2964)
  [\#15](https://github.com/ParaConUK/advtraj/pull/15) @leifdenby

- Switch to using pre-commit for linting (to automatically have linting run on
  each git commit) and use pre-commit for linting during continuous
  integration. Also update README and add development notes to give
  instructions on how to use pre-commit and contribute to advtraj.
  [\#4](https://github.com/ParaConUK/advtraj/pull/4) @leifdenby

- fix to support xarray version `>=0.19.0`
  [\#5](https://github.com/ParaConUK/advtraj/pull/5) @leifdenby




## [v0.3.0](https://github.com/ParaConUK/advtraj/tree/v0.3.0)

[Full Changelog](https://github.com/ParaConUK/advtraj/compare/c5e3ba670...v0.3.0)
