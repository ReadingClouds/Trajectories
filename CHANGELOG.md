# Changelog

## [Unreleased](https://github.com/ParaConUK/advtraj/tree/HEAD)

[Full Changelog](https://github.com/ParaConUK/advtraj/compare/v0.4.0...HEAD)

*Use of 'time' as DimCoord.

- Make 'time' the coordinate of trajectory points instead of a variable.

- Requires addition of 'ref_time' as a coord, indicating the staring point of
  forward and back trajectories.

- Introduces 'trajectory_no' as DimCoord for trajectories

- Removes need for .stack.apply to loop over trajectories. Code works for
  arbitrary number. [\#8]

*Support for MONC*

Add support for MONC grid by implementing its c-grid configuration for where
the position scalars are stored. (https://github.com/ParaConUK/advtraj/pull/10)

*maintenance*

- Switch to using pre-commit for linting (to automatically have linting run on
  each git commit) and use pre-commit for linting during continuous
  integration. Also update README and add development notes to give
  instructions on how to use pre-commit and contribute to advtraj.
  [\#4](https://github.com/ParaConUK/advtraj/pull/4)

- fix to support xarray version `>=0.19.0`
  [\#5](https://github.com/ParaConUK/advtraj/pull/5) @leifdenby


## [v0.4.0](https://github.com/ParaConUK/advtraj/tree/v0.4.0)

[Full Changelog](https://github.com/ParaConUK/advtraj/compare/v0.3.0...v0.4.0)


## [v0.3.0](https://github.com/ParaConUK/advtraj/tree/v0.3.0)

[Full Changelog](https://github.com/ParaConUK/advtraj/compare/c5e3ba670...v0.3.0)
