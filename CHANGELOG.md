# Changelog

## [Unreleased](https://github.com/ParaConUK/advtraj/tree/HEAD)

[Full Changelog](https://github.com/ParaConUK/advtraj/compare/v0.4.0...HEAD)

*Support for MONC*
Primarily changes to support the MONC grid. This has:
- First u point at first p point + dx/2, so it makes sense to have p[0] at x=0.
- First v point at first p point + dy/2, so it makes sense to have p[0] at y=0.
- Virtual p point at z = -dz/2.

Main changes are in creating synthetic data fro testing.

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
