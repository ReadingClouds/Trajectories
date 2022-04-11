# Changelog

## [Unreleased](https://github.com/ParaConUK/advtraj/tree/HEAD)

[Full Changelog](https://github.com/ParaConUK/advtraj/compare/v0.4.0...HEAD)

*new features*

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


## [v0.4.0](https://github.com/ParaConUK/advtraj/tree/v0.4.0)

[Full Changelog](https://github.com/ParaConUK/advtraj/compare/v0.3.0...v0.4.0)


## [v0.3.0](https://github.com/ParaConUK/advtraj/tree/v0.3.0)

[Full Changelog](https://github.com/ParaConUK/advtraj/compare/c5e3ba670...v0.3.0)
