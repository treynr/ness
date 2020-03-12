
Changelog
=========

Unreleased
----------

Fixed
'''''

- Fix memory usage issue when running permutations over massive (>100GB) networks


1.1.0 - 2020.01.06
------------------

Added
'''''

- Add changelog.
- Add CLI option, :code:`--graph`, for saving the heterogeneous graph to an output file.
- Add option for starting the walk from multiple seeds at once with permutation testing.

Fixed
'''''

- Fix bug where using permutation testing for one random walk per seed would instead
  start from all seeds at once.


1.0.0 - 2019.12.19
------------------

Initial public release.
