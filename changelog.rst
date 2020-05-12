
Changelog
=========

Unreleased
----------

Added
'''''

- Add an API module containing simplified, easy-to-use functions for calling NESS
  graph building, matrix creation, and random walk code from external Python code.
- Add option to specify a temporary directory when running a distributed random walk.
  This is probably necessary if using a cluster of machines.

Changed
'''''''

- Alter the function for distributing permutation tests across cores to take in a
  variable that can be used to specify the temporary directory for temp files.
  This is necessary if using NESS as a library and distributing work across
  a cluster of nodes--most nodes will not have access to the temp directories of other
  nodes.

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
