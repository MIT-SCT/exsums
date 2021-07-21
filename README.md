# Multidimensional Included and Excluded Sums

This work is to appear at ACDA21. If you use this software please cite us. Once
published, this citation will be updated to match the official ACDA21 proceedings.
In the meantime, please use:

```
@misc{xu2021multidimensional,
      title={Multidimensional Included and Excluded Sums},
      author={Helen Xu and Sean Fraser and Charles E. Leiserson},
      year={2021},
      eprint={2106.00124},
      archivePrefix={arXiv},
      primaryClass={cs.DS}
}
```

Build
-------
We recommend compiling with clang++. An example Makefile is provided to build.
The testing library requires `googletest` and `Boost` libraries.

```bash
make clean && make
./exsum
./exsum_memory
```

Contributing
------------
Contributions via GitHub pull requests are welcome.

Authors
-------
- Helen Xu <hjxu@mit.edu>
- Sean Fraser <sfraser@mit.edu>
- Charles E Leiserson <cel@mit.edu>
