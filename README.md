
# calamus

A lightweight library for scientific computing, designed to have no dependency. Named after the root of a feather.


## Objective


I already hear the old wise guys (or girls) say:

*But! Wouldn't LAPACK / BLAS / (insert the fast C-Fortran library of your choice) be faster?? What's the point!*

And in the current libraries' state (and probably all future state), they would be correct. Using legacy, possibly memory unsafe, code made by math wizards in the 80s or something would be faster. I don't pretend I can beat them. But isn't the point to try to innovate to at least *try* ? If we didn't reinvent the wheel, we'd still have wooden wheels! 

As such, the goal of this library is to provide as performant as possible, pure Rust code for scientific computation, with no dependency.



## Features

- Dense linear algebra
- Sparse linear algebra
- Multivariate function root finding
- Multivariate function minimization
- Explicit Rk45 adaptive Ode solver
- Implicit Bdf23 adaptive Ode solver
- Functions


## Design methodology

- **Traits**
When multiple solvers or method are implemented, a generic trait covers their functionality. 

- **Memory allocations**
All solvers that require temporary data allocate it only one at the structs creation, and don't reallocate in future solve steps.

