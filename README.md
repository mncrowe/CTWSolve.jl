# CTWSolve.jl
Julia implementation of CTWSpectralMethods.m

Work in progress. The code mostly works and examples are working. Documentation missing for most functions.

Seems to require arpack version 0.5.3, rather than the newer 0.5.4 (which is [broken](https://github.com/JuliaLinearAlgebra/Arpack.jl/issues/147)). Install using

```julia
add Arpack@v0.5.3
```
