"""
Basic shelf wave example

Actually works :o

This script determines the dispersion curve ω = ω(k) for 
k ∈ [0.1, 3]. We use ω₀ as an initial guess for ω and take
ω₀ = 0.5*tanh.(k/2) here. Simpler ω₀ will also work, e.g.
ω₀ = 0.5*ones(length(k)) but will generally be slower.

"""

include("CTWSolve.jl")

# Grid parameters:

Ny, Nz = 31, 21
Ly = [0, 4]
type = :laguerre

# Numerical EVP parameters:

k = 0.1:0.1:3
ω₀ = 0.5*tanh.(k/2)
n = 6

# Problem parameters:

f = 1
H₀ = 0.7
H(y) = H₀ + (1 - H₀) * tanh.(y)
U(y, z) = 0
N²(y, z) = 1

# Create grid:

println("Creating grid ...")
grid = CreateGrid(Ny, Nz, Ly, H; type)

# Create EVP:

println("Building EVP ...")
prob = CreateProblem(grid; f, U, N²)

# Solve EVP for dispersion curves:

println("Solving EVP ...")
ω = DispersionCurve(prob, k; n, ω₀, method = :all)

nothing