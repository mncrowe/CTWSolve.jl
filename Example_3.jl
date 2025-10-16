"""
Instability example

Uses none BCs in far field with Laguerre points

"""

include("CTWSolve.jl")

# Grid parameters:

Ny, Nz = [21, 11, 11, 21], 31
Ly = [-5, -1, 0, 1, 5]
type = [:laguerre, :laguerre]

# Numerical EVP parameters:

k = 1.0
ω₀ = 0.1im
n = 1

# Problem parameters:

f = 1
S = -0.2

U(y, z) = S*y
Uy(y, z) = S
Uz(y, z) = 0

N²(y, z) = 1

H₀ = 0.17
H(y) = abs(y) < 1 ? 1 - H₀ * sin(π * (y + 1) / 2) : 1
Hy(y) = abs(y) < 1 ? - H₀ * π * cos(π * (y + 1) / 2) / 2 : 0

NormalFlowBCs = [:noflow, :noflow, :none, :none]

# Create grid:

println("Creating grid ...")
grid = CreateGrid(Ny, Nz, Ly, H; Hy, type)

# Create EVP:

println("Building EVP ...")
prob = CreateProblem(grid; f, U, Uy, Uz, N², NormalFlowBCs)

# Solve EVP:

println("Solving EVP ...")
ω, p = SolveProblem(prob, k; ω₀, n)

nothing