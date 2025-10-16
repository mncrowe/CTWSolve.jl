"""
Waves in a channel

Uses noflow BC on the walls

"""


include("CTWSolve.jl")

# Grid parameters:

Ny, Nz = [21, 21], 41
Ly = [-0.1, 0, 0.1]
type = [:chebyshev, :chebyshev]

# Numerical EVP parameters:

k = 1.0
ω₀ = 0.3
n = 5

# Problem parameters:

f = 1
N²(y, z) = 1
H(y) = 1 + 0.1 * tanh(y)

# Boundary conditions [top, bottom, left, right]:

NormalFlowBCs = [:noflow, :noflow, :noflow, :noflow]

# Create grid:

println("Creating grid ...")
grid = CreateGrid(Ny, Nz, Ly, H; type)

# Create EVP:

println("Building EVP ...")
prob = CreateProblem(grid; f, N², NormalFlowBCs)

# Solve EVP:

println("Solving EVP ...")
ω, p = SolveProblem(prob, k; ω₀, n)

nothing

# Plot a mode:

# heatmap(grid.y[:, 1], grid.z[1, :], real(p[:, :, 1]))



