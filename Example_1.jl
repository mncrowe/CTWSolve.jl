include("CTWSolve.jl")

# Grid parameters:

Ny, Nz = 41, 41
Ly = [0, 4]
type = :chebyshev

# Numerical EVP parameters:

k = 1.0
ω₀ = 0.1 - 0.01im
n = 5

# Problem parameters:

f = 1
H₀ = 1
H(y) = H₀ + (1 - H₀) * tanh.(y)
U(y, z) = exp.(-y)
M²(y, z) = 0
N²(y, z) = 1
νv(y, z), νh(y, z) = 0.01, 0.01
κv(y, z), κh(y, z) = 0, 0

# Boundary conditions [top, bottom, left, right]:

NormalFlowBCs = [:noflow, :noflow, :noflow, :noflow]
NormalStressBCs = [:nostress, :nostress, :nostress, :noflow]
NormalFluxBCs = [:none, :none, :none, :none]

# Create grid:

println("Creating grid ...")
grid = CreateGrid(Ny, Nz, Ly, H; type)

# Create EVP:

println("Building EVP ...")
prob = CreateProblem(grid; f, U, N², νv, νh, κv, κh, NormalFlowBCs, NormalStressBCs, NormalFluxBCs)

# Solve EVP:

println("Solving EVP ...")
ω, p = SolveProblem(prob, k; ω₀, n)

# Plot a mode:

# heatmap(grid.y[:, 1], grid.z[1, :], real(p[:, :, 1]))



