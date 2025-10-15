# Actually works :o

include("CTWSolve.jl")

# Grid parameters:

Ny, Nz = 41, 41
Ly = [0, 3]
type = :laguerre

# Numerical EVP parameters:

k = 1.0
ω₀ = 0.3
n = 5

# Problem parameters:

f = 1
H₀ = 0.2
H(y) = H₀ + (1 - H₀) * tanh.(y)
U(y, z) = 0
M²(y, z) = 0
N²(y, z) = 1

# Create grid:

println("Creating grid ...")
grid = CreateGrid(Ny, Nz, Ly, H; type)

# Create EVP:

println("Building EVP ...")
prob = CreateProblem(grid; f, U, N²)

# Solve EVP:

println("Solving EVP ...")
ω, p = SolveProblem(prob, k; ω₀, n)

# Plot a mode:

# heatmap(grid.y[:, 1], grid.z[1, :], real(p[:, :, 1]))