using LinearAlgebra, SparseArrays, Arpack


"""
    SpecMatrix(x, w, w′)



"""
function SpecMatrix(x::AbstractVector, w::AbstractVector, w′::AbstractVector)

    N = length(x)
    D = x' .- x

    D₁ = D
    D₁[D .== 0] .= 1

    A = ones(N, 1) * (w' .* prod(D₁; dims = 1))

    M = A' ./ (A .* D')
    M[1:(N+1):N^2] .= sum(1 ./ D₁; dims = 1)' .- 1 + (w′ ./ w)

    return M

end

"""
    SpecMatrix(x)



"""
SpecMatrix(x) = SpecMatrix(x, ones(length(x)), zeros(length(x)))


"""
    GridChebyshev(N, L)

Creates a grid of Chebyshev points of the second kind and a spectral collocation
differentiation matrix.

# Arguments:
 - `N`: number of gridpoints
 - `L`: vector of endpoints, ``x ∈ [L₁, L₂]``

"""
function GridChebyshev(N::Int, L::Vector)

    x = (L[1] + L[2]) / 2 .- (L[2] - L[1]) / 2 * cos.((0:(N-1))*π/(N-1))

    M = SpecMatrix(x)

    return x, M

end

"""
    GridLaguerre(N, L)


"""
function GridLaguerre(N::Int, L::Vector)

    J = diagm(0 => 1:2:2*N-3) - diagm(1 => 1:N-2) - diagm(-1 => 1:N-2)
    p = sort(eigvals(J))
    x = L[1] .+ [0; p / p[end] * (L[2] - L[1])]
    w = [1; exp.(-p/2)]                          # weight function w(x)
    w′ = -p[end] / (L[2] - L[1]) / 2 * w         # dw/dx = dw/dp * dp/dx

    M = SpecMatrix(x, w, w′)

    return x, M

end

"""
    GridComposite(N, L, type)


"""

function GridComposite(N::Vector, L::Vector, type::Vector{Symbol})

    for i in 1:2; @assert type[i] in (:chebyshev, :laguerre); end

    n = length(N)

    # first segment:

    if type[1] == :chebyshev
        
        x, M = GridChebyshev(N[1], [L[1], L[2]])

    end

    if type[1] == :laguerre

        x, M = GridLaguerre(N[1], [L[2], L[1]])
        x, M = x[end:-1:1], M[end:-1:1, end:-1:1]    # flip segment

    end

    # middle Chebyshev segments:

    for i in 2:n-1

        x′, M′ = GridChebyshev(N[i], [L[i], L[i+1]])
        x, M = JoinSegment(x, M, x′, M′)

    end

    # final segment:

    if type[2] == :chebyshev
        
        x′, M′ = GridChebyshev(N[n], [L[n], L[n+1]])
        x, M = JoinSegment(x, M, x′, M′)

    end

    if type[2] == :laguerre

        x′, M′ = GridLaguerre(N[n], [L[n], L[n+1]])
        x, M = JoinSegment(x, M, x′, M′)

    end

    return x, M

end


"""
    JoinSegment(x1, M1, x2, M2)


"""
function JoinSegment(x1::Vector, M1::Array, x2::Vector, M2::Array)

    if x1[end] != x2[1]; @error "Grid endpoints do not match."; end

    x = [x1[1:end]; x2[2:end]]
    M = zeros(length(x), length(x))
    M[1:length(x1)-1, 1:length(x1)] = M1[1:end-1, :]
    M[length(x1), 1:length(x1)] = M1[end, :] / 2
    M[length(x1), length(x1):end] += M2[1, :] / 2
    M[length(x1)+1:end, length(x1):end] = M2[2:end, :]

    return x, M

end

"""
    GridFourier(N, L)

"""
function GridFourier(N::Int, L::Vector)

    if isodd(N); @warn "Even values of n are recommended."; end

    x = Vector(LinRange(L[1], L[2], N+1)[1:N])
    h = (L[2] - L[1]) / N
    M = π / (L[2] - L[1]) * (-1 + 0im) .^ ((x .- x') / h) ./ tan.(π * (x .- x') / (L[2] - L[1]))
    M[1:(N+1):N^2] .= 0
    M = real(M)

    return x, M

end


"""
    GridStruct

    type is vector of :bounded, :infinite, :periodic

"""
struct GridStruct
    Ny::Union{Int,Vector{Int}}
    Nz::Int
    y::Array
    z::Array
    Ly::Vector
    H::Function
    Hy::Union{Function,Nothing}
    λ::Vector
    ζ::Vector
    Mλ::SparseMatrixCSC
    Mζ::SparseMatrixCSC
    type::Union{Symbol,Vector{Symbol}}
end

"""
    ParamsStruct

Store the problem parameters

"""
struct ParamsStruct
    f::Number
    g::Number
    δh::Number
    δa::Number
    U::Union{Nothing,Array,Function}
    V::Union{Nothing,Array,Function}
    W::Union{Nothing,Array,Function}
    N²::Union{Nothing,Array,Function}
    M²::Union{Nothing,Array,Function}
    νh::Union{Nothing,Array,Function}
    νv::Union{Nothing,Array,Function}
    κh::Union{Nothing,Array,Function}
    κv::Union{Nothing,Array,Function}
    Uy::Union{Nothing,Array,Function}
    Uz::Union{Nothing,Array,Function}
    Vy::Union{Nothing,Array,Function}
    Vz::Union{Nothing,Array,Function}
    Wy::Union{Nothing,Array,Function}
    Wz::Union{Nothing,Array,Function}
end

"""
   FieldStruct

Store the matrix fields for building the problem

"""
struct FieldStruct
    H::Vector
    Hy::Vector
    U::SparseMatrixCSC
    V::SparseMatrixCSC
    W::SparseMatrixCSC
    N²::SparseMatrixCSC
    M²::SparseMatrixCSC
    νh::SparseMatrixCSC
    νv::SparseMatrixCSC
    κh::SparseMatrixCSC
    κv::SparseMatrixCSC
    Uy::SparseMatrixCSC
    Uz::SparseMatrixCSC
    Vy::SparseMatrixCSC
    Vz::SparseMatrixCSC
    Wy::SparseMatrixCSC
    Wz::SparseMatrixCSC
    Dy::SparseMatrixCSC
    Dz::SparseMatrixCSC
end

"""
    ProblemStruct

Stores the matrices in a k-independent way (i.e. k coefficients Li in L = L0 + k*L1 + k^2*L2 etc).

"""
struct ProblemStruct
    grid::GridStruct
    params::ParamsStruct
    fields::FieldStruct
    D::SparseMatrixCSC
    L₀::SparseMatrixCSC
    L₁::SparseMatrixCSC
    L₂::SparseMatrixCSC
end

"""
    CreateGrid(Ny, Nz, Ly, H; Hy, type)

type argument: :chebyshev, :laguerre, [:.., :..] (for composite), :fourier

Define the numerical grid as a [`GridStruct`](@ref)

# Arguments:
 - `Ny`, `Nz`: number of gridpoints in y and z directions, Integers
 - `Ly`, `Lz`: ...
 - `H`: function

# Keyword arguments:
 - `Hy`: function or nothing
 - `type`: :chebyshev or :laguerre
"""
function CreateGrid(
    Ny::Union{Int,Vector{Int}},
    Nz::Int,
    Ly::Vector,
    H::Function = y -> 1;
    Hy::Union{Function,Nothing} = nothing,
    type = :laguerre
)

    if type isa Vector
        for i in 1:2; @assert type[i] in (:chebyshev, :laguerre); end

        λ, Mλ = GridComposite(Ny, Ly, type)

    else
        @assert type in (:chebyshev, :laguerre, :fourier)

        if type == :chebyshev

            λ, Mλ = GridChebyshev(Ny, Ly)
        end

        if type == :laguerre
            λ, Mλ = GridLaguerre(Ny, Ly)
        end

        if type == :fourier
            λ, Mλ = GridFourier(Ny, Ly)
        end

    end

    ζ, Mζ = GridChebyshev(Nz, [-1, 0])  

    y, z = λ .* ones(1, Nz), H.(λ) .* ζ'

    return GridStruct(Ny, Nz, y, z, Ly, H, Hy, λ, ζ, sparse(Mλ), sparse(Mζ), type)    

end

"""
    CreateProblem



Normal Flow BCS:
    - top: :noflow, :freesurface or :none
    - bottom: :noflow or :none
    - left: :noflow or :none
    - right: :noflow or :none

Normal Stress BCs:
    - top: :nostress, :noflow or :none
    - bottom: :nostress, :noflow or :none
    - left: :nostress, :noflow or :none
    - right: :nostress, :noflow or :none

Normal Flux BCs:
    - top: :noflux, :none
    - bottom: :noflux, :none
    - left: :noflux, :none
    - right: :noflux, :none


"""
function CreateProblem(grid;
                       f::Number = 0.0,
                       g::Number = 9.8,
                       δh::Number = 0.0,
                       δa::Number = 0.0,
                       U::Union{Nothing,Array,Function} = nothing,
                       V::Union{Nothing,Array,Function} = nothing,
                       W::Union{Nothing,Array,Function} = nothing,
                       N²::Union{Nothing,Array,Function} = nothing,
                       M²::Union{Nothing,Array,Function} = nothing,
                       νh::Union{Nothing,Array,Function} = nothing,
                       νv::Union{Nothing,Array,Function} = nothing,
                       κh::Union{Nothing,Array,Function} = nothing,
                       κv::Union{Nothing,Array,Function} = nothing,
                       Uy::Union{Nothing,Array,Function} = nothing,
                       Vy::Union{Nothing,Array,Function} = nothing,
                       Wy::Union{Nothing,Array,Function} = nothing,
                       Uz::Union{Nothing,Array,Function} = nothing,
                       Vz::Union{Nothing,Array,Function} = nothing,
                       Wz::Union{Nothing,Array,Function} = nothing,
                       NormalFlowBCs::Vector{Symbol} = [:noflow, :noflow, :noflow, :none],
                       NormalStressBCs::Vector{Symbol} = [:none, :none, :none, :none],
                       NormalFluxBCs::Vector{Symbol} = [:none, :none, :none, :none],
)

    # Validate inputs:

    @assert NormalFlowBCs[1] in (:noflow, :freesurface, :none)
    for i in 2:4; @assert NormalFlowBCs[i] in (:noflow, :none); end
    for i in 1:4; @assert NormalStressBCs[i] in (:nostress, :noflow, :none); end
    for i in 1:4; @assert NormalFluxBCs[i] in (:noflux, :none); end

    # Get grid size and arrays:

    Ny, Nz = grid.Ny, grid.Nz
    y, z = grid.y, grid.z

    # Calculate total length in y direction:
    Ny = sum(Ny) - length(Ny) + 1

    # Use grid to create Dy and Dz:

    Iy, Iz, I₀ = sparse(I(Ny)), sparse(I(Nz)), sparse(I(Ny * Nz))
    Oy, Oz, O = spzeros(Ny, Ny), spzeros(Nz, Nz), spzeros(Ny * Nz, Ny * Nz)

    H = grid.H.(grid.λ)
    
    if grid.Hy isa Nothing
        Hy = grid.Mλ * (H .- H[end])
    else
        Hy = grid.Hy.(grid.λ)
    end

    Dy = kron(Iz, grid.Mλ) - kron(spdiagm(grid.ζ) * grid.Mζ, spdiagm(Hy ./ H))
    Dz = kron(grid.Mζ, spdiagm(1 ./ H))

    # Create background flow fields:

    U₀ = CreateDiagField(U, y, z)
    V₀ = CreateDiagField(V, y, z)
    W₀ = CreateDiagField(W, y, z)

    # Create stratification fields:

    N²₀ = CreateDiagField(N², y, z)
    M²₀ = CreateDiagField(M², y, z)

    # Create viscosity and diffusivity fields:

    νh₀ = CreateDiagField(νh, y, z)
    νv₀ = CreateDiagField(νv, y, z)
    κh₀ = CreateDiagField(κh, y, z)
    κv₀ = CreateDiagField(κv, y, z)

    # Create gradients of background flow:

    if Uy isa Nothing
        Uy₀ = spdiagm(Dy * (diag(U₀) .- U₀[end, end]))
    else
        Uy₀ = CreateDiagField(Uy, y, z)
    end

    if Vy isa Nothing
        Vy₀ = spdiagm(Dy * (diag(V₀) .- V₀[end, end]))
    else
        Vy₀ = CreateDiagField(Vy, y, z)
    end

    if Wy isa Nothing
        Wy₀ = spdiagm(Dy * (diag(W₀) .- W₀[end, end]))
    else
        Wy₀ = CreateDiagField(Wy, y, z)
    end

    if Uz isa Nothing
        Uz₀ = spdiagm(Dz * diag(U₀))
    else
        Uz₀ = CreateDiagField(Uz, y, z)
    end

    if Vz isa Nothing
        Vz₀ = spdiagm(Dz * diag(V₀))
    else
        Vz₀ = CreateDiagField(Vz, y, z)
    end

    if Wz isa Nothing
        Wz₀ = spdiagm(Dz * diag(W₀))
    else
        Wz₀ = CreateDiagField(Wz, y, z)
    end
 
    # Create parameter structure:

    params = ParamsStruct(f, g, δh, δa, U, V, W, N², M², νh, νv, κh, κv, Uy, Uz, Vy, Vz, Wy, Wz)

    # Create field structure:

    fields = FieldStruct(H, Hy, U₀, V₀, W₀, N²₀, M²₀, νh₀, νv₀, κh₀, κv₀, Uy₀, Uz₀, Vy₀, Vz₀, Wy₀, Wz₀, Dy, Dz)

    # Build linear system matrices:

    Lν = V₀ * Dy + W₀ * Dz - Dy * νh₀ * Dy - Dz * νv₀ * Dz
    Lκ = V₀ * Dy + W₀ * Dz - Dy * κh₀ * Dy - Dz * κv₀ * Dz

    D = [I₀ O O O O; O I₀ O O O; O O δh*I₀ O O; O O O I₀ O; O O O O O]
    L₀ = [O Uy₀-f*I₀ Uz₀ O O; -f*I₀ O O O -Dy; O O O I₀ -Dz; O M²₀ N²₀ O O; O Dy Dz O O] -
         im * [Lν O O O O; O Lν+Vy₀ Vz₀ O O; O δh*Wy₀ δh*(Lν+Wz₀) O O; O O O Lκ O; O O O O O]
    L₁ = [U₀ O O O I₀; O U₀ O O O; O O δh*U₀ O O; O O O U₀ O; I₀ O O O O]
    L₂ = -im * [νh₀ O O O O; O νh₀ O O O; O O δh*νh₀ O O; O O O κh₀ O; O O O O O]

    # Apply normal flow BCs:

    # Top - free surface:

    if NormalFlowBCs[1] == :freesurface

        if δa == 0; @warn "Free surface boundary conditions are selected but δa = 0"; end

        ApplyBC!(D; Ny, Nz, direction = :z, location = :max, equation = 5, value = [Oy, Oy, Oy, Oy, δa*Iy/g])
        ApplyBC!(L₀; Ny, Nz, direction = :z, location = :max, equation = 5, value = [Oy, Oy, -Iy, Oy, -im*δa/g*V₀*Dy])
        ApplyBC!(L₁; Ny, Nz, direction = :z, location = :max, equation = 5, value = [Oy, Oy, Oy, Oy, δa/g*U₀])
        ApplyBC!(L₂; Ny, Nz, direction = :z, location = :max, equation = 5)

    end

    # Top - no normal flow:

    if NormalFlowBCs[1] == :noflow

        ApplyBC!(D; Ny, Nz, direction = :z, location = :max, equation = 5)
        ApplyBC!(L₀; Ny, Nz, direction = :z, location = :max, equation = 5, value = [Oy, Oy, -Iy, Oy, Oy])
        ApplyBC!(L₁; Ny, Nz, direction = :z, location = :max, equation = 5)
        ApplyBC!(L₂; Ny, Nz, direction = :z, location = :max, equation = 5)

    end

    # Apply normal stress BCs:

    # Top - no stress:

    if NormalStressBCs[1] == :nostress

        ApplyBC!(D; Ny, Nz, direction = :z, location = :max, equation = 1)
        ApplyBC!(L₀; Ny, Nz, direction = :z, location = :max, equation = 1, value = [νv₀*Dz, Oy, Oy, Oy, Oy])
        ApplyBC!(L₁; Ny, Nz, direction = :z, location = :max, equation = 1)
        ApplyBC!(L₂; Ny, Nz, direction = :z, location = :max, equation = 1)

        ApplyBC!(D; Ny, Nz, direction = :z, location = :max, equation = 2)
        ApplyBC!(L₀; Ny, Nz, direction = :z, location = :max, equation = 2, value = [Oy, νv₀*Dz, Oy, Oy, Oy])
        ApplyBC!(L₁; Ny, Nz, direction = :z, location = :max, equation = 2)
        ApplyBC!(L₂; Ny, Nz, direction = :z, location = :max, equation = 2)

    end

    # Top - no tangential flow:

    if NormalStressBCs[1] == :noflow

        ApplyBC!(D; Ny, Nz, direction = :z, location = :max, equation = 1)
        ApplyBC!(L₀; Ny, Nz, direction = :z, location = :max, equation = 1, value = [Iy, Oy, Oy, Oy, Oy])
        ApplyBC!(L₁; Ny, Nz, direction = :z, location = :max, equation = 1)
        ApplyBC!(L₂; Ny, Nz, direction = :z, location = :max, equation = 1)

        ApplyBC!(D; Ny, Nz, direction = :z, location = :max, equation = 2)
        ApplyBC!(L₀; Ny, Nz, direction = :z, location = :max, equation = 2, value = [Oy, Iy, Oy, Oy, Oy])
        ApplyBC!(L₁; Ny, Nz, direction = :z, location = :max, equation = 2)
        ApplyBC!(L₂; Ny, Nz, direction = :z, location = :max, equation = 2)

    end

    # Bottom - no stress:

    if NormalStressBCs[2] == :nostress

        #τu = kron(Iz,spdiagm(Hy)) * νh₀*Dy + νv₀*Dz
        #τv = 2 * kron(Iz,spdiagm(Hy)) * νh₀*Dy + (I₀-kron(Iz,spdiagm(Hy.^2))) * νv₀*Dz
        #τw = (I₀-kron(Iz,spdiagm(Hy.^2))) * νh₀*Dy - 2 * kron(Iz,spdiagm(Hy)) * νv₀*Dz

        ApplyBC!(D; Ny, Nz, direction = :z, location = :min, equation = 1)
        ApplyBC!(L₀; Ny, Nz, direction = :z, location = :min, equation = 1, value = [kron(Iz,spdiagm(Hy))*νh₀*Dy+νv₀*Dz, Oy, Oy, Oy, Oy])
        ApplyBC!(L₁; Ny, Nz, direction = :z, location = :min, equation = 1)
        ApplyBC!(L₂; Ny, Nz, direction = :z, location = :min, equation = 1)

        ApplyBC!(D; Ny, Nz, direction = :z, location = :min, equation = 2)
        ApplyBC!(L₀; Ny, Nz, direction = :z, location = :min, equation = 2, value = [Oy, kron(Iz,spdiagm(Hy))*νh₀*Dy+νv₀*Dz, Oy, Oy, Oy])
        ApplyBC!(L₁; Ny, Nz, direction = :z, location = :min, equation = 2)
        ApplyBC!(L₂; Ny, Nz, direction = :z, location = :min, equation = 2)

    end

    # Bottom - no tangential flow:

    if NormalStressBCs[2] == :noflow

        ApplyBC!(D; Ny, Nz, direction = :z, location = :min, equation = 1)
        ApplyBC!(L₀; Ny, Nz, direction = :z, location = :min, equation = 1, value = [Iy, Oy, Oy, Oy, Oy])
        ApplyBC!(L₁; Ny, Nz, direction = :z, location = :min, equation = 1)
        ApplyBC!(L₂; Ny, Nz, direction = :z, location = :min, equation = 1)

        ApplyBC!(D; Ny, Nz, direction = :z, location = :min, equation = 2)
        ApplyBC!(L₀; Ny, Nz, direction = :z, location = :min, equation = 2, value = [Oy, Iy, -spdiagm(Hy), Oy, Oy])
        ApplyBC!(L₁; Ny, Nz, direction = :z, location = :min, equation = 2)
        ApplyBC!(L₂; Ny, Nz, direction = :z, location = :min, equation = 2)

    end

    # Left - no stress:

    if NormalStressBCs[3] == :nostress

        ApplyBC!(D; Ny, Nz, direction = :y, location = :min, equation = 1)
        ApplyBC!(L₀; Ny, Nz, direction = :y, location = :min, equation = 1, value = [νh₀*Dy, Oz, Oz, Oz, Oz])
        ApplyBC!(L₁; Ny, Nz, direction = :y, location = :min, equation = 1)
        ApplyBC!(L₂; Ny, Nz, direction = :y, location = :min, equation = 1)

        if δh > 0
            ApplyBC!(D; Ny, Nz, direction = :y, location = :min, equation = 3)
            ApplyBC!(L₀; Ny, Nz, direction = :y, location = :min, equation = 3, value = [Oz, Oz, νh₀*Dy, Oz, Oz])
            ApplyBC!(L₁; Ny, Nz, direction = :y, location = :min, equation = 3)
            ApplyBC!(L₂; Ny, Nz, direction = :y, location = :min, equation = 3)
        end

    end

    # Left - no tangential flow:

    if NormalStressBCs[3] == :noflow

        ApplyBC!(D; Ny, Nz, direction = :y, location = :min, equation = 1)
        ApplyBC!(L₀; Ny, Nz, direction = :y, location = :min, equation = 1, value = [Iz, Oz, Oz, Oz, Oz])
        ApplyBC!(L₁; Ny, Nz, direction = :y, location = :min, equation = 1)
        ApplyBC!(L₂; Ny, Nz, direction = :y, location = :min, equation = 1)

        if δh > 0
            ApplyBC!(D; Ny, Nz, direction = :y, location = :min, equation = 3)
            ApplyBC!(L₀; Ny, Nz, direction = :y, location = :min, equation = 3, value = [Oz, Oz, Iz, Oz, Oz])
            ApplyBC!(L₁; Ny, Nz, direction = :y, location = :min, equation = 3)
            ApplyBC!(L₂; Ny, Nz, direction = :y, location = :min, equation = 3)
        end

    end

    # Right - no stress:

    if NormalStressBCs[4] == :nostress

        ApplyBC!(D; Ny, Nz, direction = :y, location = :max, equation = 1)
        ApplyBC!(L₀; Ny, Nz, direction = :y, location = :max, equation = 1, value = [νh₀*Dy, Oz, Oz, Oz, Oz])
        ApplyBC!(L₁; Ny, Nz, direction = :y, location = :max, equation = 1)
        ApplyBC!(L₂; Ny, Nz, direction = :y, location = :max, equation = 1)

        if δh > 0
            ApplyBC!(D; Ny, Nz, direction = :y, location = :max, equation = 3)
            ApplyBC!(L₀; Ny, Nz, direction = :y, location = :max, equation = 3, value = [Oz, Oz, νh₀*Dy, Oz, Oz])
            ApplyBC!(L₁; Ny, Nz, direction = :y, location = :max, equation = 3)
            ApplyBC!(L₂; Ny, Nz, direction = :y, location = :max, equation = 3)
        end

    end

    # Right - no tangential flow:

    if NormalStressBCs[4] == :noflow

        ApplyBC!(D; Ny, Nz, direction = :y, location = :max, equation = 1)
        ApplyBC!(L₀; Ny, Nz, direction = :y, location = :max, equation = 1, value = [Iz, Oz, Oz, Oz, Oz])
        ApplyBC!(L₁; Ny, Nz, direction = :y, location = :max, equation = 1)
        ApplyBC!(L₂; Ny, Nz, direction = :y, location = :max, equation = 1)

        if δh > 0
            ApplyBC!(D; Ny, Nz, direction = :y, location = :max, equation = 3)
            ApplyBC!(L₀; Ny, Nz, direction = :y, location = :max, equation = 3, value = [Oz, Oz, Iz, Oz, Oz])
            ApplyBC!(L₁; Ny, Nz, direction = :y, location = :max, equation = 3)
            ApplyBC!(L₂; Ny, Nz, direction = :y, location = :max, equation = 3)
        end

    end

    # Apply normal flux BCs:

    # Top - no flux:

    if NormalFluxBCs[1] == :noflux

        ApplyBC!(D; Ny, Nz, direction = :z, location = :max, equation = 4)
        ApplyBC!(L₀; Ny, Nz, direction = :z, location = :max, equation = 4, value = [Oy, Oy, Oy, κv₀*Dz, Oy])
        ApplyBC!(L₁; Ny, Nz, direction = :z, location = :max, equation = 4)
        ApplyBC!(L₂; Ny, Nz, direction = :z, location = :max, equation = 4)

    end

    # Bottom - no flux:

    if NormalFluxBCs[2] == :noflux

        ApplyBC!(D; Ny, Nz, direction = :z, location = :min, equation = 4)
        ApplyBC!(L₀; Ny, Nz, direction = :z, location = :min, equation = 4, value = [Oy, Oy, Oy, kron(Iz,spdiagm(Hy))*κh₀*Dy+κv₀*Dz, Oy])
        ApplyBC!(L₁; Ny, Nz, direction = :z, location = :min, equation = 4)
        ApplyBC!(L₂; Ny, Nz, direction = :z, location = :min, equation = 4)

    end

    # Left - no flux:

    if NormalFluxBCs[3] == :noflux

        ApplyBC!(D; Ny, Nz, direction = :y, location = :min, equation = 4)
        ApplyBC!(L₀; Ny, Nz, direction = :y, location = :min, equation = 4, value = [Oz, Oz, Oz, κh₀*Dy, Oz])
        ApplyBC!(L₁; Ny, Nz, direction = :y, location = :min, equation = 4)
        ApplyBC!(L₂; Ny, Nz, direction = :y, location = :min, equation = 4)

    end

    # Right - no flux:

    if NormalFluxBCs[4] == :noflux

        ApplyBC!(D; Ny, Nz, direction = :y, location = :max, equation = 4)
        ApplyBC!(L₀; Ny, Nz, direction = :y, location = :max, equation = 4, value = [Oz, Oz, Oz, κh₀*Dy, Oz])
        ApplyBC!(L₁; Ny, Nz, direction = :y, location = :max, equation = 4)
        ApplyBC!(L₂; Ny, Nz, direction = :y, location = :max, equation = 4)

    end

    # Bottom - no normal flow (through slope):

    if NormalFlowBCs[2] == :noflow

        ApplyBC!(D; Ny, Nz, direction = :z, location = :min, equation = 3)
        ApplyBC!(L₀; Ny, Nz, direction = :z, location = :min, equation = 3, value = [Oy, spdiagm(Hy), Iy, Oy, Oy])
        ApplyBC!(L₁; Ny, Nz, direction = :z, location = :min, equation = 3)
        ApplyBC!(L₂; Ny, Nz, direction = :z, location = :min, equation = 3)

    end

    # Left - no normal flow:

    if NormalFlowBCs[3] == :noflow

        ApplyBC!(D; Ny, Nz, direction = :y, location = :min, equation = 2)
        ApplyBC!(L₀; Ny, Nz, direction = :y, location = :min, equation = 2, value = [Oz, Iz, Oz, Oz, Oz])
        ApplyBC!(L₁; Ny, Nz, direction = :y, location = :min, equation = 2)
        ApplyBC!(L₂; Ny, Nz, direction = :y, location = :min, equation = 2)

    end

    # Right - no normal flow:

    if NormalFlowBCs[4] == :noflow

        ApplyBC!(D; Ny, Nz, direction = :y, location = :max, equation = 2)
        ApplyBC!(L₀; Ny, Nz, direction = :y, location = :max, equation = 2, value = [Oz, Iz, Oz, Oz, Oz])
        ApplyBC!(L₁; Ny, Nz, direction = :y, location = :max, equation = 2)
        ApplyBC!(L₂; Ny, Nz, direction = :y, location = :max, equation = 2)

    end

    # Convert matrices to real if all entries are real:

    if maximum(abs.(imag(L₀))) == 0; L₀ = real(L₀); end
    if maximum(abs.(imag(L₂))) == 0; L₂ = real(L₂); end

    # Return problem as structure:

    return ProblemStruct(grid, params, fields, D, L₀, L₁, L₂)

end

"""
    ApplyBC!(M; Ny, Nz, direction, location, equation, value)

value is either 'nothing' or a vector of matrices. these matrices should either match
the size of the BC block being replaced (i.e. size Ny or Nz for z and y boundaries
respectively), or correspond to the full problem domain size (Ny * Nz). If the second
of these is chose, the relevant rows/columns will be extracted using the 'direction'
and 'location' information and used to set the BC value

direction is :y or :z

location is :min or :max

equation determines which equation rows are replaced




"""
function ApplyBC!(M;
                 Ny::Int,
                 Nz::Int,
                 direction::Symbol,
                 location::Symbol,
                 equation::Int,
                 value::Union{Vector,Nothing} = nothing,
)

    # Set index of rows to apply BCs to:

    if direction == :y

        if location == :min; index = 1:Ny:Ny*(Nz-1)+1; end
        if location == :max; index = Ny:Ny:Ny*Nz; end

    end

    if direction == :z

        if location == :min; index = 1:Ny; end
        if location == :max; index = Ny*Nz-Ny+1:Ny*Nz; end

    end

    rows = (equation - 1) * Ny*Nz .+ index

    # Set all rows to zero:

    M[rows, :] .= 0

    # If BC values provided, set these:

    if !isnothing(value)

        # Loop through blocks in system and apply values to each:

        for block in 1:length(value)

            # If a value is provided in the full domain, the relevant boundary part is extracted:

            if length(value[block]) == length(index)^2
                columns = (block - 1) * Ny * Nz .+ index
                R = value[block]
            else
                columns = ((block - 1) * Ny * Nz + 1):(block * Ny * Nz)
                R = value[block][index, :]
            end

            # Set BC rows:

            M[rows, columns] = R

        end

    end

end

"""
    CreateDiagField(F, y, z)



"""
function CreateDiagField(F::Union{Nothing,Array,Function},
                         y::Array,
                         z::Array
)

    Ny, Nz = size(y)

    if F isa Nothing; F₀ = spzeros(Ny * Nz, Ny * Nz); end
    if F isa Array; F₀ = spdiagm(reshape(F, Ny * Nz)); end
    if F isa Function; F₀ = spdiagm(reshape(F.(y, z), Ny * Nz)); end

    return F₀

end

"""
    ToField(F, Ny, Nz)



"""
function ToField(F::Array, Ny::Int, Nz::Int)

    return reshape(diag(F), Ny, Nz)

end

ToField(F::SparseMatrixCSC, Ny, Nz) = ToField(Array(F), Ny, Nz)

"""
    SolveProblem(problem, k; n, ω₀)

solves problem for a given wavenumber k, returns frequency and mode structure


"""
function SolveProblem(prob::ProblemStruct,
                      k::Number;
                      n::Int = 10,
                      ω₀::Number = prob.params.f / π
)

    # Get problem size:
    Ny, Nz = grid.Ny, grid.Nz

    # Calculate total length in y direction:
    Ny = sum(Ny) - length(Ny) + 1

    # Calculate RHS matrix L for given k value:
    L = prob.L₀ + k * prob.L₁ + k^2 * prob.L₂

    # Solve EVP for frequency ω and mode structure ϕ:
    ω, ϕ = GenEVP(prob.D, L, n, ω₀)

    # Reshape to Ny x Nz fields:
    ϕ = reshape(ϕ, Ny, Nz, 5, n)

    # Define u ,v, w, b, p from ϕ:
    u, v, w, b, p = ϕ[:, :, 1, :], im * ϕ[:, :, 2, :], im * ϕ[:, :, 3, :], ϕ[:, :, 4, :], ϕ[:, :, 5, :]

    return ω, p, u, v, w, b

end

"""
    GenEVP(A, B, n, ω₀)

solves ωAv = Bv for (ω, v)


generalised EVP solver, copy from MATLAB version

"""
function GenEVP(A::AbstractMatrix,
                B::AbstractMatrix,
                n::Int,
                ω₀::Number)

    # Pass problem to Arpack, we look for eigenvalues close to ω₀:
    ω′, ϕ = eigs(A, B - ω₀ * A; nev = n, maxiter = 1000, which = :LM)

    # Determine ω from ω′:
    ω = ω₀ .+ 1 ./ ω′

    return ω, ϕ

end

"""
    Base.summary function for custom type GridStruct

"""
function Base.summary(g::GridStruct)
    return string("Grid structure for wave/instability problem with (Ny, Nz) = ", (g.Ny, g.Nz), ".")
end

"""
    Base.summary function for custom type ParamsStruct

"""
function Base.summary(g::ParamsStruct)
    return string("Parameter structure for wave/instability problem.")
end

"""
    Base.summary function for custom type FieldStruct

"""
function Base.summary(g::FieldStruct)
    return string("Field/operator structure for wave/instability problem.")
end

"""
    Base.summary function for custom type ProblemStruct

"""
function Base.summary(g::ProblemStruct)
    return string("Problem structure for wave/instability problem.")
end

"""
    Base.show function for custom type GridStruct

"""
function Base.show(io::IO, g::GridStruct)
    return print(io,
                 "GridStruct\n",
                 "  ├─────── size (Ny, Nz): ", (g.Ny, g.Nz), "\n",
                 "  ├───────── segments Ly: ", g.Ly, "\n",
                 "  ├──────────────── type: ", g.type, "\n",
                 "  └──────── Hy specified: ", ~(g.Hy isa Nothing),
    )
end

"""
    Base.show function for custom type ParamsStruct

"""
function Base.show(io::IO, g::ParamsStruct)
    return print(io,
                 "ParamsStruct\n",
                 "  ├───────────────── Coriolis parameter f: ", g.f, "\n",
                 "  ├───────── gravitational acceleration g: ", g.g, "\n",
                 "  ├───────────── hydrostatic parameter δh: ", g.δh, "\n",
                 "  ├──────────── free surface parameter δa: ", g.δa, "\n",
                 "  ├───── x velocity specified (U, Uy, Uz): ", (~(g.U isa Nothing), ~(g.Uy isa Nothing), ~(g.Uz isa Nothing)), "\n",
                 "  ├───── y velocity specified (V, Vy, Vz): ", (~(g.V isa Nothing), ~(g.Vy isa Nothing), ~(g.Vz isa Nothing)), "\n",
                 "  ├───── z velocity specified (W, Wy, Wz): ", (~(g.W isa Nothing), ~(g.Wy isa Nothing), ~(g.Wz isa Nothing)), "\n",
                 "  ├─ buoyancy gradient specified (N², M²): ", (~(g.N² isa Nothing), ~(g.M² isa Nothing)), "\n",
                 "  ├───────── viscosity specified (νh, νv): ", (~(g.νh isa Nothing), ~(g.νv isa Nothing)), "\n",
                 "  └─────── diffusivity specified (κh, κv): ", (~(g.κh isa Nothing), ~(g.κv isa Nothing)),
    )
end

"""
    Base.show function for custom type FieldStruct

"""
function Base.show(io::IO, g::FieldStruct)

    Ny = length(g.H)
    Nz = Int(size(g.U)[1] / Ny)

    return print(io,
                 "FieldStruct\n",
                 "  └─────── field size (Ny, Nz): ", (Ny, Nz),
    )
end

"""
    Base.show function for custom type ProblemStruct

"""
function Base.show(io::IO, g::ProblemStruct)
    return print(io,
                 "ProblemStruct\n",
                 "  ├─── real (D, L₀, L₁, L₂): ", (isreal(g.D), isreal(g.L₀), isreal(g.L₁), isreal(g.L₂)), "\n",
                 "  └──────────── matrix size: ", size(g.D),
    )
end






# Plotting (CairoMakie): surface(X, Y, zeros(size(Z)), color = Z, shading = NoShading, interpolate = true)

# Use Plots, write function to interpolate onto a rectilinear grid