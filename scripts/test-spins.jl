using Revise
using StatsPlots, DataFramesMeta
using LaTeXStrings
using LinearAlgebra

using DrWatson

include(srcdir("spin-glass.jl"))

function max_so_far(X)
    M = similar(X)
    M[1] = X[1]
    for i in 2:length(X)
        M[i] = X[i] > M[i-1] ? X[i] : M[i-1]
    end
    return M
end

#PARISI = 0.763217 

N = 100;
J = randn(N, N)/sqrt(N);
J = UpperTriangular(J) - Diagonal(J);


SK = Model(Gurobi.Optimizer)
set_optimizer_attribute(SK, "TimeLimit", 10)
@variable(SK, x[1:N], Bin);
@objective(SK, Max, x'*Matrix(J)*x / N);
optimize!(SK)


parameters = Dict(
    :couplings => J,
    :mutation_type => [:point, :argmax],
    :δ => .05,
    :m => collect(1:2:10),
    :elitism => true
)

adf, _ = paramscan(parameters, initialize; adata = adata = [(:fitness, maximum)], agent_step! = mutate!, model_step! = select!, n = 250_000, parallel = true)

begin 
    plot(dpi = 500)
    c = 1
    for m in parameters[:m]
        color = palette(:tab10)[c]
        for mutation_type in [:point, :argmax]
            plt = @df @subset(adf, :mutation_type .== mutation_type, :m .== m) plot!(
                :step, 
                max_so_far(:maximum_fitness)/N, 
                dpi = 500,
                linestyle = mutation_type == :argmax ? :solid : :dash,
                color = color,
                title = "Optimization of " * L"H = \sum_{i, j = 1}^{N} J_{ij}\sigma_i\sigma_j" * ", N = $N",
                xlabel = "Generation " * L"t",
                ylabel = L"\max_{s \leq t} H(s)/N",
                label = mutation_type == :argmax ? "m = $m" : nothing,
                legend = :bottomright
            )
        end
        c += 1
    end
    hline!([objective_value(SK)], color = "black", linewidth = 2, label = "Gurobi estimate", linestyle = :dash)
    hline!([objective_bound(SK)], color = "black", linewidth = 2, label = "Gurobi upper bound", linestyle = :solid)
end
current()
savefig(plotsdir("SK-elitism"))
