using StatsPlots, LaTeXStrings, DataFramesMeta
gr(dpi = 500)
using DrWatson
@quickactivate

include(srcdir("one-max.jl"))
moving_average(g, n) = [i < n ? mean(g[begin:i]) : mean(g[i-n+1:i]) for i in 1:length(g)]


model = initialize(L = 50, δ = .01, m = 2,
elitism = true, reset = false, mutation_type = :argmax)


adf, _ = run!(
    model, mutate!, select!, 10000;
    adata = [(:fitness, maximum), (:probabilities, mean), (:sequence, mean)]
    )

    
activity = heatmap(
reduce(hcat, [1 .- 2norm.(mp .- .5) for mp in adf.mean_probabilities]),
xlabel = "Generation", ylabel = "Site", title = L"1 - 2\vert \langle p_i\rangle - 1/2\vert"
)

quality = heatmap(
reduce(hcat, adf.mean_sequence),
xlabel = "Generation", ylabel = "Site", title = L"\langle s_i \rangle"
)

plot(activity, quality)
savefig(plotsdir("activity-quality-one-max"))

@df adf plot(:step, 
:maximum_fitness/model.L,
label = "max fitness",
xlabel = L"t"
)
plot!(adf.step,
        [corspearman(adf.mean_probabilities[t], adf.mean_sequence[t]) for t in 1:length(adf.step)],
        label = L"\textrm{corr}(\langle p\rangle, \langle s\rangle)",
        legend = :bottomright
)

