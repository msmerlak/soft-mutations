using Agents
using ViennaRNA
using Random
using Distributions

const BASES = ['A', 'C', 'U', 'G']
window!(x) = (x = max(0, min(1, x)))

function base(p)
    return BASES[argmax(p)]
end

function prob2seq(v)
    return String(base.(v))
end


mutable struct RNA <: AbstractAgent
    id::Int            
    genome::Union{String, Vector}         
    structure::String         
end

function initialize(target_structure; popsize = 100, seed = 1, genome_type)
    properties = Dict(
        :L => length(target_structure),
        :popsize => popsize, 
        :target => target_structure,
        :fitness => a -> 1/(1+bp_distance(a.structure, target_structure))
        )
    rng = Random.MersenneTwister(seed)
    model = ABM(
        RNA;
        properties, rng, scheduler = Schedulers.randomly
    )

    for n in 1:popsize
        if genome_type == "hard"
            add_agent!(model, 
            randstring(model.rng, BASES, model.L), 
            "")
        elseif genome_type == "soft"
            add_agent!(model, 
            [rand(model.rng) for _ in 1:model.L],
            ""
            )
    end
    return model
end

function mutate!(agent, model)
    if isa(agent.genome, String)
        for i in 1:model.L 
            rand(model.rng) < model.μ && agent.genome[i] = rand(model.rng, BASES)
        end
    elseif isa(agent.genome, Vector)
        for i in 1:model.L 
            agent.genome[i] .*= rand(model.rng, LogNormal(0, model.μ), length(BASES))
            window!.(agent.genome[i])
        end
    end
end

function fold!(agent, model)
    agent.structure = mfe(
        isa(agent.genome, String) ? agent.genome : prob2seq(agent.genome))
end

function agent_step!(agent, model)
    mutate!(agent, model)
    fold!(agent, model)
end

function select!(model)
    sample!(model, model.popsize, model.fitness)
end