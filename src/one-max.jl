using Agents
using Random, StatsBase
using LinearAlgebra

window(x) = max(0., min(1., x))
current_best(model) = model.agents[argmax(a -> a[2].fitness, model.agents).first]

mutable struct Tour <: AbstractAgent
    id::Int            
    probabilities::Vector{Float64}        
    sequence::Vector{Float64}
    fitness::Float64         
end

function initialize(;
    L = 10,
    m = 2, 
    δ = .1, 
    popsize = 100, 
    mutation_type = :point,
    elitism = false,
    reset = false)


    properties = Dict(
        :L => L,
        :popsize => popsize, 
        :fitness => a -> sum(a.sequence),
        :δ => δ,
        :m => m,
        :mutation_type => mutation_type,
        :elitism => elitism,
        :reset => reset
        )

    rng = Random.MersenneTwister()
    model = ABM(
        Tour;
        properties, rng, scheduler = Schedulers.fastest
    )

    for n in 1:popsize
        add_agent!(model, 
        fill(.5, model.L),
        rand(model.rng, [0, 1], model.L),
        0.
        )
    end
    if model.elitism 
        model.properties[:best_probabilities] = fill(.5, model.L)
        model.properties[:best_sequence] = rand(model.rng, [0., 1.], model.L)
        model.properties[:best_fitness] = 0.
    end

    model.properties[:mutation_map] = zeros(model.L)
    model.properties[:consensus] = zeros(model.L)
    model.properties[:step] = 1
    return model
end

function mutate!(agent, model)
    if isa(model.m, Int)
        i = sample(model.rng, 1:model.L, model.m, replace = false)
    elseif model.m == :zipf
        i = sample(model.rng, 1:model.L, 
        min(model.L, 
        sample(model.rng, 1:model.L, Weights((1:model.L).^(-2)))
        ),
        replace = false
        )
    end

    if model.mutation_type == :argmax

    agent.probabilities[i] .+= model.δ*(2rand(model.rng, model.m) .- 1.)
    @. agent.probabilities[i] = max(0, min(1, agent.probabilities[i]))
    @. model.mutation_map[i] += round(agent.probabilities[i]) != agent.sequence[i]
    @. agent.sequence[i] = round(agent.probabilities[i])

    elseif model.mutation_type == :point
        agent.sequence[i] .= 1. .- agent.sequence[i]
    end
    agent.fitness = model.fitness(agent)
end

function select!(model)

    model.step += 1
    if model.reset
        if model.step %100 == 0
            for agent in allagents(model)
                agent.probabilities = fill(0.5, model.L)
            end
        end
    end

    model.mutation_map = zeros(model.L)
    model.consensus = [mean([agent.sequence[i] for agent in allagents(model)]) for i in 1:model.L]

    if model.elitism
        c_best = current_best(model)
        if c_best.fitness > model.best_fitness
            model.best_probabilities = c_best.probabilities
            model.best_sequence = c_best.sequence
            model.best_fitness = c_best.fitness
        end
        Agents.sample!(model, model.popsize - 1, :fitness)
        add_agent!(model, 
                model.best_probabilities, 
                model.best_sequence, 
                model.best_fitness
                )
    else
        Agents.sample!(model, model.popsize, :fitness)
    end

end