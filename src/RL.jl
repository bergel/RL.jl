module RL

using Random, StructEquality

export CarState, new_car_pos, act
export q, choose_action, train
export RLContext, NGGState

abstract type AbstractState end

struct RLContext
    start_state::AbstractState
    actions_count::Int

    q_table::Dict{AbstractState, Vector{Float64}}

    # Learning rate for Q-learning. Higher alpha means that the agent learns more from new
    # experiences, while lower alpha means that it relies more on past knowledge. We can
    # use a schedule of alphas that decrease over time to allow for more exploration in
    # the beginning and more exploitation later on.
    alphas::Vector{Float64}

    # Discount factor for future rewards. Higher gamma means that the agent values future
    # rewards more, while lower gamma means that it values immediate rewards more.
    gamma::Float64

    # Epsilon for epsilon-greedy action selection. Higher epsilon means more exploration,
    # but also more randomness in the actions taken by the agent.
    eps::Float64

    # How many episodes we are considering? More episodes means more exploration, and
    # therefore a more efficient model, but also more time to train.
    n_episodes::Int

    # How many steps, at maximum, per episode? We need to make sure that this number is large
    # enough for end the episode
    max_steps_per_episode::Int

    seed::Int

    function RLContext(
        start_state::AbstractState,
        actions_count::Int; 
        min_alpha::Float64=0.02, 
        max_steps_per_episode::Int=20, 
        n_episodes::Int = 80, 
        gamma::Float64=1.0, 
        eps::Float64=0.1,
        seed::Int = 42
    )
        q_table = Dict{AbstractState, Vector{Float64}}()
        alphas = collect(range(1.0, stop=min_alpha, length=n_episodes))
        return new(
            start_state,
            actions_count,
            q_table,
            alphas, 
            gamma, 
            eps, 
            
            n_episodes,
            max_steps_per_episode,
            seed
        )
    end
end

function q(context::RLContext, state::AbstractState, action::Union{Int, Nothing}=nothing)
    if !(haskey(context.q_table, state))
        context.q_table[state] = zeros(context.actions_count)
    end

    if action === nothing
        return context.q_table[state]
    else
        return context.q_table[state][action]
    end
end

function choose_action(context::RLContext, state::AbstractState)
    rand() < context.eps && return rand(1:context.actions_count)
    return argmax(q(context, state))
end

function train(context::RLContext, io::IO=stdout)
    Random.seed!(context.seed)

    for episode in 1:context.n_episodes
        state = context.start_state
        total_reward = 0
        alpha = context.alphas[episode]

        for step in 1:context.max_steps_per_episode
            action = choose_action(context, state)
            new_state, reward, is_done = act(state, action, io)

            q(context, state)[action] += alpha * (reward + context.gamma * maximum(q(context, new_state)) - q(context, state, action))

            state = new_state
            total_reward += reward

            is_done && break
        end
        println(io, "Episode $episode: Total Reward = $total_reward")
    end
end


################################################
# CAR 
ZOMBIE = "z"
CAR = "c"
ICE_CREAM = "i"
EMPTY = "*"

@struct_hash_equal struct CarState <: AbstractState
    position::Tuple{Int, Int}
    grid::Vector{Vector{String}}
end

function CarState(; position::Tuple{Int, Int}, grid::Vector{Vector{String}})
    return CarState(position, grid)
end

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

ACTIONS = [UP, DOWN, LEFT, RIGHT]

function new_car_pos(state::CarState, action::Int)
    row, col = state.position
    if action == UP
        return (max(row - 1, 1), col)
    elseif action == DOWN
        return (min(row + 1, length(state.grid)), col)
    elseif action == LEFT
        return (row, max(col - 1, 1))
    elseif action == RIGHT
        return (row, min(col + 1, length(state.grid[1])))
    else
        error("Invalid action $action. Valid actions are: $ACTIONS")
    end
end

function Base.show(io::IO, state::CarState)
    println(io, "CarState(position=$(state.position), grid=[")
    for row in state.grid
        print(io, "    ")
        print(io, row)
        println(io, ",")
    end
    println(io, "])")
end

function act(state::CarState, action::Int, io::IO=devnull)
    reward = -1
    is_done = false
    new_grid = deepcopy(state.grid)
    new_pos = new_car_pos(state, action)
    old_pos = state.position
    grid_item = state.grid[new_pos[1]][new_pos[2]]

    if grid_item == ZOMBIE
        println(io, "The car has been eaten by a zombie! Game over.")
        is_done = true
        reward = -100
    elseif grid_item == ICE_CREAM
        println(io, "The car has collected the ice cream! You win!")
        is_done = true
        reward = 1000
    elseif grid_item == EMPTY
        reward = -1  # Small negative reward for moving to an empty cell
    elseif grid_item == CAR
        reward = -1  # Small negative reward for moving to an empty cell
    end
    new_grid[new_pos[1]][new_pos[2]] = CAR  # Move the car to the new position
    new_grid[old_pos[1]][old_pos[2]] = EMPTY  # Clear the old position

    return CarState(position=new_pos, grid=new_grid), reward, is_done
end

################################################
# MULTI-ARMED BANDIT

# In practice, We have only one state since the number of arms is fixed, but we can still
# use the same structure for consistency with the rest of the code.
struct MABState <: AbstractState 
    arms_count::Int
    MABState(arms_count::Int=3) = new(arms_count)
end

function act(state::MABState, action::Int, io::IO=devnull)
    @assert action >= 1 && action <= state.arms_count "Action must be between 1 and $(state.arms_count)"
    true_probs = [0.2, 0.5, 0.75]
    reward = rand() < true_probs[action] ? 1 : 0
    return state, reward, false
end

################################################
# Number Guessing Game
# context = RLContext(NGGState(1), 100; max_steps_per_episode = 1000, n_episodes=10000, eps=0.1)
# train(context)
# argmax(q(context, NGGState(1)))  # Should show higher values for actions 6 and 7

@struct_hash_equal struct NGGState <: AbstractState
    current_guess::Int
end 

function act(state::NGGState, action::Int, io::IO=devnull)
    target = 7
    #is_done = action == target
    is_done = false  # We want the agent to keep learning even after finding the correct answer

    reward = action == target ? 10 : -1
    # reward = state.current_guess == target ? 10 : -1

    # new guess is the action taken by the agent
    return NGGState(action), reward, is_done
end

################################################
# TIC TAC TOE

include("tictactoe.jl")


################################################
# Dynamic Pricing for a Small Hotel

# @struct_hash_equal struct DPSHState <: AbstractState
#     day_of_week::Int,
#     rooms_left::Int,
#     season::String,
#     competitor_price_bucket::String
# end 

# function act(state::DPSHState, action::Int, io::IO=devnull)
#     room_price[80, 100, 120, 140, 160][action]  # Map action to a price
#     # Simulate demand based on the state and the price
#     demand = simulate_demand(state, room_price)
#     # revenue = min(demand, state.rooms_left) * room_price
#     reward = rooms_sold * room_price

#     booking_probability =
#         sigmoid(
#             3
#             - 0.02 * price
#             + 0.5 * weekend
#             + 0.8 * summer
#         )


#     new_rooms_left = max(state.rooms_left - demand, 0)

#     new_state = DPSHState(state.day_of_week, new_rooms_left, state.season, state.competitor_price_bucket)

#     return new_state, reward, new_rooms_left == 0
# end

end # module RL
