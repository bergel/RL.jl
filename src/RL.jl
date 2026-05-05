module RL

using Random, StructEquality

export State, new_car_pos, act
export UP, DOWN, LEFT, RIGHT, ACTIONS, q, choose_action, train
export RLContext

ZOMBIE = "z"
CAR = "c"
ICE_CREAM = "i"
EMPTY = "*"

grid = [[ICE_CREAM, EMPTY],
        [ZOMBIE, CAR]]

@struct_hash_equal struct State
    position::Tuple{Int, Int}
    grid::Vector{Vector{String}}
end

function State(; position::Tuple{Int, Int}, grid::Vector{Vector{String}})
    return State(position, grid)
end

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = [UP, DOWN, LEFT, RIGHT]

function new_car_pos(state::State, action::Int)
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

function Base.show(io::IO, state::State)
    println(io, "State(position=$(state.position), grid=[")
    for row in state.grid
        print(io, "    ")
        print(io, row)
        println(io, ",")
    end
    println(io, "])")
end

function act(state::State, action::Int, io::IO=devnull)
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

    return State(position=new_pos, grid=new_grid), reward, is_done
end

struct RLContext
    q_table::Dict{State, Vector{Float64}}
    alphas::Vector{Float64}
    gamma::Float64
    eps::Float64
    start_state::State

    # How many episodes we are considering? More episodes means more exploration, and
    # therefore a more efficient model, but also more time to train.
    n_episodes::Int

    # How many steps, at maximum, per episode? We need to make sure that this number is large
    # enough for end the episode
    max_steps_per_episode::Int

    seed::Int

    function RLContext(
        start_state::State; 
        min_alpha::Float64=0.02, 
        max_steps_per_episode::Int=20, 
        n_episodes::Int = 80, 
        gamma::Float64=1.0, 
        eps::Float64=0.1,
        seed::Int = 42
    )
        return new(
            Dict{State, Vector{Float64}}(), 
            collect(range(1.0, stop=min_alpha, length=n_episodes)), 
            gamma, 
            eps, 
            start_state,
            n_episodes,
            max_steps_per_episode,
            seed
        )
    end
end

function q(context::RLContext, state::State, action::Union{Int, Nothing}=nothing)
    if !(haskey(context.q_table, state))
        context.q_table[state] = zeros(length(ACTIONS))
    end

    if action === nothing
        return context.q_table[state]
    else
        return context.q_table[state][action + 1]
    end
end

function choose_action(context::RLContext, state::State)
    rand() < context.eps && return rand(ACTIONS)
    return argmax(q(context, state)) - 1
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

            q(context, state)[action + 1] += alpha * (reward + context.gamma * maximum(q(context, new_state)) - q(context, state, action))

            state = new_state
            total_reward += reward

            is_done && break
        end
        println(io, "Episode $episode: Total Reward = $total_reward")
    end
end

end # module RL
