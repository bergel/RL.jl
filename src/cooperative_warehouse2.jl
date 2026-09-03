mutable struct CWAgent
    name::String
    position::Tuple{Int, Int}

    # behavior(agent, env, row) -> move
    behavior::Function

    score::Int
end

mutable struct WarehouseEnv
    grid::Matrix{Int}
    rows::Int
    cols::Int
end

function move_up!(agent::CWAgent, env::WarehouseEnv)
    new_row = max(1, agent.position[1] - 1)
    agent.position = (new_row, agent.position[2])
end

function move_down!(agent::CWAgent, env::WarehouseEnv)
    new_row = min(env.rows, agent.position[1] + 1)
    agent.position = (new_row, agent.position[2])
end

function move_left!(agent::CWAgent, env::WarehouseEnv)
    new_col = max(1, agent.position[2] - 1)
    agent.position = (agent.position[1], new_col)
end

function move_right!(agent::CWAgent, env::WarehouseEnv)
    new_col = min(env.cols, agent.position[2] + 1)
    agent.position = (agent.position[1], new_col)
end

function stay!(agent::CWAgent, env::WarehouseEnv)
    # Do nothing
end

function take_cell!(agent::CWAgent, env::WarehouseEnv)
    row, col = agent.position
    agent.score += env.grid[row, col]
    env.grid[row, col] = 0  # Clear the cell after taking resources
end

# const ACTIONS = [
#     move_up!,
#     move_down!,
#     move_left!,
#     move_right!,
#     stay!,
#     take_cell!
# ]