# =========================
# Cooperative Warehouse
# =========================

# Grid movement options
const MOVES = [
    (-1),  # left
    (0),   # stay
    (1)    # right
]

# -------------------------
# Agent Definition
# -------------------------

mutable struct Agent
    name::String
    col::Int

    # behavior(agent, env, row) -> move
    behavior::Function

    score::Int
end

# -------------------------
# Environment
# -------------------------

mutable struct WarehouseEnv
    grid::Matrix{Int}
    rows::Int
    cols::Int
end

function WarehouseEnv(grid::Matrix{Int})
    r, c = size(grid)
    WarehouseEnv(grid, r, c)
end

# -------------------------
# Utility
# -------------------------

function clamp_col(col, cols)
    return max(1, min(cols, col))
end

# -------------------------
# Simulation
# -------------------------

function run_simulation(env::WarehouseEnv,
                        agent1::Agent,
                        agent2::Agent)

    total = 0

    for row in 1:env.rows

        # Collect resources
        if agent1.col == agent2.col
            collected = env.grid[row, agent1.col]
        else
            collected =
                env.grid[row, agent1.col] +
                env.grid[row, agent2.col]
        end

        total += collected

        println("Row $row")
        println("  $(agent1.name) at column $(agent1.col)")
        println("  $(agent2.name) at column $(agent2.col)")
        println("  Collected = $collected")
        println()

        # Last row -> stop
        if row == env.rows
            break
        end

        # Ask each behavior for next move
        move1 = agent1.behavior(agent1, env, row)
        move2 = agent2.behavior(agent2, env, row)

        # Apply movement
        agent1.col = clamp_col(agent1.col + move1, env.cols)
        agent2.col = clamp_col(agent2.col + move2, env.cols)
    end

    return total
end

# =========================
# Example Behaviors
# =========================

# Random movement
function random_behavior(agent, env, row)
    return rand(MOVES)
end

# Greedy local movement
function greedy_behavior(agent, env, row)

    next_row = row + 1

    best_move = 0
    best_value = -1

    for move in MOVES

        next_col = clamp_col(agent.col + move, env.cols)

        value = env.grid[next_row, next_col]

        if value > best_value
            best_value = value
            best_move = move
        end
    end

    return best_move
end

# Always move right
function right_behavior(agent, env, row)
    return 1
end

# =========================
# Example Usage
# =========================

warehouse = [
    4 2 3 1;
    3 5 1 2;
    7 1 4 6;
    2 5 2 3
]

env = WarehouseEnv(warehouse)

agentA = Agent(
    "GreedyBot",
    1,
    greedy_behavior,
    0
)

agentB = Agent(
    "RandomBot",
    env.cols,
    random_behavior,
    0
)

total = run_simulation(env, agentA, agentB)

println("Total collected resources = $total")