module RL

using StructEquality
export State, display_map, new_car_pos
export UP, DOWN, LEFT, RIGHT, ACTIONS, start_state

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

start_state = State((2, 2), grid)

function new_car_pos(state::State, action)
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
        error("Invalid action")
    end
end

function Base.show(io::IO, state::State)
    println(io, "State(position=$(state.position), grid=[")
    for row in state.grid
        # println(join(row, ' '))
        print(io, "    ")
        print(io, row)
        println(io, ",")
    end
    println(io, "])")
end


end # module RL
