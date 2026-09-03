export Board, AbstractPlayer, HumanPlayer, RandomPlayer, run_game, play!, swap_player, check_winner
export is_cell_empty, get_next_move
export TTTState
export AIPlayer, run_exp_against_random, run_exp_against_previous

# Two humans playing the game:
# run_game(verbose=true, players=[HumanPlayer(), HumanPlayer()])
# run_game(verbose=true, players=[RandomPlayer(), RandomPlayer()])

# With an AI player:
# fill(' ', 9)
#=
initial_cells = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
context = RLContext(TTTState(initial_cells, 'X'), 9; max_steps_per_episode = 9, n_episodes=100_000, eps= 0.1)
train(context)
run_game(verbose=true, players=[AIPlayer(context), RandomPlayer()])

run_game(verbose=true, players=[AIPlayer(context), HumanPlayer()])


ss = TTTState([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 'X')
a = argmax(q(context, ss))
ss, _, _, = act(ss, a)
=#
abstract type AbstractPlayer end
struct HumanPlayer <: AbstractPlayer end

struct AIPlayer <: AbstractPlayer
    context::RLContext
end

struct RandomPlayer <: AbstractPlayer end

mutable struct Board
    cells::Vector{Char}  # ' ' for empty, 'X' for player 1, 'O' for player 2
    players::Vector{AbstractPlayer}
    current_player_mark::Char

    function Board(players::Vector{T} = [HumanPlayer(), HumanPlayer()]) where T <: AbstractPlayer
        return Board(fill(' ', 9), players, 'X')
    end

    function Board(
        cells::Vector{Char},
        players::Vector{T},
        current_player_mark::Char,
    ) where T <: AbstractPlayer
        @assert length(cells) == 9 "Board must have 9 cells"
        @assert current_player_mark in ('X', 'O') "Current player must be 1 or 2"
        new(cells, players, current_player_mark)
    end
end

function swap_player(board::Board)
    board.current_player_mark = board.current_player_mark == 'X' ? 'O' : 'X'
end

is_cell_empty(board::Board, position::Int) = board.cells[position] == ' '

function play!(board::Board, player::Char, position::Int)
    @assert position >= 1 && position <= 9 "Position must be between 1 and 9"
    @assert is_cell_empty(board, position) "Cell is already occupied"
    @assert player in ('X', 'O') "Player must be 'X' or 'O'"

    board.cells[position] = player
    return position
end

play!(board::Board, position::Int) = play!(board, board.current_player_mark, position)

is_board_full(board) = all(board.cells .!= ' ')

# Return 0: draw
# Return 1: player 1 wins
# Return 2: player 2 wins
function run_game(board::Board; verbose::Bool = false)
    while true
        verbose && println(board)
        verbose && println("Player: ", board.current_player_mark)
        verbose && println()

        if is_board_full(board)
            verbose && println("It's a draw!")
            return 0
            break
        end

        play!(board)
        if check_winner(board) != false
            verbose && println(board)
            verbose && println("Player ", board.current_player_mark, " wins!")
            return board.current_player_mark == 'X' ? 1 : 2
        end
        swap_player(board)
    end
end
function run_game(
    ;
    initial_cells::Vector{Char} = fill(' ', 9),
    verbose::Bool = false,
    players::Vector{T} = [HumanPlayer(), RandomPlayer()],
) where T <: AbstractPlayer
    run_game(Board(initial_cells, players, 'X'); verbose=verbose)
end

# Return 'X', 'O' or false
function check_winner(board::Board)
    winning_combinations = [
        [1, 2, 3], [4, 5, 6], [7, 8, 9],  # Rows
        [1, 4, 7], [2, 5, 8], [3, 6, 9],  # Columns
        [1, 5, 9], [3, 5, 7]              # Diagonals
    ]

    for combination in winning_combinations
        if board.cells[combination[1]] == board.cells[combination[2]] ==
           board.cells[combination[3]] && board.cells[combination[1]] != ' '
            return board.cells[combination[1]]
        end
    end
    return false
end

function get_next_move(player_mark::Char, board::Board)
    index = player_mark == 'X' ? 1 : 2
    return get_next_move(board.players[index], board)
end

function get_next_move(player::HumanPlayer, board::Board)
    print("Enter position (1-9): ")
    return parse(Int, readline(stdin))
end

function get_next_move(player::RandomPlayer, board::Board)
    available_positions = findall(c -> c == ' ', board.cells)
    return rand(available_positions)
end

# function get_next_move(player::AIPlayer, board::Board)
#     for r in player.rules
#         if r.cells == board.cells
#             return r.where_to_play
#         end
#     end

#     # @info "No hit!"
#     available_positions = findall(c -> c == ' ', board.cells)
#     return rand(available_positions)
# end

function play!(board::Board, verbose::Bool = false)
    verbose && println("Current player: ", board.current_player_mark, ". Enter position (1-9): ")

    position = get_next_move(board.current_player_mark, board)
    @assert position >= 1 && position <= 9 "Position must be between 1 and 9"
    @assert board.cells[position] == ' ' "Cell is already occupied"

    return play!(board, board.current_player_mark, position)
end

function Base.show(io::IO, board::Board)
    for i in 1:3
        println(io, join(board.cells[(i-1)*3+1:i*3], " | "))
        if i < 3
            println(io, "---------")
        end
    end
    println("Current player: $(board.current_player_mark)")
end

################################################
# RL on the game

@struct_hash_equal struct TTTState <: AbstractState
    cells::Vector{Char}
    current_player_mark::Char
end 

function get_next_move(player::AIPlayer, board::Board)
    state = TTTState(copy(board.cells), board.current_player_mark)
    vs = q(player.context, state)
    # @info vs
    return argmax(vs)
end

function Base.show(io::IO, state::TTTState)
    for i in 1:3
        println(io, join(state.cells[(i-1)*3+1:i*3], " | "))
        if i < 3
            println(io, "---------")
        end
    end
    println("Current player: $(state.current_player_mark)")
end

# Reward = -1 if the move is wrong or if it leads to an opponent win, +10 if the move is correct, 0 if it's a draw
function act(state::TTTState, action::Int, io::IO=devnull)
    @assert action >= 1 && action <= 9 "Action must be between 1 and 9: $action"
    # @assert state.current_player_mark in ('X', 'O') "Current player must be 'X' or 'O': $(state.current_player_mark)"
    @assert state.current_player_mark == 'X' "Current player must be 'X': $(state.current_player_mark)"

    board = Board(copy(state.cells), [RandomPlayer(), RandomPlayer()], state.current_player_mark)

    # make sure that the cell is free to play
    if !is_cell_empty(board, action)
        return state, -10, true  # Invalid move, end the episode with a negative reward
    end

    play!(board, state.current_player_mark, action)
    result = check_winner(board)

    # If the game is ended
    if result == 'X'
        return TTTState(copy(board.cells), board.current_player_mark), 100, true
    elseif result == 'O'
        return TTTState(copy(board.cells), board.current_player_mark), -1, true
    elseif is_board_full(board)
        return TTTState(copy(board.cells), board.current_player_mark), 0, true
    end

    # Else, we let the other play.
    swap_player(board)
    @assert board.current_player_mark == 'O'
    play!(board, false)

    result = check_winner(board)
    # if the game is ended
    if result == 'X'
        return TTTState(copy(board.cells), board.current_player_mark), 100, true
    elseif result == 'O'
        return TTTState(copy(board.cells), board.current_player_mark), -1, true
    elseif is_board_full(board)
        return TTTState(copy(board.cells), board.current_player_mark), 0, true
    end        

    swap_player(board)
    @assert board.current_player_mark == 'X'
    return TTTState(copy(board.cells), board.current_player_mark), -1, false
end

function eval_context(context::RLContext, nb_tries::Int = 100)
    score_0 = 0
    score_1 = 0
    score_2 = 0
    nb_errors = 0
    for _ in 1:nb_tries
        try
            result = run_game(verbose=false, players=[AIPlayer(context), RandomPlayer()])
            if result == 0
                score_0 += 1
            elseif result == 1
                score_1 += 1
            else
                score_2 += 1
            end
        catch e
            nb_errors += 1
        end
    end
    @info "Scores after $n_episodes episodes: Errors: $nb_errors, Draws: $score_0, AI wins: $score_1, Random wins: $score_2, q-table size: $(length(context.q_table))"    
end

# run_game(verbose=true, players=[AIPlayer(context), RandomPlayer()])
# run_game(verbose=true, players=[AIPlayer(context), HumanPlayer()])

# run_game(verbose=true, players=[HumanPlayer(), AIPlayer(context)])
# context = run_exp_against_random()
function run_exp_against_random()
    initial_cells = fill(' ', 9)
    episode_max = 2_000_000
    episode_max_steps = 100_000
    context = nothing
    for n_episodes in 0:episode_max_steps:episode_max
        @info "Training with $n_episodes episodes..."
        context = RLContext(
            TTTState(initial_cells, 'X'), 
            9; 
            max_steps_per_episode = 9, 
            n_episodes=n_episodes)
        train(context, devnull)

        eval_context(context)
    end
    return context
end


# function run_exp_against_previous()
#     initial_cells = fill(' ', 9)
#     episode_max = 10_000_000
#     episode_max_steps = 100_000
#     nb_tries = 100
#     previous_context = nothing
#     for n_episodes in 0:episode_max_steps:episode_max
#         @info "Training with $n_episodes episodes..."
#         context = RLContext(
#             TTTState(initial_cells, 'X'), 
#             9; 
#             max_steps_per_episode = episode_max_steps, 
#             n_episodes=n_episodes)
#         train(context, devnull)

#         score_0 = 0
#         score_1 = 0
#         score_2 = 0
#         nb_errors = 0
#         for _ in 1:nb_tries
#             try
#                 result = run_game(verbose=false, players=[AIPlayer(context), RandomPlayer()])
#                 if result == 0
#                     score_0 += 1
#                 elseif result == 1
#                     score_1 += 1
#                 else
#                     score_2 += 1
#                 end
#             catch e
#                 nb_errors += 1
#             end
#         end
#         @info "Scores after $n_episodes episodes: Errors: $nb_errors, Draws: $score_0, AI wins: $score_1, Random wins: $score_2"
#     end
# end