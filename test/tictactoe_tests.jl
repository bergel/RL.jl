@testset "Tic tac toe" begin
    @testset "Initialization" begin
        board = Board()
        @test length(board.cells) == 9
        @test all(board.cells .== ' ')
        @test board.current_player_mark == 'X'

        custom_board = Board(['X', 'O', ' ', ' ', 'X', 'O', ' ', ' ', ' '], [HumanPlayer(), HumanPlayer()], 'X')
        @test custom_board.cells == ['X', 'O', ' ', ' ', 'X', 'O', ' ', ' ', ' ']
        @test custom_board.current_player_mark == 'X'
    end

    @testset "Cell Access" begin
        board = Board()
        board.cells[1] = 'X'
        @test board.cells[1] == 'X'
        @test board.cells[2] == ' '
    end

    @testset "Current Player" begin
        board = Board()
        @test board.current_player_mark == 'X'
        board.current_player_mark = 'O'
        @test board.current_player_mark == 'O'

        swap_player(board)
        @test board.current_player_mark == 'X'
    end

    @testset "Play" begin
        board = Board()
        @test is_cell_empty(board, 1)
        play!(board, 'X', 1)
        @test board.cells[1] == 'X'
        @test board.current_player_mark == 'X'
        @test_throws AssertionError play!(board, 'X', 1)  # Cell already occupied
        @test_throws AssertionError play!(board, 'Y', 2)  # Invalid player
        @test_throws AssertionError play!(board, 'X', 10)

        play!(board, 'O', 2)
        @test board.cells[2] == 'O'
        @test board.current_player_mark == 'X'  # Current player should not change after play

        swap_player(board)
        r = play!(board, 'X', 3)
        @test r == 3
        @test board.cells[3] == 'X'
        @test board.current_player_mark == 'O'  # Current player should change after swap
    end

    @testset "printing" begin
        board = Board()
        io = IOBuffer()
        show(io, board)
        output = String(take!(io))
        expected_output = "  |   |  \n---------\n  |   |  \n---------\n  |   |  \n"
        @test output == expected_output

        board = Board(['X', 'O', ' ', ' ', 'X', 'O', ' ', ' ', ' '], [HumanPlayer(), HumanPlayer()], 'X')
        io = IOBuffer()
        show(io, board)
        output = String(take!(io))
        expected_output = "X | O |  \n---------\n  | X | O\n---------\n  |   |  \n"
        @test output == expected_output
    end

    @testset "run_game" begin
        for _ in 1:50
            result = run_game(verbose=false, players=[RandomPlayer(), RandomPlayer()])
            @test result in (0, 1, 2)
        end
    end

    @testset "RL Training" begin
        initial_cells = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
        context = RLContext(TTTState(initial_cells, 'X'), 9; max_steps_per_episode = 9, n_episodes=300)
        train(context)
    end
end