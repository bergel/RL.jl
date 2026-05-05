using Test

using RL

ZOMBIE = "z"
CAR = "c"
ICE_CREAM = "i"
EMPTY = "*"

grid = [[ICE_CREAM, EMPTY],
        [ZOMBIE, CAR]]

grid2 = [[ICE_CREAM, ZOMBIE],
        [ZOMBIE, CAR]]      

@testset "State" begin
    state = State((1, 2), grid)

    @testset "Creation" begin
        @test state.position == (1, 2)
        @test state.grid == grid
    end    

    @testset "Comparison" begin
        @testset "Equal" begin
            state2 = State((1, 2), grid)
            @test state == state2
        end

        @testset "Not Equal" begin
            state3 = State((2, 1), grid)
            @test state != state3
        end

        @testset "Not Equal" begin
            state3 = State((1, 2), grid2)
            @test state != state3
        end        

        @testset "Hash" begin
            state2 = State((1, 2), grid)
            @test hash(state) == hash(state2)

            state3 = State((2, 1), grid)
            @test hash(state) != hash(state3)
        end

        @testset "Hash" begin
            state2 = State((1, 2), grid)
            @test hash(state) == hash(state2)

            state3 = State((2, 1), grid)
            @test hash(state) != hash(state3)
            
            state4 = State((1, 2), grid2)
            @test hash(state) != hash(state4)
        end        
    end

    @testset "Display" begin
        io = IOBuffer()
        show(io, state)
        output = String(take!(io))
        @test output == """
State(position=(1, 2), grid=[
    ["i", "*"],
    ["z", "c"],
])
"""
    end 

    @testset "new_car_pos" begin
        state = State((2, 2), [[EMPTY, EMPTY, EMPTY, EMPTY], 
                                [EMPTY, EMPTY, EMPTY, EMPTY], 
                                [EMPTY, EMPTY, EMPTY, EMPTY], 
                                [EMPTY, EMPTY, EMPTY, EMPTY]])

        @testset "Expected Positions" begin
            @test new_car_pos(state, UP) == (1, 2)
            @test new_car_pos(state, DOWN) == (3, 2)
            @test new_car_pos(state, LEFT) == (2, 1)
            @test new_car_pos(state, RIGHT) == (2, 3)
        end

        @testset "Invalid Action" begin
            @test_throws ErrorException new_car_pos(state, 99)
        end

        @testset "Boundary Conditions" begin
            state = State((1, 1), grid)
            @test new_car_pos(state, UP) == (1, 1)
            @test new_car_pos(state, LEFT) == (1, 1)

            state = State((2, 2), grid)
            @test new_car_pos(state, DOWN) == (2, 2)
            @test new_car_pos(state, RIGHT) == (2, 2)
        end
    end

    @testset "act" begin
        @testset "Move to ice cream cell" begin
            test_grid = [[EMPTY, ICE_CREAM],
                         [ZOMBIE, CAR]]
            state = State((2, 2), test_grid)
            result, _, _ = act(state, UP)

            @test result.position == (1, 2)
            @test result.grid == [[EMPTY, CAR],
                                   [ZOMBIE, EMPTY]]
        end

        @testset "Move to zombie cell" begin
            test_grid = [[EMPTY, ICE_CREAM],
                         [ZOMBIE, CAR]]
            state = State((2, 2), test_grid)

            result, _, _ = act(state, LEFT)
            @test result.position == (2, 1)
            @test result.grid == [[EMPTY, ICE_CREAM],
                                   [CAR, EMPTY]]
        end

        @testset "Move to ice cream cell" begin
            test_grid = [[CAR, ICE_CREAM],
                         [EMPTY, EMPTY]]
            state = State((1, 1), test_grid)

            result, _, _ = act(state, RIGHT)
            @test result.position == (1, 2)
            @test result.grid == [[EMPTY, CAR],
                                   [EMPTY, EMPTY]]
        end
    end

end

@testset "q" begin
    state = State((1, 1), [[EMPTY, ICE_CREAM], [ZOMBIE, CAR]])
    context = RLContext(state)
    @testset "Initialization" begin
        q_values = RL.q(context, state)
        @test length(q_values) == length(RL.ACTIONS)
        @test q_values == zeros(length(RL.ACTIONS))
    end

    @testset "Action lookup" begin
        @test RL.q(context, state, RL.UP) == 0.0
        @test RL.q(context, state, RL.RIGHT) == 0.0
        @test RL.q(context, state, RL.DOWN) == 0.0
        @test RL.q(context, state, RL.LEFT) == 0.0
    end
end

@testset "choose_action" begin
    state = State((1, 1), [[EMPTY, ICE_CREAM], [ZOMBIE, CAR]])

    @testset "Epsilon-greedy" begin
        # Set eps to 1 to always choose a random action
        context = RLContext(state; eps=1.0)
        random_actions = [choose_action(context, state) for _ in 1:100]
        @test all(action -> action in RL.ACTIONS, random_actions)

        context = RLContext(state; eps=0.0)
        @test choose_action(context, state) == 0
    end
end

@testset "train" begin
    start_state = State((2, 2), [[EMPTY, ICE_CREAM], [ZOMBIE, CAR]])
    context = RLContext(start_state)
    train(context, devnull)

    @test q(context, start_state) != zeros(length(RL.ACTIONS))
    @test argmax(q(context, start_state)) == 1
end