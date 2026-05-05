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

end
