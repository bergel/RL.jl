using Random

# =========================================================
# TILES
# =========================================================

abstract type TileType end

struct Plains <: TileType end
struct Forest <: TileType end
struct Mountain <: TileType end
struct Water <: TileType end
struct Village <: TileType end

tile_symbol(::Plains) = "."
tile_symbol(::Forest) = "F"
tile_symbol(::Mountain) = "M"
tile_symbol(::Water) = "~"
tile_symbol(::Village) = "V"

# =========================================================
# UNIT TYPES
# =========================================================

abstract type UnitType end

struct Militia <: UnitType end
struct Archer <: UnitType end
struct Knight <: UnitType end

unit_name(::Militia) = "Militia"
unit_name(::Archer) = "Archer"
unit_name(::Knight) = "Knight"

unit_symbol(::Militia) = "M"
unit_symbol(::Archer) = "A"
unit_symbol(::Knight) = "K"

unit_stats(::Militia) = (10, 3)
unit_stats(::Archer) = (8, 4)
unit_stats(::Knight) = (15, 6)

unit_cost(::Militia) = (food=3, wood=0, iron=2)
unit_cost(::Archer) = (food=3, wood=2, iron=2)
unit_cost(::Knight) = (food=5, wood=0, iron=5)

# =========================================================
# UNIT
# =========================================================

mutable struct Unit
    x::Int
    y::Int
    hp::Int
    atk::Int
    owner::Int
    kind::UnitType
end

# =========================================================
# PLAYER
# =========================================================

mutable struct Player
    id::Int
    food::Int
    wood::Int
    iron::Int
end

# =========================================================
# GAME
# =========================================================

mutable struct Game
    width::Int
    height::Int
    map::Matrix{TileType}
    units::Vector{Unit}
    players::Vector{Player}
    turn::Int
end

# =========================================================
# MAP GENERATION
# =========================================================

function random_tile()

    r = rand()

    if r < 0.05
        return Water()
    elseif r < 0.10
        return Forest()
    elseif r < 0.12
        return Mountain()
    elseif r < 0.15
        return Village()
    else
        return Plains()
    end
end

generate_map(w,h) = [random_tile() for y in 1:h, x in 1:w]

# =========================================================
# DISPLAY
# =========================================================

function display(game::Game)

    println("\n=== TURN $(game.turn) ===")

    for y in 1:game.height

        row = ""

        for x in 1:game.width

            unit_here = nothing

            for u in game.units
                if u.hp > 0 && u.x == x && u.y == y
                    unit_here = u
                    break
                end
            end

            if unit_here !== nothing
                row *= string(unit_here.owner)
            else
                row *= tile_symbol(game.map[y,x])
            end

            row *= " "
        end

        println(row)
    end

    println()

    for p in game.players
        println("Player $(p.id)")
        println(" Food=$(p.food) Wood=$(p.wood) Iron=$(p.iron)")
    end

    println()
end

# =========================================================
# RESOURCE COLLECTION
# =========================================================

function collect_resources!(game::Game)

    for u in game.units

        player = game.players[u.owner]

        tile = game.map[u.y, u.x]

        if tile isa Forest
            player.wood += 2
        elseif tile isa Mountain
            player.iron += 2
        elseif tile isa Village
            player.food += 3
        else
            player.food += 1
        end
    end
end

# =========================================================
# CREATE UNIT
# =========================================================

function create_unit(x, y, owner, kind::UnitType)

    hp, atk = unit_stats(kind)

    Unit(
        x,
        y,
        hp,
        atk,
        owner,
        kind
    )
end

# =========================================================
# RECRUITMENT
# =========================================================

function can_afford(player::Player, kind::UnitType)

    c = unit_cost(kind)

    return (
        player.food >= c.food &&
        player.wood >= c.wood &&
        player.iron >= c.iron
    )
end

function pay_cost!(player::Player, kind::UnitType)

    c = unit_cost(kind)

    player.food -= c.food
    player.wood -= c.wood
    player.iron -= c.iron
end

function recruit_units!(game::Game)

    for p in game.players

        # Find villages occupied by player's units
        villages = []

        for u in game.units

            if u.owner == p.id &&
               game.map[u.y, u.x] isa Village

                push!(villages, (u.x, u.y))
            end
        end

        # Recruit randomly
        for (x,y) in villages

            choices = [Militia(), Archer(), Knight()]
            kind = rand(choices)

            if can_afford(p, kind)

                pay_cost!(p, kind)

                new_unit = create_unit(x, y, p.id, kind)

                push!(game.units, new_unit)

                println("Player $(p.id) recruits $(unit_name(kind))")
            end
        end
    end
end

# =========================================================
# MOVEMENT
# =========================================================

function move_unit!(game::Game, unit::Unit)

    dirs = [
        (1,0),
        (-1,0),
        (0,1),
        (0,-1)
    ]

    dx, dy = rand(dirs)

    nx = clamp(unit.x + dx, 1, game.width)
    ny = clamp(unit.y + dy, 1, game.height)

    if game.map[ny, nx] isa Water
        return
    end

    unit.x = nx
    unit.y = ny
end

# =========================================================
# BATTLE
# =========================================================

function battle!(a::Unit, b::Unit)

    println(
        "$(unit_name(a.kind)) P$(a.owner) vs " *
        "$(unit_name(b.kind)) P$(b.owner)"
    )

    damage_to_b = max(1, a.atk + rand(-1:2))
    damage_to_a = max(1, b.atk + rand(-1:2))

    b.hp -= damage_to_b
    a.hp -= damage_to_a

    println("  A deals $damage_to_b")
    println("  B deals $damage_to_a")
end

function resolve_battles!(game::Game)

    n = length(game.units)

    for i in 1:n-1
        for j in i+1:n

            a = game.units[i]
            b = game.units[j]

            if a.hp > 0 &&
               b.hp > 0 &&
               a.owner != b.owner &&
               a.x == b.x &&
               a.y == b.y

                battle!(a,b)
            end
        end
    end

    filter!(u -> u.hp > 0, game.units)
end

# =========================================================
# AI
# =========================================================

function ai_turn!(game::Game)

    for u in game.units
        move_unit!(game, u)
    end
end

# =========================================================
# WINNER
# =========================================================

function check_winner(game::Game)

    alive = unique(u.owner for u in game.units)

    if length(alive) == 1
        return first(alive)
    end

    nothing
end

# =========================================================
# SETUP
# =========================================================

function create_game()

    width = 12
    height = 12

    map = generate_map(width, height)

    players = [
        Player(1, 10, 10, 10),
        Player(2, 10, 10, 10)
    ]

    units = Unit[]

    push!(units, create_unit(2,2,1,Militia()))
    #push!(units, create_unit(3,2,1,Archer()))

    push!(units, create_unit(10,10,2,Militia()))
    #push!(units, create_unit(9,10,2,Knight()))

    Game(
        width,
        height,
        map,
        units,
        players,
        1
    )
end

# =========================================================
# MAIN LOOP
# =========================================================

function run_game()

    game = create_game()

    while true

        display(game)

        collect_resources!(game)

        recruit_units!(game)

        ai_turn!(game)

        resolve_battles!(game)

        winner = check_winner(game)

        if winner !== nothing
            println("\nPLAYER $winner WINS!")
            break
        end

        game.turn += 1

        sleep(1)
    end
end

run_game()