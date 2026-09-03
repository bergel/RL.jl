# =========================================================
# Tiny Strategy Game Prototype (Julia)
# =========================================================
# Features:
# - Grid map
# - Resources
# - Units
# - Basic battles
# - Turn system
#
# Run:
#   julia game.jl
# =========================================================

using Random

# ---------------------------------------------------------
# TILE TYPES
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# UNIT
# ---------------------------------------------------------

mutable struct Unit
    x::Int
    y::Int
    hp::Int
    atk::Int
    owner::Int
end

# ---------------------------------------------------------
# PLAYER
# ---------------------------------------------------------

mutable struct Player
    id::Int
    food::Int
    wood::Int
    iron::Int
end

# ---------------------------------------------------------
# GAME STATE
# ---------------------------------------------------------

mutable struct Game
    width::Int
    height::Int
    map::Matrix{TileType}
    units::Vector{Unit}
    players::Vector{Player}
    turn::Int
end

# ---------------------------------------------------------
# MAP GENERATION
# ---------------------------------------------------------

function random_tile()
    r = rand()

    if r < 0.05
        return Water()
    elseif r < 0.10
        return Forest()
    elseif r < 0.15
        return Mountain()
    elseif r < 0.20
        return Village()
    else
        return Plains()
    end
end

function generate_map(w, h)
    [random_tile() for y in 1:h, x in 1:w]
end

# ---------------------------------------------------------
# DISPLAY MAP
# ---------------------------------------------------------

function display(game::Game)

    println("\n=== TURN $(game.turn) ===")

    for y in 1:game.height
        row = ""

        for x in 1:game.width

            # Check units first
            unit_here = nothing

            for u in game.units
                if u.x == x && u.y == y && u.hp > 0
                    unit_here = u
                    break
                end
            end

            if unit_here !== nothing
                row *= string(unit_here.owner)
            else
                row *= tile_symbol(game.map[y, x])
            end

            row *= " "
        end

        println(row)
    end

    println()

    for p in game.players
        println("Player $(p.id): Food=$(p.food) Wood=$(p.wood) Iron=$(p.iron)")
    end
end

# ---------------------------------------------------------
# RESOURCE COLLECTION
# ---------------------------------------------------------

function collect_resources!(game::Game)

    for p in game.players

        for u in game.units

            if u.owner == p.id && u.hp > 0

                tile = game.map[u.y, u.x]

                if tile isa Forest
                    p.wood += 2
                elseif tile isa Mountain
                    p.iron += 2
                elseif tile isa Village
                    p.food += 3
                else
                    p.food += 1
                end
            end
        end
    end
end

# ---------------------------------------------------------
# MOVEMENT
# ---------------------------------------------------------

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

    # Cannot enter water
    if game.map[ny, nx] isa Water
        return
    end

    unit.x = nx
    unit.y = ny
end

# ---------------------------------------------------------
# BATTLE
# ---------------------------------------------------------

function battle!(a::Unit, b::Unit)

    println("Battle: Player $(a.owner) vs Player $(b.owner)")

    damage_to_b = max(1, a.atk + rand(-1:2))
    damage_to_a = max(1, b.atk + rand(-1:2))

    b.hp -= damage_to_b
    a.hp -= damage_to_a

    println("  A deals $damage_to_b")
    println("  B deals $damage_to_a")

    if a.hp <= 0
        println("  Player $(a.owner) unit destroyed")
    end

    if b.hp <= 0
        println("  Player $(b.owner) unit destroyed")
    end
end

# ---------------------------------------------------------
# FIND BATTLES
# ---------------------------------------------------------

function resolve_battles!(game::Game)

    for i in 1:length(game.units)
        for j in (i+1):length(game.units)

            a = game.units[i]
            b = game.units[j]

            if a.hp > 0 &&
               b.hp > 0 &&
               a.owner != b.owner &&
               a.x == b.x &&
               a.y == b.y

                battle!(a, b)
            end
        end
    end

    # Remove dead units
    filter!(u -> u.hp > 0, game.units)
end

# ---------------------------------------------------------
# AI TURN
# ---------------------------------------------------------

function ai_turn!(game::Game)

    for u in game.units
        move_unit!(game, u)
    end
end

# ---------------------------------------------------------
# WIN CHECK
# ---------------------------------------------------------

function check_winner(game::Game)

    alive_players = unique(u.owner for u in game.units)

    if length(alive_players) == 1
        return first(alive_players)
    end

    return nothing
end

# ---------------------------------------------------------
# GAME SETUP
# ---------------------------------------------------------

function create_game()

    width = 10
    height = 10

    map = generate_map(width, height)

    players = [
        Player(1, 10, 10, 10),
        Player(2, 10, 10, 10)
    ]

    units = [
        Unit(2, 2, 10, 4, 1),
        Unit(3, 2, 10, 4, 1),

        Unit(9, 9, 10, 4, 2),
        Unit(8, 9, 10, 4, 2)
    ]

    Game(
        width,
        height,
        map,
        units,
        players,
        1
    )
end

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------

function run_game()

    game = create_game()

    while true

        display(game)

        collect_resources!(game)

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

# ---------------------------------------------------------
# START
# ---------------------------------------------------------

run_game()
