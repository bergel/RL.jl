#=
rl_context = RLContext(MapCrawlerState(), 10; max_steps_per_episode = 20, n_episodes=500)
training = train(rl_context);

result = []
reward = 0
s = MapCrawlerState()
for _ in 1:1000
    a = RL.pick_best_action(rl_context, s)
    push!(result, a)
    s, reward, is_done = RL.act(s, a)
    is_done && break
end
(reward, result)
=#

export MapCrawlerState

@struct_hash_equal struct MapCrawlerState <: AbstractState
    x::Int64
    y::Int64
    MapCrawlerState(x, y) = new(x, y)
    MapCrawlerState() = MapCrawlerState(0, 0)
end

function available_actions(context::RLContext, ::MapCrawlerState)
    return 1:4
end

function act(state::MapCrawlerState, action::Int, io::IO=devnull)
    x = state.x
    y = state.y
    if action == 1
        x -= 1
    elseif action == 2
        x += 1
    elseif action == 3
        y -= 1
    elseif action == 4
        y += 1
    else
        error("Should not be here $(action)")
    end

    if x == 5 && y == 5
        return MapCrawlerState(x, y), 1, true
    end
    return MapCrawlerState(x, y), -1, false
end
