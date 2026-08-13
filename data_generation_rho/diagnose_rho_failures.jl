using Pkg
Pkg.activate(@__DIR__)

using JSON
using JuMP
using Ipopt
using PowerModels
using Random

include(joinpath(@__DIR__, "src", "OPFLearn.jl"))

const CASE_TO_PATH = Dict(
    "case14" => "pglib/pglib_opf_case14_ieee.m",
    "case30" => "pglib/pglib_opf_case30_ieee.m",
    "case57" => "pglib/pglib_opf_case57_ieee.m",
    "case118" => "pglib/pglib_opf_case118_ieee.m",
    "case500" => "pglib/pglib_opf_case500_goc.m",
    "case2000" => "pglib/pglib_opf_case2000_goc.m",
)

function parse_float_list(value::AbstractString)
    value = strip(value)
    if occursin(":", value)
        parts = split(value, ":")
        @assert length(parts) == 3 "Range format must be start:step:stop"
        start_value = parse(Float64, parts[1])
        step_value = parse(Float64, parts[2])
        stop_value = parse(Float64, parts[3])
        values = Float64[]
        current = start_value
        while current <= stop_value + step_value / 2
            push!(values, round(current; digits=10))
            current += step_value
        end
        return values
    end
    return [parse(Float64, strip(x)) for x in split(value, ",") if strip(x) != ""]
end

function default_config()
    return Dict{String, Any}(
        "case_name" => "case30",
        "branch_id" => "auto",
        "seed" => 1,
        "rho_attempts" => 500,
        "branch_attempts" => 100,
        "train_rhos" => parse_float_list("0:0.01:1"),
        "load_sampler" => "polytope",
        "load_dist_perc" => 0.2,
        "pf_min" => 0.7071,
        "pf_lagging" => true,
        "print_level" => 0,
        "output_dir" => joinpath(@__DIR__, "diagnostics"),
    )
end

function set_config!(cfg, key::AbstractString, value::AbstractString)
    key = replace(lowercase(strip(key)), "-" => "_")
    value = strip(value)
    if key in ["case", "case_name"]
        cfg["case_name"] = value
    elseif key == "branch_id"
        cfg["branch_id"] = value
    elseif key == "seed"
        cfg["seed"] = parse(Int, value)
    elseif key == "rho_attempts"
        cfg["rho_attempts"] = parse(Int, value)
    elseif key == "branch_attempts"
        cfg["branch_attempts"] = parse(Int, value)
    elseif key in ["train_rhos", "rhos"]
        cfg["train_rhos"] = parse_float_list(value)
    elseif key == "load_sampler"
        cfg["load_sampler"] = lowercase(value)
    elseif key == "load_dist_perc"
        cfg["load_dist_perc"] = parse(Float64, value)
    elseif key == "pf_min"
        cfg["pf_min"] = parse(Float64, value)
    elseif key == "pf_lagging"
        cfg["pf_lagging"] = lowercase(value) in ["1", "true", "yes", "y"]
    elseif key == "print_level"
        cfg["print_level"] = parse(Int, value)
    elseif key == "output_dir"
        cfg["output_dir"] = isabspath(value) ? value : joinpath(@__DIR__, value)
    else
        error("Unknown argument --$(key)")
    end
end

function parse_args(args)
    cfg = default_config()
    i = 1
    while i <= length(args)
        arg = args[i]
        @assert startswith(arg, "--") "Arguments must use --key=value or --key value"
        body = arg[3:end]
        eq_pos = findfirst(isequal('='), body)
        if isnothing(eq_pos)
            key = body
            i += 1
            @assert i <= length(args) "Missing value for --$(key)"
            value = args[i]
        else
            key = body[1:eq_pos-1]
            value = body[eq_pos+1:end]
        end
        set_config!(cfg, key, value)
        i += 1
    end
    return cfg
end

function load_case(case_name::AbstractString)
    case_name = String(case_name)
    @assert haskey(CASE_TO_PATH, case_name) "Unknown case $(case_name)"
    return PowerModels.parse_file(joinpath(@__DIR__, CASE_TO_PATH[case_name]))
end

function make_solver()
    return JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol" => OPFLearn.TOL)
end

function make_load_sampler(network::Dict, cfg::Dict, rng)
    net_info = OPFLearn.gather_net_info(network)
    base_load = vec(net_info["base_load"])
    if cfg["load_sampler"] == "uniform"
        return Dict{Symbol, Any}(
            :method => "uniform",
            :base_load => base_load,
            :rng => rng,
            :dist_perc => cfg["load_dist_perc"],
        )
    end

    A, b, x, _, _, _, _ = OPFLearn.initialize(
        network,
        cfg["pf_min"],
        cfg["pf_lagging"],
        nothing,
        nothing,
        OPFLearn.DEFAULT_INPUTS,
        OPFLearn.DEFAULT_OUTPUTS,
        OPFLearn.DEFAULT_DUALS,
        false,
        false,
        false,
        0,
        false,
        [],
        [],
        OPFLearn.sample_polytope_cprnd,
        Dict{Symbol, Any}(),
        network["name"],
        cfg["output_dir"],
        OPFLearn.PM.QCLSPowerModel,
        make_solver(),
        0,
        cfg["print_level"],
    )
    return Dict{Symbol, Any}(:method => "polytope", :A => A, :b => b, :x => x, :rng => rng)
end

function next_load_setting!(sampler::Dict)
    rng = sampler[:rng]
    if sampler[:method] == "uniform"
        base_load = sampler[:base_load]
        num_loads = Int(length(base_load) / 2)
        factors = 1 .+ sampler[:dist_perc] .* (2 .* rand(rng, num_loads) .- 1)
        pd = base_load[1:num_loads] .* factors
        qd = base_load[num_loads+1:end] .* factors
        return reshape(vcat(pd, qd), :, 1)
    end
    sampler[:x] = OPFLearn.sample_polytope_cprnd(
        sampler[:A],
        sampler[:b],
        sampler[:x],
        1;
        seed=rand(rng, 1:10^9),
    )
    return sampler[:x]
end

function select_branch(network::Dict, branch_id)
    loop_keys = OPFLearn.loop_branch_keys(network)
    @assert !isempty(loop_keys) "No loop branch found"
    selected = string(branch_id) == "auto" ? loop_keys[1] : string(branch_id)
    @assert selected in loop_keys "Branch $(selected) is not a loop branch"
    return selected, loop_keys
end

function solve_status(base_network, load_setting, branch_id, rho, cfg)
    network = deepcopy(base_network)
    OPFLearn.set_network_load!(network, load_setting, scale_load=false)
    OPFLearn.apply_branch_rho!(network, branch_id, rho)
    OPFLearn.change_bus_type!(network)
    _, feasible, raw_result = OPFLearn.run_ac_opf_pfdelta(
        network;
        print_level=cfg["print_level"],
        solver=make_solver(),
    )
    return feasible, string(get(raw_result, "termination_status", "unknown"))
end

function inc!(dict, key)
    dict[key] = get(dict, key, 0) + 1
end

function diagnose_current_branch(base_network, branch_id, cfg)
    rng = Random.MersenneTwister(cfg["seed"])
    sampler = make_load_sampler(base_network, cfg, rng)
    rhos = Float64.(cfg["train_rhos"])
    first_failed_rho_counts = Dict{String, Int}()
    status_counts = Dict{String, Int}()
    accepted = 0
    records = Vector{Dict{String, Any}}()

    for attempt in 1:cfg["rho_attempts"]
        load_setting = next_load_setting!(sampler)
        failed_rho = nothing
        failed_status = nothing
        solved_count = 0
        for rho in rhos
            feasible, status = solve_status(base_network, load_setting, branch_id, rho, cfg)
            if feasible
                solved_count += 1
            else
                failed_rho = rho
                failed_status = status
                break
            end
        end
        if isnothing(failed_rho)
            accepted += 1
            inc!(first_failed_rho_counts, "none")
        else
            inc!(first_failed_rho_counts, string(failed_rho))
            inc!(status_counts, failed_status)
        end
        push!(records, Dict(
            "attempt" => attempt,
            "accepted" => isnothing(failed_rho),
            "first_failed_rho" => isnothing(failed_rho) ? nothing : failed_rho,
            "termination_status" => failed_status,
            "solved_count_before_failure" => solved_count,
        ))
        println("rho-scan attempt $(attempt)/$(cfg["rho_attempts"]): accepted=$(accepted), failed_rho=$(failed_rho)")
        flush(stdout)
    end

    return Dict(
        "branch_id" => branch_id,
        "attempts" => cfg["rho_attempts"],
        "accepted" => accepted,
        "acceptance_rate" => accepted / cfg["rho_attempts"],
        "first_failed_rho_counts" => first_failed_rho_counts,
        "termination_status_counts" => status_counts,
        "records" => records,
    )
end

function diagnose_loop_branches(base_network, loop_keys, cfg)
    rng = Random.MersenneTwister(cfg["seed"] + 1000)
    sampler = make_load_sampler(base_network, cfg, rng)
    load_settings = [next_load_setting!(sampler) for _ in 1:cfg["branch_attempts"]]
    rows = Vector{Dict{String, Any}}()

    for (idx, branch_id) in enumerate(loop_keys)
        accepted = 0
        status_counts = Dict{String, Int}()
        branch = base_network["branch"][branch_id]
        for load_setting in load_settings
            feasible, status = solve_status(base_network, load_setting, branch_id, 0.0, cfg)
            accepted += feasible ? 1 : 0
            feasible || inc!(status_counts, status)
        end
        row = Dict(
            "branch_id" => branch_id,
            "f_bus" => branch["f_bus"],
            "t_bus" => branch["t_bus"],
            "attempts" => cfg["branch_attempts"],
            "accepted" => accepted,
            "acceptance_rate" => accepted / cfg["branch_attempts"],
            "failure_status_counts" => status_counts,
        )
        push!(rows, row)
        println("branch-scan $(idx)/$(length(loop_keys)): branch=$(branch_id), accepted=$(accepted)/$(cfg["branch_attempts"])")
        flush(stdout)
    end

    sort!(rows, by=x -> (-x["acceptance_rate"], parse(Int, x["branch_id"])))
    return rows
end

function write_json(path, data)
    open(path, "w") do io
        JSON.print(io, data, 2)
    end
end

function main(args)
    cfg = parse_args(args)
    mkpath(cfg["output_dir"])
    base_network = load_case(cfg["case_name"])
    selected_branch, loop_keys = select_branch(base_network, cfg["branch_id"])
    println("Selected branch: $(selected_branch)")
    println("Loop branch count: $(length(loop_keys))")

    current = diagnose_current_branch(base_network, selected_branch, cfg)
    branch_rows = diagnose_loop_branches(base_network, loop_keys, cfg)

    result = Dict(
        "config" => cfg,
        "selected_branch" => selected_branch,
        "loop_branches" => loop_keys,
        "current_branch_rho_scan" => current,
        "rho0_branch_scan" => branch_rows,
    )
    output_path = joinpath(cfg["output_dir"], "rho_failure_diagnostics.json")
    write_json(output_path, result)
    println("DONE")
    println("Output: $(output_path)")
end

main(ARGS)
