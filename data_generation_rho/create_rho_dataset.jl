using Pkg
Pkg.activate(@__DIR__)

using Dates
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

function parse_bool(value::AbstractString)
    return lowercase(strip(value)) in ["1", "true", "yes", "y"]
end

function parse_float_list(value::AbstractString)
    value = strip(value)
    if occursin(":", value)
        parts = split(value, ":")
        @assert length(parts) == 3 "Range format must be start:step:stop"
        start_value = parse(Float64, parts[1])
        step_value = parse(Float64, parts[2])
        stop_value = parse(Float64, parts[3])
        @assert step_value > 0.0 "rho range step must be positive"

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
        "task_name" => "rho_task_1_1_single_edge",
        "output_dir" => "",
        "branch_id" => "auto",
        "seed" => 1,
        "train_loads" => 100,
        "val_loads" => 20,
        "test_loads" => 20,
        "train_rhos" => parse_float_list("0:0.1:1"),
        "val_rhos" => [0.0],
        "test_rhos" => [0.0],
        "load_sampler" => "polytope",
        "load_dist_perc" => 0.2,
        "pf_min" => 0.7071,
        "pf_lagging" => true,
        "perturb_costs" => "none",
        "max_attempts_per_load" => 200,
        "print_level" => 0,
        "clean" => false,
    )
end

function set_config!(cfg, key::AbstractString, value::AbstractString)
    key = replace(lowercase(strip(key)), "-" => "_")
    value = strip(value)

    if key in ["case", "case_name"]
        cfg["case_name"] = value
    elseif key in ["task", "task_name"]
        cfg["task_name"] = value
    elseif key in ["output", "output_dir"]
        cfg["output_dir"] = value
    elseif key == "branch_id"
        cfg["branch_id"] = value
    elseif key == "seed"
        cfg["seed"] = parse(Int, value)
    elseif key == "train_loads"
        cfg["train_loads"] = parse(Int, value)
    elseif key == "val_loads"
        cfg["val_loads"] = parse(Int, value)
    elseif key == "test_loads"
        cfg["test_loads"] = parse(Int, value)
    elseif key == "train_rhos"
        cfg["train_rhos"] = parse_float_list(value)
    elseif key == "val_rhos"
        cfg["val_rhos"] = parse_float_list(value)
    elseif key == "test_rhos"
        cfg["test_rhos"] = parse_float_list(value)
    elseif key == "load_sampler"
        cfg["load_sampler"] = lowercase(value)
    elseif key == "load_dist_perc"
        cfg["load_dist_perc"] = parse(Float64, value)
    elseif key == "pf_min"
        cfg["pf_min"] = parse(Float64, value)
    elseif key == "pf_lagging"
        cfg["pf_lagging"] = parse_bool(value)
    elseif key == "perturb_costs"
        cfg["perturb_costs"] = lowercase(value)
    elseif key == "max_attempts_per_load"
        cfg["max_attempts_per_load"] = parse(Int, value)
    elseif key == "print_level"
        cfg["print_level"] = parse(Int, value)
    elseif key == "clean"
        cfg["clean"] = parse_bool(value)
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
            if key == "clean"
                value = "true"
            else
                i += 1
                @assert i <= length(args) "Missing value for --$(key)"
                value = args[i]
            end
        else
            key = body[1:eq_pos-1]
            value = body[eq_pos+1:end]
        end
        set_config!(cfg, key, value)
        i += 1
    end

    if cfg["output_dir"] == ""
        cfg["output_dir"] = joinpath(@__DIR__, cfg["case_name"], cfg["task_name"])
    elseif !isabspath(cfg["output_dir"])
        cfg["output_dir"] = joinpath(@__DIR__, cfg["output_dir"])
    end

    for rho in vcat(cfg["train_rhos"], cfg["val_rhos"], cfg["test_rhos"])
        @assert 0.0 <= rho <= 1.0 "rho values must be in [0, 1]"
    end

    @assert cfg["load_sampler"] in ["polytope", "uniform"] "load_sampler must be polytope or uniform"
    @assert cfg["perturb_costs"] in ["none", "shuffle"] "perturb_costs must be none or shuffle"
    return cfg
end

function load_case(case_name::String)
    @assert haskey(CASE_TO_PATH, case_name) "Unknown case $(case_name)"
    return PowerModels.parse_file(joinpath(@__DIR__, CASE_TO_PATH[case_name]))
end

function select_branch(network::Dict, branch_id)
    loop_keys = OPFLearn.loop_branch_keys(network)
    @assert !isempty(loop_keys) "No loop branch found. Cannot build a rho line-outage dataset."

    if string(branch_id) == "auto"
        selected = loop_keys[1]
    else
        selected = string(branch_id)
        candidate_text = join(loop_keys, ", ")
        @assert selected in loop_keys "Branch $(selected) is not a loop branch. Choose from $(candidate_text)."
    end

    branch = network["branch"][selected]
    return selected, loop_keys, Dict(
        "branch_id" => selected,
        "index" => branch["index"],
        "f_bus" => branch["f_bus"],
        "t_bus" => branch["t_bus"],
        "br_r" => branch["br_r"],
        "br_x" => branch["br_x"],
        "g_fr" => branch["g_fr"],
        "b_fr" => branch["b_fr"],
        "g_to" => branch["g_to"],
        "b_to" => branch["b_to"],
    )
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

    return Dict{Symbol, Any}(
        :method => "polytope",
        :A => A,
        :b => b,
        :x => x,
        :base_load => base_load,
        :rng => rng,
    )
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

function load_vector(load_setting)
    return [float(load_setting[i, 1]) for i in 1:size(load_setting, 1)]
end

function write_json(path::String, data)
    open(path, "w") do io
        JSON.print(io, data)
    end
end

function solve_rho_candidate(base_network::Dict, load_setting, branch_id::String, rho::Float64, cfg::Dict)
    network = deepcopy(base_network)
    OPFLearn.set_network_load!(network, load_setting, scale_load=false)
    OPFLearn.perturb_costs!(network; method=cfg["perturb_costs"])
    rho_meta = OPFLearn.apply_branch_rho!(network, branch_id, rho)
    OPFLearn.change_bus_type!(network)

    result, feasible, raw_result = OPFLearn.run_ac_opf_pfdelta(
        network;
        print_level=cfg["print_level"],
        solver=make_solver(),
    )
    return feasible, network, raw_result, result, rho_meta
end

function generate_split!(
    split_name::String,
    num_load_settings::Int,
    rhos::Vector{Float64},
    base_network::Dict,
    branch_id::String,
    cfg::Dict,
    sampler::Dict,
    raw_path::String,
    counters::Dict,
)
    records = Vector{Dict{String, Any}}()
    load_records = Vector{Dict{String, Any}}()
    failures = Vector{Dict{String, Any}}()
    max_attempts = max(num_load_settings * cfg["max_attempts_per_load"], num_load_settings)
    attempts = 0

    while length(load_records) < num_load_settings && attempts < max_attempts
        attempts += 1
        load_setting = next_load_setting!(sampler)
        load_setting_id = counters["load_setting_id"]
        candidate_samples = Vector{Dict{String, Any}}()
        failed_rho = nothing
        failed_status = nothing

        for (rho_index, rho) in enumerate(rhos)
            feasible, network, raw_result, _, rho_meta = solve_rho_candidate(
                base_network,
                load_setting,
                branch_id,
                rho,
                cfg,
            )

            if feasible
                rho_meta["split"] = split_name
                rho_meta["task_name"] = cfg["task_name"]
                rho_meta["case_name"] = cfg["case_name"]
                rho_meta["load_setting_id"] = load_setting_id
                rho_meta["rho_index"] = rho_index
                rho_meta["load_sampler"] = sampler[:method]
                push!(candidate_samples, Dict{String, Any}(
                    "network" => network,
                    "result" => raw_result,
                    "rho_meta" => rho_meta,
                ))
            else
                failed_rho = rho
                failed_status = string(get(raw_result, "termination_status", "unknown"))
                break
            end
        end

        if isnothing(failed_rho)
            sample_ids = Int[]
            for sample in candidate_samples
                sample_id = counters["sample_id"]
                sample["rho_meta"]["sample_id"] = sample_id
                OPFLearn.store_feasible_sample_json(
                    sample_id,
                    sample["network"],
                    sample["result"],
                    raw_path;
                    rho_meta=sample["rho_meta"],
                )

                push!(sample_ids, sample_id)
                push!(records, Dict{String, Any}(
                    "sample_id" => sample_id,
                    "file" => "raw/sample_$(sample_id).json",
                    "split" => split_name,
                    "load_setting_id" => load_setting_id,
                    "rho" => sample["rho_meta"]["rho"],
                    "rho_index" => sample["rho_meta"]["rho_index"],
                    "branch_id" => sample["rho_meta"]["branch_id"],
                    "br_status" => sample["rho_meta"]["br_status"],
                ))
                counters["sample_id"] += 1
            end

            push!(load_records, Dict{String, Any}(
                "load_setting_id" => load_setting_id,
                "split" => split_name,
                "sample_ids" => sample_ids,
                "rhos" => rhos,
                "load_vector" => load_vector(load_setting),
            ))
            counters["load_setting_id"] += 1

            println("$(split_name): accepted load $(length(load_records)) / $(num_load_settings), samples=$(length(records))")
            flush(stdout)
        else
            push!(failures, Dict{String, Any}(
                "split" => split_name,
                "attempt" => attempts,
                "load_setting_id" => load_setting_id,
                "failed_rho" => failed_rho,
                "termination_status" => failed_status,
            ))
        end
    end

    @assert length(load_records) == num_load_settings "Only accepted $(length(load_records)) / $(num_load_settings) load settings for $(split_name). Increase max_attempts_per_load or change the selected branch."

    return Dict{String, Any}(
        "split" => split_name,
        "records" => records,
        "load_settings" => load_records,
        "failures" => failures,
        "attempts" => attempts,
    )
end

function prepare_output_dir(cfg::Dict)
    output_dir = cfg["output_dir"]
    if isdir(output_dir)
        if cfg["clean"]
            rm(output_dir; recursive=true)
        else
            error("Output directory already exists: $(output_dir). Pass --clean to replace it.")
        end
    end

    mkpath(joinpath(output_dir, "raw"))
    mkpath(joinpath(output_dir, "splits"))
end

function main(args)
    cfg = parse_args(args)
    prepare_output_dir(cfg)

    output_dir = cfg["output_dir"]
    raw_path = joinpath(output_dir, "raw")
    splits_path = joinpath(output_dir, "splits")

    Random.seed!(cfg["seed"])
    rng = Random.MersenneTwister(cfg["seed"])

    base_network = load_case(cfg["case_name"])
    selected_branch, loop_keys, selected_branch_meta = select_branch(base_network, cfg["branch_id"])
    from_bus = selected_branch_meta["f_bus"]
    to_bus = selected_branch_meta["t_bus"]
    println("Selected rho branch: $(selected_branch), bus $(from_bus) -> $(to_bus)")

    sampler = make_load_sampler(base_network, cfg, rng)
    counters = Dict{String, Any}(
        "sample_id" => 1,
        "load_setting_id" => 1,
    )

    meta = Dict{String, Any}(
        "status" => "running",
        "created_at" => string(Dates.now()),
        "config" => cfg,
        "selected_branch" => selected_branch_meta,
        "loop_branch_candidates" => loop_keys,
        "rho_definition" => "For the selected branch, series admittance is scaled by rho. For rho > 0, br_r and br_x are divided by rho, line charging terms are multiplied by rho, and br_status remains 1. For rho = 0, br_status is set to 0.",
        "splits" => Dict{String, Any}(),
    )
    write_json(joinpath(output_dir, "meta.json"), meta)

    split_specs = [
        ("train", cfg["train_loads"], cfg["train_rhos"]),
        ("val", cfg["val_loads"], cfg["val_rhos"]),
        ("test", cfg["test_loads"], cfg["test_rhos"]),
    ]

    for (split_name, num_loads, rhos) in split_specs
        split_result = generate_split!(
            split_name,
            num_loads,
            Float64.(rhos),
            base_network,
            selected_branch,
            cfg,
            sampler,
            raw_path,
            counters,
        )
        write_json(joinpath(splits_path, "$(split_name).json"), split_result)
        meta["splits"][split_name] = Dict(
            "load_settings" => length(split_result["load_settings"]),
            "samples" => length(split_result["records"]),
            "rhos" => rhos,
            "attempts" => split_result["attempts"],
            "failures" => length(split_result["failures"]),
        )
        write_json(joinpath(output_dir, "meta.json"), meta)
    end

    meta["status"] = "finished"
    meta["finished_at"] = string(Dates.now())
    meta["total_samples"] = counters["sample_id"] - 1
    meta["total_load_settings"] = counters["load_setting_id"] - 1
    write_json(joinpath(output_dir, "meta.json"), meta)

    println("DONE")
    println("Output: $(output_dir)")
    total_samples = meta["total_samples"]
    println("Total samples: $(total_samples)")
end

main(ARGS)
