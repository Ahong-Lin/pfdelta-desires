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

function parse_string_list(value::AbstractString)
    value = strip(value)
    return [strip(x) for x in split(value, ",") if strip(x) != ""]
end

function default_config()
    return Dict{String, Any}(
        "case_name" => "case30",
        "task_name" => "rho_bank_normal_loop_edges",
        "output_dir" => "",
        "branch_mode" => "normal-loop",
        "branch_ids" => String[],
        "seed" => 1,
        "train_loads" => 30,
        "val_loads" => 5,
        "test_loads" => 5,
        "rhos" => parse_float_list("0:0.1:1"),
        "load_sampler" => "polytope",
        "load_dist_perc" => 0.2,
        "pf_min" => 0.7071,
        "pf_lagging" => true,
        "perturb_costs" => "none",
        "print_level" => 0,
        "clean" => false,
        "append" => false,
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
    elseif key == "branch_mode"
        cfg["branch_mode"] = lowercase(value)
    elseif key == "branch_ids"
        cfg["branch_ids"] = parse_string_list(value)
    elseif key == "seed"
        cfg["seed"] = parse(Int, value)
    elseif key == "train_loads"
        cfg["train_loads"] = parse(Int, value)
    elseif key == "val_loads"
        cfg["val_loads"] = parse(Int, value)
    elseif key == "test_loads"
        cfg["test_loads"] = parse(Int, value)
    elseif key in ["rhos", "rho_grid"]
        cfg["rhos"] = parse_float_list(value)
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
    elseif key == "print_level"
        cfg["print_level"] = parse(Int, value)
    elseif key == "clean"
        cfg["clean"] = parse_bool(value)
    elseif key == "append"
        cfg["append"] = parse_bool(value)
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
            if key in ["clean", "append"]
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

    for rho in cfg["rhos"]
        @assert 0.0 <= rho <= 1.0 "rho values must be in [0, 1]"
    end

    @assert cfg["load_sampler"] in ["polytope", "uniform"] "load_sampler must be polytope or uniform"
    @assert cfg["perturb_costs"] in ["none", "shuffle"] "perturb_costs must be none or shuffle"
    @assert cfg["branch_mode"] in ["normal-loop", "all-loop", "explicit"] "branch_mode must be normal-loop, all-loop, or explicit"
    return cfg
end

function load_case(case_name::AbstractString)
    case_name = String(case_name)
    @assert haskey(CASE_TO_PATH, case_name) "Unknown case $(case_name)"
    return PowerModels.parse_file(joinpath(@__DIR__, CASE_TO_PATH[case_name]))
end

function is_transformer_branch(branch)
    tap = get(branch, "tap", 1.0)
    shift = get(branch, "shift", 0.0)
    return abs(tap - 1.0) > 1e-9 || abs(shift) > 1e-9
end

function select_branch_keys(network::Dict, cfg::Dict)
    loop_keys = OPFLearn.loop_branch_keys(network)
    if cfg["branch_mode"] == "explicit"
        branch_keys = string.(cfg["branch_ids"])
        candidate_text = join(loop_keys, ", ")
        for key in branch_keys
            @assert key in loop_keys "Branch $(key) is not a loop branch. Choose from $(candidate_text)."
        end
    elseif cfg["branch_mode"] == "all-loop"
        branch_keys = loop_keys
    else
        branch_keys = [key for key in loop_keys if !is_transformer_branch(network["branch"][key])]
    end
    return branch_keys, loop_keys
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
        JSON.print(io, data, 2)
    end
end

function append_jsonl(path::String, data)
    open(path, "a") do io
        JSON.print(io, data)
        write(io, "\n")
    end
end

function prepare_output_dir(cfg)
    output_dir = cfg["output_dir"]
    if cfg["clean"] && isdir(output_dir)
        rm(output_dir; recursive=true)
    end
    mkpath(joinpath(output_dir, "raw"))
    mkpath(joinpath(output_dir, "manifests"))
    mkpath(joinpath(output_dir, "splits"))
end

function load_existing_counters(output_dir::String)
    samples_path = joinpath(output_dir, "manifests", "samples.jsonl")
    loads_path = joinpath(output_dir, "manifests", "load_settings.jsonl")
    sample_id = 1
    load_setting_id = 1

    if isfile(samples_path)
        for line in eachline(samples_path)
            isempty(strip(line)) && continue
            rec = JSON.parse(line)
            sample_id = max(sample_id, Int(rec["sample_id"]) + 1)
        end
    end

    if isfile(loads_path)
        for line in eachline(loads_path)
            isempty(strip(line)) && continue
            rec = JSON.parse(line)
            load_setting_id = max(load_setting_id, Int(rec["load_setting_id"]) + 1)
        end
    end

    return Dict{String, Any}(
        "sample_id" => sample_id,
        "load_setting_id" => load_setting_id,
    )
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

function generation_plan(cfg)
    return [
        ("train", cfg["train_loads"]),
        ("val", cfg["val_loads"]),
        ("test", cfg["test_loads"]),
    ]
end

function existing_split_records(output_dir::String, split_name::String)
    path = joinpath(output_dir, "splits", "$(split_name).json")
    if isfile(path)
        data = JSON.parsefile(path)
        return get(data, "records", Any[]), get(data, "load_settings", Any[]), get(data, "failures", Any[])
    end
    return Any[], Any[], Any[]
end

function generate_split!(
    split_name::String,
    num_load_settings::Int,
    branch_keys::Vector,
    rhos::Vector{Float64},
    base_network::Dict,
    cfg::Dict,
    sampler::Dict,
    counters::Dict,
)
    output_dir = cfg["output_dir"]
    raw_path = joinpath(output_dir, "raw")
    samples_manifest = joinpath(output_dir, "manifests", "samples.jsonl")
    failures_manifest = joinpath(output_dir, "manifests", "failures.jsonl")
    loads_manifest = joinpath(output_dir, "manifests", "load_settings.jsonl")

    records, load_records, failures = existing_split_records(output_dir, split_name)

    for local_load_idx in 1:num_load_settings
        load_setting = next_load_setting!(sampler)
        load_setting_id = counters["load_setting_id"]
        load_rec = Dict{String, Any}(
            "load_setting_id" => load_setting_id,
            "split" => split_name,
            "case_name" => cfg["case_name"],
            "sampler" => sampler[:method],
            "seed" => cfg["seed"],
            "load_vector" => load_vector(load_setting),
            "created_at" => string(Dates.now()),
        )
        append_jsonl(loads_manifest, load_rec)

        accepted_for_load = 0
        failed_for_load = 0
        sample_ids = Int[]

        for branch_id in branch_keys
            branch = base_network["branch"][string(branch_id)]
            for (rho_index, rho) in enumerate(rhos)
                feasible, network, raw_result, _, rho_meta = solve_rho_candidate(
                    base_network,
                    load_setting,
                    string(branch_id),
                    rho,
                    cfg,
                )
                rho_meta["split"] = split_name
                rho_meta["task_name"] = cfg["task_name"]
                rho_meta["case_name"] = cfg["case_name"]
                rho_meta["load_setting_id"] = load_setting_id
                rho_meta["rho_index"] = rho_index
                rho_meta["load_sampler"] = sampler[:method]
                rho_meta["branch_mode"] = cfg["branch_mode"]

                if feasible
                    sample_id = counters["sample_id"]
                    rho_meta["sample_id"] = sample_id
                    OPFLearn.store_feasible_sample_json(
                        sample_id,
                        network,
                        raw_result,
                        raw_path;
                        rho_meta=rho_meta,
                    )
                    file_name = "raw/sample_$(sample_id).json"
                    status = string(get(raw_result, "termination_status", "unknown"))
                    objective = get(raw_result, "objective", nothing)
                    rec = Dict{String, Any}(
                        "sample_id" => sample_id,
                        "file" => file_name,
                        "split" => split_name,
                        "case_name" => cfg["case_name"],
                        "load_setting_id" => load_setting_id,
                        "branch_id" => string(branch_id),
                        "f_bus" => branch["f_bus"],
                        "t_bus" => branch["t_bus"],
                        "rho" => rho,
                        "rho_index" => rho_index,
                        "rho_grid" => join(string.(rhos), ","),
                        "br_status" => rho_meta["br_status"],
                        "termination_status" => status,
                        "objective" => objective,
                        "created_at" => string(Dates.now()),
                    )
                    append_jsonl(samples_manifest, rec)
                    push!(records, rec)
                    push!(sample_ids, sample_id)
                    counters["sample_id"] += 1
                    accepted_for_load += 1
                else
                    status = string(get(raw_result, "termination_status", "unknown"))
                    fail = Dict{String, Any}(
                        "split" => split_name,
                        "case_name" => cfg["case_name"],
                        "load_setting_id" => load_setting_id,
                        "branch_id" => string(branch_id),
                        "f_bus" => branch["f_bus"],
                        "t_bus" => branch["t_bus"],
                        "rho" => rho,
                        "rho_index" => rho_index,
                        "rho_grid" => join(string.(rhos), ","),
                        "termination_status" => status,
                        "reason" => "opf_failed",
                        "created_at" => string(Dates.now()),
                    )
                    append_jsonl(failures_manifest, fail)
                    push!(failures, fail)
                    failed_for_load += 1
                end
            end
        end

        push!(load_records, Dict{String, Any}(
            "load_setting_id" => load_setting_id,
            "split" => split_name,
            "sample_ids" => sample_ids,
            "num_success" => accepted_for_load,
            "num_failed" => failed_for_load,
            "load_vector" => load_vector(load_setting),
        ))

        counters["load_setting_id"] += 1
        total_for_load = accepted_for_load + failed_for_load
        println("$(split_name): load $(local_load_idx)/$(num_load_settings), load_setting_id=$(load_setting_id), success=$(accepted_for_load)/$(total_for_load), total_success_records=$(length(records))")
        flush(stdout)

        write_json(joinpath(output_dir, "splits", "$(split_name).json"), Dict(
            "split" => split_name,
            "records" => records,
            "load_settings" => load_records,
            "failures" => failures,
        ))
    end

    return Dict(
        "load_settings" => length(load_records),
        "samples" => length(records),
        "failures" => length(failures),
    )
end

function main(args)
    cfg = parse_args(args)
    @assert !(cfg["clean"] && cfg["append"]) "Use either --clean or --append, not both"
    prepare_output_dir(cfg)

    output_dir = cfg["output_dir"]
    Random.seed!(cfg["seed"])
    rng = Random.MersenneTwister(cfg["seed"])

    base_network = load_case(cfg["case_name"])
    branch_keys, loop_keys = select_branch_keys(base_network, cfg)
    sampler = make_load_sampler(base_network, cfg, rng)
    counters = cfg["append"] ? load_existing_counters(output_dir) : Dict{String, Any}("sample_id" => 1, "load_setting_id" => 1)

    meta_path = joinpath(output_dir, "meta.json")
    meta = Dict{String, Any}(
        "status" => "running",
        "created_at" => string(Dates.now()),
        "config" => cfg,
        "num_total_branches" => length(base_network["branch"]),
        "loop_branch_candidates" => loop_keys,
        "selected_branch_ids" => branch_keys,
        "num_selected_branches" => length(branch_keys),
        "rho_definition" => "Each sample controls exactly one selected branch. All branches carry an edge-level rho field. Uncontrolled branches have rho=1.0. For the controlled branch, rho scales series admittance; rho=0 sets br_status=0.",
        "splits" => Dict{String, Any}(),
    )
    write_json(meta_path, meta)

    rhos = cfg["rhos"]
    println("Output: $(output_dir)")
    println("Selected branches ($(length(branch_keys))): $(join(branch_keys, ","))")
    println("Rhos ($(length(rhos))): $(join(string.(rhos), ","))")

    for (split_name, num_loads) in generation_plan(cfg)
        result = generate_split!(
            split_name,
            num_loads,
            branch_keys,
            Float64.(cfg["rhos"]),
            base_network,
            cfg,
            sampler,
            counters,
        )
        meta["splits"][split_name] = result
        write_json(meta_path, meta)
    end

    meta["status"] = "finished"
    meta["finished_at"] = string(Dates.now())
    meta["next_sample_id"] = counters["sample_id"]
    meta["next_load_setting_id"] = counters["load_setting_id"]
    write_json(meta_path, meta)

    println("DONE")
    println("Output: $(output_dir)")
end

main(ARGS)
