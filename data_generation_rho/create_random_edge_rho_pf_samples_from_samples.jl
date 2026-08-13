using Pkg
Pkg.activate(@__DIR__)

using Dates
using JSON
using PowerModels
using Random

include(joinpath(@__DIR__, "src", "OPFLearn.jl"))

function parse_bool(value::AbstractString)
    return lowercase(strip(value)) in ["1", "true", "yes", "y"]
end

function parse_float_list(value::AbstractString)
    value = strip(value)
    if occursin(":", value)
        parts = split(value, ":")
        @assert length(parts) == 3 "Range format must be start:step:stop"
        start_value = parse(Float64, parts[1])
        step = parse(Float64, parts[2])
        stop_value = parse(Float64, parts[3])
        values = Float64[]
        current = start_value
        while current <= stop_value + step / 2
            push!(values, round(current; digits=10))
            current += step
        end
        return values
    end
    return [parse(Float64, strip(x)) for x in split(value, ",") if strip(x) != ""]
end

function parse_string_list(value::AbstractString)
    return [strip(x) for x in split(strip(value), ",") if strip(x) != ""]
end

function repo_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function default_config()
    return Dict{String, Any}(
        "case_name" => "case30",
        "source_dir" => joinpath(repo_root(), "data", "pfdelta_data", "case30", "n", "raw"),
        "output_dir" => joinpath(@__DIR__, "case30", "rho_pf_random_edge_random_rho_extra_from10001"),
        "task_name" => "rho_pf_random_edge_random_rho_extra_from10001",
        "start_index" => 10001,
        "target_success" => 5000,
        "max_loads" => 8000,
        "rhos" => parse_float_list("0:0.1:1"),
        "rho_mode" => "grid",
        "load_id_offset" => 0,
        "branch_mode" => "normal-loop",
        "branch_ids" => String[],
        "split" => "train",
        "seed" => 11,
        "max_attempts_per_load" => 20,
        "exclude_roots" => String[],
        "exclude_splits" => ["test"],
        "shuffle_sources" => false,
        "clean" => false,
        "print_every" => 100,
        "flat_start" => false,
    )
end

function set_config!(cfg, key::AbstractString, value::AbstractString)
    key = replace(lowercase(strip(key)), "-" => "_")
    value = strip(value)
    if key in ["case", "case_name"]
        cfg["case_name"] = value
    elseif key == "source_dir"
        cfg["source_dir"] = abspath(value)
    elseif key in ["output", "output_dir"]
        cfg["output_dir"] = abspath(value)
    elseif key in ["task", "task_name"]
        cfg["task_name"] = value
    elseif key == "start_index"
        cfg["start_index"] = parse(Int, value)
    elseif key == "target_success"
        cfg["target_success"] = parse(Int, value)
    elseif key == "max_loads"
        cfg["max_loads"] = parse(Int, value)
    elseif key in ["rhos", "rho_grid"]
        cfg["rhos"] = parse_float_list(value)
    elseif key == "rho_mode"
        cfg["rho_mode"] = lowercase(value)
    elseif key == "load_id_offset"
        cfg["load_id_offset"] = parse(Int, value)
    elseif key == "branch_mode"
        cfg["branch_mode"] = lowercase(value)
    elseif key == "branch_ids"
        cfg["branch_ids"] = parse_string_list(value)
    elseif key == "split"
        cfg["split"] = value
    elseif key == "seed"
        cfg["seed"] = parse(Int, value)
    elseif key == "max_attempts_per_load"
        cfg["max_attempts_per_load"] = parse(Int, value)
    elseif key == "exclude_roots"
        cfg["exclude_roots"] = parse_string_list(value)
    elseif key == "exclude_splits"
        cfg["exclude_splits"] = parse_string_list(value)
    elseif key in ["shuffle_sources", "shuffle_source"]
        cfg["shuffle_sources"] = parse_bool(value)
    elseif key == "clean"
        cfg["clean"] = parse_bool(value)
    elseif key == "print_every"
        cfg["print_every"] = parse(Int, value)
    elseif key == "flat_start"
        cfg["flat_start"] = parse_bool(value)
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
            if key in ["clean", "flat-start", "flat_start"]
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
    @assert cfg["target_success"] > 0
    @assert cfg["max_loads"] > 0
    @assert cfg["max_attempts_per_load"] > 0
    @assert cfg["rho_mode"] in ["grid", "uniform-open"] "rho-mode must be grid or uniform-open"
    if cfg["rho_mode"] == "grid"
        @assert !isempty(cfg["rhos"]) "grid rho-mode needs at least one rho value"
        for rho in cfg["rhos"]
            @assert 0.0 <= rho <= 1.0
        end
    end
    return cfg
end

function sample_file_id(path::AbstractString)
    m = match(r"sample_(\d+)\.json$", basename(path))
    isnothing(m) && return typemax(Int)
    return parse(Int, m.captures[1])
end

function source_files(cfg)
    files = filter(isfile, readdir(cfg["source_dir"]; join=true))
    files = filter(path -> occursin(r"sample_\d+\.json$", basename(path)), files)
    sort!(files, by=sample_file_id)
    start_index = cfg["start_index"]
    stop_index = min(length(files), start_index + cfg["max_loads"] - 1)
    @assert start_index <= length(files) "start-index exceeds available source samples"
    return files[start_index:stop_index]
end

function branch_sort_key(net, key)
    return get(net["branch"][key], "index", parse(Int, string(key)))
end

function is_transformer_like(branch)
    tap = Float64(get(branch, "tap", 1.0))
    shift = Float64(get(branch, "shift", 0.0))
    return abs(tap - 1.0) > 1e-12 || abs(shift) > 1e-12
end

function selected_branch_keys(net, cfg)
    if !isempty(cfg["branch_ids"])
        return sort(string.(cfg["branch_ids"]), by=k -> branch_sort_key(net, k))
    end
    loop_keys = OPFLearn.loop_branch_keys(net)
    if cfg["branch_mode"] == "normal-loop"
        return [k for k in loop_keys if !is_transformer_like(net["branch"][k])]
    elseif cfg["branch_mode"] == "loop"
        return loop_keys
    elseif cfg["branch_mode"] == "all-active"
        keys = [k for k in keys(net["branch"]) if get(net["branch"][k], "br_status", 1) == 1]
        return sort(keys, by=k -> branch_sort_key(net, k))
    else
        error("Unknown branch-mode $(cfg["branch_mode"])")
    end
end

function seed_pf_controls_from_solution!(net::Dict, solution::Dict)
    bus_solution = solution["bus"]
    gen_solution = solution["gen"]
    branch_solution = get(solution, "branch", Dict())
    for (bus_id, bus) in net["bus"]
        if haskey(bus_solution, bus_id)
            sol = bus_solution[bus_id]
            bus["vm"] = sol["vm"]
            bus["va"] = sol["va"]
            bus["vm_start"] = sol["vm"]
            bus["va_start"] = sol["va"]
        end
    end
    for (gen_id, gen) in net["gen"]
        if get(gen, "gen_status", 1) == 1 && haskey(gen_solution, gen_id)
            sol = gen_solution[gen_id]
            gen["pg"] = sol["pg"]
            gen["qg"] = sol["qg"]
            gen["pg_start"] = sol["pg"]
            gen["qg_start"] = sol["qg"]
            gen_bus = string(gen["gen_bus"])
            if haskey(bus_solution, gen_bus)
                gen["vg"] = bus_solution[gen_bus]["vm"]
            end
        end
    end
    for (branch_id, branch) in net["branch"]
        if haskey(branch_solution, branch_id)
            sol = branch_solution[branch_id]
            branch["pf_start"] = sol["pf"]
            branch["qf_start"] = sol["qf"]
            branch["pt_start"] = sol["pt"]
            branch["qt_start"] = sol["qt"]
        end
    end
end

function pf_converged(result)
    status = get(result, "termination_status", false)
    if status === true
        return true
    end
    status_str = uppercase(string(status))
    return status_str in ["TRUE", "LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED", "ALMOST_OPTIMAL"]
end

function status_string(result)
    return string(get(result, "termination_status", "unknown"))
end

function solve_pf_rho(base_sample::Dict, branch_id::String, rho::Float64, cfg::Dict)
    net = deepcopy(base_sample["network"])
    solution = base_sample["solution"]["solution"]
    seed_pf_controls_from_solution!(net, solution)
    rho_meta = OPFLearn.apply_branch_rho!(net, branch_id, rho)
    try
        result = PowerModels.compute_ac_pf(net; flat_start=cfg["flat_start"])
        converged = pf_converged(result)
        result["solver_type"] = "AC_PF"
        result["pf_termination_status"] = get(result, "termination_status", nothing)
        result["termination_status"] = converged ? "PF_CONVERGED" : status_string(result)
        return converged, net, result, rho_meta
    catch err
        result = Dict{String, Any}(
            "solver_type" => "AC_PF",
            "termination_status" => "EXCEPTION",
            "error" => sprint(showerror, err),
        )
        return false, net, result, rho_meta
    end
end

function write_json(path::String, payload)
    open(path, "w") do io
        write(io, JSON.json(payload))
    end
end

function write_jsonl(io, payload)
    write(io, JSON.json(payload))
    write(io, "\n")
end

function load_vector_from_sample(base_sample)
    loads = base_sample["network"]["load"]
    keys_sorted = sort(collect(keys(loads)), by=k -> parse(Int, string(k)))
    pd = [loads[k]["pd"] for k in keys_sorted]
    qd = [loads[k]["qd"] for k in keys_sorted]
    return vcat(pd, qd)
end

function sample_open_unit_rho(rng)
    rho = rand(rng)
    while rho <= 0.0 || rho >= 1.0
        rho = rand(rng)
    end
    return rho
end

function sample_branch_rho(rng, branch_keys, rhos, cfg)
    branch_id = rand(rng, branch_keys)
    if cfg["rho_mode"] == "uniform-open"
        return branch_id, sample_open_unit_rho(rng)
    end
    return branch_id, rand(rng, rhos)
end

function rho_source_label(rhos, cfg)
    if cfg["rho_mode"] == "uniform-open"
        return "uniform-open(0,1)"
    end
    return join(string.(rhos), ",")
end

function collect_ids_from_records!(excluded::Set{Int}, records)
    for rec in records
        for key in ["load_setting_id", "source_load_setting_id", "base_sample_id"]
            if haskey(rec, key)
                try
                    push!(excluded, Int(rec[key]))
                catch
                end
            end
        end
    end
end

function excluded_load_ids(cfg)
    excluded = Set{Int}()
    for root in cfg["exclude_roots"]
        for split in cfg["exclude_splits"]
            split_path = joinpath(root, "splits", "$(split).json")
            if !isfile(split_path)
                continue
            end
            payload = JSON.parsefile(split_path)
            collect_ids_from_records!(excluded, get(payload, "records", Any[]))
            collect_ids_from_records!(excluded, get(payload, "load_settings", Any[]))
        end
    end
    return excluded
end

function main()
    cfg = parse_args(ARGS)
    output_dir = cfg["output_dir"]
    if cfg["clean"] && isdir(output_dir)
        rm(output_dir; recursive=true, force=true)
    elseif isdir(output_dir)
        error("Output directory already exists. Use --clean to overwrite: $(output_dir)")
    end
    mkpath(joinpath(output_dir, "raw"))
    mkpath(joinpath(output_dir, "manifests"))
    mkpath(joinpath(output_dir, "splits"))
    mkpath(joinpath(output_dir, "analysis"))

    files = source_files(cfg)
    first_sample = JSON.parsefile(files[1])
    branch_keys = selected_branch_keys(first_sample["network"], cfg)
    rhos = Float64.(cfg["rhos"])
    rng = MersenneTwister(cfg["seed"])
    excluded = excluded_load_ids(cfg)
    if cfg["shuffle_sources"]
        files = shuffle(rng, files)
    end

    samples_path = joinpath(output_dir, "manifests", "samples.jsonl")
    failures_path = joinpath(output_dir, "manifests", "failures.jsonl")
    loads_path = joinpath(output_dir, "manifests", "load_settings.jsonl")
    sample_id = 1
    success_loads = 0
    attempted_loads = 0
    records = Any[]
    load_records = Any[]

    meta = Dict{String, Any}(
        "created_at" => string(Dates.now()),
        "case_name" => cfg["case_name"],
        "task_name" => cfg["task_name"],
        "source_dir" => cfg["source_dir"],
        "output_dir" => output_dir,
        "start_index" => cfg["start_index"],
        "target_success" => cfg["target_success"],
        "max_loads" => cfg["max_loads"],
        "branch_mode" => cfg["branch_mode"],
        "branch_ids" => branch_keys,
        "rhos" => rhos,
        "rho_mode" => cfg["rho_mode"],
        "load_id_offset" => cfg["load_id_offset"],
        "seed" => cfg["seed"],
        "shuffle_sources" => cfg["shuffle_sources"],
        "exclude_roots" => cfg["exclude_roots"],
        "exclude_splits" => cfg["exclude_splits"],
        "excluded_load_settings" => length(excluded),
        "selection" => "For each source load_setting, randomly try one branch/rho pair at a time until the first successful AC-PF sample is found.",
    )
    write_json(joinpath(output_dir, "meta.json"), meta)

    open(samples_path, "w") do samples_io
    open(failures_path, "w") do failures_io
    open(loads_path, "w") do loads_io
        for path in files
            success_loads >= cfg["target_success"] && break
            attempted_loads += 1
            base_sample = JSON.parsefile(path)
            base_sample_id = sample_file_id(path)
            load_setting_id = base_sample_id + cfg["load_id_offset"]
            if load_setting_id in excluded
                continue
            end
            tried = Set{Tuple{String, Float64}}()
            grid_space = cfg["rho_mode"] == "grid" ? length(branch_keys) * length(rhos) : cfg["max_attempts_per_load"]
            max_attempts = min(cfg["max_attempts_per_load"], grid_space)
            accepted = false

            for attempt in 1:max_attempts
                branch_id, rho = sample_branch_rho(rng, branch_keys, rhos, cfg)
                pair = (branch_id, rho)
                while cfg["rho_mode"] == "grid" && pair in tried && length(tried) < length(branch_keys) * length(rhos)
                    branch_id, rho = sample_branch_rho(rng, branch_keys, rhos, cfg)
                    pair = (branch_id, rho)
                end
                push!(tried, pair)

                converged, net, result, rho_meta = solve_pf_rho(base_sample, branch_id, rho, cfg)
                branch = base_sample["network"]["branch"][branch_id]
                rho_meta["sample_id"] = converged ? sample_id : nothing
                rho_meta["base_sample_id"] = base_sample_id
                rho_meta["load_setting_id"] = load_setting_id
                rho_meta["source_file"] = path
                rho_meta["split"] = cfg["split"]
                rho_meta["task_name"] = cfg["task_name"]
                rho_meta["case_name"] = cfg["case_name"]
                rho_meta["source_topology"] = "n"
                rho_meta["rho_grid"] = rho_source_label(rhos, cfg)
                rho_meta["rho_mode"] = cfg["rho_mode"]
                rho_meta["solver_type"] = "AC_PF"
                rho_meta["branch_mode"] = cfg["branch_mode"]
                rho_meta["random_attempt"] = attempt

                if converged
                    sample = Dict{String, Any}(
                        "network" => net,
                        "solution" => result,
                        "parent_seed" => -1,
                        "rho_meta" => rho_meta,
                    )
                    file_name = "raw/sample_$(sample_id).json"
                    write_json(joinpath(output_dir, file_name), sample)
                    rec = Dict{String, Any}(
                        "sample_id" => sample_id,
                        "file" => file_name,
                        "split" => cfg["split"],
                        "case_name" => cfg["case_name"],
                        "base_sample_id" => base_sample_id,
                        "load_setting_id" => load_setting_id,
                        "source_file" => path,
                        "branch_id" => branch_id,
                        "f_bus" => branch["f_bus"],
                        "t_bus" => branch["t_bus"],
                        "rho" => rho,
                        "rho_grid" => rho_source_label(rhos, cfg),
                        "rho_mode" => cfg["rho_mode"],
                        "br_status" => rho_meta["br_status"],
                        "solver_type" => "AC_PF",
                        "termination_status" => result["termination_status"],
                        "created_at" => string(Dates.now()),
                        "random_attempts_used" => attempt,
                    )
                    write_jsonl(samples_io, rec)
                    push!(records, rec)
                    load_rec = Dict{String, Any}(
                        "split" => cfg["split"],
                        "load_setting_id" => load_setting_id,
                        "base_sample_id" => base_sample_id,
                        "source_file" => path,
                        "load_vector" => load_vector_from_sample(base_sample),
                        "sample_ids" => [sample_id],
                        "selected_branch_ids" => branch_id,
                        "rho_values" => string(rho),
                        "random_attempts_used" => attempt,
                    )
                    write_jsonl(loads_io, load_rec)
                    push!(load_records, load_rec)
                    sample_id += 1
                    success_loads += 1
                    accepted = true
                    break
                else
                    fail = Dict{String, Any}(
                        "split" => cfg["split"],
                        "case_name" => cfg["case_name"],
                        "base_sample_id" => base_sample_id,
                        "load_setting_id" => load_setting_id,
                        "source_file" => path,
                        "branch_id" => branch_id,
                        "f_bus" => branch["f_bus"],
                        "t_bus" => branch["t_bus"],
                        "rho" => rho,
                        "rho_grid" => rho_source_label(rhos, cfg),
                        "rho_mode" => cfg["rho_mode"],
                        "br_status" => rho_meta["br_status"],
                        "solver_type" => "AC_PF",
                        "termination_status" => status_string(result),
                        "random_attempt" => attempt,
                        "created_at" => string(Dates.now()),
                    )
                    write_jsonl(failures_io, fail)
                end
            end

            if cfg["print_every"] > 0 && attempted_loads % cfg["print_every"] == 0
                println("attempted_loads=$(attempted_loads) success_loads=$(success_loads) last_accepted=$(accepted)")
                flush(stdout)
            end
        end
    end
    end
    end

    write_json(joinpath(output_dir, "splits", "$(cfg["split"]).json"), Dict(
        "split" => cfg["split"],
        "records" => records,
        "load_settings" => load_records,
        "failures" => Any[],
    ))
    write_json(joinpath(output_dir, "splits", "val.json"), Dict("split" => "val", "records" => Any[], "load_settings" => Any[], "failures" => Any[]))
    write_json(joinpath(output_dir, "splits", "test.json"), Dict("split" => "test", "records" => Any[], "load_settings" => Any[], "failures" => Any[]))

    final_meta = merge(meta, Dict{String, Any}(
        "status" => success_loads >= cfg["target_success"] ? "complete" : "partial",
        "attempted_loads" => attempted_loads,
        "success_loads" => success_loads,
        "samples" => length(records),
    ))
    write_json(joinpath(output_dir, "meta.json"), final_meta)
    println(JSON.json(final_meta))
end

main()
