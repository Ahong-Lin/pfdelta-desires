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
        "output_random_dir" => joinpath(@__DIR__, "case30", "rho_pf_random_edge_random_rho_14000"),
        "output_rho1_dir" => joinpath(@__DIR__, "case30", "rho_pf_random_edge_rho1_matched_14000"),
        "task_random_name" => "rho_pf_random_edge_random_rho_14000",
        "task_rho1_name" => "rho_pf_random_edge_rho1_matched_14000",
        "start_index" => 1,
        "max_loads" => 56000,
        "train_count" => 14000,
        "val_count" => 1000,
        "rhos" => parse_float_list("0:0.1:1"),
        "branch_mode" => "normal-loop",
        "branch_ids" => String[],
        "seed" => 11,
        "max_attempts_per_load" => 30,
        "exclude_roots" => String[],
        "exclude_splits" => ["test"],
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
    elseif key == "output_random_dir"
        cfg["output_random_dir"] = abspath(value)
    elseif key == "output_rho1_dir"
        cfg["output_rho1_dir"] = abspath(value)
    elseif key == "task_random_name"
        cfg["task_random_name"] = value
    elseif key == "task_rho1_name"
        cfg["task_rho1_name"] = value
    elseif key == "start_index"
        cfg["start_index"] = parse(Int, value)
    elseif key == "max_loads"
        cfg["max_loads"] = parse(Int, value)
    elseif key == "train_count"
        cfg["train_count"] = parse(Int, value)
    elseif key == "val_count"
        cfg["val_count"] = parse(Int, value)
    elseif key in ["rhos", "rho_grid"]
        cfg["rhos"] = parse_float_list(value)
    elseif key == "branch_mode"
        cfg["branch_mode"] = lowercase(value)
    elseif key == "branch_ids"
        cfg["branch_ids"] = parse_string_list(value)
    elseif key == "seed"
        cfg["seed"] = parse(Int, value)
    elseif key == "max_attempts_per_load"
        cfg["max_attempts_per_load"] = parse(Int, value)
    elseif key == "exclude_roots"
        cfg["exclude_roots"] = parse_string_list(value)
    elseif key == "exclude_splits"
        cfg["exclude_splits"] = parse_string_list(value)
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
    @assert cfg["train_count"] > 0
    @assert cfg["val_count"] >= 0
    @assert cfg["max_loads"] > 0
    @assert cfg["max_attempts_per_load"] > 0
    for rho in cfg["rhos"]
        @assert 0.0 <= rho <= 1.0
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

function prepare_root(root::String, clean::Bool)
    if clean && isdir(root)
        rm(root; recursive=true, force=true)
    elseif isdir(root)
        error("Output directory already exists. Use --clean to overwrite: $(root)")
    end
    for subdir in ["raw", "manifests", "splits", "analysis"]
        mkpath(joinpath(root, subdir))
    end
end

function make_record(sample_id, file_name, split, cfg, task_name, base_sample_id, load_setting_id, path, branch_id, branch, rho, rho_meta, result, attempts)
    return Dict{String, Any}(
        "sample_id" => sample_id,
        "file" => file_name,
        "split" => split,
        "case_name" => cfg["case_name"],
        "task_name" => task_name,
        "base_sample_id" => base_sample_id,
        "load_setting_id" => load_setting_id,
        "source_file" => path,
        "branch_id" => branch_id,
        "f_bus" => branch["f_bus"],
        "t_bus" => branch["t_bus"],
        "rho" => rho,
        "rho_grid" => join(string.(cfg["rhos"]), ","),
        "br_status" => rho_meta["br_status"],
        "solver_type" => "AC_PF",
        "termination_status" => result["termination_status"],
        "created_at" => string(Dates.now()),
        "random_attempts_used" => attempts,
        "paired_selection_id" => "$(load_setting_id):$(branch_id)",
    )
end

function write_sample(root, sample_id, net, result, rho_meta)
    sample = Dict{String, Any}(
        "network" => net,
        "solution" => result,
        "parent_seed" => -1,
        "rho_meta" => rho_meta,
    )
    file_name = "raw/sample_$(sample_id).json"
    write_json(joinpath(root, file_name), sample)
    return file_name
end

function main()
    cfg = parse_args(ARGS)
    random_root = cfg["output_random_dir"]
    rho1_root = cfg["output_rho1_dir"]
    prepare_root(random_root, cfg["clean"])
    prepare_root(rho1_root, cfg["clean"])

    files = source_files(cfg)
    first_sample = JSON.parsefile(files[1])
    branch_keys = selected_branch_keys(first_sample["network"], cfg)
    rhos = Float64.(cfg["rhos"])
    rng = MersenneTwister(cfg["seed"])
    excluded = excluded_load_ids(cfg)

    shuffled_files = shuffle(rng, files)
    split_targets = Dict("train" => cfg["train_count"], "val" => cfg["val_count"], "test" => 0)
    split_order = ["train", "val"]
    split_records_random = Dict(split => Any[] for split in ["train", "val", "test"])
    split_records_rho1 = Dict(split => Any[] for split in ["train", "val", "test"])
    split_loads_random = Dict(split => Any[] for split in ["train", "val", "test"])
    split_loads_rho1 = Dict(split => Any[] for split in ["train", "val", "test"])

    random_samples_path = joinpath(random_root, "manifests", "samples.jsonl")
    rho1_samples_path = joinpath(rho1_root, "manifests", "samples.jsonl")
    random_loads_path = joinpath(random_root, "manifests", "load_settings.jsonl")
    rho1_loads_path = joinpath(rho1_root, "manifests", "load_settings.jsonl")
    random_failures_path = joinpath(random_root, "manifests", "failures.jsonl")
    rho1_failures_path = joinpath(rho1_root, "manifests", "failures.jsonl")

    meta_common = Dict{String, Any}(
        "created_at" => string(Dates.now()),
        "case_name" => cfg["case_name"],
        "source_dir" => cfg["source_dir"],
        "start_index" => cfg["start_index"],
        "max_loads" => cfg["max_loads"],
        "train_count" => cfg["train_count"],
        "val_count" => cfg["val_count"],
        "branch_mode" => cfg["branch_mode"],
        "branch_ids" => branch_keys,
        "rhos" => rhos,
        "seed" => cfg["seed"],
        "exclude_roots" => cfg["exclude_roots"],
        "exclude_splits" => cfg["exclude_splits"],
        "excluded_load_settings" => length(excluded),
        "pairing" => "The random-rho and rho=1 datasets share the same accepted load_setting_id and selected branch_id.",
        "pf_definition" => "AC-PF is solved with PowerModels.compute_ac_pf after pfdelta OPFLearn.apply_branch_rho! and base OPF controls/warm starts.",
    )
    write_json(joinpath(random_root, "meta.json"), merge(meta_common, Dict("task_name" => cfg["task_random_name"], "rho_policy" => "random rho sampled from rho grid")))
    write_json(joinpath(rho1_root, "meta.json"), merge(meta_common, Dict("task_name" => cfg["task_rho1_name"], "rho_policy" => "matched selected branch, rho=1")))

    sample_id_random = 1
    sample_id_rho1 = 1
    attempted_loads = 0
    skipped_excluded = 0
    accepted_total = 0
    split_index = 1
    current_split = split_order[split_index]

    open(random_samples_path, "w") do random_samples_io
    open(rho1_samples_path, "w") do rho1_samples_io
    open(random_loads_path, "w") do random_loads_io
    open(rho1_loads_path, "w") do rho1_loads_io
    open(random_failures_path, "w") do random_failures_io
    open(rho1_failures_path, "w") do rho1_failures_io
        for path in shuffled_files
            split_index > length(split_order) && break
            current_split = split_order[split_index]
            if length(split_records_random[current_split]) >= split_targets[current_split]
                split_index += 1
                split_index > length(split_order) && break
                current_split = split_order[split_index]
            end

            base_sample_id = sample_file_id(path)
            if base_sample_id in excluded
                skipped_excluded += 1
                continue
            end
            attempted_loads += 1
            base_sample = JSON.parsefile(path)
            load_setting_id = base_sample_id
            load_vector = load_vector_from_sample(base_sample)
            accepted = false
            tried = Set{Tuple{String, Float64}}()
            max_attempts = min(cfg["max_attempts_per_load"], length(branch_keys) * length(rhos))

            for attempt in 1:max_attempts
                branch_id = rand(rng, branch_keys)
                rho_random = rand(rng, rhos)
                pair = (branch_id, rho_random)
                while pair in tried && length(tried) < length(branch_keys) * length(rhos)
                    branch_id = rand(rng, branch_keys)
                    rho_random = rand(rng, rhos)
                    pair = (branch_id, rho_random)
                end
                push!(tried, pair)
                branch = base_sample["network"]["branch"][branch_id]

                ok_random, net_random, result_random, meta_random = solve_pf_rho(base_sample, branch_id, rho_random, cfg)
                ok_rho1, net_rho1, result_rho1, meta_rho1 = solve_pf_rho(base_sample, branch_id, 1.0, cfg)

                if ok_random && ok_rho1
                    meta_random["sample_id"] = sample_id_random
                    meta_random["base_sample_id"] = base_sample_id
                    meta_random["load_setting_id"] = load_setting_id
                    meta_random["source_file"] = path
                    meta_random["split"] = current_split
                    meta_random["task_name"] = cfg["task_random_name"]
                    meta_random["case_name"] = cfg["case_name"]
                    meta_random["source_topology"] = "n"
                    meta_random["rho_grid"] = join(string.(rhos), ",")
                    meta_random["solver_type"] = "AC_PF"
                    meta_random["branch_mode"] = cfg["branch_mode"]
                    meta_random["random_attempt"] = attempt
                    meta_random["paired_rho1_sample_id"] = sample_id_rho1

                    meta_rho1["sample_id"] = sample_id_rho1
                    meta_rho1["base_sample_id"] = base_sample_id
                    meta_rho1["load_setting_id"] = load_setting_id
                    meta_rho1["source_file"] = path
                    meta_rho1["split"] = current_split
                    meta_rho1["task_name"] = cfg["task_rho1_name"]
                    meta_rho1["case_name"] = cfg["case_name"]
                    meta_rho1["source_topology"] = "n"
                    meta_rho1["rho_grid"] = "1"
                    meta_rho1["solver_type"] = "AC_PF"
                    meta_rho1["branch_mode"] = cfg["branch_mode"]
                    meta_rho1["random_attempt"] = attempt
                    meta_rho1["paired_random_sample_id"] = sample_id_random
                    meta_rho1["paired_random_rho"] = rho_random

                    file_random = write_sample(random_root, sample_id_random, net_random, result_random, meta_random)
                    file_rho1 = write_sample(rho1_root, sample_id_rho1, net_rho1, result_rho1, meta_rho1)

                    rec_random = make_record(sample_id_random, file_random, current_split, cfg, cfg["task_random_name"], base_sample_id, load_setting_id, path, branch_id, branch, rho_random, meta_random, result_random, attempt)
                    rec_rho1 = make_record(sample_id_rho1, file_rho1, current_split, cfg, cfg["task_rho1_name"], base_sample_id, load_setting_id, path, branch_id, branch, 1.0, meta_rho1, result_rho1, attempt)
                    rec_random["paired_rho1_sample_id"] = sample_id_rho1
                    rec_rho1["paired_random_sample_id"] = sample_id_random
                    rec_rho1["paired_random_rho"] = rho_random

                    load_rec_random = Dict{String, Any}(
                        "split" => current_split,
                        "load_setting_id" => load_setting_id,
                        "base_sample_id" => base_sample_id,
                        "source_file" => path,
                        "load_vector" => load_vector,
                        "sample_ids" => [sample_id_random],
                        "selected_branch_ids" => branch_id,
                        "rho_values" => string(rho_random),
                        "paired_rho1_sample_id" => sample_id_rho1,
                        "random_attempts_used" => attempt,
                    )
                    load_rec_rho1 = Dict{String, Any}(
                        "split" => current_split,
                        "load_setting_id" => load_setting_id,
                        "base_sample_id" => base_sample_id,
                        "source_file" => path,
                        "load_vector" => load_vector,
                        "sample_ids" => [sample_id_rho1],
                        "selected_branch_ids" => branch_id,
                        "rho_values" => "1.0",
                        "paired_random_sample_id" => sample_id_random,
                        "paired_random_rho" => rho_random,
                        "random_attempts_used" => attempt,
                    )

                    write_jsonl(random_samples_io, rec_random)
                    write_jsonl(rho1_samples_io, rec_rho1)
                    write_jsonl(random_loads_io, load_rec_random)
                    write_jsonl(rho1_loads_io, load_rec_rho1)
                    push!(split_records_random[current_split], rec_random)
                    push!(split_records_rho1[current_split], rec_rho1)
                    push!(split_loads_random[current_split], load_rec_random)
                    push!(split_loads_rho1[current_split], load_rec_rho1)
                    sample_id_random += 1
                    sample_id_rho1 += 1
                    accepted_total += 1
                    accepted = true
                    break
                else
                    fail_common = Dict{String, Any}(
                        "split" => current_split,
                        "case_name" => cfg["case_name"],
                        "base_sample_id" => base_sample_id,
                        "load_setting_id" => load_setting_id,
                        "source_file" => path,
                        "branch_id" => branch_id,
                        "f_bus" => branch["f_bus"],
                        "t_bus" => branch["t_bus"],
                        "random_rho" => rho_random,
                        "random_attempt" => attempt,
                        "created_at" => string(Dates.now()),
                    )
                    if !ok_random
                        write_jsonl(random_failures_io, merge(fail_common, Dict("rho" => rho_random, "termination_status" => status_string(result_random))))
                    end
                    if !ok_rho1
                        write_jsonl(rho1_failures_io, merge(fail_common, Dict("rho" => 1.0, "termination_status" => status_string(result_rho1))))
                    end
                end
            end

            if cfg["print_every"] > 0 && attempted_loads % cfg["print_every"] == 0
                println("attempted_loads=$(attempted_loads) accepted_total=$(accepted_total) split=$(current_split) split_count=$(length(split_records_random[current_split])) skipped_excluded=$(skipped_excluded) last_accepted=$(accepted)")
                flush(stdout)
            end
        end
    end
    end
    end
    end
    end
    end

    for split in ["train", "val", "test"]
        write_json(joinpath(random_root, "splits", "$(split).json"), Dict("split" => split, "records" => split_records_random[split], "load_settings" => split_loads_random[split], "failures" => Any[]))
        write_json(joinpath(rho1_root, "splits", "$(split).json"), Dict("split" => split, "records" => split_records_rho1[split], "load_settings" => split_loads_rho1[split], "failures" => Any[]))
    end

    function write_summary(root, records_by_split)
        rows = String["split,samples,unique_load_settings,branches,rho_values,branch_counts,rho_counts"]
        for split in ["train", "val", "test"]
            records = records_by_split[split]
            loads = Set{Int}()
            branch_counts = Dict{String, Int}()
            rho_counts = Dict{String, Int}()
            for rec in records
                push!(loads, Int(rec["load_setting_id"]))
                branch = string(rec["branch_id"])
                rho = string(rec["rho"])
                branch_counts[branch] = get(branch_counts, branch, 0) + 1
                rho_counts[rho] = get(rho_counts, rho, 0) + 1
            end
            branch_text = join(["$(k):$(branch_counts[k])" for k in sort(collect(keys(branch_counts)), by=x -> parse(Int, x))], ";")
            rho_text = join(["$(k):$(rho_counts[k])" for k in sort(collect(keys(rho_counts)), by=x -> parse(Float64, x))], ";")
            push!(rows, join([split, string(length(records)), string(length(loads)), string(length(branch_counts)), string(length(rho_counts)), branch_text, rho_text], ","))
        end
        open(joinpath(root, "analysis", "split_summary.csv"), "w") do io
            write(io, join(rows, "\n"))
            write(io, "\n")
        end
    end

    write_summary(random_root, split_records_random)
    write_summary(rho1_root, split_records_rho1)

    final_common = Dict{String, Any}(
        "status" => length(split_records_random["train"]) >= cfg["train_count"] && length(split_records_random["val"]) >= cfg["val_count"] ? "complete" : "partial",
        "attempted_loads" => attempted_loads,
        "accepted_total" => accepted_total,
        "skipped_excluded" => skipped_excluded,
        "train_samples" => length(split_records_random["train"]),
        "val_samples" => length(split_records_random["val"]),
    )
    write_json(joinpath(random_root, "meta.json"), merge(meta_common, final_common, Dict("task_name" => cfg["task_random_name"], "rho_policy" => "random rho sampled from rho grid")))
    write_json(joinpath(rho1_root, "meta.json"), merge(meta_common, final_common, Dict("task_name" => cfg["task_rho1_name"], "rho_policy" => "matched selected branch, rho=1")))
    println(JSON.json(merge(final_common, Dict("output_random_dir" => random_root, "output_rho1_dir" => rho1_root))))
end

main()
