using Pkg
Pkg.activate(@__DIR__)

using Dates
using JSON
using PowerModels

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
    return [strip(x) for x in split(strip(value), ",") if strip(x) != ""]
end

function repo_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function default_config()
    return Dict{String, Any}(
        "case_name" => "case30",
        "source_dir" => joinpath(repo_root(), "data", "pfdelta_data", "case30", "n", "raw"),
        "output_dir" => joinpath(@__DIR__, "case30", "rho_pf_bank_normal_loop_edges_10000_0p1"),
        "task_name" => "rho_pf_bank_normal_loop_edges",
        "max_loads" => 10000,
        "start_index" => 1,
        "rhos" => parse_float_list("0:0.1:1"),
        "branch_mode" => "normal-loop",
        "branch_ids" => String[],
        "split" => "all",
        "clean" => false,
        "append" => false,
        "print_every" => 1,
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
    elseif key in ["max_loads", "loads"]
        cfg["max_loads"] = parse(Int, value)
    elseif key == "start_index"
        cfg["start_index"] = parse(Int, value)
    elseif key in ["rhos", "rho_grid"]
        cfg["rhos"] = parse_float_list(value)
    elseif key == "branch_mode"
        cfg["branch_mode"] = lowercase(value)
    elseif key == "branch_ids"
        cfg["branch_ids"] = parse_string_list(value)
    elseif key == "split"
        cfg["split"] = value
    elseif key == "clean"
        cfg["clean"] = parse_bool(value)
    elseif key == "append"
        cfg["append"] = parse_bool(value)
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
            if key in ["clean", "append", "flat-start", "flat_start"]
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

    for rho in cfg["rhos"]
        @assert 0.0 <= rho <= 1.0 "rho values must be in [0, 1]"
    end
    @assert cfg["max_loads"] > 0 "--max-loads must be positive"
    @assert cfg["start_index"] > 0 "--start-index must be positive"
    @assert cfg["print_every"] > 0 "--print-every must be positive"
    @assert !(cfg["clean"] && cfg["append"]) "Use either --clean or --append, not both"
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

    if cfg["branch_mode"] == "normal-loop"
        loop_keys = OPFLearn.loop_branch_keys(net)
        return [k for k in loop_keys if !is_transformer_like(net["branch"][k])]
    elseif cfg["branch_mode"] == "loop"
        return OPFLearn.loop_branch_keys(net)
    elseif cfg["branch_mode"] == "all-active"
        keys = [k for k in keys(net["branch"]) if get(net["branch"][k], "br_status", 1) == 1]
        return sort(keys, by=k -> branch_sort_key(net, k))
    else
        error("Unknown branch-mode $(cfg["branch_mode"]). Use normal-loop, loop, all-active, or --branch-ids.")
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
        return converged, net, result, rho_meta, nothing
    catch err
        return false, net, Dict{String, Any}(
            "solver_type" => "AC_PF",
            "termination_status" => "EXCEPTION",
            "error" => sprint(showerror, err),
        ), rho_meta, err
    end
end

function write_jsonl(io, record)
    write(io, JSON.json(record))
    write(io, "\n")
end

function write_sample_json(path::String, sample)
    open(path, "w") do io
        write(io, JSON.json(sample))
    end
end

function write_csv(path::String, rows, cols::Vector{String})
    open(path, "w") do io
        write(io, join(cols, ","), "\n")
        for row in rows
            vals = [replace(string(get(row, c, "")), "," => ";") for c in cols]
            write(io, join(vals, ","), "\n")
        end
    end
end

function manifest_line_count(path::String)
    !isfile(path) && return 0
    count = 0
    open(path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            count += 1
        end
    end
    return count
end

function max_manifest_sample_id(path::String)
    !isfile(path) && return 0
    max_id = 0
    open(path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            rec = JSON.parse(line)
            if haskey(rec, "sample_id")
                max_id = max(max_id, Int(rec["sample_id"]))
            end
        end
    end
    return max_id
end

function inc!(dict::Dict, key, value=1)
    dict[key] = get(dict, key, 0) + value
end

function main()
    cfg = parse_args(ARGS)
    output_dir = cfg["output_dir"]
    raw_dir = joinpath(output_dir, "raw")
    manifests_dir = joinpath(output_dir, "manifests")
    analysis_dir = joinpath(output_dir, "analysis")

    append_mode = cfg["append"]

    if cfg["clean"] && isdir(output_dir)
        rm(output_dir; recursive=true, force=true)
    elseif isdir(output_dir)
        if !append_mode
            error("Output directory already exists. Use --clean to overwrite or --append to continue: $(output_dir)")
        end
    end
    mkpath(raw_dir)
    mkpath(manifests_dir)
    mkpath(analysis_dir)

    samples_path = joinpath(manifests_dir, "samples.jsonl")
    failures_path = joinpath(manifests_dir, "failures.jsonl")
    loads_path = joinpath(manifests_dir, "load_settings.jsonl")
    curves_path = joinpath(manifests_dir, "curves.jsonl")

    sample_id = append_mode ? max_manifest_sample_id(samples_path) + 1 : 1
    output_mode = append_mode ? "a" : "w"
    resume_stamp = replace(replace(string(Dates.now()), ":" => ""), "." => "_")
    meta_path = joinpath(output_dir, "meta.json")
    run_meta_path = append_mode ? joinpath(output_dir, "meta_resume_$(resume_stamp).json") : meta_path

    files = source_files(cfg)
    @assert !isempty(files) "No source sample files found"

    first_sample = JSON.parsefile(files[1])
    branch_keys = selected_branch_keys(first_sample["network"], cfg)
    @assert !isempty(branch_keys) "No selected branch keys"

    rhos = Float64.(cfg["rhos"])
    meta = Dict{String, Any}(
        "created_at" => string(Dates.now()),
        "status" => "running",
        "case_name" => cfg["case_name"],
        "task_name" => cfg["task_name"],
        "solver_type" => "AC_PF",
        "source_dir" => cfg["source_dir"],
        "output_dir" => output_dir,
        "append" => append_mode,
        "starting_sample_id" => sample_id,
        "num_source_files" => length(files),
        "max_loads" => cfg["max_loads"],
        "start_index" => cfg["start_index"],
        "branch_mode" => cfg["branch_mode"],
        "branch_ids" => branch_keys,
        "num_branches" => length(branch_keys),
        "rhos" => rhos,
        "rho_definition" => "All branches carry edge-level rho. Uncontrolled branches have rho=1.0 and rho_controlled=false. The controlled branch has rho set to the current grid value; rho=0 sets br_status=0, rho>0 divides br_r/br_x by rho and multiplies shunt terms by rho.",
        "pf_definition" => "AC-PF is solved from the base OPF sample controls: load Pd/Qd, generator Pg, PV/slack voltage magnitudes, and base solution warm starts. Successful PF points are saved as regression samples; failed points are recorded only in failures.jsonl.",
    )
    write_sample_json(run_meta_path, meta)

    total_attempts = 0
    total_success = 0
    total_failed = 0
    success_by_rho = Dict{Float64, Int}()
    failure_by_rho = Dict{Float64, Int}()
    success_by_branch = Dict{String, Int}()
    failure_by_branch = Dict{String, Int}()

    open(samples_path, output_mode) do samples_io
    open(failures_path, output_mode) do failures_io
    open(loads_path, output_mode) do loads_io
    open(curves_path, output_mode) do curves_io
        for (local_idx, path) in enumerate(files)
            base_sample = JSON.parsefile(path)
            base_sample_id = sample_file_id(path)
            load_setting_id = base_sample_id
            load_success = 0
            load_failed = 0

            for branch_id in branch_keys
                branch = base_sample["network"]["branch"][branch_id]
                curve_success_rhos = Float64[]
                curve_failed_rhos = Float64[]
                curve_sample_ids = Int[]

                for (rho_index, rho) in enumerate(rhos)
                    total_attempts += 1
                    converged, net, result, rho_meta, err = solve_pf_rho(base_sample, branch_id, rho, cfg)
                    rho_meta["sample_id"] = converged ? sample_id : nothing
                    rho_meta["base_sample_id"] = base_sample_id
                    rho_meta["load_setting_id"] = load_setting_id
                    rho_meta["source_file"] = path
                    rho_meta["split"] = cfg["split"]
                    rho_meta["task_name"] = cfg["task_name"]
                    rho_meta["case_name"] = cfg["case_name"]
                    rho_meta["source_topology"] = "n"
                    rho_meta["rho_index"] = rho_index
                    rho_meta["rho_grid"] = join(string.(rhos), ",")
                    rho_meta["solver_type"] = "AC_PF"
                    rho_meta["branch_mode"] = cfg["branch_mode"]

                    if converged
                        sample = Dict{String, Any}(
                            "network" => net,
                            "solution" => result,
                            "parent_seed" => -1,
                            "rho_meta" => rho_meta,
                        )
                        file_name = "raw/sample_$(sample_id).json"
                        write_sample_json(joinpath(output_dir, file_name), sample)

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
                            "rho_index" => rho_index,
                            "rho_grid" => join(string.(rhos), ","),
                            "br_status" => rho_meta["br_status"],
                            "solver_type" => "AC_PF",
                            "termination_status" => result["termination_status"],
                            "created_at" => string(Dates.now()),
                        )
                        write_jsonl(samples_io, rec)
                        push!(curve_success_rhos, rho)
                        push!(curve_sample_ids, sample_id)
                        sample_id += 1
                        total_success += 1
                        load_success += 1
                        inc!(success_by_rho, rho)
                        inc!(success_by_branch, branch_id)
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
                            "rho_index" => rho_index,
                            "rho_grid" => join(string.(rhos), ","),
                            "br_status" => rho_meta["br_status"],
                            "solver_type" => "AC_PF",
                            "termination_status" => status_string(result),
                            "reason" => isnothing(err) ? "pf_not_converged" : "pf_exception",
                            "error" => get(result, "error", nothing),
                            "created_at" => string(Dates.now()),
                        )
                        write_jsonl(failures_io, fail)
                        push!(curve_failed_rhos, rho)
                        total_failed += 1
                        load_failed += 1
                        inc!(failure_by_rho, rho)
                        inc!(failure_by_branch, branch_id)
                    end
                end

                curve = Dict{String, Any}(
                    "case_name" => cfg["case_name"],
                    "base_sample_id" => base_sample_id,
                    "load_setting_id" => load_setting_id,
                    "source_file" => path,
                    "branch_id" => branch_id,
                    "f_bus" => branch["f_bus"],
                    "t_bus" => branch["t_bus"],
                    "success_count" => length(curve_success_rhos),
                    "failure_count" => length(curve_failed_rhos),
                    "success_rhos" => join(string.(curve_success_rhos), ","),
                    "failed_rhos" => join(string.(curve_failed_rhos), ","),
                    "sample_ids" => join(string.(curve_sample_ids), ","),
                    "min_success_rho" => isempty(curve_success_rhos) ? nothing : minimum(curve_success_rhos),
                    "max_success_rho" => isempty(curve_success_rhos) ? nothing : maximum(curve_success_rhos),
                    "created_at" => string(Dates.now()),
                )
                write_jsonl(curves_io, curve)
            end

            load_rec = Dict{String, Any}(
                "case_name" => cfg["case_name"],
                "base_sample_id" => base_sample_id,
                "load_setting_id" => load_setting_id,
                "source_file" => path,
                "split" => cfg["split"],
                "num_success" => load_success,
                "num_failed" => load_failed,
                "num_attempts" => load_success + load_failed,
                "created_at" => string(Dates.now()),
            )
            write_jsonl(loads_io, load_rec)

            if local_idx % cfg["print_every"] == 0
                println("load $(local_idx)/$(length(files)), base_sample_id=$(base_sample_id), success=$(load_success), failed=$(load_failed), total_success=$(total_success), total_failed=$(total_failed)")
                flush(stdout)
                flush(samples_io)
                flush(failures_io)
                flush(loads_io)
                flush(curves_io)
            end
        end
    end
    end
    end
    end

    success_rho_rows = [Dict("rho" => k, "success" => v, "failure" => get(failure_by_rho, k, 0)) for (k, v) in sort(collect(success_by_rho), by=x -> x[1])]
    for (rho, v) in sort(collect(failure_by_rho), by=x -> x[1])
        if !haskey(success_by_rho, rho)
            push!(success_rho_rows, Dict("rho" => rho, "success" => 0, "failure" => v))
        end
    end
    sort!(success_rho_rows, by=row -> row["rho"])
    analysis_prefix = append_mode ? "resume_$(resume_stamp)_" : ""
    write_csv(joinpath(analysis_dir, "$(analysis_prefix)success_by_rho.csv"), success_rho_rows, ["rho", "success", "failure"])

    branch_rows = Dict{String, Dict{String, Any}}()
    for branch_id in union(keys(success_by_branch), keys(failure_by_branch))
        branch_rows[branch_id] = Dict(
            "branch_id" => branch_id,
            "success" => get(success_by_branch, branch_id, 0),
            "failure" => get(failure_by_branch, branch_id, 0),
        )
    end
    write_csv(joinpath(analysis_dir, "$(analysis_prefix)success_by_branch.csv"), [branch_rows[k] for k in sort(collect(keys(branch_rows)), by=x -> parse(Int, x))], ["branch_id", "success", "failure"])

    meta["status"] = "finished"
    meta["finished_at"] = string(Dates.now())
    meta["run_num_attempts"] = total_attempts
    meta["run_num_success_samples"] = total_success
    meta["run_num_failures"] = total_failed
    meta["total_success_samples"] = manifest_line_count(samples_path)
    meta["total_failures"] = manifest_line_count(failures_path)
    meta["total_load_settings"] = manifest_line_count(loads_path)
    meta["total_curves"] = manifest_line_count(curves_path)
    meta["next_sample_id"] = sample_id
    write_sample_json(run_meta_path, meta)
    write_sample_json(meta_path, meta)

    println("DONE")
    println("Output: $(output_dir)")
    println("Attempts: $(total_attempts), success: $(total_success), failed: $(total_failed)")
end

main()
