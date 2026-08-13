using Pkg
Pkg.activate(@__DIR__)

using Dates
using JSON
using PowerModels
using Random

include(joinpath(@__DIR__, "src", "OPFLearn.jl"))

function parse_args(args)
    cfg = Dict{String, Any}(
        "source_dir" => "",
        "plan" => "",
        "output_dir" => "",
        "min_load_id" => 1,
        "max_load_id" => typemax(Int),
        "seed" => 20260731,
        "clean" => false,
        "print_every" => 100,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        @assert startswith(arg, "--") "Expected --key value"
        key = replace(arg[3:end], "-" => "_")
        if key == "clean"
            cfg["clean"] = true
        else
            i += 1
            @assert i <= length(args) "Missing value for --$(key)"
            value = args[i]
            if key in ["min_load_id", "max_load_id", "seed", "print_every"]
                cfg[key] = parse(Int, value)
            elseif haskey(cfg, key)
                cfg[key] = value
            else
                error("Unknown argument --$(key)")
            end
        end
        i += 1
    end
    for key in ["source_dir", "plan", "output_dir"]
        @assert !isempty(cfg[key]) "--$(replace(key, "_" => "-")) is required"
    end
    return cfg
end

function read_plan(path)
    rows = Dict{Int, String}()
    open(path) do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            row = JSON.parse(line)
            load_id = Int(row["load_setting_id"])
            haskey(rows, load_id) && error("Duplicate load setting in plan: $(load_id)")
            rows[load_id] = string(row["branch_id"])
        end
    end
    return rows
end

function branch_sort_key(net, key)
    return get(net["branch"][key], "index", parse(Int, string(key)))
end

function is_transformer_like(branch)
    tap = Float64(get(branch, "tap", 1.0))
    shift = Float64(get(branch, "shift", 0.0))
    return abs(tap - 1.0) > 1e-12 || abs(shift) > 1e-12
end

function selected_branch_keys(net)
    loop_keys = OPFLearn.loop_branch_keys(net)
    return [key for key in loop_keys if !is_transformer_like(net["branch"][key])]
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
    status === true && return true
    return uppercase(string(status)) in ["TRUE", "LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED", "ALMOST_OPTIMAL"]
end

function solve_outage(base_sample, branch_id)
    net = deepcopy(base_sample["network"])
    seed_pf_controls_from_solution!(net, base_sample["solution"]["solution"])
    rho_meta = OPFLearn.apply_branch_rho!(net, branch_id, 0.0)
    try
        result = PowerModels.compute_ac_pf(net; flat_start=false)
        converged = pf_converged(result)
        result["solver_type"] = "AC_PF"
        result["pf_termination_status"] = get(result, "termination_status", nothing)
        result["termination_status"] = converged ? "PF_CONVERGED" : string(get(result, "termination_status", "unknown"))
        return converged, net, result, rho_meta
    catch err
        return false, net, Dict{String, Any}("solver_type" => "AC_PF", "termination_status" => "EXCEPTION", "error" => sprint(showerror, err)), rho_meta
    end
end

function make_normal(base_sample, branch_id)
    net = deepcopy(base_sample["network"])
    seed_pf_controls_from_solution!(net, base_sample["solution"]["solution"])
    rho_meta = OPFLearn.apply_branch_rho!(net, branch_id, 1.0)
    result = deepcopy(base_sample["solution"])
    result["solver_type"] = "AC_PF"
    result["termination_status"] = "PF_CONVERGED"
    return net, result, rho_meta
end

function write_json(path, payload)
    open(path, "w") do io
        write(io, JSON.json(payload))
    end
end

function write_jsonl(io, payload)
    write(io, JSON.json(payload))
    write(io, "\n")
end

function rho_record(sample_id, file_name, load_id, source_file, branch, rho, rho_meta, result, planned_branch, fallback_attempt)
    return Dict{String, Any}(
        "sample_id" => sample_id,
        "file" => file_name,
        "split" => "train",
        "case_name" => "case30",
        "base_sample_id" => load_id,
        "load_setting_id" => load_id,
        "source_file" => source_file,
        "branch_id" => branch,
        "planned_branch_id" => planned_branch,
        "f_bus" => rho_meta["f_bus"],
        "t_bus" => rho_meta["t_bus"],
        "rho" => rho,
        "rho_grid" => "0.0,1.0",
        "rho_mode" => "endpoint-paired",
        "br_status" => rho_meta["br_status"],
        "solver_type" => "AC_PF",
        "termination_status" => result["termination_status"],
        "created_at" => string(Dates.now()),
        "fallback_attempt" => fallback_attempt,
    )
end

function prepare_output(cfg)
    output_dir = cfg["output_dir"]
    if isdir(output_dir)
        cfg["clean"] || error("Output exists: $(output_dir). Use --clean.")
        rm(output_dir; recursive=true, force=true)
    end
    for name in ["raw", "manifests", "splits", "analysis"]
        mkpath(joinpath(output_dir, name))
    end
end

function main()
    cfg = parse_args(ARGS)
    plan = read_plan(cfg["plan"])
    load_ids = sort([id for id in keys(plan) if cfg["min_load_id"] <= id <= cfg["max_load_id"]])
    isempty(load_ids) && error("No planned loads in the requested ID range")
    prepare_output(cfg)

    samples = Any[]
    load_rows = Any[]
    failures = Any[]
    rng = MersenneTwister(cfg["seed"])
    sample_id = 1
    fallback_count = 0
    samples_io = open(joinpath(cfg["output_dir"], "manifests", "samples.jsonl"), "w")
    failures_io = open(joinpath(cfg["output_dir"], "manifests", "failures.jsonl"), "w")
    try
        for (position, load_id) in enumerate(load_ids)
            source_file = joinpath(cfg["source_dir"], "sample_$(load_id).json")
            isfile(source_file) || error("Missing source normal PF sample: $(source_file)")
            base_sample = JSON.parsefile(source_file)
            planned_branch = plan[load_id]
            branches = selected_branch_keys(base_sample["network"])
            planned_branch in branches || error("Planned branch $(planned_branch) is invalid for load $(load_id)")
            alternatives = [branch for branch in branches if branch != planned_branch]
            shuffle!(rng, alternatives)
            candidates = vcat([planned_branch], alternatives)

            solved = false
            selected_branch = ""
            outage_net = nothing
            outage_result = nothing
            outage_meta = nothing
            selected_attempt = 0
            for (attempt, branch) in enumerate(candidates)
                converged, net, result, meta = solve_outage(base_sample, branch)
                if converged
                    solved = true
                    selected_branch = branch
                    outage_net = net
                    outage_result = result
                    outage_meta = meta
                    selected_attempt = attempt
                    fallback_count += attempt > 1 ? 1 : 0
                    break
                end
                failure = Dict("load_setting_id" => load_id, "planned_branch_id" => planned_branch, "branch_id" => branch, "fallback_attempt" => attempt, "termination_status" => result["termination_status"])
                push!(failures, failure)
                write_jsonl(failures_io, failure)
            end
            solved || error("No converged outage found for load $(load_id)")

            normal_net, normal_result, normal_meta = make_normal(base_sample, selected_branch)
            for (rho, net, result, meta, role) in [
                (1.0, normal_net, normal_result, normal_meta, "normal_rho1"),
                (0.0, outage_net, outage_result, outage_meta, "outage_rho0"),
            ]
                meta["sample_id"] = sample_id
                meta["load_setting_id"] = load_id
                meta["base_sample_id"] = load_id
                meta["source_file"] = source_file
                meta["split"] = "train"
                meta["task_name"] = "case30_endpoint_scaling_v1_master"
                meta["case_name"] = "case30"
                meta["source_topology"] = "n"
                meta["rho_grid"] = "0.0,1.0"
                meta["rho_mode"] = "endpoint-paired"
                meta["planned_branch_id"] = planned_branch
                meta["source_role"] = role
                file_name = "raw/sample_$(sample_id).json"
                write_json(joinpath(cfg["output_dir"], file_name), Dict("network" => net, "solution" => result, "parent_seed" => -1, "rho_meta" => meta))
                record = rho_record(sample_id, file_name, load_id, source_file, selected_branch, rho, meta, result, planned_branch, selected_attempt)
                push!(samples, record)
                write_jsonl(samples_io, record)
                sample_id += 1
            end
            push!(load_rows, Dict("split" => "train", "load_setting_id" => load_id, "base_sample_id" => load_id, "source_file" => source_file, "sample_ids" => [sample_id - 2, sample_id - 1], "selected_branch_ids" => selected_branch, "planned_branch_ids" => planned_branch, "rho_values" => "0.0,1.0"))
            if cfg["print_every"] > 0 && position % cfg["print_every"] == 0
                println("completed_loads=$(position)/$(length(load_ids)) fallback_loads=$(fallback_count) failed_attempts=$(length(failures))")
                flush(stdout)
            end
        end
    finally
        close(samples_io)
        close(failures_io)
    end

    output_dir = cfg["output_dir"]
    write_json(joinpath(output_dir, "splits", "train.json"), Dict("split" => "train", "records" => samples, "load_settings" => load_rows, "failures" => failures))
    write_json(joinpath(output_dir, "splits", "val.json"), Dict("split" => "val", "records" => Any[], "load_settings" => Any[], "failures" => Any[]))
    write_json(joinpath(output_dir, "splits", "test.json"), Dict("split" => "test", "records" => Any[], "load_settings" => Any[], "failures" => Any[]))
    open(joinpath(output_dir, "manifests", "load_settings.jsonl"), "w") do io
        for row in load_rows
            write_jsonl(io, row)
        end
    end
    write_json(joinpath(output_dir, "meta.json"), Dict(
        "protocol" => "case30_endpoint_scaling_v1",
        "source_dir" => cfg["source_dir"],
        "plan" => cfg["plan"],
        "load_range" => [minimum(load_ids), maximum(load_ids)],
        "loads" => length(load_ids),
        "samples" => length(samples),
        "fallback_loads" => fallback_count,
        "failed_attempts" => length(failures),
    ))
    println("complete loads=$(length(load_ids)) samples=$(length(samples)) fallback_loads=$(fallback_count)")
end

main()
