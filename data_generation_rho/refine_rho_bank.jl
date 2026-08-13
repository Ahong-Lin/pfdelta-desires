using Pkg
Pkg.activate(@__DIR__)

using CSV
using DataFrames
using Dates
using JSON
using JuMP
using Ipopt
using PowerModels

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
    return [strip(x) for x in split(strip(value), ",") if strip(x) != ""]
end

function default_config()
    return Dict{String, Any}(
        "case_name" => "case30",
        "task_name" => "rho_bank_normal_loop_edges_fine_step0p01",
        "source_dir" => "",
        "output_dir" => "",
        "coarse_rhos" => parse_float_list("0:0.1:1"),
        "fine_rhos" => parse_float_list("0:0.01:1"),
        "splits" => ["train", "val", "test"],
        "load_setting_ids" => String[],
        "branch_ids" => String[],
        "max_curves" => 0,
        "perturb_costs" => "none",
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
    elseif key == "source_dir"
        cfg["source_dir"] = value
    elseif key in ["output", "output_dir"]
        cfg["output_dir"] = value
    elseif key == "coarse_rhos"
        cfg["coarse_rhos"] = parse_float_list(value)
    elseif key == "fine_rhos"
        cfg["fine_rhos"] = parse_float_list(value)
    elseif key == "splits"
        cfg["splits"] = parse_string_list(value)
    elseif key == "load_setting_ids"
        cfg["load_setting_ids"] = parse_string_list(value)
    elseif key == "branch_ids"
        cfg["branch_ids"] = parse_string_list(value)
    elseif key == "max_curves"
        cfg["max_curves"] = parse(Int, value)
    elseif key == "perturb_costs"
        cfg["perturb_costs"] = lowercase(value)
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

    @assert cfg["source_dir"] != "" "--source-dir is required"
    if !isabspath(cfg["source_dir"])
        cfg["source_dir"] = joinpath(@__DIR__, cfg["source_dir"])
    end

    if cfg["output_dir"] == ""
        cfg["output_dir"] = joinpath(@__DIR__, cfg["case_name"], "rho_bank_normal_loop_edges_30_0.01")
    elseif !isabspath(cfg["output_dir"])
        cfg["output_dir"] = joinpath(@__DIR__, cfg["output_dir"])
    end

    for rho in vcat(cfg["coarse_rhos"], cfg["fine_rhos"])
        @assert 0.0 <= rho <= 1.0 "rho values must be in [0, 1]"
    end
    @assert cfg["perturb_costs"] in ["none", "shuffle"] "perturb_costs must be none or shuffle"
    return cfg
end

function rho_key(rho)
    return string(round(Float64(rho); digits=10))
end

function read_jsonl(path::String)
    rows = Any[]
    if !isfile(path)
        return rows
    end
    for line in eachline(path)
        line = strip(line)
        isempty(line) && continue
        push!(rows, JSON.parse(line))
    end
    return rows
end

function append_jsonl(path::String, data)
    open(path, "a") do io
        JSON.print(io, data)
        write(io, "\n")
    end
end

function write_json(path::String, data)
    open(path, "w") do io
        JSON.print(io, data, 2)
    end
end

function load_case(case_name::AbstractString)
    @assert haskey(CASE_TO_PATH, String(case_name)) "Unknown case $(case_name)"
    return PowerModels.parse_file(joinpath(@__DIR__, CASE_TO_PATH[String(case_name)]))
end

function make_solver()
    return JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol" => OPFLearn.TOL)
end

function load_vector_to_matrix(load_vector)
    values = Float64[Float64(x) for x in load_vector]
    return reshape(values, :, 1)
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

function prepare_output_dir(cfg)
    output_dir = cfg["output_dir"]
    if cfg["clean"] && isdir(output_dir)
        rm(output_dir; recursive=true)
    end
    mkpath(joinpath(output_dir, "raw"))
    mkpath(joinpath(output_dir, "manifests"))
    mkpath(joinpath(output_dir, "splits"))
    mkpath(joinpath(output_dir, "analysis"))
end

function build_source_index(source_dir::String)
    samples = read_jsonl(joinpath(source_dir, "manifests", "samples.jsonl"))
    loads = read_jsonl(joinpath(source_dir, "manifests", "load_settings.jsonl"))

    sample_by_key = Dict{Tuple{String, Int, String, String}, Any}()
    successful_rhos = Dict{Tuple{String, Int, String}, Set{String}}()

    for rec in samples
        split = String(rec["split"])
        load_setting_id = Int(rec["load_setting_id"])
        branch_id = string(rec["branch_id"])
        rkey = rho_key(rec["rho"])
        sample_by_key[(split, load_setting_id, branch_id, rkey)] = rec
        curve_key = (split, load_setting_id, branch_id)
        if !haskey(successful_rhos, curve_key)
            successful_rhos[curve_key] = Set{String}()
        end
        push!(successful_rhos[curve_key], rkey)
    end

    load_by_id = Dict{Int, Any}()
    for rec in loads
        load_by_id[Int(rec["load_setting_id"])] = rec
    end

    return sample_by_key, successful_rhos, load_by_id
end

function split_rank(split::String)
    return get(Dict("train" => 0, "val" => 1, "test" => 2), split, 9)
end

function select_complete_curves(successful_rhos, cfg)
    expected = Set(rho_key.(cfg["coarse_rhos"]))
    split_filter = Set(String.(cfg["splits"]))
    load_filter = Set(String.(cfg["load_setting_ids"]))
    branch_filter = Set(String.(cfg["branch_ids"]))

    curves = Tuple{String, Int, String}[]
    for (curve_key, rhos) in successful_rhos
        split, load_setting_id, branch_id = curve_key
        split in split_filter || continue
        if !isempty(load_filter) && !(string(load_setting_id) in load_filter)
            continue
        end
        if !isempty(branch_filter) && !(branch_id in branch_filter)
            continue
        end
        if expected ⊆ rhos
            push!(curves, curve_key)
        end
    end

    sort!(curves, by=x -> (split_rank(x[1]), x[2], parse(Int, x[3])))
    if cfg["max_curves"] > 0 && length(curves) > cfg["max_curves"]
        curves = curves[1:cfg["max_curves"]]
    end
    return curves
end

function copy_source_sample(source_dir::String, source_rec, sample_id::Int, cfg::Dict, rho_meta_extra)
    source_file = joinpath(source_dir, String(source_rec["file"]))
    sample = JSON.parsefile(source_file)
    if !haskey(sample, "rho_meta") || isnothing(sample["rho_meta"])
        sample["rho_meta"] = Dict{String, Any}()
    end
    sample["rho_meta"]["sample_id"] = sample_id
    sample["rho_meta"]["task_name"] = cfg["task_name"]
    sample["rho_meta"]["source_sample_id"] = source_rec["sample_id"]
    sample["rho_meta"]["source_file"] = source_rec["file"]
    sample["rho_meta"]["refined_from"] = cfg["source_dir"]
    for (key, value) in rho_meta_extra
        sample["rho_meta"][key] = value
    end

    output_file = joinpath(cfg["output_dir"], "raw", "sample_$(sample_id).json")
    open(output_file, "w") do io
        write(io, JSON.json(sample))
    end
end

function make_success_record(sample_id::Int, split::String, load_setting_id::Int, branch_id::String, branch, rho::Float64, rho_index::Int, status, objective, source_kind::String; source_rec=nothing)
    rec = Dict{String, Any}(
        "sample_id" => sample_id,
        "file" => "raw/sample_$(sample_id).json",
        "split" => split,
        "case_name" => "case30",
        "load_setting_id" => load_setting_id,
        "branch_id" => branch_id,
        "f_bus" => branch["f_bus"],
        "t_bus" => branch["t_bus"],
        "rho" => rho,
        "rho_index" => rho_index,
        "br_status" => rho == 0.0 ? 0 : 1,
        "termination_status" => string(status),
        "objective" => objective,
        "source_kind" => source_kind,
        "created_at" => string(Dates.now()),
    )
    if !isnothing(source_rec)
        rec["source_sample_id"] = source_rec["sample_id"]
        rec["source_file"] = source_rec["file"]
    end
    return rec
end

function make_failure_record(split::String, load_setting_id::Int, branch_id::String, branch, rho::Float64, rho_index::Int, status)
    return Dict{String, Any}(
        "split" => split,
        "case_name" => "case30",
        "load_setting_id" => load_setting_id,
        "branch_id" => branch_id,
        "f_bus" => branch["f_bus"],
        "t_bus" => branch["t_bus"],
        "rho" => rho,
        "rho_index" => rho_index,
        "termination_status" => string(status),
        "reason" => "opf_failed",
        "created_at" => string(Dates.now()),
    )
end

function write_analysis(output_dir::String, records, failures, curves, fine_rhos)
    rows = Dict{String, Any}[]
    success_by_curve = Dict{Tuple{String, Int, String}, Set{String}}()
    failure_by_curve = Dict{Tuple{String, Int, String}, Set{String}}()
    branch_info = Dict{String, Tuple{Any, Any}}()

    for rec in records
        key = (String(rec["split"]), Int(rec["load_setting_id"]), string(rec["branch_id"]))
        if !haskey(success_by_curve, key)
            success_by_curve[key] = Set{String}()
        end
        push!(success_by_curve[key], rho_key(rec["rho"]))
        branch_info[string(rec["branch_id"])] = (rec["f_bus"], rec["t_bus"])
    end
    for rec in failures
        key = (String(rec["split"]), Int(rec["load_setting_id"]), string(rec["branch_id"]))
        if !haskey(failure_by_curve, key)
            failure_by_curve[key] = Set{String}()
        end
        push!(failure_by_curve[key], rho_key(rec["rho"]))
        branch_info[string(rec["branch_id"])] = (rec["f_bus"], rec["t_bus"])
    end

    fine_keys = Set(rho_key.(fine_rhos))
    for curve in curves
        split, load_setting_id, branch_id = curve
        successes = get(success_by_curve, curve, Set{String}())
        failures_for_curve = get(failure_by_curve, curve, Set{String}())
        f_bus, t_bus = get(branch_info, branch_id, ("", ""))
        push!(rows, Dict{String, Any}(
            "split" => split,
            "load_setting_id" => load_setting_id,
            "branch_id" => branch_id,
            "f_bus" => f_bus,
            "t_bus" => t_bus,
            "success_rho_count" => length(successes),
            "success_rhos" => join(sort(collect(successes), by=x -> parse(Float64, x)), ","),
            "failed_rho_count" => length(failures_for_curve),
            "failed_rhos" => join(sort(collect(failures_for_curve), by=x -> parse(Float64, x)), ","),
            "is_fine_complete" => fine_keys ⊆ successes,
        ))
    end

    df = DataFrame(rows)
    CSV.write(joinpath(output_dir, "analysis", "fine_success_by_curve.csv"), df)
    if nrow(df) > 0
        CSV.write(joinpath(output_dir, "analysis", "fine_full_rho_success_curves.csv"), filter(:is_fine_complete => ==(true), df))
    end
end

function main(args)
    cfg = parse_args(args)
    prepare_output_dir(cfg)

    source_dir = cfg["source_dir"]
    output_dir = cfg["output_dir"]
    base_network = load_case(cfg["case_name"])
    sample_by_key, successful_rhos, load_by_id = build_source_index(source_dir)
    curves = select_complete_curves(successful_rhos, cfg)
    @assert !isempty(curves) "No complete coarse curves found under $(source_dir)"

    meta_path = joinpath(output_dir, "meta.json")
    meta = Dict{String, Any}(
        "status" => "running",
        "created_at" => string(Dates.now()),
        "config" => cfg,
        "source_dir" => source_dir,
        "num_selected_curves" => length(curves),
        "coarse_rhos" => cfg["coarse_rhos"],
        "fine_rhos" => cfg["fine_rhos"],
        "rho_definition" => "Input curves are selected only if all coarse rho values are successful. Coarse points are copied from the source bank; missing fine-grid rho points are newly solved with AC OPF.",
    )
    write_json(meta_path, meta)

    samples_manifest = joinpath(output_dir, "manifests", "samples.jsonl")
    failures_manifest = joinpath(output_dir, "manifests", "failures.jsonl")
    loads_manifest = joinpath(output_dir, "manifests", "load_settings.jsonl")

    records = Dict{String, Any}[]
    failures = Dict{String, Any}[]
    split_records = Dict(split => Dict("records" => Any[], "failures" => Any[], "load_settings" => Any[]) for split in cfg["splits"])
    written_loads = Set{Int}()
    sample_id = 1

    println("Source: $(source_dir)")
    println("Output: $(output_dir)")
    println("Selected complete coarse curves: $(length(curves))")
    println("Fine rhos: $(length(cfg["fine_rhos"])) values")

    for (curve_index, curve) in enumerate(curves)
        split, load_setting_id, branch_id = curve
        @assert haskey(load_by_id, load_setting_id) "Missing load_setting_id $(load_setting_id) in source load manifest"
        load_rec = deepcopy(load_by_id[load_setting_id])
        load_setting = load_vector_to_matrix(load_rec["load_vector"])
        branch = base_network["branch"][branch_id]
        curve_success = 0
        curve_fail = 0

        if !(load_setting_id in written_loads)
            append_jsonl(loads_manifest, load_rec)
            if haskey(split_records, split)
                push!(split_records[split]["load_settings"], load_rec)
            end
            push!(written_loads, load_setting_id)
        end

        for (rho_index, rho) in enumerate(Float64.(cfg["fine_rhos"]))
            key = (split, load_setting_id, branch_id, rho_key(rho))
            if haskey(sample_by_key, key)
                source_rec = sample_by_key[key]
                copy_source_sample(source_dir, source_rec, sample_id, cfg, Dict(
                    "rho_index" => rho_index,
                    "fine_grid" => true,
                    "source_kind" => "copied_coarse",
                ))
                rec = make_success_record(
                    sample_id,
                    split,
                    load_setting_id,
                    branch_id,
                    branch,
                    rho,
                    rho_index,
                    get(source_rec, "termination_status", "copied"),
                    get(source_rec, "objective", nothing),
                    "copied_coarse";
                    source_rec=source_rec,
                )
                append_jsonl(samples_manifest, rec)
                push!(records, rec)
                if haskey(split_records, split)
                    push!(split_records[split]["records"], rec)
                end
                sample_id += 1
                curve_success += 1
            else
                feasible, network, raw_result, _, rho_meta = solve_rho_candidate(
                    base_network,
                    load_setting,
                    branch_id,
                    rho,
                    cfg,
                )
                rho_meta["split"] = split
                rho_meta["task_name"] = cfg["task_name"]
                rho_meta["case_name"] = cfg["case_name"]
                rho_meta["load_setting_id"] = load_setting_id
                rho_meta["rho_index"] = rho_index
                rho_meta["fine_grid"] = true
                rho_meta["source_kind"] = "solved_fine"
                rho_meta["refined_from"] = source_dir

                if feasible
                    rho_meta["sample_id"] = sample_id
                    OPFLearn.store_feasible_sample_json(
                        sample_id,
                        network,
                        raw_result,
                        joinpath(output_dir, "raw");
                        rho_meta=rho_meta,
                    )
                    rec = make_success_record(
                        sample_id,
                        split,
                        load_setting_id,
                        branch_id,
                        branch,
                        rho,
                        rho_index,
                        get(raw_result, "termination_status", "unknown"),
                        get(raw_result, "objective", nothing),
                        "solved_fine",
                    )
                    append_jsonl(samples_manifest, rec)
                    push!(records, rec)
                    if haskey(split_records, split)
                        push!(split_records[split]["records"], rec)
                    end
                    sample_id += 1
                    curve_success += 1
                else
                    fail = make_failure_record(
                        split,
                        load_setting_id,
                        branch_id,
                        branch,
                        rho,
                        rho_index,
                        get(raw_result, "termination_status", "unknown"),
                    )
                    append_jsonl(failures_manifest, fail)
                    push!(failures, fail)
                    if haskey(split_records, split)
                        push!(split_records[split]["failures"], fail)
                    end
                    curve_fail += 1
                end
            end
        end

        println("curve $(curve_index)/$(length(curves)): $(split) load_setting_id=$(load_setting_id) branch_id=$(branch_id), success=$(curve_success)/$(length(cfg["fine_rhos"])), failed=$(curve_fail)")
        flush(stdout)
    end

    for (split, data) in split_records
        write_json(joinpath(output_dir, "splits", "$(split).json"), data)
    end
    write_analysis(output_dir, records, failures, curves, Float64.(cfg["fine_rhos"]))

    meta["status"] = "finished"
    meta["finished_at"] = string(Dates.now())
    meta["num_success_samples"] = length(records)
    meta["num_failures"] = length(failures)
    meta["next_sample_id"] = sample_id
    write_json(meta_path, meta)

    println("DONE")
    println("Output: $(output_dir)")
    println("success_samples=$(length(records)), failures=$(length(failures))")
end

main(ARGS)
