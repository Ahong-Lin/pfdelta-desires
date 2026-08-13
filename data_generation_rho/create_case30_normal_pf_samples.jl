using Pkg

repo_root = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(@__DIR__)

using JSON
using PowerModels
using Random

include(joinpath(@__DIR__, "src", "OPFLearn.jl"))

function parse_args(args)
    cfg = Dict("output_dir" => "", "num_loads" => 0, "seed" => 20260801)
    i = 1
    while i <= length(args)
        key = replace(args[i], "--" => "", "-" => "_")
        haskey(cfg, key) || error("Unknown argument $(args[i])")
        i += 1
        i <= length(args) || error("Missing value for --$(key)")
        cfg[key] = key in ["num_loads", "seed"] ? parse(Int, args[i]) : args[i]
        i += 1
    end
    isempty(cfg["output_dir"]) && error("--output-dir is required")
    cfg["num_loads"] > 0 || error("--num-loads must be positive")
    return cfg
end

function main()
    cfg = parse_args(ARGS)
    output_dir = abspath(cfg["output_dir"])
    ispath(output_dir) && error("Output already exists: $(output_dir)")
    Random.seed!(cfg["seed"])

    case_path = joinpath(@__DIR__, "pglib", "pglib_opf_case30_ieee.m")
    network = PowerModels.parse_file(case_path)
    mkpath(joinpath(output_dir, "raw"))
    results = OPFLearn.create_samples(
        network,
        cfg["num_loads"];
        save_path=output_dir,
        perturb_costs_method="shuffle",
        perturb_topology_method="none",
        net_path=output_dir,
        save_max_load=true,
    )
    open(joinpath(output_dir, "results.json"), "w") do io
        JSON.print(io, results)
    end
    println("DONE normal PF samples: ", output_dir, ", count=", cfg["num_loads"])
end

main()
