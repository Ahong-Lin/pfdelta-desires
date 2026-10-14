#!/usr/bin/env julia
# Portable OPFLearn/custom-ACOPF input generation + explicit Newton PF labels.
using Distributed, JSON, PowerModels, Random, LinearAlgebra, SHA, Dates, JuMP, Ipopt

const PACKAGE_ROOT=normpath(joinpath(@__DIR__,".."))
include(joinpath(@__DIR__,"lib","helpers.jl"))
using .Case500Helpers

function parse_args(args)
    o=Dict{String,Any}("config"=>joinpath(PACKAGE_ROOT,"config.json"),
        "output"=>"","mode"=>"smoke","workers"=>nothing,"plan"=>false)
    i=1
    while i<=length(args)
        a=args[i]
        if a in ("-h","--help")
            println("Usage: julia --project=data_generation data_generation/generate_case500.jl --config config.json --output PATH --mode smoke|full [--workers N] [--plan]")
            exit(0)
        elseif a=="--plan"
            o["plan"]=true; i+=1; continue
        end
        startswith(a,"--") && haskey(o,a[3:end]) || error("Unknown option $a")
        i<length(args) || error("Missing value after $a")
        o[a[3:end]]=a=="--workers" ? parse(Int,args[i+1]) : args[i+1]
        i+=2
    end
    o["mode"] in ("smoke","full") || error("--mode must be smoke or full")
    o
end

function reference_protocol!(net,raw)
    original=[k for k in numeric_keys(raw["bus"]) if raw["bus"][k]["bus_type"]==3]
    original_status=Dict(k=>[Dict("generator_id"=>gid,"status"=>g["gen_status"])
        for (gid,g) in raw["gen"] if string(g["gen_bus"])==k] for k in original)
    parsed=[k for k in numeric_keys(net["bus"]) if net["bus"][k]["bus_type"]==3]
    invalid=isempty(original) || any(!any(get(g,"gen_status",1)==1 && string(g["gen_bus"])==k
        for g in values(raw["gen"])) for k in original)
    policy="PowerModels_automatic_bus_type_correction"
    if invalid
        active=[(k,g) for (k,g) in net["gen"] if get(g,"gen_status",1)==1]
        isempty(active) && error("No active generator available for a reference bus")
        sort!(active;by=x->(-Float64(x[2]["pmax"]),parse(Int,x[1])))
        chosen=string(first(active)[2]["gen_bus"])
        for b in values(net["bus"])
            b["bus_type"]==3 && (b["bus_type"]=2)
        end
        net["bus"][chosen]["bus_type"]=3
        policy="invalid_original_reference_replaced_by_largest_active_Pmax_then_smallest_generator_id"
    end
    refs=[k for k in numeric_keys(net["bus"]) if net["bus"][k]["bus_type"]==3]
    length(refs)==1 || error("Expected exactly one generating Slack bus after reference correction")
    all(any(g["gen_status"]==1 && string(g["gen_bus"])==k for g in values(net["gen"]))
        for k in refs) || error("Reference bus has no active generator")
    Dict("original_reference_bus_ids"=>original,"original_reference_generator_status"=>original_status,
        "PowerModels_automatic_reference_bus_ids"=>parsed,"reference_bus_ids"=>refs,
        "reference_policy"=>policy,"fixed_reference_angle_radians"=>0.0)
end

function load_environment(worker_count)
    worker_count>=0 || error("workers must be nonnegative")
    nprocs()==1 || error("Start this script without julia -p; use --workers instead")
    worker_count>0 && addprocs(worker_count;exeflags=`--project=$(@__DIR__) --threads=1`)
    src=joinpath(@__DIR__,"src","OPFLearn.jl")
    helper=joinpath(@__DIR__,"lib","helpers.jl")
    @everywhere begin
        using JSON, PowerModels, Random, LinearAlgebra, SHA
        LinearAlgebra.BLAS.set_num_threads(1)
        include($src)
        if !isdefined(Main,:Case500Helpers)
            include($helper)
        end
        using .Case500Helpers
    end
end

function atomic_copy(src,dest)
    mkpath(dirname(dest)); tmp=dest*".tmp."*string(getpid())
    cp(src,tmp;force=true); Base.Filesystem.rename(tmp,dest)
end

function read_matrix_rows(x)
    isempty(x) && return zeros(0,0)
    reduce(vcat,[permutedims(Float64.(row)) for row in x])
end

function save_sampling(path,A,b,batch)
    atomic_json(path,Dict("source_batch"=>batch,
        "A_rows"=>[collect(row) for row in eachrow(A)],"b"=>vec(b)))
end

function source_batch(net,config,out,batch,startid,requested,seed)
    dir=joinpath(out,"sources","batch_$(lpad(batch,6,'0'))")
    ready_path=joinpath(dir,"ready.json")
    if isfile(ready_path)
        ready=JSON.parsefile(ready_path)
        ready["start_id"]==startid && ready["requested"]==requested || error("Staged source batch does not match resume state")
        all(isfile(joinpath(dir,"raw","sample_$(i).json")) for i in ready["source_ids"]) || error("Staged source files are missing")
        return dir,Int.(ready["source_ids"])
    end
    # Files without ready.json were never committed. Rebuild only this package's
    # own unfinished staging directory; committed endpoint files are untouched.
    isdir(dir) && rm(dir;recursive=true)
    mkpath(joinpath(dir,"raw")); mkpath(joinpath(out,"sampler"))
    batch_seed=seed+batch*1_000_003
    @everywhere Random.seed!($batch_seed + myid())
    sampling_path=joinpath(out,"sampling_state.json")
    A=Float64[]; b=Float64[]
    if isfile(sampling_path)
        saved=JSON.parsefile(sampling_path)
        A=read_matrix_rows(saved["A_rows"]); b=reshape(Float64.(saved["b"]),:,1)
    end
    source_mode=string(get(config,"source_sampling_mode","parallel"))
    source_mode in ("parallel","serial") || error("source_sampling_mode must be parallel or serial")
    parallel=source_mode=="parallel" && nprocs()>=5
    seconds=Float64(get(config,"opf_max_cpu_time_seconds",120))
    seconds>0 || error("opf_max_cpu_time_seconds must be positive")
    solver=JuMP.optimizer_with_attributes(Ipopt.Optimizer,"tol"=>1e-6,"max_cpu_time"=>seconds)
    kwargs=(save_path=dir,net_path=joinpath(out,"sampler"),save_max_load=true,
        save_certs=true,returnAnb=true,starting_k=startid-1,
        perturb_costs_method="shuffle",perturb_topology_method="none",
        A=A,b=b,print_level=0,opf_solver=solver,r_solver=solver,
        max_iter=Int(get(config,"source_max_iterations_per_batch",max(1000,requested*50))),
        T=Float64(get(config,"source_time_limit_seconds",7200)))
    println("Building source batch $batch: up to $requested ACOPF inputs, mode=$(parallel ? "parallel" : "serial")")
    flush(stdout)
    generator=parallel ? OPFLearn.dist_create_samples : OPFLearn.create_samples
    # OPFLearn was included after entering main(), so invoke its newly loaded
    # methods in the latest world (and load the same module on every worker).
    results,Anb=Base.invokelatest(generator,net,requested;kwargs...)
    # save_certs returns the producer's latest cuts even in distributed mode.
    if haskey(results,"polytope") && haskey(results["polytope"],"A")
        A=results["polytope"]["A"]; b=results["polytope"]["b"]
    else
        A,b=Anb
    end
    save_sampling(joinpath(dir,"sampling_after.json"),A,b,batch)
    ids=Int[]
    for f in readdir(joinpath(dir,"raw"))
        m=match(r"^sample_(\d+)\.json$",f)
        m===nothing || push!(ids,parse(Int,m.captures[1]))
    end
    sort!(ids)
    isempty(ids) && error("OPFLearn produced no ACOPF inputs within this batch's limits; inspect source logs before retrying")
    ids==collect(startid:(startid+length(ids)-1)) || error("Non-contiguous source IDs")
    length(ids)<=requested || error("Source count exceeds requested batch")
    atomic_json(ready_path,Dict("source_batch"=>batch,"start_id"=>startid,
        "requested"=>requested,"source_ids"=>ids,"seed"=>batch_seed,
        "sampling_mode"=>parallel ? "parallel" : "serial"))
    GC.gc()
    dir,ids
end

function write_manifests(out,journals)
    parent_tmp=joinpath(out,"manifests","parents.jsonl.tmp")
    attempt_tmp=joinpath(out,"manifests","attempts.jsonl.tmp")
    open(parent_tmp,"w") do parents
        open(attempt_tmp,"w") do attempts
            for file in journals
                for row in JSON.parsefile(file)["results"]
                    JSON.print(attempts,Dict(k=>v for (k,v) in row if k!="input_features"));write(attempts,'\n')
                    if row["accepted"]
                        JSON.print(parents,row);write(parents,'\n')
                    end
                end
            end
        end
    end
    Base.Filesystem.rename(parent_tmp,joinpath(out,"manifests","parents.jsonl"))
    Base.Filesystem.rename(attempt_tmp,joinpath(out,"manifests","attempts.jsonl"))
end

function implementation_fingerprints()
    files=[joinpath(@__DIR__,name) for name in ("generate_case500.jl","Project.toml","Manifest.toml")]
    for subdir in ("src","lib")
        for (dir,_,names) in walkdir(joinpath(@__DIR__,subdir))
            append!(files,[joinpath(dir,n) for n in names if endswith(n,".jl")])
        end
    end
    Dict(relpath(path,PACKAGE_ROOT)=>case_file_hash(path) for path in sort!(unique(files)))
end

function run_generation(net,config,out,plan,workers;diagnostics=Dict())
    for d in ("raw","sources","journal","manifests","sampler")
        mkpath(joinpath(out,d))
    end
    planpath=joinpath(out,"plan.json")
    configpath=joinpath(out,"generation_config.json")
    provenance=Dict("config"=>config,"plan"=>plan,"workers"=>workers,
        "implementation_sha256"=>implementation_fingerprints(),
        "julia_version"=>string(VERSION),"PowerModels_version"=>string(Base.pkgversion(PowerModels)),
        "NLsolve_version"=>string(Base.pkgversion(PowerModels.NLsolve)))
    if isfile(configpath)
        JSON.parsefile(configpath)==provenance || error("Resume configuration differs; use a new output directory")
    else
        atomic_json(configpath,provenance)
    end
    if isfile(planpath)
        JSON.parsefile(planpath)==plan || error("Saved plan differs from the current generation plan")
    else
        atomic_json(planpath,plan)
    end
    # Record the parser's incidental reference selection separately. A tie in
    # PowerModels' dictionary iteration may choose another automatic reference
    # on another host, while the frozen final reference remains identical.
    diagnostics_path=joinpath(out,"initial_environment_diagnostics.json")
    if !isfile(diagnostics_path)
        atomic_json(diagnostics_path,merge(Dict("recorded_at_utc"=>string(now(UTC)),
            "julia_version"=>string(VERSION),"platform"=>Sys.MACHINE),diagnostics))
    end
    branches=String.(plan["eligible_branch_ids"])
    targets=Dict(b=>Int(plan["branch_parent_targets"][b]) for b in branches)
    counts=Dict(b=>0 for b in branches); attempts=Dict(b=>0 for b in branches)
    seen=Set{String}(); processed=0; nextid=1
    journal_dir=joinpath(out,"journal")
    journals=sort!([joinpath(journal_dir,f) for f in readdir(journal_dir) if occursin(r"^batch_\d+\.json$",f)])
    for (expected,file) in enumerate(journals)
        doc=JSON.parsefile(file); Int(doc["source_batch"])==expected || error("Journal batch sequence has a gap")
        for row in doc["results"]
            bid=string(row["branch_id"]); attempts[bid]+=1; processed+=1
            nextid=max(nextid,Int(row["parent_id"])+1)
            if row["accepted"]
                counts[bid]+=1
                row["input_fingerprint"] in seen && error("Duplicate accepted input in journal")
                push!(seen,row["input_fingerprint"])
                all(isfile(joinpath(out,f)) for f in row["files"]) || error("A committed endpoint file is missing")
            end
        end
    end
    # Complete a checkpoint interrupted after journal commit but before sampler
    # checkpoint replacement. A journal is the authoritative commit record.
    if !isempty(journals)
        lastbatch=length(journals)
        statepath=joinpath(out,"sampling_state.json")
        savedbatch=isfile(statepath) ? Int(JSON.parsefile(statepath)["source_batch"]) : 0
        if savedbatch<lastbatch
            staged=joinpath(out,"sources","batch_$(lpad(lastbatch,6,'0'))","sampling_after.json")
            isfile(staged) || error("Committed batch has no recoverable sampler checkpoint")
            atomic_copy(staged,statepath)
        end
    end
    all(counts[b]<=targets[b] for b in branches) || error("Committed counts exceed configured targets")
    target=Int(plan["target_parents"])
    multiplier=Float64(get(config,"max_candidate_multiplier",5))
    multiplier>=1 || error("max_candidate_multiplier must be >= 1")
    limit=ceil(Int,target*multiplier)
    seed=Int(get(config,"seed",20261014))
    priority=shuffle(MersenneTwister(seed),copy(branches))
    batchsize=Int(get(config,"source_batch_size",128)); batchsize>0 || error("source_batch_size must be positive")
    sum(values(counts))<target && load_environment(workers)
    println("Resume: accepted=$(sum(values(counts)))/$target, processed=$processed, next source ID=$nextid")
    while sum(values(counts))<target && processed<limit
        batch=length(journals)+1
        requested=min(batchsize,target-sum(values(counts)),limit-processed)
        reserved=copy(counts); assigned=String[]
        for _ in 1:requested
            available=[b for b in priority if reserved[b]<targets[b]]
            bid=available[argmin([reserved[b]/targets[b] for b in available])]
            push!(assigned,bid); reserved[bid]+=1
        end
        dir,ids=source_batch(net,config,out,batch,nextid,requested,seed)
        jobs=[(joinpath(dir,"raw","sample_$(sid).json"),out,sid,assigned[i],batch,plan["case_sha256"],plan["reference_bus_ids"])
            for (i,sid) in enumerate(ids)]
        rows=nprocs()>1 ? pmap(Case500Helpers.solve_pair,jobs) : map(Case500Helpers.solve_pair,jobs)
        for row in rows
            bid=string(row["branch_id"]); attempts[bid]+=1; processed+=1
            nextid=max(nextid,Int(row["parent_id"])+1)
            if row["accepted"]
                if row["input_fingerprint"] in seen
                    row["accepted"]=false;row["reason"]="duplicate_fixed_input"
                    # These files are uncommitted products of this attempted pair.
                    for f in row["files"];rm(joinpath(out,f);force=true);end
                    delete!(row,"files");delete!(row,"input_features")
                else
                    push!(seen,row["input_fingerprint"]);counts[bid]+=1
                end
            end
        end
        journal=joinpath(journal_dir,"batch_$(lpad(batch,6,'0')).json")
        atomic_json(journal,Dict("source_batch"=>batch,"results"=>rows))
        push!(journals,journal)
        atomic_copy(joinpath(dir,"sampling_after.json"),joinpath(out,"sampling_state.json"))
        accepted=sum(values(counts))
        atomic_json(joinpath(out,"state.json"),Dict("completed"=>false,
            "phase"=>accepted==target ? "finalizing" : "generating",
            "accepted_parents"=>accepted,"accepted_endpoint_records"=>2*accepted,
            "target_parents"=>target,"processed_sources"=>processed,
            "committed_batches"=>batch,"branch_accepted_counts"=>counts,
            "branch_attempt_counts"=>attempts,"updated_at_utc"=>string(now(UTC))))
        if !Bool(get(config,"keep_sources",false))
            rm(dir;recursive=true)
        end
        println("Committed batch $batch: accepted=$accepted/$target; attempted=$processed/$limit")
        flush(stdout);GC.gc()
    end
    write_manifests(out,journals)
    counts==targets || error("Candidate limit reached with unfilled line quotas. See state.json and manifests/attempts.jsonl; do not silently substitute other lines.")
    finalstate=Dict("completed"=>true,"phase"=>"complete",
        "accepted_parents"=>sum(values(counts)),"accepted_endpoint_records"=>2*sum(values(counts)),
        "target_parents"=>target,"processed_sources"=>processed,
        "committed_batches"=>length(journals),"branch_accepted_counts"=>counts,
        "branch_attempt_counts"=>attempts,"updated_at_utc"=>string(now(UTC)))
    atomic_json(joinpath(out,"state.json"),finalstate)
    println("COMPLETE: $(sum(values(counts))) paired parents / $(2*sum(values(counts))) endpoint records at $out")
end

function main()
    opts=parse_args(ARGS)
    config=JSON.parsefile(abspath(opts["config"]))
    casepath=string(get(config,"case_file","data_generation/pglib/pglib_opf_case500_goc.m"))
    isabspath(casepath) || (casepath=joinpath(PACKAGE_ROOT,casepath))
    isfile(casepath) || error("Case file missing: $casepath")
    raw=PowerModels.parse_file(casepath;validate=false)
    net=PowerModels.parse_file(casepath)
    reference=reference_protocol!(net,raw)
    diagnostics=Dict("PowerModels_automatic_reference_bus_ids"=>pop!(reference,"PowerModels_automatic_reference_bus_ids"),
        "final_reference_bus_ids"=>reference["reference_bus_ids"])
    length(net["bus"])==500 || error("This package expects the supplied 500-bus case")
    all(get(b,"bus_type",1)!=4 for b in values(net["bus"])) || error("Inactive buses are not supported")
    casehash=case_file_hash(casepath)
    plan=network_plan(net,config,opts["mode"],casehash)
    merge!(plan,reference)
    if opts["plan"]
        JSON.print(stdout,plan,2);println();return
    end
    out=isempty(opts["output"]) ? joinpath(PACKAGE_ROOT,"outputs",opts["mode"]) : abspath(opts["output"])
    workers=opts["workers"]===nothing ? Int(get(config,"workers",4)) : Int(opts["workers"])
    run_generation(net,config,out,plan,workers;diagnostics=diagnostics)
end

try
    main()
finally
    nprocs()>1 && rmprocs(workers())
end
