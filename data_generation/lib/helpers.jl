module Case500Helpers

using JSON, PowerModels, Random, LinearAlgebra, SHA
export atomic_json, connected, eligible_branches, seed_controls!, input_description,
       solve_pair, numeric_keys, network_plan, audit_pf, case_file_hash

numeric_keys(d) = sort!(collect(keys(d)); by=k -> parse(Int, k))
case_file_hash(path) = bytes2hex(open(sha256, path))

function atomic_json(path, payload)
    mkpath(dirname(path))
    tmp = path * ".tmp." * string(getpid())
    open(tmp, "w") do io
        JSON.print(io, payload)
        write(io, '\n')
    end
    # Same-directory rename atomically replaces a file on the target Linux
    # filesystem. Julia mv(...; force=true) would unlink the old file first.
    # This is process-interruption recovery, not an fsync/power-loss guarantee.
    Base.Filesystem.rename(tmp, path)
end

function connected(net; off="")
    active = [k for (k,b) in net["bus"] if get(b,"bus_type",1) != 4]
    isempty(active) && return false
    adj = Dict(k => String[] for k in active)
    for (k,b) in net["branch"]
        (k == off || get(b,"br_status",1) == 0) && continue
        f,t = string(b["f_bus"]),string(b["t_bus"])
        haskey(adj,f) && haskey(adj,t) || return false
        push!(adj[f],t); push!(adj[t],f)
    end
    seen = Set([first(active)]); todo = [first(active)]
    while !isempty(todo)
        for k in adj[pop!(todo)]
            k in seen && continue
            push!(seen,k); push!(todo,k)
        end
    end
    length(seen) == length(active)
end

function eligible_branches(net)
    connected(net) || error("Base network is disconnected")
    # A unity-ratio transformer is still a transformer. The old case118 tap-only
    # filter is deliberately not used for this case500 protocol.
    [k for k in numeric_keys(net["branch"]) if begin
        b=net["branch"][k]
        get(b,"br_status",1)==1 && !get(b,"transformer",false) &&
        abs(get(b,"tap",1.0)-1.0)<=1e-12 && abs(get(b,"shift",0.0))<=1e-12 &&
        abs(complex(b["br_r"],b["br_x"]))>0 && connected(net; off=k)
    end]
end

function network_plan(net, config, mode, casehash)
    branches = eligible_branches(net)
    isempty(branches) && error("No eligible non-islanding, non-transformer branches")
    train = Int(get(config,"train_endpoint_records",320000))
    train > 0 && iseven(train) || error("train_endpoint_records must be a positive even number")
    val = Int(get(config,"val_parents_per_branch",50))
    test = Int(get(config,"test_parents_per_branch",50))
    min(val,test)>=0 || error("Validation/test counts must be nonnegative")
    if mode == "smoke"
        rng = MersenneTwister(Int(get(config,"seed",20261014)))
        branches = shuffle(rng,branches)[1:min(length(branches),Int(get(config,"smoke_branch_count",4)))]
        sort!(branches; by=k->parse(Int,k))
        n = Int(get(config,"smoke_train_parents_per_branch",2))
        n>0 || error("smoke_train_parents_per_branch must be positive")
        val=Int(get(config,"smoke_val_parents_per_branch",1))
        test=Int(get(config,"smoke_test_parents_per_branch",1))
        quotas = Dict(b=>n+val+test for b in branches)
        train = 2*n*length(branches)
    else
        q,r = divrem(train÷2,length(branches))
        extra = Set(branches[1:r])
        quotas = Dict(b=>q+(b in extra)+val+test for b in branches)
    end
    Dict("case_name"=>"case500", "mode"=>mode, "case_sha256"=>casehash,
        "bus_count"=>length(net["bus"]), "branch_count"=>length(net["branch"]),
        "eligible_branch_count"=>length(branches), "eligible_branch_ids"=>branches,
        "eligible_branches"=>branches,
        "branch_policy"=>"active_nontransformer_nonbridge_zero_phase_shift",
        "train_endpoint_records"=>train, "val_parents_per_branch"=>val,
        "test_parents_per_branch"=>test, "branch_parent_targets"=>quotas,
        "train_parents"=>train÷2,
        "full_train_quota"=>Dict(b=>quotas[b]-val-test for b in branches),
        "train_endpoint_scales"=>mode=="smoke" ? [train] :
            get(config,"train_endpoint_scales",get(config,"train_scales",[10000,20000,40000,45000,50000,55000,60000,80000,90000,100000,110000,120000,160000,320000])),
        "target_parents"=>sum(values(quotas)), "target_endpoint_records"=>2*sum(values(quotas)),
        "input_feature_schema"=>"bus-major [Pd,Qd,Pg_if_PV,Vm_set_if_PV_or_Slack,Va_set_if_Slack,bus_type]",
        "feature_dimension"=>6*length(net["bus"]),
        "input_feature_bus_ids"=>numeric_keys(net["bus"]),
        "rho_values"=>[1.0,0.0], "source_method"=>"OPFLearn_custom_ACOPF_without_CPF",
        "label_method"=>"analytic_Jacobian_damped_Newton_Raphson_BackTracking",
        "label_balance_tolerance"=>1e-7,"fixed_control_tolerance"=>1e-8,
        "stability_certified"=>false,
        "engineering_limit_filter"=>"none; finite_positive_voltages_and_numerical_checks_only")
end

function seed_controls!(net, sol)
    for (k,b) in net["bus"]
        get(b,"bus_type",1)==4 && continue
        s=sol["bus"][k]
        # PowerModels 0.18.4's direct PF fixes reference angle to zero. Its
        # custom ACOPF source also uses theta_ref=0; reject arbitrary nonzero
        # references instead of silently changing the fixed PF input.
        b["bus_type"]==3 && abs(s["va"])>1e-8 && error("Source Slack angle is not zero")
        b["vm"]=s["vm"]; b["va"]=s["va"]
        b["vm_start"]=s["vm"]; b["va_start"]=s["va"]
    end
    for (k,g) in net["gen"]
        get(g,"gen_status",1)==1 || continue
        s=sol["gen"][k]
        g["pg"]=s["pg"]; g["qg"]=s["qg"]
        g["pg_start"]=s["pg"]; g["qg_start"]=s["qg"]
        g["vg"]=sol["bus"][string(g["gen_bus"])]["vm"]
    end
end

function input_description(net, casehash)
    ks=numeric_keys(net["bus"])
    demand=Dict(k=>zeros(2) for k in ks); pg=Dict(k=>0.0 for k in ks)
    for l in values(net["load"])
        get(l,"status",1)==1 || continue
        demand[string(l["load_bus"])] .+= [Float64(l["pd"]),Float64(l["qd"])]
    end
    for g in values(net["gen"])
        get(g,"gen_status",1)==1 || continue
        pg[string(g["gen_bus"])]+=Float64(g["pg"])
    end
    features=Float64[]
    for k in ks
        b=net["bus"][k]; typ=Int(b["bus_type"])
        append!(features,[demand[k][1],demand[k][2],typ==2 ? pg[k] : 0.0,
            typ in (2,3) ? Float64(b["vm"]) : 0.0,typ==3 ? Float64(b["va"]) : 0.0,Float64(typ)])
    end
    all(isfinite,features) || error("Nonfinite fixed inputs")
    # Costs and the unknown PQ voltage / Slack Pg / generator Qg are omitted.
    # Static case hash + active device status prevent false matches between
    # different physical networks. Selected outage and rho are omitted so all
    # variants of the same underlying input are grouped together at splitting.
    status = [[k,Int(get(net["gen"][k],"gen_status",1))] for k in numeric_keys(net["gen"])]
    fingerprint=bytes2hex(sha256(JSON.json([casehash,ks,features,status])))
    features,fingerprint
end

function apply_endpoint!(net,bid,rho)
    rho in (0.0,1.0) || error("Only rho endpoints are implemented")
    for b in values(net["branch"])
        for f in ("br_r","br_x","g_fr","b_fr","g_to","b_to","br_status")
            b[f*"_base"]=b[f]
        end
        b["rho"]=1.0; b["rho_controlled"]=false
    end
    b=net["branch"][bid]
    b["br_status"]==1 || error("Chosen source branch is inactive")
    b["rho"]=rho; b["rho_controlled"]=true
    b["br_status"]=rho==0.0 ? 0 : 1
end

function audit_pf(net,result; residual_tolerance=1e-7,control_tolerance=1e-8)
    sol=result["solution"]; ks=numeric_keys(net["bus"])
    ix=Dict(k=>i for (i,k) in enumerate(ks))
    vm=Float64[sol["bus"][k]["vm"] for k in ks]
    va=Float64[sol["bus"][k]["va"] for k in ks]
    all(isfinite,vm) && all(isfinite,va) && minimum(vm)>0 || error("Nonfinite or nonpositive voltage")
    v=vm .* exp.(im .* va)
    power=zeros(ComplexF64,length(ks)); gen=copy(power); load=copy(power)
    expected_pg=zeros(length(ks)); flow=Dict{String,Any}()
    for (k,b) in net["branch"]
        if get(b,"br_status",1)==0
            flow[k]=Dict("pf"=>0.0,"qf"=>0.0,"pt"=>0.0,"qt"=>0.0)
            continue
        end
        f=ix[string(b["f_bus"])]; t=ix[string(b["t_bus"])]
        y=inv(complex(b["br_r"],b["br_x"])); tap=b["tap"]*exp(im*b["shift"])
        yff=(y+complex(b["g_fr"],b["b_fr"]))/abs2(tap)
        ytt=y+complex(b["g_to"],b["b_to"])
        sf=v[f]*conj(yff*v[f]-y/conj(tap)*v[t])
        st=v[t]*conj(-y/tap*v[f]+ytt*v[t])
        power[f]+=sf; power[t]+=st
        flow[k]=Dict("pf"=>real(sf),"qf"=>imag(sf),"pt"=>real(st),"qt"=>imag(st))
    end
    for s in values(net["shunt"])
        get(s,"status",1)==1 || continue
        i=ix[string(s["shunt_bus"])]; power[i]+=vm[i]^2*complex(s["gs"],-s["bs"])
    end
    for l in values(net["load"])
        get(l,"status",1)==1 || continue
        load[ix[string(l["load_bus"])]]+=complex(l["pd"],l["qd"])
    end
    for (k,g) in net["gen"]
        get(g,"gen_status",1)==1 || continue
        i=ix[string(g["gen_bus"])]; s=sol["gen"][k]
        gen[i]+=complex(s["pg"],s["qg"]); expected_pg[i]+=g["pg"]
    end
    pgerr=0.0; vmerr=0.0; vaerr=0.0
    for (k,b) in net["bus"]
        i=ix[k]; typ=Int(b["bus_type"])
        typ!=3 && (pgerr=max(pgerr,abs(real(gen[i])-expected_pg[i])))
        typ in (2,3) && (vmerr=max(vmerr,abs(vm[i]-b["vm"])))
        typ==3 && (vaerr=max(vaerr,abs(va[i]-b["va"])))
    end
    residual=power-gen+load
    all(isfinite,residual) || error("Nonfinite power balance")
    err=max(maximum(abs.(real.(residual))),maximum(abs.(imag.(residual))))
    err<=residual_tolerance || error("Power balance residual $err exceeds $residual_tolerance")
    max(pgerr,vmerr,vaerr)<=control_tolerance || error("Fixed PF controls changed")
    connected(net) || error("Disconnected endpoint")
    sol["branch"]=flow
    Dict("max_balance_residual"=>err,"fixed_bus_pg_error"=>pgerr,
        "fixed_vm_error"=>vmerr,"fixed_slack_va_error"=>vaerr,
        "min_vm"=>minimum(vm),"max_vm"=>maximum(vm))
end

function solve_newton(net)
    result=PowerModels.compute_ac_pf(net;flat_start=false,finite_differencing=false,
        method=:newton,linesearch=PowerModels.NLsolve.LineSearches.BackTracking(),
        ftol=1e-9,xtol=0.0,iterations=100)
    get(result,"termination_status",false)===true || error("Newton AC-PF did not converge")
    qa=audit_pf(net,result)
    merge!(result,Dict("pf_termination_status"=>true,"termination_status"=>"PF_CONVERGED",
        "solver_type"=>"AC_PF","solver_method"=>"newton",
        "numerical_algorithm"=>"damped_Newton_Raphson_with_BackTracking",
        "solver_linesearch"=>"BackTracking","solver_jacobian"=>"analytic",
        "solver_ftol"=>1e-9,"solver_xtol"=>0.0,"solver_max_iterations"=>100,
        "solver_initialization"=>"source_ACOPF_solution_fixed_controls",
        "solver_fallback"=>"none","validation"=>qa))
    result,qa
end

function solve_pair(job)
    sourcefile,outdir,sid,bid,batch,casehash,expected_refs=job
    row=Dict{String,Any}("parent_id"=>sid,"load_setting_id"=>sid,
        "source_sample_id"=>sid,"source_batch"=>batch,"branch_id"=>bid,"accepted"=>false)
    try
        row["source_sha256"]=case_file_hash(sourcefile)
        src=JSON.parsefile(sourcefile); base=deepcopy(src["network"])
        refs=[k for k in numeric_keys(base["bus"]) if base["bus"][k]["bus_type"]==3]
        refs==expected_refs || error("Source Slack differs from the frozen reference bus")
        isempty(get(base,"dcline",Dict())) || error("DC lines are not supported in this AC-only audit")
        seed_controls!(base,src["solution"]["solution"])
        features,fingerprint=input_description(base,casehash)
        row["input_fingerprint"]=fingerprint
        normal=deepcopy(base); outage=deepcopy(base)
        apply_endpoint!(normal,bid,1.0); apply_endpoint!(outage,bid,0.0)
        normal["load"]==outage["load"] && normal["gen"]==outage["gen"] &&
        normal["bus"]==outage["bus"] && normal["shunt"]==outage["shunt"] || error("Paired controls differ")
        nr,nq=solve_newton(normal); ores,oq=solve_newton(outage)
        input_description(normal,casehash)[2]==fingerprint &&
        input_description(outage,casehash)[2]==fingerprint || error("Input changed during PF solve")
        files=String[]
        for (rho,net,res,sampleid) in ((1.0,normal,nr,2*sid-1),(0.0,outage,ores,2*sid))
            rel=joinpath("raw","sample_$(sampleid).json"); push!(files,rel)
            b=net["branch"][bid]
            meta=Dict{String,Any}("case_name"=>"case500","sample_id"=>sampleid,
                "load_setting_id"=>sid,"base_sample_id"=>sid,"parent_id"=>sid,
                "branch_id"=>bid,"f_bus"=>b["f_bus"],"t_bus"=>b["t_bus"],
                "rho"=>rho,"rho_grid"=>"0.0,1.0","rho_mode"=>"endpoint-paired",
                "solver_type"=>"AC_PF","solver_method"=>"newton","solver_fallback"=>"none",
                "source_topology"=>"n","source_batch"=>batch,
                "input_fingerprint"=>fingerprint,"source_sha256"=>row["source_sha256"],
                "task_name"=>"case500_endpoint_NR_v1")
            for f in ("br_r","br_x","g_fr","b_fr","g_to","b_to","br_status")
                meta[f*"_base"]=b[f*"_base"]
            end
            atomic_json(joinpath(outdir,rel),Dict("network"=>net,"solution"=>res,
                "parent_seed"=>get(src,"parent_seed",-1),"rho_meta"=>meta))
        end
        merge!(row,Dict("accepted"=>true,"files"=>files,"rho_values"=>[1.0,0.0],
            "input_features"=>features,"normal_validation"=>nq,"outage_validation"=>oq))
    catch err
        row["reason"]="pair_rejected"; row["error"]=sprint(showerror,err)
    end
    row
end

end # module
