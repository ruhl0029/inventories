using Distributions, CompEcon, Plots, LinearAlgebra, CSV, Tables
include("setup.jl");
include("vfuncs.jl");
include("wrapup.jl")

function main()    
    pd = 12;                            # frequency (12 = monthly)
    beta = 0.94^(1/pd);                 # discount factor
    sigma = 1.5;                        # price Elasticity  
    omega = (sigma-1)/sigma;            # import price; normalized so unconstarined price = 1
    delta = 0.025;                      # rate at which inventory  depreciates
    sigmav = 1.15;                      # standard deviation of Gaussian demand shocks
    f = 0.095;                          # the fixed cost of ordering
    Cm = 1;                             # partial equilibrium, normalize to 1                          
    Pm = 1;                             # partial equilibrium, normalize to 1
    smin_val = 1e-4;                    # minimum inventory grid point
    smax_val = 21;                      # maximum inventory grid point
    
    vfi_its = 3;                        # number of vfi iterations in solve_v()
    pfi_max_its = 30;                   # max number of pfi iterations in solve_v()
    vfi_tol = 1e-5;                     # tolerance in vfi loop in solve_v()
    pfi_tol = 1e-7;                     # tolerance in pfi loop in solve_v()
    tolp = 1e-5;                        # tolerance for price in golden search
    toli = 1e-5                         # tolerance for order size in golden search
    n_nodes_s = 31;                     # number of colocation nodes for inventory
    n_nodes_e = 21;                     # number of colocation nodes for demand shocks
    nv = 21;                            # number of points to pass to init_shocks
    ns = (n_nodes_s+2)*n_nodes_e;       # number of possible states; splines add 2 points to n_nodes_s 
    
    params = Dict{String, Any}("sigma"=>sigma, "beta"=>beta, "omega"=>omega, "f"=>f, "Cm"=>Cm, "Pm"=>Pm,
                               "delta"=>delta, "vfi_its"=>vfi_its, "pfi_max_its"=>pfi_max_its, "vfi_tol"=>vfi_tol, 
                               "pfi_tol"=>pfi_tol, "tolp"=>tolp, "toli"=>toli, "smin_val"=>smin_val, 
                               "smax_val"=>smax_val, "n_nodes_s"=>n_nodes_s, "n_nodes_e"=>n_nodes_e, "nv"=>nv, "ns"=>ns); 

    
    # Step 1.
    e, w = init_shocks(sigmav, nv);
    
    # Step 2.
    fspace, s, params["smin"], params["smax"], params["Phi"] = init_state_space(smin_val, smax_val, e[1], e[end], 0, params);
     
    # Step 3.
    vnew, cnew = solve_v(s, e, w, fspace, params);
    
    # Step 4. 
    med_shock = Int(median(range(1, n_nodes_e)));
    plot_vfuncs(med_shock, s, vnew, params);
    
    v1, v2, xa, xn = saveBelmax(:not, cnew, fspace, s, e, w, params)
    plot_policy_funcs(med_shock, s, xa, xn, vnew, params);
      
    return e, w, fspace, s, vnew, cnew, params, xa, xn;
    
end

e, w, fspace, s, vnew, cnew, params, xa, xn = main();