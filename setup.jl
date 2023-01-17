function init_shocks(vbar, k)
    d = Normal(0, vbar);
    e, w = qnwnorm(k, 0, vbar^2);          # e are the nodes and w the weights
    w = w[abs.(e).<=quantile(d, 0.999)];   # quantile(x,y): compute quantile y of vector x
    e = e[abs.(e).<=quantile(d, 0.999)];
    w /= sum(w);
    return e, w;
end

################################################################################
function init_state_space(smin_val, smax_val, emin_val, emax_val, showme, params)
    smin=[smin_val, emin_val];    
    smax=[smax_val, emax_val];    
    sigma, n_nodes_s, n_nodes_e = params["sigma"], params["n_nodes_s"], params["n_nodes_e"];    
    scale = 1-1/sigma;    
    
    # Plot the colocation points to see how scale operates
    if showme == 1 plot_scale_diagram(smin, smax, scale, n_nodes[1]); end
    
    fspace = fundef([:spli, range(smin[1]^scale, smax[1]^scale, length=n_nodes_s).^(1/scale), 0, 3], 
                    [:lin,  range(smin[2], smax[2],             length=n_nodes_e), 0]);
    
    s   = funnode(fspace)[1];        # matrix of all possible combinations of inventory and demand
    Phi = funbase(fspace, s);      # basis functions evaluated at the colocation nodes
    
    return fspace, s, smin, smax, Phi;
    
end

################################################################################
function plot_scale_diagram(smin, smax, scale, Ns)
    plt  = plot(range(smin[1], smax[1], length=Ns), range(0, 0, length=Ns), seriestype=:scatter, 
       label="Original, uniform spaced", title="The effect of scale on the placement of nodes")
       
   plot!(range(smin[1]^scale, smax[1]^scale, length=Ns), range(0.1, 0.1, length=31), 
           seriestype=:scatter, label="Scale endpoints, uniform spaced")
       
   plot!(range(smin[1]^scale, smax[1]^scale, length=Ns).^(1/scale), range(0.2, 0.2, length=Ns), 
           seriestype=:scatter, label="Scaled")
       
   plot!(legend=:right, foreground_color_legend = nothing)
   display(plt)
end