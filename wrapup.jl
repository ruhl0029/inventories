function plot_vfuncs(ei, s, v, params)
    
    N_shocks, N_inv = params["n_nodes_e"], params["n_nodes_s"]+2;
    start, stop = (ei-1)*N_inv+1, ei*N_inv;
    lw = 2.5;
    
    # Plot the value functions
    plt1  = plot(s[start:stop, 1], v[start:stop, 1],
        label="Value adjusting", title="Value functions", linewidth=lw);

    plot!(s[start:stop, 1], v[start:stop, 2],
        label="Value not adjusting", title="Value functions", linewidth=lw);
    
    xl, xh = Plots.xlims(plt1)
    yl, yh = Plots.ylims(plt1)
    annotate!(xh*0.95, yl+(yh-yl)*0.25, text("For demand shock $ei of $N_shocks", :right, 10))

    plot!(legend=:right, foreground_color_legend = nothing, xlabel="Inventory");
    
    # Plot the order thresholds
    cutoffs = zeros(N_shocks, 2);
    adjust_01 = v[:,1].>=v[:,2];

    for ei in 1:N_shocks
        start, stop = (ei-1)*N_inv+1, (ei)*N_inv;
        for si in stop:-1:start
            if (adjust_01[si,1]==1)
                cutoffs[ei,1], cutoffs[ei,2] = s[si,1], s[si,2];
                break;
            end
        end
    end

    plt2 = plot(cutoffs[:,1], cutoffs[:,2], xlabel="Inventory", ylabel="log demand", legend=false, linewidth=lw,
    title="Order thresholds");
    xl, xh = Plots.xlims(plt2)
    yl, yh = Plots.ylims(plt2)
    annotate!(xl+(xh-xl)*0.05, yh*0.95, text("Place an order", :left, 10))
    annotate!(xh*0.95, yl+(yh-yl)*0.05, text("Do not place an order", :right, 10))
    
    
    display(plot(plt1, plt2, size=(900,400)));
end

################################################################################
function plot_policy_funcs(ei, s, xa, xn, v, params)
    
    N_shocks, N_inv = params["n_nodes_e"], params["n_nodes_s"]+2;
    start, stop = (ei-1)*N_inv+1, ei*N_inv;
    adjust_01 = v[:,1].>=v[:,2];
    
    
    lw = 2.5;
        
    # Prices
    plt1  = plot(s[start:stop, 1], xa[start:stop, 1],
       label="Price if adjusting", title="Price functions", linewidth=lw, markershape=:circle,);
     
    plot!(s[start:stop, 1], xn[start:stop, 1],
       label="Price if not adjusting", linewidth=lw, markershape=:circle,);
    
    xl, xh = Plots.xlims(plt1)
    yl, yh = Plots.ylims(plt1)
    annotate!(xh*0.95, yh*0.9, text("For demand shock $ei of $N_shocks", :right, 10))
        
    plot!(legend=:right, foreground_color_legend = nothing);
    
    plt3 = plot(s[start:stop, 1], xa[start:stop, 1].*adjust_01[start:stop] + xn[start:stop, 1].*(1 .- adjust_01[start:stop]),
           title="Price functions", linewidth=lw, markershape=:circle,legend=false, xlabel="Inventory");
    
    # Order size
    plt2  = plot(s[start:stop, 1], xa[start:stop, 2], markershape=:circle,
            title="Order size if adjusting", linewidth=lw, legend=false);
        
    plt4 = plot(s[start:stop, 1], xa[start:stop, 2].*adjust_01[start:stop], markershape=:circle,
            title="Order size", linewidth=lw, legend=false, xlabel="Inventory");
        
    display(plot(plt1, plt2, plt3, plt4, size=(900,650)));
end