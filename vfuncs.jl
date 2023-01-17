function solve_v(s, e, w, fspace, params)
    sigma, omega, beta, ns = params["sigma"], params["omega"], params["beta"], params["ns"]
    vfi_its, pfi_max_its, vfi_tol, pfi_tol = params["vfi_its"], params["pfi_max_its"], params["vfi_tol"], params["pfi_tol"]

    # Compute the basis coefficients that approximate the three value functions `[va, va, va]` at the points `s`.
    # Then compute the intial guess of the value function in this interpolation scheme.
    reward = 1 / sigma
    va = omega * s[:, 1] .+ exp.(s[:, 2]) ./ sigma .+ beta * reward / (1 - beta)
    cnew = funfitxy(fspace, s, [va va va])[1]
    vnew = funeval(cnew, fspace, s)[1][:, :]

    # VFI for a few iterations to get a good guess for Newton's method
    for it = 1:vfi_its
        cold = copy(cnew)
        vold = copy(vnew)

        v1, v2 = saveBelmax(:not, cold, fspace, s, e, w, params)
        cnew[:, 1:2] = funfitxy(fspace, s, [v1 v2])[1]
        v3 = valfunc3(cnew, fspace, s, [], e, w, params)
        cnew[:, 3] = funfitxy(fspace, s, v3)[1]

        vnew = funeval(cnew, fspace, s)[1][:, :]

        println([it, round(norm(vold - vnew), digits=2)])
        if (norm(vold - vnew) < vfi_tol) | (it == vfi_its)
            #CSV.write("vnew.csv", Tables.table(vnew), writeheader=false);
            break
        end
    end


    # PFI using Newton's method
    cnew = cnew[:]    # transform array cnew to a vector (column-major ordering)
    for it = 1:pfi_max_its
        cold = copy(cnew)
        vold = copy(vnew)

        bel, beljac = solvebel(cold, fspace, s, e, w, params)
        cnew = cold - (beljac \ bel)

        vnew = funeval([cnew[1:ns, :] cnew[ns+1:2*ns, :] cnew[2*ns+1:3*ns, :]], fspace, s)[1]

        println([it round(norm(vold - vnew), sigdigits=2)])

        if norm(vold - vnew) < pfi_tol
            CSV.write("vnew-kr.csv", Tables.table(vnew[:, :]), writeheader=false)
            break
        end

    end

    return vnew, [cnew[1:ns, :] cnew[ns+1:2*ns, :] cnew[2*ns+1:3*ns, :]]
end

################################################################################
function saveBelmax(case::Symbol, c, fspace, s, e, w, params)
    # In the :not case, given c, update the policy function with solve 1 and solve 2
    # then compute the new value functions v1 and v2 given the update policy function.
    # In the :full case do everything in the :not case but also compute the derivatives of v wrt c.

    beta, ns = params["beta"], params["ns"]

    # Given c, find optimal policy when ordering (xadjust)
    x = solve1(c, fspace, s, [], e, w, params)
    xadjust = [x[1] x[2]]
    v1 = valfunc1(c, fspace, s, xadjust, e, w, params)

    # Given c, find optimal policy when not ordering (xnadjust)
    x = solve2(c, fspace, s, x, e, w, params)
    xnadjust = x
    v2 = valfunc2(c, fspace, s, xnadjust, e, w, params)

    if case != :full
        return v1, v2, xadjust, xnadjust
    end


    if case == :full

        # Given c, compute V3
        v3 = valfunc3(c, fspace, s, [], e, w, params)

        v13, v23, v31, v32 = zeros(Float64, ns, ns), zeros(Float64, ns, ns), zeros(Float64, ns, ns), zeros(Float64, ns, ns)

        # Derivative of V1 wrt the coefficents in V3
        g = law_of_motion_adj(s, xadjust, params)
        v13 = beta .* funbase(fspace, g)

        # Derivative of V2 wrt the coefficents in V3
        g = law_of_motion_nadj(s, xnadjust, params)
        v23 = beta .* funbase(fspace, g)

        # Derivatives of V3 wrt to the coefficents in V1 and V2
        # Only take the coefficents from the V1 or V2 for the chosen action.
        for k in eachindex(w)

            g = [s[:, 1] e[k] .* ones(ns, 1)]

            vint1 = funeval(c[:, 1], fspace, g)[1]
            vint2 = funeval(c[:, 2], fspace, g)[1]

            c1 = vint1 .>= vint2
            c2 = vint2 .> vint1

            v31 = v31 .+ w[k] .* (funbase(fspace, g) .* repeat(c1, 1, ns))
            v32 = v32 .+ w[k] .* (funbase(fspace, g) .* repeat(c2, 1, ns))

        end
        return v1, v2, v3, v13, v23, v31, v32
    end

end

################################################################################
function solvebel(c, fspace, s, e, w, params)

    ns = params["ns"]
    c = [c[1:ns, :] c[ns+1:2*ns, :] c[2*ns+1:3*ns, :]]

    # saveBelmax returns the optimized Bellman equations (RHS) v1, v2, v3 and the derivatives
    v1, v2, v3, v13, v23, v31, v32 = saveBelmax(:full, c, fspace, s, e, w, params)

    # The stacked up Bellman equations: LHS-RHS
    bel = zeros(Float64, 3 * ns, 1)
    LHS = funeval(c, fspace, s)[1]                                         # LHS of Bellman equation

    bel[1:ns, :] = LHS[:, 1] .- v1                                  # LHS-RHS of Bellman equation for adjustment
    bel[ns+1:2*ns, :] = LHS[:, 2] .- v2                                  # LHS-RHS of bellman equation for non-adjustment
    bel[2*ns+1:3*ns, :] = LHS[:, 3] .- v3                                  # LHS-RHS of bellman equation exp value of firm

    # The Jacobian matrix of bel wrt. c
    beljac = zeros(Float64, 3 * ns, 3 * ns)

    beljac[1:ns, :] = [params["Phi"] zeros(ns, ns) -v13]
    beljac[ns+1:2*ns, :] = [zeros(ns, ns) params["Phi"] -v23]
    beljac[2*ns+1:3*ns, :] = [-v31 -v32 params["Phi"]]

    return bel, beljac

end

################################################################################
function solve1(c, fspace, s, x, e, w, params)
    # Compute order size and price, conditional on making an order
    # TODO: In the inner loop over prices, we solve for all the optimal order size
    # even though some of them are not not relevant to the "case". Maybe it speeds up
    # if we only run best_order_size() of the subset of states which need solutions?

    tolp, toli = params["tolp"], params["toli"]
    alpha1 = (3 - sqrt(5)) / 2
    alpha2 = (sqrt(5) - 1) / 2

    # Bounds for prices and inventories 
    ap, bp = get_price_bounds(params)
    ai, bi = get_inventory_bounds(s, params)

    # The initial trial points for prices and order sizes. We always use the same inital 
    # bounds when searching in order-size space, so these show up often below. 
    x1p0, x2p0, dp = init_guess(ap, bp, alpha1, alpha2)
    x1i0, x2i0, di = init_guess(ai, bi, alpha1, alpha2)

    # Find optimal order size given initial lower price bound (x1p0)
    xi = best_order_size(x1p0, di, x1i0, x2i0, toli, alpha2, c, fspace, s, e, w, params)
    f1p = valfunc1(c, fspace, s, [x1p0 xi], e, w, params)

    # Find optimal order size given initla upper price bound (x2p0)
    xi = best_order_size(x2p0, di, x1i0, x2i0, toli, alpha2, c, fspace, s, e, w, params)
    f2p = valfunc1(c, fspace, s, [x2p0 xi], e, w, params)


    # This is nested golden search. The outer loop is searching over price space.
    # For each given candidate price, search over order size (inner loop).
    f1pnew, f2pnew = f1p, f2p
    x1pnew, x2pnew = x1p0, x2p0

    # In need to declare these outside of the loop to make them available after the loop.
    xic1, xic2 = Array{Float64}(undef, params["ns"], 1), Array{Float64}(undef, params["ns"], 1)

    while sum(dp .> tolp) > 0
        f1p, f2p = f1pnew, f2pnew
        x1p, x2p = x1pnew, x2pnew
        dp = dp * alpha2

        # Find the next trial point. Note the two cases: 
        #   1. When f2p < f1p:  x2pnew = x1p and x1pnew = x1p-dp 
        #   2. When f2p >= f1p: x2pnew = x2p+dp and x1pnew = x2p 
        x2pnew = x1p .* (f2p .< f1p) .+ (x2p + dp) .* (f2p .>= f1p)
        x1pnew = (x1p - dp) .* (f2p .< f1p) .+ x2p .* (f2p .>= f1p)

        # Case 1
        # For the states in which f2p < f1p, need to find optimal order size for x1pnew = x1p-dp
        xic1 = best_order_size(x1p .- dp, di, x1i0, x2i0, toli, alpha2, c, fspace, s, e, w, params)
        f1pnew = valfunc1(c, fspace, s, [x1p .- dp xic1], e, w, params) .* (f2p .< f1p) .+ f2p .* (f2p .>= f1p)

        # Case 2
        # For the states in which f2p >= f1p, need to find optimal order size for x2pnew = x2p+dp
        xic2 = best_order_size(x2p .+ dp, di, x1i0, x2i0, toli, alpha2, c, fspace, s, e, w, params)
        f2pnew = f1p .* (f2p .< f1p) .+ valfunc1(c, fspace, s, [x2p .+ dp xic2], e, w, params) .* (f2p .>= f1p)

    end
    # These are the optimal xp and xi
    xp = x1pnew .* (f1pnew .> f2pnew) .+ x2pnew .* (f1pnew .<= f2pnew)
    xi = xic1 .* (f1pnew .> f2pnew) .+ xic2 .* (f1pnew .<= f2pnew)

    return [xp, xi]
end

################################################################################
function init_guess(a, b, alpha1, alpha2)
    # Return initial trial points (x1, x2) and the inerval length (d).
    d = b .- a
    return a .+ alpha1 * d, a .+ alpha2 * d, alpha1 * alpha2 * d
end

################################################################################
function best_order_size(xp, di, x1inew, x2inew, toli, alpha2, c, fspace, s, e, w, params)
    # Given price xp, and the two x and two f values that bracket the max, 
    # find the order size xi that maximizes the valfunc1. Uses golden section search.

    f1inew = valfunc1(c, fspace, s, [xp x1inew], e, w, params)
    f2inew = valfunc1(c, fspace, s, [xp x2inew], e, w, params)

    while sum(di .> toli) > 0
        f1i, f2i = f1inew, f2inew
        x1i, x2i = x1inew, x2inew
        di = di * alpha2

        x2inew = x1i .* (f2i .< f1i) .+ (x2i .+ di) .* (f2i .>= f1i)
        f2inew = f1i .* (f2i .< f1i) .+ valfunc1(c, fspace, s, [xp x2i .+ di], e, w, params) .* (f2i .>= f1i)

        x1inew = (x1i .- di) .* (f2i .< f1i) .+ x2i .* (f2i .>= f1i)
        f1inew = valfunc1(c, fspace, s, [xp x1i .- di], e, w, params) .* (f2i .< f1i) .+ f2i .* (f2i .>= f1i)
    end

    xi = x2inew .* (f2inew .>= f1inew) .+ x1inew .* (f2inew .< f1inew)

    return xi
end

################################################################################
function solve2(c, fspace, s, x, e, w, params)
    # Use golden section search to find the optimal response if not placing an order. 
    # The only choice is price.
    # We are choosing price to maximize valfunc2().

    # Get bounds of price and set constants
    ap, bp = get_price_bounds(params)
    tolp = params["tolp"]
    alpha1 = (3 - sqrt(5)) / 2
    alpha2 = (sqrt(5) - 1) / 2

    # Initial guess
    d = bp .- ap
    x1new = ap .+ alpha1 * d
    x2new = ap .+ alpha2 * d
    f1new = valfunc2(c, fspace, s, x1new, e, w, params)
    f2new = valfunc2(c, fspace, s, x2new, e, w, params)

    d = alpha1 * alpha2 * d

    while sum(d .> tolp) > 0
        f1, f2 = f1new, f2new
        x1, x2, = x1new, x2new
        d = d * alpha2

        x2new = x1 .* (f2 .< f1) .+ (x2 .+ d) .* (f2 .>= f1)
        f2new = f1 .* (f2 .< f1) .+ valfunc2(c, fspace, s, x2 .+ d, e, w, params) .* (f2 .>= f1)

        x1new = (x1 .- d) .* (f2 .< f1) .+ x2 .* (f2 .>= f1)
        f1new = valfunc2(c, fspace, s, x1 .- d, e, w, params) .* (f2 .< f1) .+ f2 .* (f2 .>= f1)
    end

    x = x2new .* (f2new .>= f1new) .+ x1new .* (f2new .< f1new)

    return x

end

################################################################################
function valfunc1(c, fspace, s, x, e, w, params)
    # For a given policy x, return V1, the value of adjusting.
    beta = params["beta"]
    g = law_of_motion_adj(s, x, params)

    return profit_adj(s, x, params) .+ beta .* funeval(c[:, 3], fspace, g)[1]
end

################################################################################
function valfunc2(c, fspace, s, x, e, w, params)
    # For a given policy x, return the V2, the value of not adjusting.
    beta = params["beta"]
    g = law_of_motion_nadj(s, x, params)
    return profit_nadj(s, x, params) .+ beta .* funeval(c[:, 3], fspace, g)[1]
end

################################################################################
function valfunc3(c, fspace, s, x, e, w, params)
    # Computes the expected value (E_e') of the max over adjust and not adjust vfuncs
    #                V3 = E_e' max{V1(g), V2(g)}
    # Since e is iid, it does not depend on current e.   
    # V3 has size = ns x 1, but the value of V3 is the same for every s, regardless
    # of the value of e. 

    K = length(w)
    ns = params["ns"]
    v = zeros(Float64, ns, 1)

    for k in 1:K
        g = [s[:, 1] e[k] .* ones(ns, 1)]
        v = v .+ w[k] .* maximum(funeval(c[:, 1:2], fspace, g)[1][:, 1:2], dims=2)
    end

    return v
end

################################################################################
function law_of_motion_nadj(s, x, params)
    # Do not order. 
    Pm, Cm, sigma, delta = params["Pm"], params["Cm"], params["sigma"], params["delta"]
    v = exp.(s[:, 2])

    p = x[:, 1]
    q = v .* minimum([(p ./ Pm) .^ (-sigma) .* Cm s[:, 1] ./ v], dims=2)

    return [(1 - delta) * (s[:, 1] .- q) s[:, 2]]
end

################################################################################
function law_of_motion_adj(s, x, params)
    # Place and order.
    Pm, Cm, sigma, delta = params["Pm"], params["Cm"], params["sigma"], params["delta"]
    v = exp.(s[:, 2])

    p = x[:, 1]
    i = x[:, 2]

    q = v .* minimum([(p ./ Pm) .^ (-sigma) .* Cm (s[:, 1]) ./ v], dims=2)

    return [(1 - delta) * (s[:, 1] .- q + i) s[:, 2]]
end

################################################################################
function profit_nadj(s, x, params)
    # Profit when not ordering. 
    Pm, Cm, sigma = params["Pm"], params["Cm"], params["sigma"]
    v = exp.(s[:, 2])

    p = x[:, 1]
    q = v .* minimum([(p ./ Pm) .^ (-sigma) .* Cm s[:, 1] ./ v], dims=2)

    return p .* q
end

################################################################################    
function profit_adj(s, x, params)
    # Profit when ordering. 
    Pm, Cm, sigma, omega, f = params["Pm"], params["Cm"], params["sigma"], params["omega"], params["f"]
    v = exp.(s[:, 2])

    p = x[:, 1]
    i = x[:, 2]

    q = v .* minimum([(p ./ Pm) .^ (-sigma) .* Cm (s[:, 1]) ./ v], dims=2)

    return p .* q .- omega .* i .- f
end

################################################################################
function get_inventory_bounds(s, params)
    # Return the bounds that bracket the inventory argmax. Used in the golden search methods. 

    smin, smax, ns = params["smin"], params["smax"], params["ns"]
    v = exp.(s[:, 2])

    invlow = maximum([smin[1] .* ones(ns, 1) .- (maximum([s[:, 1] .- v zeros(ns, 1)], dims=2)) zeros(ns, 1)], dims=2)
    invhig = maximum([smax[1] .* ones(ns, 1) .- (maximum([s[:, 1] .- v zeros(ns, 1)], dims=2)) zeros(ns, 1)], dims=2)

    return invlow, invhig
end

function get_price_bounds(params)
    # Return the bounds that bracket the price argmax. Used in the golden search methods. 
    return 0.1 .* ones(params["ns"], 1), 5.0 .* ones(params["ns"], 1)
end