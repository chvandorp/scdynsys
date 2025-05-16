# check the structural identifiability of the model

using StructuralIdentifiability


# ODE model with 2 state variables (and 2 auxiliary variables)

ode2 = @ODEmodel(
    x1'(t) = -z1(t) * x1(t) + q21 * x2(t) - q12 * x1(t),
    x2'(t) = -z2(t) * x2(t) + q12 * x1(t) - q21 * x2(t),
    z1'(t) = -u * (z1(t) - eta1),
    z2'(t) = -u * (z2(t) - eta2),
    y1(t) = x1(t),
    y2(t) = x2(t)
)

assess_identifiability(ode2, p=0.999)


# use eval and Meta.parse to generate the ODE model
# I know, it's not pretty, but it works
function generate_ODE_model(n)
    x_eqn_strings = []
    for i in 1:n
        eqn_str = "x$i'(t) = -z$i(t) * x$i(t)"
        for j in 1:n
            if i != j
                eqn_str *= " + q_$(j)_$i * x$j(t)"
            end
        end
        eqn_str *= " - ("
        plus = ""
        for j in 1:n
            if i != j
                eqn_str *= plus * "q_$(i)_$j"
                plus = " + "
            end
        end
        eqn_str *= ") * x$i(t)"
        push!(x_eqn_strings, eqn_str)
    end
    z_eqn_strings = []
    for i in 1:n
        eqn_str = "z$i'(t) = -u * (z$i(t) - eta$i)"
        push!(z_eqn_strings, eqn_str)
    end
    y_eqn_strings = []
    for i in 1:n
        eqn_str = "y$i(t) = x$i(t)"
        push!(y_eqn_strings, eqn_str)
    end
    eqn_strings = vcat(x_eqn_strings, z_eqn_strings, y_eqn_strings)
    eqn_str = join(eqn_strings, ", ")

    expr = Meta.parse("@ODEmodel($eqn_str)")

    ode = eval(expr)
    
    return ode
end

odes = [
    generate_ODE_model(n)
    for n in 2:12
]

# check the identifiability of the model

ids = [
    assess_identifiability(ode, p=0.999)
    for ode in odes
]

# check the identifiability of the model
for (i, id) in enumerate(ids)
    res = all([id[k] == :globally for k in keys(id)])
    println(i+1, res)
end