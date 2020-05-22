import numpy as np
from tqdm import tqdm


def minimize(function, x0, n_iters, coef_r=1, coef_e=2, coef_ci=-0.5, coef_co=0.5, coef_s=0.5):
    n_dims = len(x0)
    simplex = np.zeros((n_dims + 1, n_dims), dtype=x0.dtype)
    simplex[0] = x0
    for k in range(n_dims):
        y = np.array(x0, copy=True)
        if y[k] != 0:
            y[k] = (1 + 0.1) * y[k]
        else:
            y[k] = 0.005
        simplex[k + 1] = y

    simplex_values = np.array([function(x) for x in simplex])
    simplex_traj = []
    simplex_traj_mae = []
    for _ in tqdm(range(n_iters)):
        simplex_traj_mae.append(np.mean(np.abs(simplex_values - 0.0)))
        print(np.min(simplex_values))
        sort_idx = np.argsort(simplex_values)
        simplex_values = simplex_values[sort_idx]
        simplex = simplex[sort_idx]

        simplex_centroid = np.mean(simplex[:-1], axis=0)
        reflect_point = generate_point(simplex, simplex_centroid, coef_r)
        reflect_val = function(reflect_point)

        if reflect_val < simplex_values[-2]:
            simplex_values[-1] = reflect_val
            simplex[-1] = reflect_point
            simplex_traj.append(simplex)
            continue
        if reflect_val < simplex_values[0]:
            expand_point = generate_point(simplex, simplex_centroid, coef_e)
            expand_value = function(expand_point)

            if expand_value < reflect_val:
                simplex_values[-1] = expand_value
                simplex[-1] = expand_point
                simplex_traj.append(simplex)
                continue
            else:
                simplex_values[-1] = reflect_val
                simplex[-1] = reflect_point
        elif reflect_val > simplex_values[-2]:
            if reflect_val <= simplex_values[-1]:
                contract_point = generate_point(simplex, simplex_centroid, coef_co)
                contract_value = function(contract_point)
                if contract_value < reflect_val:
                    simplex_values[-1] = contract_value
                    simplex[-1] = contract_point
                    simplex_traj.append(simplex)
                    continue
                simplex[-1] = reflect_point
                simplex_values = reflect_val
            elif reflect_val > simplex_values[-1]:
                contract_point = generate_point(simplex, simplex_centroid, coef_ci)
                contract_value = function(contract_point)
                if contract_value < simplex_values[-1]:
                    simplex_values[-1] = contract_value
                    simplex[-1] = contract_point
                    simplex_traj.append(simplex)
                    continue

            for i in range(len(simplex) - 1):
                s_point = simplex[0] + coef_s * (simplex[i + 1] - simplex[0])
                s_val = function(s_point)
                simplex[i + 1] = s_point
                simplex_values[i + 1] = s_val
            simplex_traj.append(simplex)
    return simplex_traj, simplex_traj_mae


def generate_point(simplex, centroid, coefficient):
    return centroid + coefficient * (centroid - simplex[-1])


if __name__ == '__main__':
    rsbrock = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    st, se = minimize(function=rsbrock,
                      x0=np.array([-2.1, -1.1]),
                      n_iters=250)
