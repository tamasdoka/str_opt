from pyeasyga import pyeasyga
import numpy as np
import optimizer_env_v2 as opt


env = opt.StrOptEnv()
env.reset()

data = np.zeros(42)
best_solutions = []

range_coords = np.array([[-350, -50],
                        [-250, -100],
                        [-450, -100],
                        [-250, -50]])

check_coords = np.zeros((range_coords.shape))

for i in range(range_coords.shape[0]):
    check_coords[i, 0] = range_coords[i, 0] - 10
    check_coords[i, 1] = range_coords[i, 1] + 10


def valid_state(range_coords):
    state = []
    for i in range(range_coords.shape[0]):
        state.append(np.random.randint(range_coords[i, 0], range_coords[i, 1]))

    return state


ga = pyeasyga.GeneticAlgorithm(range_coords,
                               population_size=150,
                               generations=200,
                               crossover_probability=0.75,
                               mutation_probability=0.15,
                               elitism=True,
                               maximise_fitness=True)


def fitness_func(genes, range_coords):

    state = np.asarray(genes)

    check, diff = in_range(state, range_coords)

    env.state = state
    env.step(0)

    dx, dy, ax, ay = state
    # calculating arm lengths
    arm_length = np.sqrt((ax - env.KPLX) ** 2 + (ay - env.KPLY) ** 2)
    tierod_length = np.sqrt((ax - dx) ** 2 + (ay - dy) ** 2)
    rack_length = 2*abs(dx)

    length_factor = 1/(2*arm_length + 0.7*tierod_length + rack_length)*env.TW

    if env.error is None or env.max_r is None or env.border_ang is None:
        fitness = 0
        return fitness

    check_decrease = 1
    ang_decrease = 1

    if ~check:
        check_decrease = np.exp(-diff)
        # print(check, diff)
        # print(check_decrease)

    if env.max_r < env.border_ang:
        ang_decrease = np.exp(-abs(env.border_ang - env.max_r)*20)

    factor = check_decrease * ang_decrease * length_factor

    if env.error == 0:
        fitness = 100000 * factor
    elif 1 / env.error > 100000:
        fitness = 100000 * factor
    else:
        fitness = 1 / env.error * factor

    return fitness


ga.fitness_function = fitness_func


def in_range(state, range_coords):
    status = True
    diff = 0
    for i in range(len(state)):
        new_stat = range_coords[i, 0] <= state[i] <= range_coords[i, 1]
        status = status and new_stat

        if ~status:
            for j in range(len(state)):
                diff += (min(abs(range_coords[j, 0] - state[j]), abs(range_coords[j, 1] - state[j])))

    diff = diff / len(state)
    return status, diff

# if __name__ == '__main__':
#     ind = ga.run()
#
#     print("best individual: ", ga.best_individual())
#
#     for individual in ga.last_generation():
#        print("last_generation inds: ", individual)
#
#     best_sol = []
#     for i in ga.best_solutions:
#         best_sol.append(i)
#     for i in best_sol:
#         env.reset()
#         env.state = i
#         env.step(0)
#
#         if env.check_error is None or env.check_r is None:
#             print("none")
#         else:
#             env.save_plot(env.check_error, env.check_r)

