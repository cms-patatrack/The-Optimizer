import numpy as np
import optimizer

class zdt1:
    num_params = 30
    lb = [0.] * num_params
    ub = [1.] * num_params

    def zdt1_objective(x):
        f1 = x[0]
        g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h
        return f1, f2

    objective = optimizer.ElementWiseObjective(zdt1_objective, 2)

    def get_optimal_pareto(num_points):
        real_x = (np.linspace(0, 1, num_points))
        real_y = 1 - np.sqrt(real_x)
        return real_x, real_y 

class zdt2:
    num_params = 30
    lb = [0.] * num_params
    ub = [1.] * num_params

    def zdt2_objective1(x):
        return x[0]


    def zdt2_objective2(x):
        f1 = x[0]
        g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
        h = 1.0 - np.power((f1 * 1.0 / g), 2)
        f2 = g * h
        return f2

    objective = optimizer.ElementWiseObjective([zdt2_objective1, zdt2_objective2])

    def get_optimal_pareto(num_points):
        real_x = (np.linspace(0, 1, num_points))
        real_y = 1 - np.power(real_x, 2)
        return real_x, real_y
    
class zdt3:
    num_params = 30
    lb = [0.] * num_params
    ub = [1.] * num_params

    def zdt3_objective1(x):
        return x[0]

    def zdt3_objective2(x):
        f1 = x[0]
        g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
        h = (1.0 - np.power(f1 * 1.0 / g, 0.5) -
            (f1 * 1.0 / g) * np.sin(10 * np.pi * f1))
        f2 = g * h
        return f2

    objective = optimizer.ElementWiseObjective([zdt3_objective1, zdt3_objective2])

    def get_optimal_pareto(num_points):
        regions = [[0, 0.0830015349],
                   [0.182228780, 0.2577623634],
                   [0.4093136748, 0.4538821041],
                   [0.6183967944, 0.6525117038],
                   [0.8233317983, 0.8518328654]]

        pf = []

        for r in regions:
            x1 = np.linspace(r[0], r[1], int(num_points / len(regions)))
            x2 = 1 - np.sqrt(x1) - x1 * np.sin(10 * np.pi * x1)
            pf.append([x1, x2])

        real_x = np.concatenate([x for x, _ in pf])
        real_y = np.concatenate([y for _, y in pf])
        return real_x, real_y
    
class zdt4:
    num_params = 10
    lb = [0.] + [-5.] * (num_params - 1)
    ub = [1.] + [5.] * (num_params - 1)

    optimizer.Logger.setLevel('INFO')

    optimizer.Randomizer.rng = np.random.default_rng(46)

    def zdt4_objective1(x):
        return x[0]

    def zdt4_objective2(x):
        f1 = x[0]
        g = 1.0 + 10 * (len(x) - 1) + sum([i**2 - 10 * np.cos(4 * np.pi * i) for i in x[1:]])
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h
        return f2

    objective = optimizer.ElementWiseObjective([zdt4_objective1, zdt4_objective2])

    def get_optimal_pareto(num_points):
        real_x = (np.linspace(0, 1, num_points))
        real_y = 1 - np.sqrt(real_x)
        return real_x, real_y
    
class zdt5:
    num_params = 11
    lb = [0] * num_params
    ub = [1073741825] * num_params

    @classmethod
    def u(x):
        c = 0
        while x:
            c += 1
            x &= x - 1
        return c

    @classmethod
    def v(x):
        un = u(x)
        if un < 5:
            return 2 + un
        elif un == 5:
            return 1
        else:
            return 0
        
    def zdt5_objective1(x):
        return 1 + u(x[0])

    def zdt5_objective2(x):
        f1 = 1 + u(x[0])
        g = sum([v(u(i)) for i in x[1:]])
        h = 1.0 / f1
        f2 = g * h
        return f2

    objective = optimizer.ElementWiseObjective([zdt5_objective1, zdt5_objective2])

    def get_optimal_pareto(num_points):
        real_x = (np.linspace(0, 1, num_points))
        real_y = 1 - np.sqrt(real_x)
        return real_x, real_y
        
