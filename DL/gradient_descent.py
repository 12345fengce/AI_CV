def gradient_descent(formula, init, epoch, lr, delta=1e-8):
    """formula: f(1), f(2), f(3)
        w: the weight of init
        epoch: the epoch for iterations
        lr: the step"""
    for i in range(epoch):
        f1 = formula(init - delta)
        f2 = formula(init + delta)
        g = (f2 - f1) / (2 * delta)
        init -= g * lr
    return init


def f(x):
    return (x+3)**2+5


w = gradient_descent(f, 3, 1000, 0.1)
print(w)




