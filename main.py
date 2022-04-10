from Models import train
import numpy as np
for hidden_neural in range(5,11,1):
    hidden_neural = 2 ** hidden_neural
    for lr in range(1,11,1):
        lr = lr/10
        for regularization in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            train(hidden = hidden_neural, reg = regularization, lr_init = lr, save_path = "./experiments_0408/lr_{}_hidden_{}_reg_{}".format(lr, hidden_neural, regularization))
