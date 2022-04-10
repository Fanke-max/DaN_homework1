from Models import test
import numpy as np

test_acc_list = []
test_loss_list = []
parameters = []
for hidden_neural in range(5,11,1):
    hidden_neural = 2 ** hidden_neural
    for lr in range(1,11,1):
        lr = lr/10
        for regularization in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            print("lr:{:2.2f}\thidden_neurons:{}\trregularization:{}".format(lr, hidden_neural, regularization),end="\t")
            test_acc,test_loss = test(load_file = "./experiments_0408/lr_{}_hidden_{}_reg_{}/epoch_19".format(lr, hidden_neural, regularization))
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
