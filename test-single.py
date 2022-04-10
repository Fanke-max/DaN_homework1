from Models import test

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default= 1.0)
parser.add_argument('--hidden_neurons', type=int, default= 1024)
parser.add_argument('--regularization', type=float, default= 1e-5)
args = parser.parse_args()

print("lr:{:2.2f}\thidden_neurons:{}\trregularization:{}".format(args.lr, args.hidden_neurons, args.regularization),end="\t")
test_acc,test_loss = test(load_file = "./experiments_0408/lr_{}_hidden_{}_reg_{}/epoch_19".format(args.lr, args.hidden_neurons, args.regularization))