import xgboost as xgb
import pandas as pd
import numpy as np

data = pd.read_csv('iris.csv')

df = pd.DataFrame(data)

print("data frame : \n", df)

train = np.array(df[::4])
test = np.array(df[::4])
print("Training data : \n", train)
print("Test data : \n", test)


print("training model shape", train.shape, test.shape)

#weights
w = np.random.rand(38,1)

dtrain = xgb.DMatrix(train, label=train, missing=np.NaN, weight=w)
dtest = xgb.DMatrix(test, label=test, missing=np.NaN, weight=w)

print("\nxgb DMatrix : \n", dtrain, "\nweights : \n", w)

print("\ndtrain\n", dtrain, "\ndtest : \n", dtest, "\nweights ; \n", w, "\n\n")

param = {'max_depth': 2, 'eta': 1, 'nthread': 4}

num_rounds = 100

evallist = [(dtrain, 'train'), (dtest, 'eval')]

bst = xgb.train(param, dtrain, num_rounds, evallist)

print("\nTrained Model : \n", bst)

#save model
bst.save_model('0001.model')

bst.dump_model("dump.raw.txt")

bst = xgb.Booster({'nthread': 4})
bst.load_model('0001.model')
print(bst.get_score)

#perform predictions
ypred = bst.predict(dtest)

print("\nPrediction from Random data :\n", ypred)


#plots
#xgb.to_graphviz(bst, num_trees=2)
#xgb.plot_importance(bst)
#xgb.plot_tree(bst, num_trees=2)

