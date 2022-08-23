from pickle import load
import yaml
import os
os.makedirs(os.path.join("score"), exist_ok=True)
params = yaml.safe_load(open("params.yaml"))["inference"]

model = load(open('model/model_v001.pkl', 'rb'))
with open('score/result.txt','w') as oof:
    oof.write("Test, Score\n" )

    for i in range(1,params["n_test"]):
        X, y = load(open('dataset/datasets_v%03d.pkl' % i, 'rb'))
    
        score = model.score(X, y)

        oof.write("%d, %f\n" % (i,score))


