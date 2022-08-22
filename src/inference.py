from pickle import load

model = load(open('../model/model_v001.pkl', 'rb'))

for i in range(1,12):
    X, y = load(open('../dataset/datasets_v%03d.pkl' % i, 'rb'))

    score = model.score(X, y)

    print(score)

