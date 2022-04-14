from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection


mlp = MLPClassifier(random_state = 12, max_iter = 300, hidden_layer_sizes = 300)
pipe = Pipeline(steps=[('scaler', MinMaxScaler()),
                ('mlp',
                 MLPClassifier(activation='logistic', hidden_layer_sizes=500,
                               max_iter=400, random_state=12))])
