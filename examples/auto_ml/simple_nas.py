import pandas as pd
import numpy as np

import torch
from torch import nn

from src.cl3s import SpecificationBuilder, Constructor, Literal, Var, SearchSpaceSynthesizer, DataGroup
from src.cl3s.scikit.bayesian_optimization import BayesianOptimization


def load_iris(path):
    iris = pd.read_csv(path)

    # load training data
    train_input = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()

    # construct labels manually since data is ordered by class
    train_labels = np.array([0]*50 + [1]*50 + [2]*50).reshape(-1)

    # one-hot encode 3 classes
    train_labels = np.identity(3)[train_labels]

    return train_input, train_labels

class Simple_DNN_Repository:

    def __init__(self, learning_rates: list[float],
                 dimensions: list[int],
                 max_hidden: int,
                 batch_sizes: list[int],
                 epochs: list[int],
                 dataset: torch.utils.data.TensorDataset):
        self.learning_rates = learning_rates
        self.dimensions = dimensions
        self.max_hidden = max_hidden
        self.batch_size = batch_sizes
        self.epochs = epochs
        self.train_dataset = dataset
        self.initial_weights = {}

    def gamma(self):
        dimension = DataGroup("dimension", self.dimensions)
        bool = DataGroup("bool", [True, False])
        learning_rate = DataGroup("learning_rate", self.learning_rates)
        hidden = DataGroup("hidden", list(range(0, self.max_hidden + 1, 1)))
        batch_size = DataGroup("batch_size", self.batch_size)
        epochs = DataGroup("epochs", self.epochs)
        return {
            "Layer": SpecificationBuilder()
            .parameter("n", dimension)
            .parameter("bias", bool)  # Constructor("Bias", Var("n")))
            .argument("af", Constructor("activation_function"))
            .suffix(Constructor("layer", Var("n"))),

            # "Bias": DSL()
            # .parameter("n", "dimension")
            # .suffiy(Constructor("Bias", Var("n"))),

            "Model": SpecificationBuilder()
            .parameter("in", dimension)
            .parameter("out", dimension)
            .argument("l", Constructor("layer", Var("out")))
            .suffix(
                Constructor("model",
                            Constructor("input", Var("in"))
                            & Constructor("output", Var("out")))
                & Constructor("hidden", Literal(0))
            ),

            "Model_cons": SpecificationBuilder()
            .parameter("in", dimension)
            .parameter("out", dimension)
            .parameter("neurons", dimension)
            .parameter("n", hidden)
            .parameter("m", hidden, lambda vs: [vs["n"]-1])
            .argument("layer", Constructor("layer", Var("neurons")))
            .argument("model",
                 Constructor("model",
                             Constructor("input", Var("neurons"))
                             & Constructor("output", Var("out")))
                 & Constructor("hidden", Var("m"))
                 )
            .suffix(
                Constructor("model",
                            Constructor("input", Var("in"))
                            & Constructor("output", Var("out"))
                            )
                & Constructor("hidden", Var("n"))
            ),

            "ReLu": Constructor("activation_function"),

            "ELU": Constructor("activation_function"),

            "Sigmoid": Constructor("activation_function"),

            "MSE": Constructor("loss_function"),

            "CrossEntropy": Constructor("loss_function"),

            "L1": Constructor("loss_function"),

            "DataLoader": SpecificationBuilder()
            .parameter("bs", batch_size)
            .suffix(Constructor("data", Constructor("batch_size", Var("bs")))),

            #"Adagrad": DSL()
            #.Use("lr", "learning_rate")
            #.In(Constructor("optimizer", Constructor("learning_rate", LVar("lr")))),

            "Adam": SpecificationBuilder()
            .parameter("lr", learning_rate)
            .suffix(Constructor("optimizer", Constructor("learning_rate", Var("lr")))),

            "SGD": SpecificationBuilder()
            .parameter("lr", learning_rate)
            .suffix(Constructor("optimizer", Constructor("learning_rate", Var("lr")))),

            "System": SpecificationBuilder()
            .parameter("in", dimension)
            .parameter("out", dimension)
            .parameter("n", hidden)
            .parameter("lr", learning_rate)
            .parameter("ep", epochs)
            .parameter("bs", batch_size)
            .argument("data", Constructor("data", Constructor("batch_size", Var("bs"))))
            .argument("m", Constructor("model", Constructor("input", Var("in")) & Constructor("output", Var("out"))) & Constructor("hidden", Var("n")))
            .argument("opt", Constructor("optimizer", Constructor("learning_rate", Var("lr"))))
            .argument("l", Constructor("loss_function"))
            .suffix(
                Constructor("system",
                            Constructor("input_dim", Var("in"))
                            & Constructor("output_dim", Var("out"))
                            )
                & Constructor("learning_rate", Var("lr"))
                & Constructor("hidden_layer", Var("n"))
                & Constructor("epochs", Var("ep"))
                & Constructor("batch_size", Var("bs"))
            ),
        }

    @staticmethod
    def train_loop(dataloader, model, loss_fn, opti, batch_size):
        size = len(dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        # print(f"Model structure: {model}\n\n")
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X = X.float()
            y = y.float()
            pred = model(X)
            loss = loss_fn(pred, y)

            # build the optimizer
            optimizer = opti(model.parameters())

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    @staticmethod
    def test_loop(dataloader, model, loss_fn):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in dataloader:
                X = X.float()
                y = y.float()
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        #print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return 100 * correct + 0.0001, test_loss

    def system(self, dataloader, model, loss_fn, optimizer, batch_size, epochs):
        for epoch in range(epochs):
            # print(f"Epoch {epoch + 1}/{epochs} \n -------------------------------")
            self.train_loop(dataloader, model, loss_fn, optimizer, batch_size)
            #  self.test_loop(dataloader, model, loss_fn)
        return self.test_loop(dataloader, model, loss_fn)

    # Try to make the obj_fun a bit more deterministic by reusing the initial weights for the same layer dimensions
    def linear_layer(self, i, o, b):
        if self.initial_weights.get((i, o)) is None:
            layer = nn.Linear(i, o, b)
            self.initial_weights[(i, o)] = layer.weight
            return layer
        else:
            layer = nn.Linear(i, o, b)
            layer.weight = self.initial_weights[(i, o)]
            return layer


    def torch_algebra(self):
        return {
            "System": (lambda i, o, n, lr, ep, bs, data, m, opt, l: self.system(data, m, l, opt, bs, ep)),
            "Layer": (lambda n, b, af: (b, af)),
            "Model": (lambda i, o, l: nn.Sequential(nn.Linear(i, o, l[0]), nn.Softmax(dim=1))),  # , l[1])),
            "Model_cons": (lambda i, o, neurons, n, m, l, model:
                           nn.Sequential(nn.Linear(i, neurons, l[0]), l[1]).extend(model)),
            "ReLu": nn.ReLU(),
            "ELU": nn.ELU(),
            "Sigmoid": nn.Sigmoid(),
            "MSE": nn.MSELoss(),
            "CrossEntropy": nn.CrossEntropyLoss(),
            "L1": nn.L1Loss(),
            #"Adagrad": (lambda lr, params: torch.optim.Adagrad(params, lr=lr)),
            "Adam": (lambda lr, params: torch.optim.Adam(params, lr=lr)),
            "SGD": (lambda lr, params: torch.optim.SGD(params, lr=lr)),
            "DataLoader": (lambda bs: torch.utils.data.DataLoader(self.train_dataset, batch_size=bs)),
        }

train_input, train_labels = load_iris('./data/iris.csv')
x_train_tensor = torch.tensor(train_input)
y_train_tensor = torch.tensor(train_labels)

dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)

repo = Simple_DNN_Repository([0.01, 0.001, 0.0001], [3, 4] + list(range(5, 30, 5)), 5, [8], [1000], dataset)

target = (Constructor("system",
                      Constructor("input_dim", Literal(4))
                      & Constructor("output_dim", Literal(3))
                      )
          #& Constructor("learning_rate", Literal(0.01, "learning_rate"))
          #& Constructor("hidden_layer", Literal(3, "hidden"))
          #& Constructor("epochs", Literal(100, "epochs"))
          #& Constructor("batch_size", Literal(8, "batch_size"))
          )

synthesizer = SearchSpaceSynthesizer(repo.gamma(), {})

search_space = synthesizer.construct_search_space(target).prune()

optimizer = BayesianOptimization(search_space, target)

obj_fun = lambda t: t.interpret(repo.torch_algebra())[0] # accuracy

if __name__ == "__main__":
    """"
    x_list = []
    y_list = []
    x0 = search_space.sample(10, target)
    for tree in x0:
        x_list.append(tree)
        y_list.append(obj_fun(tree))

    xp = np.array(x_list)
    yp = np.array(y_list)

    model = GaussianProcessRegressor(kernel=WeisfeilerLehmanKernel(),
                                     alpha=1e-10,
                                     # n_restarts_optimizer=10,
                                     optimizer=None,  # we currently need this, to prevent derivation of the kernel
                                     normalize_y=True)

    model.fit(xp, yp)

    acq_fun = SimplifiedExpectedImprovement(model, greater_is_better=True)

    search = TournamentSelection(search_space, target, acq_fun, population_size=10, reproduction_rate=0.2,
                                 generation_limit=2, greater_is_better=True)

    best = search.optimize()
    """

    for t in search_space.enumerate_trees(target, 10):
        print(t)
        print(obj_fun(t))

    best, xp, yp = optimizer.bayesian_optimisation(5, obj_fun, n_pre_samples=10,
                                                   greater_is_better=True)
    print(best)
    list_x = list(xp)
    list_y = list(yp)
    print(list_y[list_x.index(best)])



