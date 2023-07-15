---
title           : "Benchmarking Model Latency"
description     : "Benchmarking Model Latency"
katex           : true
date: 2023-07-15
katexExtensions : [ mhchem, copy-tex ]
---

In Data Science we are obsessed with model performance. If you're dealing with a classification task one might look at Area Under Curve (AUC), in a regression task at Mean Absolute Error (MAE).

Typically, Data Scientists create an offline evaluation pipeline where each model is benchmarked in different scenarios.
The model (or input features) are typically iterated so that we beat iteratively our performance metrics and until we have enough confidence to roll out our model into production (and if you are fancy enough to an A/B test).

**My question is: are these performance metrics enough for real-life scenarios ??**

Lets take a step back. Most of ML models don't simply live in the vacum of offline evaluations, they usually make part of a product and serve real customers.
Think for example of providing recommendations in an e-commerce, answers in a chatbot or detecting fraud in payments.

When interacting with customers at real-time there are other metrics that are important besides model performance.
For example, no one wants to navigate in a slow e-commerce website and It is known that [website load time directly impacts the conversion rate](https://www.cloudflare.com/learning/performance/more/website-performance-conversion-rates/#:~:text=Website%20performance%20has%20a%20large,quickly%20should%20a%20webpage%20load%3F).
If your `model.predict` is adding to the load time of the webpage, then it means that your model taking 1s vs 100ms (aka model latency) might make a big difference.

But how can we measure this model latency in a similar way to what we do with the DS model performance metrics discussed later ? **How can we have a fast feedback loop for model latency ?**

The short answer is: using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/) .
*pytest-benchmark* is a pytest plugin that allows you to easily benchmark parts of your code by writing (you can guess) tests.

Lets jump into an example :)
All the code used in this blogpost can be found in this [repo](https://github.com/candeiasalexandre/pytest-benchmark-example/)

Imagine you have a simple torch model.

``` python
class SimpleRegressionModel(torch.nn.Module):
    def __init__(
        self,
        embeddings: List[EmbeddingConfig],
        hidden_layers: List[int],
        numerical_cols: List[str],
        dropout: Optional[float] = None,
    ) -> None:
    ...

    def forward(self, x: SimpleRegressionModelData) -> torch.FloatTensor:
    ...
```

Without getting into much detail, this is a simple neural network (NN) for regression that can receive both categorical and numerical features. If you want to know more details regarding the model check it [here](https://github.com/candeiasalexandre/pytest-benchmark-example/blob/main/pytest_benchmark_example/model.py#L45).

The important part is that the model's input is a abstracted as `x` which is a `SimpleRegressionModelData`. This wrapper around the input data makes it easy to understand what are the inputs for the model.

``` python
@dataclass
class NamedTensor:
    columns: Tuple[str]
    data: torch.Tensor
    _column_idx_map: Dict[str, int] = field(init=False)

    def __post_init__(self):
        self._validate_data()
        self._column_idx_map = {
            column: idx for idx, column in enumerate(self.columns)
        }

    def _validate_data(self):
        if len(self.data.shape) != 2:
            raise RuntimeError("NamedTensor only supports data of dim=2!")
        if len(self.columns) != self.data.shape[1]:
            raise RuntimeError(
                "Number of columns should be the same as number size of dim=2!"
            )

    def get_data(self, columns: List[str]) -> torch.Tensor:
        idx_columns = [self._column_idx_map[column] for column in columns]
        return self.data[:, idx_columns]

    def get_data_at_idx(self, idx: int) -> "NamedTensor":
        return NamedTensor(self.columns, self.data[idx, :])


@dataclass
class SimpleRegressionModelData:
    numericals: NamedTensor
    categoricals: NamedTensor
```

it's now easy to use *pytest-benchmark* to create a latency benchmark for the forward method. The first step is to generate input data.

``` python
def generate_data(
    categorical_data_info: List[EmbeddingConfig],
    numerical_data_info: List[str],
    n_rows: int,
) -> SimpleRegressionModelData:

    _categorical_data = []
    _categorical_data_columns = []
    for embedding_cfg in categorical_data_info:
        _categorical_data.append(
            torch.randint(0, embedding_cfg.cardinality - 1, (n_rows, 1))
        )
        _categorical_data_columns.append(embedding_cfg.name)

    _numerical_data = torch.randn((n_rows, len(numerical_data_info)))

    return SimpleRegressionModelData(
        numericals=NamedTensor(tuple(numerical_data_info), _numerical_data),
        categoricals=NamedTensor(
            tuple(_categorical_data_columns), torch.cat(_categorical_data, dim=1)
        ),
    )
```

We also need to instantiate a model in order to benchmark it. We can wrap up the model creation and the data generation in another function.

``` python
def setup_model_and_data(
    categorical_data_info: List[EmbeddingConfig],
    numerical_data_info: List[str],
    n_hidden_layers: int,
    batch_size: int,
) -> Tuple[SimpleRegressionModel, SimpleRegressionModelData]:
    data = generate_data(categorical_data_info, numerical_data_info, batch_size)

    hidden_layers = [2**n for n in range(1, n_hidden_layers)]
    hidden_layers.reverse()
    dummy_model = SimpleRegressionModel(
        embeddings=categorical_data_info,
        hidden_layers=hidden_layers,
        numerical_cols=numerical_data_info,
    )

    return dummy_model, data
```

Having these two functions is now fairly easy to create a latency benchmark.

``` python
def test_benchmark_model_batch_1(
    benchmark,
    categorical_data_info: List[EmbeddingConfig],
    numerical_data_info: List[str],
) -> None:
    model, data = setup_model_and_data(categorical_data_info, numerical_data_info, 4, 1)

    model.eval()
    with torch.no_grad():
        benchmark(model.forward, data)
```

This test generates an input with 1 instance and a model containing 4 hidden layers, then it uses that model and data to run the `forward` method multiple times and acquires how much time it spends in each of these times. All of that by using the `benchmark` fixture provided by `pytest-benchmark`!

To run it, we can use the command `pytest <yourtestfile.py>` and it will give you an output as the one bellow.

![](/posts/img/benchmarking-model-latency/output_pytest_benchmark.png)
We can see that the benchmark was run for 1175 rounds and that in average the forward pass of the model takes \~180us.

Since *pytest-benchmark* is a pytest plugin, we can use all the pytest cool features.
One might want to run the benchmark with models of different sizes, i.e changing the number of hidden layers, or simply see how the model latency scales with the batch size.
This can be easily done using the `pytest.parametrize` decorator.

``` python
@pytest.mark.parametrize(
    "n_hidden_layers",
    [
        pytest.param(8, id="n_hidden_layers=8"),
        pytest.param(4, id="n_hidden_layers=4"),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_benchmark_model_parametrized(
    benchmark,
    batch_size: int,
    n_hidden_layers: int,
    categorical_data_info: List[EmbeddingConfig],
    numerical_data_info: List[str],
) -> None:

    model, data = setup_model_and_data(
        categorical_data_info, numerical_data_info, n_hidden_layers, batch_size
    )

    model.eval()
    with torch.no_grad():
        benchmark(model.forward, data)
```

Finally, we can also use *pytest-benchmark* to generate an histogram `.svg` with all the parametrised tests. For that we should run the benchmark using the command `pytest <yourtestfile.py> --benchmark-histogram`.

{{< svg  "content/posts/img/benchmarking-model-latency/histogram_example.svg" >}}

This ends our journey about *pytest-benchmark* and how to measure your model's latency in an easy, reproducible and standardised way. By using this approach a Data Scientist can simply run its benchmarks just like any other unit-test or evaluation pipeline and have a fast feedback loop.

There are many more things that you can do with it *pytest-benchmark*, like configure it to run in your CI/CD, keep track of the performance tests at each build, etc.. for more details I invite you to check the [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/en/latest/).

Hope you enjoyed, thanks for reading :)
