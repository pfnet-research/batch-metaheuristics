# batchopt

![license:MIT](https://github.com/pfnet/i20_meit/blob/release/license.svg)

このリポジトリは、PFN summer internship において開発された、メタヒューリスティクスを用いたブラックボックス最適化のためのコード群です。

## Usage

以下のコマンドを実行してください。

```sh
git clone git@github.com:pfnet-research/batch_metaheuristics.git
cd batch_metaheuristics
pip install ./batchopt
python example.py
```

以下のような出力が確認できれば OK です(ブラックボックス最適化はランダム要素を含むため、完全に同じ出力が得られるとは限りません)。

```
> Epoch: 1, Best fit: 884.7695325180321
> Epoch: 2, Best fit: -63.122234265670585
> Epoch: 3, Best fit: -85.04995732445039
> Epoch: 4, Best fit: -85.04995732445039
> Epoch: 5, Best fit: -111.70023680346456
> Epoch: 6, Best fit: -127.10859058276085
> Epoch: 7, Best fit: -131.96113684754317
> Epoch: 8, Best fit: -135.051712626202
> Epoch: 9, Best fit: -137.46424504437974
> Epoch: 10, Best fit: -138.53373567844406
```

`example.py`のコードは次のようになっています。`objective_function` などを変更すれば、独自のブラックボックス最適化問題を解くことができます。

```python
from batchopt.benchfunctions import batch_styblinski_tang
from batchopt.optimizers.whale_optimization_algorithm import WhaleOptimizationAlgorithm

opt = WhaleOptimizationAlgorithm(
    domain_range=[(-20, 20), (-20, 20), (-20, 20), (-20, 20)],
    pop_size=50,
    epoch=10,
    log=True,
)
objective_function = batch_styblinski_tang
# must be function 2-dimension-array to 1 dimension array

opt.optimize(objective_function)
```

ここでは`WhaleOptimizationAlgorithm`を用いましたが、 batchopt では 20 以上のメタヒューリスティクスを使用可能です(`batchopt/optimizers`以下を参照してください)。
実装されている全てのメタヒューリスティクスは、以下のような引数で初期化されます。

- `domain_range`: 目的関数の定義域のリスト
- `pop_size`: 同時に生成される候補の個数
- `epoch`: 最適化プロセスを繰り返す回数
- `log`: ログを出力するかどうか

メタヒューリスティクスのインスタンスを生成したら、`optimize` メソッドで最適化を行います。
目的関数は、複数の候補点に対し、候補点と同じだけのスカラー値を返すようにしてください。つまり、目的関数は二次元配列から一次元配列への写像である必要があります。
