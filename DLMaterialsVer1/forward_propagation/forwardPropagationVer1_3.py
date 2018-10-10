import sys, os
sys.path.append('../../') # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common import functions


# 順伝播（3層・複数ユニット）

# ウェイトとバイアスを設定
# ネートワークを作成
def init_network():
    print("##### ネットワークの初期化 #####")
    network = {}

    # 試してみよう
    # _各パラメータのshapeを表示
    # _ネットワークの初期値ランダム生成

    network['W1'] = np.array([
        [0.1, 0.3, 0.5],
        [0.2, 0.4, 0.6]
    ])
    network['W2'] = np.array([
        [0.1, 0.4],
        [0.2, 0.5],
        [0.3, 0.6]
    ])
    network['W3'] = np.array([
        [0.1, 0.3],
        [0.2, 0.4]
    ])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['b2'] = np.array([0.1, 0.2])
    network['b3'] = np.array([1, 2])

    functions.print_vec("重み1", network['W1'])
    functions.print_vec("重み2", network['W2'])
    functions.print_vec("重み3", network['W3'])
    functions.print_vec("バイアス1", network['b1'])
    functions.print_vec("バイアス2", network['b2'])
    functions.print_vec("バイアス3", network['b3'])

    return network


# プロセスを作成
# x：入力値
def forward(network, x):
    print("##### 順伝播開始 #####")

    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 1層の総入力
    u1 = np.dot(x, W1) + b1

    # 1層の総出力
    z1 = functions.relu(u1)

    # 2層の総入力
    u2 = np.dot(z1, W2) + b2

    # 2層の総出力
    z2 = functions.relu(u2)

    # 出力層の総入力
    u3 = np.dot(z2, W3) + b3

    # 出力層の総出力
    y = u3

    functions.print_vec("総入力1", u1)
    functions.print_vec("中間層出力1", z1)
    functions.print_vec("総入力2", u2)
    functions.print_vec("出力1", z1)
    print("出力合計: " + str(np.sum(z1)))

    return y, z1, z2


# 入力値
x = np.array([1., 2.])
functions.print_vec("入力", x)

# ネットワークの初期化
network = init_network()

y, z1, z2 = forward(network, x)