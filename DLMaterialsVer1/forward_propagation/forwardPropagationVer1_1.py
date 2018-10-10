import sys, os
sys.path.append('../../') # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common import functions

# 順伝播（単層・単ユニット）

# 重み
W = np.array([[0.1], [0.2]])

## 試してみよう_配列の初期化
#W = np.zeros(2)
#W = np.ones(2)
#W = np.random.rand(2)
#W = np.random.randint(5, size=(2))

functions.print_vec("重み", W)


# バイアス
b = 0.5

## 試してみよう_数値の初期化
#b = np.random.rand() # 0~1のランダム数値
#b = np.random.rand() * 10 -5  # -5~5のランダム数値

functions.print_vec("バイアス", b)

# 入力値
x = np.array([2, 3])
functions.print_vec("入力", x)


# 総入力
u = np.dot(x, W) + b
functions.print_vec("総入力", u)

# 中間層出力
z = functions.relu(u)
functions.print_vec("中間層出力", z)
