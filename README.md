# CNN_Image_Classification
## 概要
これは、CNN（畳み込みニューラルネットワーク）を用いて２種類の画像分類を行うモデルを作成するツールです。
任意のEpoch数で学習を行い、モデル（torch script）を出力することが出来ます。
学習がどのように進んだかを示すグラフを出力することが出来ます。出力されるグラフは学習における損失値の推移、学習における分類精度の推移を示します。
学習終了時に、最終Epochにおけるtest結果が表示されます。結果は５枚ずつ表示され、Trueが正解、Falseが不正解だった画像です。


## 使いかた
1.trainのフォルダ内にあるtrain_Aとtrain_Bに学習に使いたい画像類を入れてください。（サンプルとしてStable difusion で生成した某キャラクターを雑に模した画像が入っています。）

2.valのフォルダ内にあるval_Aとval_Bに学習モデルのテストに使いたい画像類を入れてください。（サンプルとしてStable difusion で生成した某キャラクターを雑に模した画像が入っています。）

3.お手元の環境でCNN-Image-Classification.pyを実行してください。

4.画像が正しく読み込めているかを確かめるテストを行うかどうかを尋ねられます。必要に応じて行ってください。テスト結果のウィンドウを閉じると次のステップに進みます。

5.Epoch数をいくつにするかを尋ねられます。任意の自然数を入力してください。一般にEpoch数を大きくするほど分類精度は高くなりますが、学習時間が長くなり、過学習を引き起こすリスクも上がります。

6.学習が終わるまで待ちます。画像の枚数、Epoch数、実行環境によって長時間の処理が必要になることがあります。学習とテストについての損失値と分類精度が1epochごとにそれぞれ出力されます。

7.学習がどのように進んだかを示すグラフが２種類（学習における損失値の推移、学習における分類精度の推移）表示されます。必要に応じて保存してください。

8.学習したモデル（torch script）を出力するかを尋ねられます。必要に応じて出力してください。保存場所はソースコードと同じ場所です。

## デフォルトの設定
アーキテクチャ：AlexNet
最適化関数：Adam
学習率:1e-4
画像の前処理：画像サイズを224×224ピクセルに変更
データ拡張：50%の確立で水平方向に反転

## 動作環境
＜OS＞日本語版Windows10
＜IDE＞Visual Studio 2019
＜言語＞Python3.10


## Contact
早稲田大学基幹理工学部表現工学科尾形研究室
氏名：堀和希
最終更新11/26
