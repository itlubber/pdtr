# 自动决策树规则挖掘工具包

在笔者金融风控的日常工作中，很多时候需要根据数据集内的诸多特征（有很多其他称呼，比如因子、变量、自变量、解释变量等）来挖掘一些有用的规则和组合策略，在保证通过率的基础上尽可能多的拒绝坏客户。面对成千上万的特征，如何从数据集中找到有效的规则和组合策略，一直以来都是金融风控搬砖工的日常工作。 `pdtr` 旨在帮助读者快速从高维数据中提取出有效的规则和组合策略。

> 仓库地址：https://github.com/itlubber/pdtr
> 
> 博文地址：https://itlubber.art/archives/auto-strategy-mining
> 
> 微信公共号推文：https://mp.weixin.qq.com/s/8s785MfmVznNgQyy38YnWw
> 
> pipy包：https://pypi.org/project/pdtr/

## 背景简介

金融场景风险大致可以概括为三种：系统性风险、欺诈风险（无还款意愿）、信用风险（无还款能力），而作为一名风控搬砖工，日常工作中有大量的数据挖掘工作，如何从高维数据集中挖掘出行之有效的规则、策略及模型来防范欺诈风险和信用风险每个搬砖工的基操。本仓库由笔者基于网上开源的一系列相关知识，结合实际工作中遇到的实际需求，整理得到。旨在为诸位仁兄提供一个便捷、高效、赏心悦目的决策树组合策略挖掘报告，及一系列能够实际运用到风险控制上的策略。

## 项目结构

```bash
pdtr
.
|   README.md                           # 说明文档
|   setup.py                            # 打包发布文件
|   LICENSE                             # 开源协议
|   requirements.txt                    # 项目依赖包
+---examples                            # 演示样例
|   |   combine_rules_cache             # 缓存文件
|   |   combine_rules_cache.svg         # 缓存文件
|   |   pdtr_samplts.ipynb              # 演示样例程序
|   \---model_report                    # 模型报告输出文件夹
|       |   决策树组合策略挖掘.xlsx        # 策略挖掘报告
|       +---auto_mining_rules           # 组合策略可视化存储文件夹
|       |       combiner_rules_0.png    # 决策树可视化图片
|       |       ......
|       \---bin_plots                   # 简单策略可视化存储文件夹
|               bin_vars_A.png          # 变量分箱可视化图片
|               ......
\---pdtr                                # PDTR 源码包
        template.xlsx                   # excel 模版文件
        excel_writer.py                 # excel写入公共方法
        matplot_chinese.ttf             # matplotlib 中文字体
        transforme.py                   # 策略挖掘方法
```

## 环境准备

### 创建虚拟环境（可选）

+ 通过`conda`创建虚拟环境

```bash
>> conda create -n score python==3.8.13

Collecting package metadata (current_repodata.json): done
Solving environment: failed with repodata from current_repodata.json, will retry with next repodata source.
Collecting package metadata (repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 4.10.3
  latest version: 23.3.1

Please update conda by running

    $ conda update -n base -c defaults conda



## Package Plan ##

  environment location: /Users/lubberit/anaconda3/envs/score

  added / updated specs:
    - python==3.8.13


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    ca-certificates-2023.01.10 |       hecd8cb5_0         121 KB
    ncurses-6.4                |       hcec6c5f_0        1018 KB
    openssl-1.1.1t             |       hca72f7f_0         3.3 MB
    pip-23.0.1                 |   py38hecd8cb5_0         2.5 MB
    python-3.8.13              |       hdfd78df_1        10.8 MB
    setuptools-66.0.0          |   py38hecd8cb5_0         1.2 MB
    sqlite-3.41.2              |       h6c40b1e_0         1.2 MB
    wheel-0.38.4               |   py38hecd8cb5_0          65 KB
    xz-5.4.2                   |       h6c40b1e_0         372 KB
    ------------------------------------------------------------
                                           Total:        20.5 MB

The following NEW packages will be INSTALLED:

  ca-certificates    pkgs/main/osx-64::ca-certificates-2023.01.10-hecd8cb5_0
  libcxx             pkgs/main/osx-64::libcxx-14.0.6-h9765a3e_0
  libffi             pkgs/main/osx-64::libffi-3.3-hb1e8313_2
  ncurses            pkgs/main/osx-64::ncurses-6.4-hcec6c5f_0
  openssl            pkgs/main/osx-64::openssl-1.1.1t-hca72f7f_0
  pip                pkgs/main/osx-64::pip-23.0.1-py38hecd8cb5_0
  python             pkgs/main/osx-64::python-3.8.13-hdfd78df_1
  readline           pkgs/main/osx-64::readline-8.2-hca72f7f_0
  setuptools         pkgs/main/osx-64::setuptools-66.0.0-py38hecd8cb5_0
  sqlite             pkgs/main/osx-64::sqlite-3.41.2-h6c40b1e_0
  tk                 pkgs/main/osx-64::tk-8.6.12-h5d9f67b_0
  wheel              pkgs/main/osx-64::wheel-0.38.4-py38hecd8cb5_0
  xz                 pkgs/main/osx-64::xz-5.4.2-h6c40b1e_0
  zlib               pkgs/main/osx-64::zlib-1.2.13-h4dc903c_0


Proceed ([y]/n)? y


Downloading and Extracting Packages
sqlite-3.41.2        | 1.2 MB    | ################################################################################################### | 100% 
wheel-0.38.4         | 65 KB     | ################################################################################################### | 100% 
openssl-1.1.1t       | 3.3 MB    | ################################################################################################### | 100% 
python-3.8.13        | 10.8 MB   | ################################################################################################### | 100% 
setuptools-66.0.0    | 1.2 MB    | ################################################################################################### | 100% 
ncurses-6.4          | 1018 KB   | ################################################################################################### | 100% 
xz-5.4.2             | 372 KB    | ################################################################################################### | 100% 
ca-certificates-2023 | 121 KB    | ################################################################################################### | 100% 
pip-23.0.1           | 2.5 MB    | ################################################################################################### | 100% 
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate score
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

+ 通过`pyenv`创建虚拟环境

```bash
# 安装环境
>> pyenv install -v 3.8.13
# 启动环境
>> pyenv local 3.8.13
# 卸载环境
>> pyenv uninstall 3.8.13
```


### 安装项目依赖

```bash
>> pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

Looking in indexes: http://mirrors.aliyun.com/pypi/simple/
......
Installing collected packages: webencodings, six, pytz, colour, zipp, tomli, tinycss2, threadpoolctl, python-dateutil, pyparsing, pycparser, pluggy, pillow, packaging, numpy, kiwisolver, joblib, iniconfig, graphviz, fonttools, exceptiongroup, et-xmlfile, defusedxml, cycler, scipy, pytest, patsy, pandas, openpyxl, importlib-resources, cssselect2, contourpy, cffi, statsmodels, scikit-learn, matplotlib, cairocffi, dtreeviz, category-encoders, CairoSVG
Successfully installed CairoSVG-2.7.0 cairocffi-1.5.1 category-encoders-2.6.0 cffi-1.15.1 colour-0.1.5 contourpy-1.0.7 cssselect2-0.7.0 cycler-0.11.0 defusedxml-0.7.1 dtreeviz-2.2.1 et-xmlfile-1.1.0 exceptiongroup-1.1.1 fonttools-4.39.4 graphviz-0.20.1 importlib-resources-5.12.0 iniconfig-2.0.0 joblib-1.2.0 kiwisolver-1.4.4 matplotlib-3.7.1 numpy-1.22.2 openpyxl-3.0.7 packaging-23.1 pandas-1.5.3 patsy-0.5.3 pillow-9.5.0 pluggy-1.0.0 pycparser-2.21 pyparsing-3.0.9 pytest-7.3.1 python-dateutil-2.8.2 pytz-2023.3 scikit-learn-1.2.2 scipy-1.10.1 six-1.11.0 statsmodels-0.14.0 threadpoolctl-3.1.0 tinycss2-1.2.1 tomli-2.0.1 webencodings-0.5.1 zipp-3.15.0
```


### `PDTR` 安装

```bash
pip install pdtr
```

### 版本介绍

+ `0.1.0`

仅包含决策树策略挖掘相关工具

+ `0.1.1`

除版本 `0.1.0` 中的决策树挖掘相关工具以外，新增了基于 `toad` 和 `optbinning` 的单变量策略挖掘相关方法


### 运行样例

+ 导入相关依赖

```python
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from pdtr import ParseDecisionTreeRules
except ModuleNotFoundError:
    import sys
    
    sys.path.append("../")
    from pdtr import ParseDecisionTreeRules
    
np.random.seed(1)
```

+ 数据集加载

```python
feature_map = {}
n_samples = 10000
ab = np.array(list('ABCDEFG'))

data = pd.DataFrame({
    'A': np.random.randint(10, size = n_samples),
    'B': ab[np.random.choice(7, n_samples)],
    'C': ab[np.random.choice(2, n_samples)],
    'D': np.random.random(size = n_samples),
    'target': np.random.randint(2, size = n_samples)
})
```

+ 数据集拆分

```python
train, test = train_test_split(data, test_size=0.3, shuffle=data["target"])
```

+ 决策树自动规则挖掘

```python
pdtr_instance = ParseDecisionTreeRules(target="target", max_iter=8, output="model_report/决策树组合策略挖掘.xlsx")
pdtr_instance.fit(train, lift=0., max_depth=2, max_samples=1., verbose=False, max_features="auto")
```

+ 规则验证

```python
all_rules = pdtr_instance.insert_all_rules(test=test)
```

+ 导出策略挖掘报告

```python
pdtr_instance.save()
```

+ 挖掘报告

[`examples/决策树组合策略挖掘.xlsx`](https://github.com/itlubber/pdtr/blob/main/examples/model_report/%E5%86%B3%E7%AD%96%E6%A0%91%E7%BB%84%E5%90%88%E7%AD%96%E7%95%A5%E6%8C%96%E6%8E%98.xlsx)


## 参考

> https://github.com/itlubber/LogisticRegressionPipeline
> 
> https://github.com/itlubber/itlubber-excel-writer
