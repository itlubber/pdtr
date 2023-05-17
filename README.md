# 自动决策树规则挖掘工具包



# 环境准备

## 创建虚拟环境（可选）

### 通过`conda`创建虚拟环境

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

### 通过`pyenv`创建虚拟环境

```bash
# 安装环境
>> pyenv install -v 3.8.13
# 启动环境
>> pyenv local 3.8.13
# 卸载环境
>> pyenv uninstall 3.8.13
```


## 安装项目依赖

```bash
>> pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

Looking in indexes: http://mirrors.aliyun.com/pypi/simple/
......
Installing collected packages: webencodings, six, pytz, colour, zipp, tomli, tinycss2, threadpoolctl, python-dateutil, pyparsing, pycparser, pluggy, pillow, packaging, numpy, kiwisolver, joblib, iniconfig, graphviz, fonttools, exceptiongroup, et-xmlfile, defusedxml, cycler, scipy, pytest, patsy, pandas, openpyxl, importlib-resources, cssselect2, contourpy, cffi, statsmodels, scikit-learn, matplotlib, cairocffi, dtreeviz, category-encoders, CairoSVG
Successfully installed CairoSVG-2.7.0 cairocffi-1.5.1 category-encoders-2.6.0 cffi-1.15.1 colour-0.1.5 contourpy-1.0.7 cssselect2-0.7.0 cycler-0.11.0 defusedxml-0.7.1 dtreeviz-2.2.1 et-xmlfile-1.1.0 exceptiongroup-1.1.1 fonttools-4.39.4 graphviz-0.20.1 importlib-resources-5.12.0 iniconfig-2.0.0 joblib-1.2.0 kiwisolver-1.4.4 matplotlib-3.7.1 numpy-1.22.2 openpyxl-3.0.7 packaging-23.1 pandas-1.5.3 patsy-0.5.3 pillow-9.5.0 pluggy-1.0.0 pycparser-2.21 pyparsing-3.0.9 pytest-7.3.1 python-dateutil-2.8.2 pytz-2023.3 scikit-learn-1.2.2 scipy-1.10.1 six-1.11.0 statsmodels-0.14.0 threadpoolctl-3.1.0 tinycss2-1.2.1 tomli-2.0.1 webencodings-0.5.1 zipp-3.15.0
```


## `PDTR` 安装

```bash
pip install pdtr
```


## 运行样例

### 数据集加载


### 数据集拆分


### 决策树自动规则挖掘


### 规则验证


### 导出策略挖掘报告
