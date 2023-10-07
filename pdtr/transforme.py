# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/15 17:55
@Author  : itlubber
@Site    : itlubber.art
"""

import warnings
import os
import re
import graphviz
import dtreeviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl.worksheet.worksheet import Worksheet

import toad
import category_encoders as ce
from optbinning import OptimalBinning
from sklearn.tree import DecisionTreeClassifier
from scorecardpipeline import ExcelWriter, get_column_letter, init_setting, Combiner, feature_bin_stats, bin_plot, dataframe2excel


class ParseDecisionTreeRules:

    def __init__(self, target="target", labels=["positive", "negative"], feature_map={}, nan=-1., max_iter=128, writer=None, combiner=None, seed=None, theme_color="2639E9"):
        """决策树自动规则挖掘工具包

        :param target: 数据集中好坏样本标签列名称，默认 target
        :param labels: 好坏样本标签名称，传入一个长度为2的列表，第0个元素为好样本标签，第1个元素为坏样本标签，默认 ["positive", "negative"]
        :param feature_map: 变量名称及其含义，在后续输出报告和策略信息时增加可读性，默认 {}
        :param nan: 在决策树策略挖掘时，默认空值填充的值，默认 -1
        :param max_iter: 最多支持在数据集上训练多少颗树模型，每次生成一棵树后，会剔除特征重要性最高的特征后，再生成树，默认 128
        :param output: excel 挖掘报告保存的路径， 默认 model_report/决策树组合策略挖掘.xlsx
        :param writer: 在之前程序运行时生成的 ExcelWriter，可以支持传入一个已有的writer，后续所有内容将保存至该workbook中，默认 None
        :param combiner: 可以传入提前训练好的 combiner，支持 toad.transform.Combiner 和 笔者重写的 Combiner
        """
        self.seed = seed
        self.theme_color = theme_color
        self.target = target
        self.labels = labels
        self.feature_map = feature_map
        self.nan = nan
        self.max_iter = max_iter
        self.decision_trees = []
        if combiner:
            self.combiner = combiner
        else:
            self.combiner = Combiner()
        self.target_enc = None
        self.feature_names = None
        self.dt_rules = pd.DataFrame()
        self.end_row = 2
        self.start_col = 2
        self.describe_columns = ["组合策略", "命中数", "命中率", "好样本数", "好样本占比", "坏样本数", "坏样本占比", "坏率", "样本整体坏率", "LIFT值"]

        init_setting(seed=self.seed)

        if writer:
            self.writer = writer
        else:
            self.writer = ExcelWriter(theme_color=self.theme_color)

    def encode_cat_features(self, X, y):
        cat_features = list(set(X.select_dtypes(include=[object, pd.CategoricalDtype]).columns))
        cat_features_index = [i for i, f in enumerate(X.columns) if f in cat_features]

        if len(cat_features) > 0:
            if self.target_enc is None:
                self.target_enc = ce.TargetEncoder(cols=cat_features)
                self.target_enc.fit(X[cat_features], y)
                self.target_enc.target_mapping = {}
                X_TE = X.join(self.target_enc.transform(X[cat_features]).add_suffix('_target'))
                for col in cat_features:
                    mapping = X_TE[[col, f"{col}_target"]].drop_duplicates()
                    self.target_enc.target_mapping[col] = dict(zip(mapping[col], mapping[f"{col}_target"]))
            else:
                X_TE = X.join(self.target_enc.transform(X[cat_features]).add_suffix('_target'))

            X_TE = X_TE.drop(columns=cat_features)
            return X_TE.rename(columns={f"{c}_target": c for c in cat_features})
        else:
            return X

    def get_dt_rules(self, tree, feature_names, total_bad_rate, total_count):
        tree_ = tree.tree_
        left = tree.tree_.children_left
        right = tree.tree_.children_right
        feature_name = [feature_names[i] if i != -2 else "undefined!" for i in tree_.feature]
        rules = dict()

        global res_df
        res_df = pd.DataFrame()

        def recurse(node, depth, parent):  # 搜每个节点的规则

            if tree_.feature[node] != -2:  # 非叶子节点,搜索每个节点的规则
                name = feature_name[node]
                thd = np.round(tree_.threshold[node], 3)
                s = "{} <= {} ".format(name, thd, node)
                # 左子
                if node == 0:
                    rules[node] = s
                else:
                    rules[node] = rules[parent] + ' & ' + s
                recurse(left[node], depth + 1, node)
                s = "{} > {}".format(name, thd)
                # 右子 
                if node == 0:
                    rules[node] = s
                else:
                    rules[node] = rules[parent] + ' & ' + s
                recurse(right[node], depth + 1, node)
            else:
                df = pd.DataFrame()
                df['组合策略'] = rules[parent],
                df['好样本数'] = tree_.value[node][0][0].astype(int)
                df['好样本占比'] = df['好样本数'] / (total_count * (1 - total_bad_rate))
                df['坏样本数'] = tree_.value[node][0][1].astype(int)
                df['坏样本占比'] = df['坏样本数'] / (total_count * total_bad_rate)
                df['命中数'] = df['好样本数'] + df['坏样本数']
                df['命中率'] = df['命中数'] / total_count
                df['坏率'] = df['坏样本数'] / df['命中数']
                df['样本整体坏率'] = total_bad_rate
                df['LIFT值'] = df['坏率'] / df['样本整体坏率']

                global res_df

                res_df = pd.concat([res_df, df], axis=0)

        recurse(0, 1, 0)

        return res_df.sort_values("LIFT值", ascending=True)[self.describe_columns].reset_index(drop=True)

    def select_dt_rules(self, decision_tree, x, y, lift=0., max_samples=1., labels=["positive", "negative"], save=None, verbose=False, drop=False):
        rules = self.get_dt_rules(decision_tree, x.columns, sum(y) / len(y), len(y))
        total_rules = len(rules)

        try:
            viz_model = dtreeviz.model(decision_tree,
                                       X_train=x,
                                       y_train=y,
                                       feature_names=x.columns,
                                       target_name=self.target,
                                       class_names=labels,
                                       )
        except AttributeError:
            print("请检查 dtreeviz 版本")

        rules = rules.query(f"LIFT值 >= {lift} & 命中率 <= {max_samples}").reset_index(drop=True)

        if len(rules) > 0:
            decision_tree_viz = viz_model.view(
                scale=1.5,
                orientation='LR',
                colors={
                    "classes": [None, None, ["#2639E9", "#F76E6C"], ["#2639E9", "#F76E6C", "#FE7715", "#FFFFFF"]],
                    "arrow": "#2639E9",
                    'text_wedge': "#F76E6C",
                    "pie": "#2639E9",
                    "tile_alpha": 1,
                    "legend_edge": "#FFFFFF",
                },
                ticks_fontsize=10,
                label_fontsize=10,
            )
            if verbose:
                from IPython.core.display_functions import display
                if self.feature_map is not None and len(self.feature_map) > 0:
                    display(rules.replace(self.feature_map, regex=True))
                else:
                    display(rules)
                display(decision_tree_viz)
            if save:
                if os.path.dirname(save) and not os.path.exists(os.path.dirname(save)):
                    os.makedirs(os.path.dirname(save))

                try:
                    decision_tree_viz.save("combine_rules_cache.svg")
                except graphviz.backend.execute.ExecutableNotFound:
                    print("请确保您已安装 graphviz 程序并且正确配置了 PATH 路径。可参考: https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft")

                try:
                    import cairosvg
                    cairosvg.svg2png(url="combine_rules_cache.svg", write_to=save, dpi=240)
                except:
                    from reportlab.graphics import renderPDF
                    from svglib.svglib import svg2rlg
                    drawing = svg2rlg("combine_rules_cache.svg")
                    renderPDF.drawToFile(drawing, save, dpi=240, fmt="PNG")

        if drop:
            if len(rules) > 0:
                return rules, decision_tree.feature_names_in_[list(decision_tree.feature_importances_).index(max(decision_tree.feature_importances_))], total_rules
            else:
                return rules, decision_tree.feature_names_in_[list(decision_tree.feature_importances_).index(min(decision_tree.feature_importances_))], total_rules
        else:
            return rules, total_rules

    def query_dt_rules(self, x, y, parsed_rules=None):
        total_count = len(y)
        total_bad_rate = y.sum() / len(y)

        rules = pd.DataFrame()

        if isinstance(parsed_rules, pd.DataFrame):
            parsed_rules = parsed_rules["组合策略"].unique()

        for rule in parsed_rules:
            select_index = x.query(rule).index
            if len(select_index) > 0:
                y_select = y[select_index]
                df = pd.Series()
                df['组合策略'] = rule
                df['好样本数'] = len(y_select) - y_select.sum()
                df['好样本占比'] = df['好样本数'] / (total_count * (1 - total_bad_rate))
                df['坏样本数'] = y_select.sum()
                df['坏样本占比'] = df['坏样本数'] / (total_count * total_bad_rate)
                df['命中数'] = df['好样本数'] + df['坏样本数']
                df['命中率'] = df['命中数'] / total_count
                df['坏率'] = df['坏样本数'] / df['命中数']
                df['样本整体坏率'] = total_bad_rate
                df['LIFT值'] = df['坏率'] / df['样本整体坏率']
            else:
                df = pd.Series({'组合策略': rule, '好样本数': 0, '好样本占比': 0., '坏样本数': 0, '坏样本占比': 0., '命中数': 0, '命中率': 0., '坏率': 0., '样本整体坏率': total_bad_rate, 'LIFT值': 0., })

            rules = pd.concat([rules, pd.DataFrame(df).T]).reset_index(drop=True)

        return rules[self.describe_columns]

    def insert_dt_rules(self, parsed_rules, end_row, start_col, save=None, sheet=None, figsize=(500, 350)):
        if isinstance(sheet, Worksheet):
            worksheet = sheet
        else:
            worksheet = self.writer.get_sheet_by_name(sheet or "决策树组合策略挖掘")

        end_row, end_col = dataframe2excel(parsed_rules, self.writer, sheet_name=worksheet, start_row=end_row + 1, start_col=start_col, percent_cols=['好样本占比', '坏样本占比', '命中率', '坏率', '样本整体坏率', 'LIFT值'], condition_cols=["坏率", "LIFT值"])

        if save is not None:
            end_row, end_col = self.writer.insert_pic2sheet(worksheet, save, (end_row + 1, start_col), figsize=figsize)

        return end_row, end_col

    def fit(self, x, y=None, max_depth=2, lift=0., max_samples=1., min_score=None, verbose=False, *args, **kwargs):
        """组合策略挖掘

        :param x: 包含标签的数据集
        :param max_depth: 决策树最大深度，即最多组合的特征个数，默认 2
        :param lift: 组合策略最小的lift值，默认 0.，即全部组合策略
        :param max_samples: 每条组合策略的最大样本占比，默认 1.0，即全部组合策略
        :param min_score: 决策树拟合时最小的auc，如果不满足则停止后续生成决策树
        :param verbose: 是否调试模式，仅在 jupyter 环境有效
        :param kwargs: DecisionTreeClassifier 参数
        """
        worksheet = self.writer.get_sheet_by_name("策略详情")

        y = x[self.target]
        X_TE = self.encode_cat_features(x.drop(columns=[self.target]), y)
        X_TE = X_TE.fillna(self.nan)

        self.feature_names = list(X_TE.columns)

        for i in range(self.max_iter):
            decision_tree = DecisionTreeClassifier(max_depth=max_depth, *args, **kwargs)
            decision_tree = decision_tree.fit(X_TE, y)

            if (min_score is not None and decision_tree.score(X_TE, y) < min_score) or len(X_TE.columns) < max_depth:
                break

            try:
                parsed_rules, remove, total_rules = self.select_dt_rules(decision_tree, X_TE, y, lift=lift, max_samples=max_samples, labels=self.labels, verbose=verbose, save=f"model_report/auto_mining_rules/combiner_rules_{i}.png", drop=True)

                if len(parsed_rules) > 0:
                    self.dt_rules = pd.concat([self.dt_rules, parsed_rules]).reset_index(drop=True)

                    if self.writer is not None:
                        if self.feature_map is not None and len(self.feature_map) > 0:
                            parsed_rules["组合策略"] = parsed_rules["组合策略"].replace(self.feature_map, regex=True)
                        self.end_row, _ = self.insert_dt_rules(parsed_rules, self.end_row, self.start_col, save=f"model_report/auto_mining_rules/combiner_rules_{i}.png", figsize=(500, 100 * total_rules), sheet=worksheet)

                X_TE = X_TE.drop(columns=remove)
                self.decision_trees.append(decision_tree)
            except:
                import traceback
                traceback.print_exc()

        if len(self.dt_rules) <= 0:
            print(f"未挖掘到有效策略, 可以考虑适当调整预设的筛选参数, 降低 lift / 提高 max_samples, 当前筛选标准为: 提取 lift >= {lift} 且 max_samples <= {max_samples} 的策略")

        return self

    def transform(self, x, y=None):
        y = x[self.target]
        X_TE = self.encode_cat_features(x.drop(columns=[self.target]), y)
        X_TE = X_TE.fillna(self.nan)
        if self.dt_rules is not None and len(self.dt_rules) > 0:
            parsed_rules = self.query_dt_rules(X_TE, y, parsed_rules=self.dt_rules)
            if self.feature_map is not None and len(self.feature_map) > 0:
                parsed_rules["组合策略"] = parsed_rules["组合策略"].replace(self.feature_map, regex=True)
            return parsed_rules
        else:
            return pd.DataFrame(columns=self.describe_columns)

    def insert_all_rules(self, val=None, test=None, sheet="策略汇总"):
        """组合策略插入excel文档

        :param val: 验证数据集
        :param test: 测试数据集

        :return: 返回每个数据集组合策略命中情况
        """
        worksheet = self.writer.get_sheet_by_name(sheet or "决策树组合策略挖掘")

        if sheet:
            self.writer.workbook.move_sheet(sheet, -1)

        parsed_rules_train = self.dt_rules.copy()
        if self.feature_map is not None and len(self.feature_map) > 0:
            parsed_rules_train["组合策略"] = parsed_rules_train["组合策略"].replace(self.feature_map, regex=True)
        self.end_row, _ = self.writer.insert_value2sheet(worksheet, (2 if sheet else self.end_row + 2, self.start_col), value="训练集决策树组合策略", style="header_middle")
        self.end_row, _ = self.insert_dt_rules(parsed_rules_train, self.end_row, self.start_col, sheet=worksheet)
        outputs = (parsed_rules_train,)

        if len(parsed_rules_train) > 0:
            if val is not None:
                parsed_rules_val = self.transform(val)
                self.end_row, _ = self.writer.insert_value2sheet(worksheet, (self.end_row + 2, self.start_col), value="验证集决策树组合策略", style="header_middle")
                self.end_row, _ = self.insert_dt_rules(parsed_rules_val, self.end_row, self.start_col, sheet=worksheet)
                outputs = outputs + (parsed_rules_val,)

            if test is not None:
                parsed_rules_test = self.transform(test)
                self.end_row, _ = self.writer.insert_value2sheet(worksheet, (self.end_row + 2, self.start_col), value="测试集决策树组合策略", style="header_middle")
                self.end_row, _ = self.insert_dt_rules(parsed_rules_test, self.end_row, self.start_col, sheet=worksheet)
                outputs = outputs + (parsed_rules_test,)
        else:
            if val is not None:
                outputs = outputs + (parsed_rules_train,)

            if test is not None:
                outputs = outputs + (parsed_rules_train,)

        return outputs

    def query_feature_rule(self, data, feature, desc="", plot=False, figsize=(10, 6), save=None, *args, **kwargs):
        """传入数据集和其中一个特征名称，输出简单策略挖掘统计信息

        :param data: 数据集
        :param feature: 特征名称
        :param desc: 特征中文含义或其他相关信息
        :param bin_plot: 是否可视化特征分箱图
        :param figsize: 图像的尺寸
        :param save: 图像保存的路径

        :return: pd.DataFrame, 特征分箱的统计信息
        """
        feature_table = feature_bin_stats(data, feature, desc=desc, *args, **kwargs)

        if plot:
            bin_plot(feature_table, desc=desc, figsize=figsize, save=save)

        return feature_table
    
    @staticmethod
    def bin_plot(*args, **kwargs):
        bin_plot(*args, **kwargs)

    def save(self, output="model_report/决策树组合策略挖掘.xlsx"):
        self.writer.save(output)


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    feature_map = {}
    n_samples = 10000
    ab = np.array(list('ABCDEFG'))

    data = pd.DataFrame({
        'A': np.random.randint(10, size=n_samples),
        'B': ab[np.random.choice(7, n_samples)],
        'C': ab[np.random.choice(2, n_samples)],
        'D': np.random.random(size=n_samples),
        'target': np.random.randint(2, size=n_samples)
    })

    train, test = train_test_split(data, test_size=0.3, shuffle=data["target"])

    pdtr = ParseDecisionTreeRules(target="target", feature_map=feature_map, max_iter=8)
    pdtr.fit(train, lift=1., max_depth=2, max_samples=0.5, verbose=False, min_samples_split=8, min_samples_leaf=5, max_features="auto")
    pdtr.insert_all_rules(test=test)
    pdtr.save()
