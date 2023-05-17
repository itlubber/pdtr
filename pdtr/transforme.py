# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/15 17:55
@Author  : itlubber
@Site    : itlubber.art
"""

import os
import re
import cairosvg
import dtreeviz
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl.utils import get_column_letter, column_index_from_string

import toad
import category_encoders as ce
from optbinning import OptimalBinning
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import _tree, DecisionTreeClassifier, plot_tree, export_graphviz

from .excel_writer import ExcelWriter


warnings.filterwarnings("ignore")


class ParseDecisionTreeRules:
    
    def __init__(self, target="target", labels=["positive", "negative"], feature_map={}, nan=-1., max_iter=128, output="model_report/决策树组合策略挖掘.xlsx", writer=None):
        """决策树自动规则挖掘工具包
        
        """
        self.target = target
        self.labels = labels
        self.feature_map = feature_map
        self.nan = nan
        self.max_iter = max_iter
        self.output = output
        self.decision_trees = []
        self.combiner = toad.transform.Combiner()
        self.target_enc = None
        self.feature_names = None
        self.dt_rules = pd.DataFrame()
        self.end_row = 2
        self.start_col = 2
        self.describe_columns = ["组合策略", "命中数", "命中率", "好样本数", "好样本占比", "坏样本数", "坏样本占比", "坏率", "样本整体坏率", "LIFT值"]
        
        self.init_setting(font_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matplot_chinese.ttf'))
        
        if output:
            if writer:
                self.writer = writer
            else:
                self.writer = ExcelWriter(theme_color="2639E9")
            
            self.worksheet = self.writer.get_sheet_by_name("决策树组合策略挖掘")
    
    @staticmethod
    def init_setting(font_path=None):
        import matplotlib
        
        pd.options.display.float_format = '{:.4f}'.format
        pd.set_option('display.max_colwidth', 300)
        plt.style.use('seaborn-ticks')
        if font_path:
            from matplotlib import font_manager
            matplotlib.font_manager.fontManager.addfont(font_path)
            matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
        else:
            matplotlib.rcParams['font.family'] = ["KaiTi"]
        matplotlib.rcParams['axes.unicode_minus'] = False
    
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
        rules=dict()

        global res_df
        res_df = pd.DataFrame()

        def recurse(node, depth, parent): # 搜每个节点的规则

            if tree_.feature[node] != -2:  # 非叶子节点,搜索每个节点的规则
                name = feature_name[node]
                thd = np.round(tree_.threshold[node],3)
                s= "{} <= {} ".format( name, thd, node )
                # 左子
                if node == 0:
                    rules[node]=s
                else:
                    rules[node]=rules[parent]+' & ' +s
                recurse(left[node], depth + 1, node)
                s="{} > {}".format(name, thd)
                # 右子 
                if node == 0:
                    rules[node]=s
                else:
                    rules[node]=rules[parent]+' & ' +s
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

                res_df = pd.concat([res_df, df], 0)

        recurse(0, 1, 0)

        return res_df.sort_values("LIFT值", ascending=True)[self.describe_columns].reset_index(drop=True)
    
    def select_dt_rules(self, decision_tree, x, y, lift=3., max_samples=0.05, labels=["positive", "negative"], save=None, verbose=False, drop=False):
        rules = self.get_dt_rules(decision_tree, x.columns, sum(y) / len(y), len(y))
        viz_model = dtreeviz.model(decision_tree,
                                   X_train=x, 
                                   y_train=y,
                                   feature_names=x.columns,
                                   target_name=self.target, 
                                   class_names=labels,
                                  )
        
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
                if self.feature_map is not None and len(self.feature_map) > 0:
                    print(rules.replace(self.feature_map, regex=True))
                else:
                    print(rules)
            if save:
                if os.path.dirname(save) and not os.path.exists(os.path.dirname(save)):
                    os.makedirs(os.path.dirname(save))

                decision_tree_viz.save("combine_rules_cache.svg")
                cairosvg.svg2png(url="combine_rules_cache.svg", write_to=save, dpi=240)

        if drop:
            return rules, decision_tree.feature_names_in_[list(decision_tree.feature_importances_).index(max(decision_tree.feature_importances_))]
        else:
            return rules
    
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
                df = pd.Series({'组合策略': rule,'好样本数': 0,'好样本占比': 0.,'坏样本数': 0,'坏样本占比': 0.,'命中数': 0,'命中率': 0.,'坏率': 0.,'样本整体坏率': total_bad_rate,'LIFT值': 0.,})

            rules = pd.concat([rules, pd.DataFrame(df).T]).reset_index(drop=True)

        return rules[self.describe_columns]
    
    def insert_dt_rules(self, parsed_rules, end_row, start_col, save=None):
        end_row, end_col = self.writer.insert_df2sheet(self.worksheet, parsed_rules, (end_row + 2, start_col))

        for c in ['好样本占比', '坏样本占比', '命中率', '坏率', '样本整体坏率', 'LIFT值']:
            conditional_column = get_column_letter(start_col + parsed_rules.columns.get_loc(c))
            self.writer.set_number_format(self.worksheet, f"{conditional_column}{end_row - len(parsed_rules)}:{conditional_column}{end_row - 1}", "0.00%")
        for c in ["坏率", "LIFT值"]:
            conditional_column = get_column_letter(start_col + parsed_rules.columns.get_loc(c))
            self.writer.add_conditional_formatting(self.worksheet, f'{conditional_column}{end_row - len(parsed_rules)}', f'{conditional_column}{end_row - 1}')
        
        if save is not None:
            end_row, end_col = self.writer.insert_pic2sheet(self.worksheet, save, (end_row + 1, start_col), figsize=(400, 300))
        
        return end_row, end_col
        
    def fit(self, x, y=None, max_depth=2, lift=3, max_samples=0.2, min_score=None, verbose=False, **kwargs):
        y = x[self.target]
        X_TE = self.encode_cat_features(x.drop(columns=[self.target]), y)
        X_TE = X_TE.fillna(self.nan)
        
        self.feature_names = list(X_TE.columns)
        
        for i in range(self.max_iter):
            decision_tree = DecisionTreeClassifier(max_depth=max_depth, **kwargs)
            decision_tree = decision_tree.fit(X_TE, y)
            
            if (min_score is not None and decision_tree.score(X_TE, y) < min_score) or len(X_TE.columns) < max_depth:
                break
            
            try:
                parsed_rules, remove = self.select_dt_rules(decision_tree, X_TE, y, lift=lift, max_samples=max_samples, labels=self.labels, verbose=verbose, save=f"model_report/auto_mining_rules/combiner_rules_{i}.png", drop=True)

                if len(parsed_rules) > 0:
                    self.dt_rules = pd.concat([self.dt_rules, parsed_rules]).reset_index(drop=True)

                    if self.writer is not None:
                        if self.feature_map is not None and len(self.feature_map) > 0:
                            parsed_rules["组合策略"] = parsed_rules["组合策略"].replace(self.feature_map, regex=True)
                        self.end_row, _ = self.insert_dt_rules(parsed_rules, self.end_row, self.start_col, save=f"model_report/auto_mining_rules/combiner_rules_{i}.png")

                X_TE = X_TE.drop(columns=remove)
                self.decision_trees.append(decision_tree)
            except:
                pass
        
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
    
    def insert_all_rules(self, val=None, test=None):
        parsed_rules_train = self.dt_rules.copy()
        if self.feature_map is not None and len(self.feature_map) > 0:
            parsed_rules_train["组合策略"] = parsed_rules_train["组合策略"].replace(self.feature_map, regex=True)
        self.end_row, _ = self.writer.insert_value2sheet(self.worksheet, (self.end_row + 2, self.start_col), value="训练集决策树组合策略")
        self.end_row, _ = self.insert_dt_rules(parsed_rules_train, self.end_row, self.start_col)
        outputs = (parsed_rules_train,)
        
        if val is not None:
            parsed_rules_val = self.transform(val)
            self.end_row, _ = self.writer.insert_value2sheet(self.worksheet, (self.end_row + 2, self.start_col), value="验证集决策树组合策略")
            self.end_row, _ = self.insert_dt_rules(parsed_rules_val, self.end_row, self.start_col)
            outputs = outputs + (parsed_rules_val,)
        
        if test is not None:
            parsed_rules_test = self.transform(test)
            self.end_row, _ = self.writer.insert_value2sheet(self.worksheet, (self.end_row + 2, self.start_col), value="测试集决策树组合策略")
            self.end_row, _ = self.insert_dt_rules(parsed_rules_test, self.end_row, self.start_col)
            outputs = outputs + (parsed_rules_test,)
            
        return outputs
    
    def save(self):
        self.writer.save(self.output)
    
    @staticmethod
    def feature_bins(bins):
        if isinstance(bins, list): bins = np.array(bins)
        EMPTYBINS = len(bins) if not isinstance(bins[0], (set, list, np.ndarray)) else -1
        
        l = []
        if np.issubdtype(bins.dtype, np.number):
            has_empty = len(bins) > 0 and np.isnan(bins[-1])
            if has_empty: bins = bins[:-1]
            sp_l = ["负无穷"] + bins.tolist() + ["正无穷"]
            for i in range(len(sp_l) - 1): l.append('['+str(sp_l[i])+' , '+str(sp_l[i+1])+')')
            if has_empty: l.append('缺失值')
        else:
            for keys in bins:
                keys_update = set()
                for key in keys:
                    if pd.isnull(key) or key == "nan":
                        keys_update.add("缺失值")
                    elif key.strip() == "":
                        keys_update.add("空字符串")
                    else:
                        keys_update.add(key)
                label = ','.join(keys_update)
                l.append(label)

        return {i if b != "缺失值" else EMPTYBINS: b for i, b in enumerate(l)}
    
    def feature_bin_stats(self, data, feature, rules={}, min_n_bins=2, max_n_bins=3, max_n_prebins=10, min_prebin_size=0.02, min_bin_size=0.05, max_bin_size=None, gamma=0.01, monotonic_trend="auto_asc_desc", desc="", method='chi', verbose=0, combiner=None, ks=False):
        if method not in ['dt', 'chi', 'quantile', 'step', 'kmeans', 'cart']:
            raise "method is the one of ['dt', 'chi', 'quantile', 'step', 'kmeans', 'cart']"
        
        if data[feature].dropna().nunique() <= min_n_bins:
            splits = []
            for v in data[feature].dropna().unique():
                splits.append(v)

            if str(data[feature].dtypes) in ["object", "string", "category"]:
                rule = {feature: [[s] for s in splits]}
                rule[feature].append([[np.nan]])
            else:
                rule = {feature: sorted(splits) + [np.nan]}
        else:
            if method == "cart":
                y = data[self.target]
                if str(data[feature].dtypes) in ["object", "string", "category"]:
                    dtype = "categorical"
                    x = data[feature].astype("category").values
                else:
                    dtype = "numerical"
                    x = data[feature].values

                _combiner = OptimalBinning(feature, dtype=dtype, min_n_bins=min_n_bins, max_n_bins=max_n_bins, max_n_prebins=max_n_prebins, min_prebin_size=min_prebin_size, min_bin_size=min_bin_size, max_bin_size=max_bin_size, monotonic_trend=monotonic_trend, gamma=gamma).fit(x, y)
                if _combiner.status == "OPTIMAL":
                    rule = {feature: [s.tolist() if isinstance(s, np.ndarray) else s for s in _combiner.splits] + [[np.nan] if dtype == "categorical" else np.nan]}
                else:
                    _combiner = toad.transform.Combiner()
                    _combiner.fit(data[[feature, self.target]].dropna(), self.target, method="chi", min_samples=min_bin_size, n_bins=max_n_bins)
                    rule = {feature: [s.tolist() if isinstance(s, np.ndarray) else s for s in _combiner[feature]] + [[np.nan] if dtype == "categorical" else np.nan]}
            else:
                _combiner = toad.transform.Combiner()
                if method in ("step", "quantile"):
                    _combiner.fit(data[[feature, self.target]].dropna(), self.target, method=method, n_bins=max_n_bins)
                else:
                    _combiner.fit(data[[feature, self.target]].dropna(), self.target, method=method, min_samples=min_bin_size, n_bins=max_n_bins)
                rule = {feature: [s.tolist() if isinstance(s, np.ndarray) else s for s in _combiner[feature]] + [[np.nan] if str(data[feature].dtypes) in ["object", "string", "category"] else np.nan]}
        
        self.combiner.update(rule)
        
        if verbose > 0:
            print(data[feature].describe())

        if rules and isinstance(rules, list): rules = {feature: rules}
        if rules and isinstance(rules, dict): self.combiner.update(rules)

        feature_bin = self.combiner[feature]
        feature_bin_dict = self.feature_bins(np.array(feature_bin))
        
        df_bin = self.combiner.transform(data[[feature, self.target]], labels=False)
        
        table = df_bin[[feature, self.target]].groupby([feature, self.target]).agg(len).unstack()
        table.columns.name = None
        table = table.rename(columns = {0 : '好样本数', 1 : '坏样本数'}).fillna(0)
        if "好样本数" not in table.columns:
            table["好样本数"] = 0
        if "坏样本数" not in table.columns:
            table["坏样本数"] = 0
        
        table["指标名称"] = feature
        table["指标含义"] = desc
        table = table.reset_index().rename(columns={feature: "分箱", "index": "分箱"})

        table['样本总数'] = table['好样本数'] + table['坏样本数']
        table['样本占比'] = table['样本总数'] / table['样本总数'].sum()
        table['好样本占比'] = table['好样本数'] / table['好样本数'].sum()
        table['坏样本占比'] = table['坏样本数'] / table['坏样本数'].sum()
        table['坏样本率'] = table['坏样本数'] / table['样本总数']
        
        table = table.fillna(0.)
        
        table['分档WOE值'] = table.apply(lambda x : np.log(x['好样本占比'] / (x['坏样本占比'] + 1e-6)),axis=1)
        table['分档IV值'] = table.apply(lambda x : (x['好样本占比'] - x['坏样本占比']) * np.log(x['好样本占比'] / (x['坏样本占比'] + 1e-6)), axis=1)
        
        table = table.replace(np.inf, 0).replace(-np.inf, 0)
        
        table['指标IV值'] = table['分档IV值'].sum()
        
        table["LIFT值"] = table['坏样本率'] / (table["坏样本数"].sum() / table["样本总数"].sum())
        table["累积LIFT值"] = (table['坏样本数'].cumsum() / table['样本总数'].cumsum()) / (table["坏样本数"].sum() / table["样本总数"].sum())
        
        if ks:
            table = table.sort_values("分箱")
            table["累积好样本数"] = table["好样本数"].cumsum()
            table["累积坏样本数"] = table["坏样本数"].cumsum()
            table["分档KS值"] = table["累积坏样本数"] / table['坏样本数'].sum() - table["累积好样本数"] / table['好样本数'].sum()
        
        table["分箱"] = table["分箱"].map(feature_bin_dict)
        table = table.set_index(['指标名称', '指标含义', '分箱']).reindex([(feature, desc, b) for b in feature_bin_dict.values()]).fillna(0).reset_index()
        
        if ks:
            return table[['指标名称', "指标含义", '分箱', '样本总数', '样本占比', '好样本数', '好样本占比', '坏样本数', '坏样本占比', '坏样本率', '分档WOE值', '分档IV值', '指标IV值', 'LIFT值', '累积LIFT值', '累积好样本数', '累积坏样本数', '分档KS值']]
        else:
            return table[['指标名称', "指标含义", '分箱', '样本总数', '样本占比', '好样本数', '好样本占比', '坏样本数', '坏样本占比', '坏样本率', '分档WOE值', '分档IV值', '指标IV值', 'LIFT值', '累积LIFT值']]

    @staticmethod
    def bin_plot(feature_table, desc="", figsize=(10, 6), colors=["#2639E9", "#F76E6C", "#FE7715"], max_len=35, save=None):
        feature_table = feature_table.copy()

        feature_table["分箱"] = feature_table["分箱"].apply(lambda x: x if re.match("^\[.*\)$", x) else str(x)[:max_len] + "..")

        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.barh(feature_table['分箱'], feature_table['好样本数'], color=colors[0], label='好样本', hatch="/")
        ax1.barh(feature_table['分箱'], feature_table['坏样本数'], left=feature_table['好样本数'], color=colors[1], label='坏样本', hatch="\\")
        ax1.set_xlabel('样本数')

        ax2 = ax1.twiny()
        ax2.plot(feature_table['坏样本率'], feature_table['分箱'], colors[2], label='坏样本率', linestyle='-.')
        ax2.set_xlabel('坏样本率: 坏样本数 / 样本总数')

        for i, rate in enumerate(feature_table['坏样本率']):
            ax2.scatter(rate, i, color=colors[2])

        for i, v in feature_table[['样本总数', '好样本数', '坏样本数', '坏样本率']].iterrows():
            ax1.text(v['样本总数'] / 2, i + len(feature_table) / 60, f"{int(v['好样本数'])}:{int(v['坏样本数'])}:{v['坏样本率']:.2%}")

        ax1.invert_yaxis()

        fig.suptitle(f'{desc}分箱图\n\n')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, 0.94), frameon=False)

        plt.tight_layout()

        if save:
            if os.path.dirname(save) and not os.path.exists(os.path.dirname(save)):
                os.makedirs(os.path.dirname(save))

            fig.savefig(save, dpi=240, format="png", bbox_inches="tight")
            
    def query_feature_rule(self, data, feature, desc="", bin_plot=False, figsize=(10, 6), save=None, *args, **kwargs):
        feature_table = self.feature_bin_stats(data, feature, desc=desc, *args, **kwargs)
        
        if bin_plot:
            self.bin_plot(feature_table, desc=desc, figsize=figsize, save=save)
        
        return feature_table


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    
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

    train, test = train_test_split(data, test_size=0.3, shuffle=data["target"])
    
    pdtr = ParseDecisionTreeRules(target="target", feature_map=feature_map, max_iter=8)
    pdtr.fit(train, lift=3., max_depth=2, max_samples=0.1, verbose=False, min_samples_split=8, min_samples_leaf=5, max_features="auto")
    pdtr.insert_all_rules(test=test)
    pdtr.save()
