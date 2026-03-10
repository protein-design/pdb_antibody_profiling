import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

class PlotPredict:

    def __init__(self, df, verbose:bool=True):
        self.df = df
        self.verbose = verbose
    
    def violin(self, ax, params):
        colx = params.get('colx')
        coly = params.get('coly')
        sns.violinplot(self.df, x=colx, y=coly, ax=ax,
            color='black', fill=False, inner='quart')
        if 'xlabel' in params:
            ax.set_xlabel(params['xlabel'])
        if 'ylabel' in params:
            ax.set_ylabel(params['ylabel'])
        xtick_labels = ax.get_xticklabels()
        ax.set_xticks(range(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, rotation=45, ha='center')

        q = params.get('quantile', .95)
        qval = np.quantile(self.df[coly], 1-q)
        print(f'{q} quantile of {colx} is {qval}')
        ax.axhline(qval, linestyle='--')
        return ax
    
    def dot(self, ax, params):
        col1 = params.get('col1')
        col2 = params.get('col2')
        chain_type = params.get('chain_type')
        df = self.df[self.df['chain_type']==chain_type] if chain_type else self.df
        sns.scatterplot(df, x=col1, y=col2, ax=ax,
            color='black', alpha=.2, s=10)
        if 'xlabel' in params:
            ax.set_xlabel(params['xlabel'])
        if 'ylabel' in params:
            ax.set_ylabel(params['ylabel'])
        if 'xlim' in params:
            ax.set_xlabel(params['xlim'])
        if 'ylim' in params:
            ax.set_ylabel(params['ylim'])

        b, m = polyfit(df[col1], df[col2], 1)
        print(f"intercept={b}, slope={m}")
        ax.plot(df[col1], b+ m * df[col1], '-', color='grey')
        coef, pval = pearsonr(df[col1], df[col2])
        ax.set_title(f"$R^2$ = {coef*coef:.4f}", fontsize=8)
        return ax
    
    def violin_plddt(self, ax):
        sns.violinplot(self.df, y='chain_type', x='avg_plddt', ax=ax,
            color='black', fill=False, inner='quart', split=True)
        ax.set_xlabel('Average pLDDT')
        ax.set_ylabel(None)
        return ax

    def violin_ptm(self, ax):
        sns.violinplot(self.df, y='chain_type', x='avg_ptm', ax=ax,
            color='black', fill=False, inner='quart', split=True)
        ax.set_xlabel('Average pTM')
        ax.set_ylabel(None)
        return ax

    def dot_plddt_ptm(self, ax):
        sns.scatterplot(self.df, x='avg_plddt', y='avg_ptm', hue='chain_type',
            ax=ax, color='grey', alpha=.5, s=10)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1,1))
        ax.set_xlabel('Average pLDDT')
        ax.set_ylabel('Average pTM')
        ax.set_xlim(60,100)
        ax.set_ylim(.5,1)
        return ax

    def dot_chain_plddt_ptm(self, ax, chain_type):
        df = self.df[self.df['chain_type']==chain_type]
        sns.scatterplot(df, x='avg_plddt', y='avg_ptm',
            ax=ax, color='black', alpha=.5, s=10)
        ax.set_xlabel('Average pLDDT')
        ax.set_ylabel('Average pTM')
        ax.set_xlim(70,100)
        ax.set_ylim(.5,1)

        b, m = polyfit(df['avg_plddt'], df['avg_ptm'], 1)
        print(f"intercept={b}, slope={m}")
        ax.plot(df['avg_plddt'], b+ m * df['avg_plddt'], '-', color='grey')
        coef, pval = pearsonr(df['avg_plddt'], df['avg_ptm'])
        ax.set_title(f"$R^2$={coef*coef:.4f}", fontsize=8)
        return ax

    def violin_pdb_rmsd(self, ax):
        self.df['rmsd_adj'] = self.df['rmsd'] + 1e-5

        sns.violinplot(self.df, x='chain_status', y='rmsd_adj', ax=ax,
            log_scale=True, inner='quart', color='black', fill=False)
        ax.set_xlim(-.6, 2.5)
        ax.set_xlabel('PDB Post-processing')
        ax.set_ylabel('Log RMSD')
        
        x_offset= -.55
        n=np.quantile(self.df['rmsd_adj'], .95)
        ax.axhline(np.log(n), linestyle='--', color='grey')
        ax.text(x_offset, np.log(n)+1, round(n, 1))
        # n=5
        # ax.axhline(np.log(n), linestyle='--', color='grey')
        # ax.text(x_offset, np.log(n)-1, n)
        # n=50
        # ax.axhline(np.log(n), linestyle='--', color='grey')
        # ax.text(x_offset, np.log(n)+1.2, 50)
        return ax
    

    def violin_pdb_tm(self, ax):
        sns.violinplot(self.df, x='chain_status', y='tm1', ax=ax,
            log_scale=True, color='black', fill=False)
        ax.set_xlim(-.6, 2.5)
        ax.set_ylim(-.1, 1.4)
        ax.set_xlabel('PDB Post-processing')
        ax.set_ylabel('TM-align')
        ax.axhline(1, linestyle='--', color='grey')
        return ax

    def dot_plddt_rmsd(self, ax, chain_type=None, q=.95):
        sdf = self.df[self.df['chain_type']==chain_type] if chain_type else self.df
        
        col1 = 'avg_plddt'
        col2 = 'rmsd'
        sns.scatterplot(sdf, x=col1, y=col2, ax=ax,
            alpha=.1, color='black', s=10)
        ax.set_xlabel('Average pLDDT')
        ax.set_ylabel('RMSD')
        ax.set_xlim(70, 100)
        ax.set_ylim(-1, 30)
        # ax.invert_yaxis()

        # fit
        b, m = polyfit(sdf[col1], sdf[col2], 1)
        print(f"intercept={b}, slop={m}")
        ax.plot(sdf[col1], b + m * sdf[col1], '-', color='blue')
        coef, p_value = pearsonr(sdf[col1], sdf[col2])
        ax.set_title(f"$R^2$ = {coef*coef:.2f}")

        # quantile
        q1 = np.quantile(sdf[col1], 1-q)
        print(f"{q} quantile of {col1} is {q1:.2f}")
        ax.axvline(q1, linestyle='--', color='grey')
        q2 = np.quantile(sdf[col2], q)
        print(f"{q} quantile of {col2} is {q2:.2f}")
        ax.axhline(q2, linestyle='--', color='grey')
        return ax

    def dot_plddt_tm(self, ax, chain_type=None, q=.95):
        sdf = self.df[self.df['chain_type']==chain_type] \
            if chain_type else self.df
        
        sns.scatterplot(sdf, x='avg_plddt', y='tm1', ax=ax,
            alpha=.1, color='black', s=10)
        ax.set_xlabel('Average pLDDT')
        ax.set_ylabel('TM-align')
        ax.set_xlim(60, 100)

        # fit
        b, m = polyfit(sdf['avg_plddt'], sdf['tm1'], 1)
        print(f"intercept={b}, coef={m}")
        ax.plot(sdf['avg_plddt'], b + m * sdf['avg_plddt'],
            '-', color='blue')
        coef, p_value = pearsonr(sdf['avg_plddt'], sdf['tm1'])
        ax.set_title(f"$R^2$ = {coef*coef:.2f}")

        # quantile
        q_plddt = np.quantile(sdf['avg_plddt'], 1-q)
        print(q_plddt)
        ax.axvline(q_plddt, linestyle='--', color='grey')
        q_tm = np.quantile(sdf['tm1'], q)
        print(q_tm)
        ax.axhline(q_tm, linestyle='--', color='grey')
        return ax
    
    def dot_plddt_rmsd2(self, ax, chain_type=None, q=.95):
        sdf = self.df[self.df['chain_type']==chain_type] \
            if chain_type else self.df
        
        col1 = 'avg_plddt'
        col2 = 'rmsd'
        sns.scatterplot(sdf, x=col1, y=col2, ax=ax,
            alpha=.1, color='black', s=10)
        ax.set_xlim(60, 100)
        ax.set_xlabel('average pLDDT')
        ax.set_ylabel('RMSD')

        # fit
        b, m = polyfit(sdf[col1], sdf[col2], 1)
        print(f"intercept={b}, coef={m}")
        ax.plot(sdf[col1], b + m * sdf[col1], '-', color='blue')
        coef, p_value = pearsonr(sdf[col1], sdf[col2])
        ax.text(60, b+m*60, round(coef*coef,2))
        ax.invert_yaxis()

        # quantile
        q_plddt = np.quantile(sdf[col1], 1-q)
        print(q_plddt)
        ax.axvline(q_plddt, linestyle='--', color='grey')
        q2 = np.quantile(sdf[col2], q)
        print(q2)
        ax.axhline(q2, linestyle='--', color='grey')
        return ax

    def hist_rmsd(self, ax, chain_type, quantiles:list=None):
        sub = self.df[self.df['chain_type']==chain_type].dropna()
        print(f"Number of {chain_type} chain: {len(sub)}")
        sns.histplot(sub, x='rmsd', ax=ax, bins=100, stat='percent', color='black')
        ax.set_xlim(-1, 40)
        ax.set_ylim(0, 50)
        ax.set_xlabel('RMSD')
        ax.set_ylabel('Percentage, %')

        if quantiles is None:
            quantiles = [.95,]
        for quant in quantiles:
            q = np.quantile(sub['rmsd'], quant)
            ax.axvline(q, color='grey', linestyle='--')
            ax.text(q+1, 10, round(q, 2))
        return ax