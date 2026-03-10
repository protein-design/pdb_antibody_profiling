import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class PlotPdb:
    def __init__(self, df:pd.DataFrame, verbose:bool=True):
        self.df = df
        self.verbose = verbose
    
    def bar_release(self, ax):
        self.df['year'] = self.df['release_date'].dt.year
        pdf = self.df[['pdb_id','year']].drop_duplicates()
        counts = pdf.groupby('year').agg({'pdb_id':len})
        counts = counts.reset_index()
        col1, col2 = 'Year', 'Structures'
        counts.columns = [col1, col2]
        counts[col1] = counts[col1].astype(int)

        x_order = counts.sort_values(col1, ascending=False)[col1]
        sns.barplot(counts, x=col1, y=col2, color='grey', order=x_order, ax=ax)
        ax.set_ylim(0, 1000)
        xtick_labels = ax.get_xticklabels()
        ax.set_xticks(range(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, rotation=80, ha='center')
        ax.set_ylabel('Number of structures')
        return ax

    def bar_count_pdb(self, ax):
        sdf = self.df[['pdb_id', 'structure_method']].drop_duplicates()
        counts = sdf['structure_method'].value_counts()
        num_pdb = len(sdf['pdb_id'].unique())

        bars = ax.bar(counts.index, counts, color='grey')
        ax.set_ylabel('Number of Structures')
        ax.set_xlim(-1, 2.2)
        ax.set_ylim(0, 7500)
        xticks_labels = ax.get_xticklabels()
        ax.set_xticks(range(len(xticks_labels)))
        ax.set_xticklabels(xticks_labels, rotation=20, ha='right')
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                str(counts.iloc[j]), ha='center', va='bottom')
        return ax

    def violin_resolution(self, ax):
        sdf = self.df[['pdb_id', 'resolution', 'structure_method']].drop_duplicates()
        rdf = sdf[sdf['resolution'].notna()]
        sns.violinplot(rdf, x='structure_method', y='resolution', log_scale=True,
            color='grey', fill=False, inner='stick', ax=ax)
        ax.set_ylabel(r'Resolution, log-$\AA$')
        ax.set_xlim(-1, 3)
        ax.set_xlabel(None)
        xtick_labels = ax.get_xticklabels()
        ax.set_xticks(range(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, rotation=20, ha='right')

        # add lines
        r = 5
        ax.axhline(np.log(r), color='green', linestyle='--')
        ax.text(2.6, np.log(r+1), str(r)+ r' $\AA$')
        r = 20
        ax.axhline(np.log(r), color='orange', linestyle='--')
        ax.text(2.6, np.log(r+5), str(r) + r' $\AA$')
        return ax

    def hist_resolution(self, ax):
        '''
        select best resolution
        '''
        rdf = self.df[['pdb_id', 'resolution', 'structure_method']].drop_duplicates()
        rdf = rdf[rdf['structure_method'].isin(['x-ray diffraction', 'electron microscopy'])]
        sns.histplot(rdf, x='resolution', hue='structure_method', alpha=.5, ax=ax)
        num_pdb = len(rdf['pdb_id'].unique())
        ax.set_xlim(0, 10)

        # text
        resolution = rdf['resolution'][rdf['resolution'].notna()]
        q = .99
        qr = round(np.quantile(resolution, q), 1)
        ax.axvline(qr, linestyle='--')
        ax.text(qr+.1, 200, f"quantile={q}\nresolution={qr}"+ ' $\AA$')
        q = .95
        qr = round(np.quantile(resolution, q), 1)
        ax.axvline(qr, linestyle='--')
        ax.text(qr+.1, 200, f"quantile={q}\nresolution={qr}"+ ' $\AA$')
        ax.set_xlabel(r'Resolution, $\AA$')
        ax.set_ylabel('Number of PDB')
        return ax

    def pie_complex(self, ax):
        num_pdb = len(self.df['pdb_id'].unique())
        g = self.df.groupby('pdb_id').agg({'chain_no':'nunique'})
        g = g.value_counts()
        n = 6
        counts = g[:n]
        counts.index = ['monomer','dimer','trimer','tetramer', 'pentamer', 'heptamer']
        counts['other'] = sum(g[n:])
        counts = counts.reset_index()
        if self.verbose:
            print('Categories:', counts)

        # light color
        # colors = plt.cm.Pastel1(np.arange(len(counts)))
        colors = ['lightgrey'] * len(counts)
        explode = [0.02, 0.02, 0.04, 0.1, 0.2, 0.3, 0.1]
        ax.pie(
            counts['count'],
            labels=counts['index'],
            textprops={'fontsize': 10},
            labeldistance=1.1,
            autopct='%.1f%%', 
            pctdistance=.7,
            colors=colors,
            explode=explode,
            startangle=0,
        )
        return ax

    def dot_bfactor(self, ax, method, quantile=.9, xlim_max:int=None):
        xdf = self.df[self.df['structure_method'].notna()]
        xdf = xdf[xdf['structure_method'].str.contains(method)]
        xdf = xdf[['chain_id', 'resolution', 'avg_bfactor']].dropna()
        xdf = xdf.drop_duplicates()
        if self.verbose:
            print(xdf.shape)

        sns.scatterplot(xdf, y='resolution', x='avg_bfactor', alpha=.1, s=10, ax=ax)
        ax.set_xlabel('Average B-factor')
        ax.set_ylabel(r'Resolution, $\AA$')
        if xlim_max:
            ax.set_xlim(0, xlim_max)

        # quantile
        bq = np.quantile(xdf['avg_bfactor'], quantile)
        rq = np.quantile(xdf['resolution'], quantile)
        ax.axvline(bq, linestyle='--', color='lightgreen')
        ax.axhline(rq, linestyle='--', color='lightgreen')
        rect = patches.Rectangle((-5, -.1), bq+5, rq+.1, facecolor='lightgreen', alpha=.3)
        ax.add_patch(rect)
        if self.verbose:
            print(f'Quantile of {quantile}, B-factor = {bq}')
            print(f'Quantile of {quantile}, resolution = {rq}')

        # plt.yscale('log')
        # invert axis
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        return ax

    def violin_equal_rmsd(self, ax):
        sns.violinplot(self.df, y='chain_type', x='rmsd', ax=ax,
            fill=False, color='black')
        ax.set_xlabel('RMSD')
        ax.set_ylabel(None)
        ax.set_yticks([0,1,2], labels=['Heavy','Kappa','Lambda'])
        return ax
    
    def pie_equal_seq(self, ax, n:int=5):
        equal_counts = self.df.groupby('first_chain_id').agg( num_equal= ('chain_id', len))
        equal_counts = equal_counts['num_equal'].value_counts()
        equal_counts = equal_counts.sort_index()
        counts = equal_counts[:n]
        counts[f'> {n}'] = sum(equal_counts[n:])
        print(counts)

        colors = ['lightgrey'] * len(counts)
        explode = [0.02,] * len(counts)
        ax.pie(
            counts,
            labels=counts.index,
            textprops={'fontsize': 8},
            labeldistance=1.1,
            autopct='%.1f%%', 
            pctdistance=.7,
            colors=colors,
            explode=explode,
            startangle=0,
        )
        return ax

    def hist_rmsd(self, ax, chain_type):
        sub = self.df[self.df['chain_type']==chain_type].dropna()
        sns.histplot(sub, x='rmsd', ax=ax, bins=100, stat='percent', color='lightgrey')
        ax.set_xlim(-1,20)
        ax.set_ylim(0,30)
        ax.set_xlabel('RMSD')
        ax.set_ylabel('Percentage, %')
        q = np.quantile(sub['rmsd'], .95)
        ax.axvline(q, color='grey', linestyle='--')
        ax.text(q+1, 25, round(q, 2))
        return ax