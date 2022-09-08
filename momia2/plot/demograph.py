import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

def demograph(sorted_midline, 
              sorted_length,
              n=256,
              max_l = 8,
              align='center',
              reorient=True,
              forced_reorientation=False,
              reorientation_sequence = [],
              uniformed=False,
              skip_end=0,
              smooth_window=0):
    demograph=[]
    _flip_sequence = []
    
    if forced_reorientation and len(reorientation_sequence) != len(sorted_midline):
        raise ValueError("Lengths don't match!")
       
    for i,ml in enumerate(sorted_midline):
        _flipped = False
        l = sorted_length[i]
        frac = min(1,l/max_l)
        half_n = int(0.5*n*frac)
        pad = int(n*0.5)-half_n
        if smooth_window>0 and smooth_window%2==1:
            ml = np.convolve(ml,np.ones(smooth_window)/smooth_window,mode='same')
        if skip_end>0:
            ml = ml[skip_end:-skip_end]
        
        if reorient:
            if not forced_reorientation:
                ml, _flipped = reorient_data(ml)
            else:
                _flipped = bool(reorientation_sequence[i])
                if _flipped:
                    ml = np.flip(ml)
        _flip_sequence.append(_flipped)    
        
        if not uniformed:
            interpolated = np.interp(x=np.linspace(0,1,2*half_n),xp=np.linspace(0,1,len(ml)),fp=ml)
            if align=='center':
                arr = np.pad(interpolated,(pad,pad),'constant')
            elif align=='left':
                arr = np.pad(interpolated,(0,2*pad),'constant')
            elif align=='right':
                arr = np.pad(interpolated,(2*pad,0),'constant')
        else:
            arr = np.interp(x=np.linspace(0,1,n),xp=np.linspace(0,1,len(ml)),fp=ml)
        demograph.append(arr)
    return np.array(demograph), np.array(_flip_sequence)
        
def reorient_data(data):
    half_l = int(round(len(data)/2))
    flip = bool(data[:half_l].mean()<data.mean())
    if flip:
        data = np.flip(data)
    return data, flip 

def grouped_scatter(merged_df,feature1, feature2):
    colors = {'Rv1830-mut':cm.get_cmap('Set2')(1),'WT':cm.get_cmap('Set2')(0)}
    vmax1 = merged_df[feature1].quantile(0.995)*1.2    
    vmin1 = merged_df[feature1].quantile(0.01)*0.4
    vmax2 = merged_df[feature2].quantile(0.995)*1.2
    vmin2 = merged_df[feature2].quantile(0.01)*0.4
    if feature1 == 'FITC_mean':
        vmin1-=100
    if feature2 == 'FITC_mean':
        vmin2-=100
    fig=plt.figure(figsize=(16,12))
    grids = gs(8,8,hspace=0.1,wspace=0.1)
    for i,r in enumerate('CDEF'):
        rowd = merged_df[merged_df['Row']==r]
        for j,w in enumerate(sorted(list(rowd['Well'].unique()))):
            if r == 'C':
                j+= int(j/2)*2
            ax = fig.add_subplot(grids[j,i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(vmin1,vmax1)
            ax.set_ylim(vmin2,vmax2)
            welld = rowd[rowd['Well']==w]
            time = welld['Time'].values[0]
            treatment = welld['INH Conc.'].values[0]
            strain = welld['Strain'].values[0]
            if j==0:
                ax.set_title('{} hours'.format(time))
            if j%2 ==0:
                ax.text(0.05,0.9,'{}MIC'.format(treatment),transform=ax.transAxes,va='center',fontweight='bold')
            if i==0 and j==0:
                ax.set_ylabel(feature2,fontsize=14)
            if i==1 and j==7:
                ax.set_xlabel(feature1,fontsize=14)
            sns.scatterplot(x=welld[feature1],
                            y=welld[feature2],
                            s=10,alpha=0.3,fc=colors[strain],ax=ax)
            sns.kdeplot(x=welld[feature1],
                        y=welld[feature2], levels=5, color="black", linewidths=1,ax=ax)
            
    legend_ax = fig.add_subplot(grids[2,0])
    legend_ax.scatter(0.1,0.8,s=50,alpha=0.8,fc=colors['Rv1830-mut'])
    legend_ax.scatter(0.1,0.2,s=50,alpha=0.8,fc=colors['WT'])
    legend_ax.set_xlim(0,1);
    legend_ax.set_ylim(0,1);
    legend_ax.axis('off');
    legend_ax.text(0.18,0.8,'Rv1830-mut',fontsize=14,va='center',ha='left')
    legend_ax.text(0.18,0.2,'WT',fontsize=14,va='center',ha='left') 
    

def grouped_hist(merged_df,feature1):
    colors = {'Rv1830-mut':cm.get_cmap('Set2')(1),'WT':cm.get_cmap('Set2')(0)}
    vmax1 = merged_df[feature1].quantile(0.995)*1.2
    vmin1 = merged_df[feature1].quantile(0.01)*0.4

    fig=plt.figure(figsize=(16,12))
    grids = gs(8,8,hspace=0.1,wspace=0.1)
    for i,r in enumerate('CDEF'):
        rowd = merged_df[merged_df['Row']==r]
        for j,w in enumerate(sorted(list(rowd['Well'].unique()))):
            if r == 'C':
                j+= int(j/2)*2
            ax = fig.add_subplot(grids[j,i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(vmin1,vmax1)
            welld = rowd[rowd['Well']==w]
            time = welld['Time'].values[0]
            treatment = welld['INH Conc.'].values[0]
            strain = welld['Strain'].values[0]
            if j==0:
                ax.set_title('{} hours'.format(time))
            if j%2 ==0:
                ax.text(0.05,0.9,'{}MIC'.format(treatment),transform=ax.transAxes,va='center',fontweight='bold')
            if i==1 and j==7:
                ax.set_xlabel(feature1,fontsize=14)
            ax.hist(x=welld[feature1], fc=colors[strain], bins=50, ec='black',lw=0.5,alpha=0.5)
            
    legend_ax = fig.add_subplot(grids[2,0])
    legend_ax.scatter(0.1,0.8,s=50,alpha=0.8,fc=colors['Rv1830-mut'])
    legend_ax.scatter(0.1,0.2,s=50,alpha=0.8,fc=colors['WT'])
    legend_ax.set_xlim(0,1);
    legend_ax.set_ylim(0,1);
    legend_ax.axis('off');
    legend_ax.text(0.18,0.8,'Rv1830-mut',fontsize=14,va='center',ha='left')
    legend_ax.text(0.18,0.2,'WT',fontsize=14,va='center',ha='left') 
    
def grouped_box_plot(merged_df,feature):
    vmax = merged_df[feature].quantile(0.995)*1.5
    vmin = merged_df[feature].quantile(0.001)*0.2
    fig=plt.figure(figsize=(10,4))
    grids=gs(1,7)
    for i,t in enumerate(np.sort(merged_df['Time'].unique())):
        if t == 0:
            ax = fig.add_subplot(grids[0,0])
        else:
            ax = fig.add_subplot(grids[i*2-1:i*2+1])
        subset = merged_df[merged_df['Time']==t]
        g=sns.boxplot(data=subset,x='INH Conc.',y=feature,hue='Strain',ax=ax,palette='Set2',hue_order=['WT','Rv1830-mut'])
        if i!=3:
            ax.legend([],[])
        if i!=0:
            ax.set_yticks([])
            ax.set_ylabel('')
        else:
            ax.set_ylabel(feature,fontsize=12)
        if i!=2:
            ax.set_xlabel('')
        else:
            ax.set_xlabel('INH MIC',loc='left',fontsize=12)
        ax.set_title('{} hours'.format(t))
        ax.set_ylim(vmin,vmax);