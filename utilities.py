## Function to plot final clustering 

from matplotlib.pyplot import tick_params
from numpy.core.fromnumeric import size
from numpy.lib.arraysetops import unique


def fixed_clustering_plot(dataframe, curves_to_plot, facies_curves):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # color = plt.cm.Paired
    # facies_colors = [color(i) for i in range(1, 10)]

    facies_colors = ['#F4D03F','#7ccc19','#196F3D','#160599','#2756c4','#3891f0','#80d4ff','#87039e','#ec90fc','#FF4500','#000000','#DC7633']
    log_colors = ['black', 'red', 'blue', 'green', 'purple','grey', 'orange']*5

    num_tracks = len(curves_to_plot)
    top = dataframe.index.min()
    bot = dataframe.index.max()

    f, ax = plt.subplots(nrows=1, ncols=num_tracks, figsize=(12, 6.5))

    for i, curve in enumerate(curves_to_plot):
        if curve in facies_curves:
            cmap_facies = colors.ListedColormap(facies_colors[0:int(dataframe[curve].max()+1)], 'indexed')
            cluster = np.repeat(np.expand_dims(dataframe[curve].values, 1), 100, 1)
            im = ax[i].imshow(cluster, interpolation='none', aspect='auto', cmap=cmap_facies,
                                vmin=int(dataframe[curve].min()), vmax=int(dataframe[curve].max()), alpha=0.95, extent=[0, 20, bot, top])

        else:
            ax[i].plot(dataframe[curve], dataframe.index, color=log_colors[i])

        ax[i].set_title(curve, fontsize=15, fontweight='bold')
        ax[i].grid(which='major', color='lightgrey', linestyle='-')
        ax[i].set_ylim(bot, top)

        if i == 0:
            ax[i].set_ylabel('DEPTH', fontsize=15, fontweight='bold')
        else:
            plt.setp(ax[i].get_yticklabels(), visible=False)

    #Legend
    facies_labels = ['SS', 'S-S', 'SH', 'MR', 'DOL','LIM', 'CH','HAL', 'AN', 'TF', 'CO']#, 'BS']

    divider = make_axes_locatable(ax[-1]) # appending lithology legend
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((5*' ').join(facies_labels), size=12)
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')

    plt.tight_layout()
    plt.show()


def scatter_plot(df, x_col, y_col, z_color):
    import plotly.express as px
    import matplotlib.colors as colors

    df[z_color] = df[z_color].astype(int)
    all_colors = ['#F4D03F','#7ccc19','#196F3D','#160599','#2756c4','#3891f0','#80d4ff','#87039e','#ec90fc','#FF4500','#000000','#DC7633']
    facies_colors = [all_colors[i] for i in df[z_color].unique()]

    # facies_code = df[z_color].unique()
    # facies_labels = ['SS', 'S-S', 'SH', 'MR', 'DOL','LIM', 'CH','HAL', 'AN', 'TF', 'CO', 'BS']
    # labels={i:facies_labels[i] for i in facies_code}
    
    df[z_color] = df[z_color].astype(str)
    # lit_numbers = {0: 'SS', 1:'S-S', 2:'SH', 3:'MR', 4:'DOL', 5:'LIM', 6:'CH', 7:'HAL', 8:'AN', 9:'TF', 10:'CO', 11:'BS'}
    # df[z_color] = df[z_color].map(lit_numbers)

    scat1 = px.scatter( df, 
                        x=x_col,
                        y=y_col,
                        color=z_color,
                        title= x_col + " vs. " + y_col,
                        marginal_x='histogram',
                        marginal_y='box',
                        color_discrete_sequence=facies_colors,
                        opacity=0.90,
                        width=580,
                        height=550)
    return scat1




def litho_confusion_matrix(y_true, y_pred):

    
    import pandas as pd
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
    from itertools import product

    facies_dict = {0:'Sandstone', 1:'Sandstone/Shale', 2:'Shale', 3:'Marl',
                    4:'Dolomite', 5:'Limestone', 6:'Chalk', 7:'Halite', 
                    8:'Anhydrite', 9:'Tuff', 10:'Coal', 11:'Basement'}

    labels = list(set(list(y_pred.unique()) + list(y_true.unique())))
    label_names = [facies_dict[k] for k in labels]
    label_names.insert(0, '-')

    #Normalizing confusion matrix
    cm = pd.DataFrame(confusion_matrix(y_true.values, y_pred.values))
    summ = cm.sum(axis=0)
    cm_norm = pd.DataFrame(np.zeros(cm.shape))
    for i in range(cm.shape[1]):
        for j in range(cm.shape[0]):
            cm_norm[i][j] = cm[i][j]*100/summ[i]

    cm_final = cm_norm.fillna(0).to_numpy()

    #disp = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=labels)
    fig, ax = plt.subplots(figsize=(4,3))
    plt.imshow(cm_final, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('NORMALIZED CONFUSION MATRIX (%)', size=6)
    tick_marks = np.arange(-1, len(label_names))
    plt.xticks(tick_marks, label_names, rotation=90, size=5)
    plt.yticks(tick_marks, label_names, size=5)
    #plt.colorbar().set_label(size=5)
        
    fmt = '.2f'
    thresh = cm_final.max() / 2.
    for i, j in product(range(cm_final.shape[0]),   range(cm_final.shape[1])):
        plt.text(j, i, format(cm_final[i, j], fmt),
                    horizontalalignment="center", size=4,
                    #verticalalignment = "bottom",
                    color="white" if cm_final[i, j] > thresh else "black")
            
    plt.ylabel('True label', size=6)
    plt.xlabel('Predicted label', size=6)


def predictions_plot(dataframe, curves_to_plot, facies_curves):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # color = plt.cm.Paired
    # facies_colors = [color(i) for i in range(1, 10)]

    facies_colors = ['#F4D03F','#7ccc19','#196F3D','#160599','#2756c4','#3891f0','#80d4ff','#87039e','#ec90fc','#FF4500','#000000','#DC7633']
    log_colors = ['black', 'red', 'blue', 'green', 'purple','grey', 'orange']*5

    num_tracks = len(curves_to_plot)
    top = dataframe.index.min()
    bot = dataframe.index.max()

    f, ax = plt.subplots(nrows=1, ncols=num_tracks, figsize=(12, 6.5))

    for i, curve in enumerate(curves_to_plot):
        if curve in facies_curves:
            cmap_facies = colors.ListedColormap(facies_colors[0:int(dataframe[curve].max()+1)], 'indexed')
            cluster = np.repeat(np.expand_dims(dataframe[curve].values, 1), 100, 1)
            im = ax[i].imshow(cluster, interpolation='none', aspect='auto', cmap=cmap_facies,
                                vmin=int(dataframe[curve].min()), vmax=int(dataframe[curve].max()), alpha=0.95, extent=[0, 20, bot, top])

        else:
            ax[i].plot(dataframe[curve], dataframe.index, color=log_colors[i])

        ax[i].set_title(curve, fontsize=15, fontweight='bold')
        ax[i].grid(which='major', color='lightgrey', linestyle='-')
        ax[i].set_ylim(bot, top)

        if i == 0:
            ax[i].set_ylabel('DEPTH', fontsize=15, fontweight='bold')
        else:
            plt.setp(ax[i].get_yticklabels(), visible=False)

    #Legend
    facies_labels = ['SS', 'S-S', 'SH', 'MR', 'DOL','LIM', 'CH','HAL', 'AN', 'TF', 'CO']#, 'BS']

    divider = make_axes_locatable(ax[-1]) # appending lithology legend
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((5*' ').join(facies_labels), size=12)
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')

    plt.tight_layout()
    plt.show()