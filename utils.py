import numpy as np
from sklearn.preprocessing import MinMaxScaler
import itertools
import joblib
from keras.utils import np_utils
import matplotlib

#pad the FTCP data along the second dimension(real space point and fourier point arrangement)
def pad (vae_x , p):
    dum_x = np.zeros((vae_x.shape[0],vae_x.shape[1]+p,vae_x.shape[2] ))
    dum_x[:,:-1*p,:] = vae_x
    return dum_x

#perform data normalizaiton for crystal representation along the second dimension(real space point and fourier point arrangement)
def minmax (X):
    
    scaler_x = MinMaxScaler()

    dim0 = X.shape[0]
    dim1 = X.shape[1]
    dim2 = X.shape[2]

    X1 = np.transpose(X,(1,0,2))
    X1 =X1.reshape(dim1,dim0*dim2)
    X1 = scaler_x.fit_transform(X1.T)
    X1 = X1.T
    X1 =X1.reshape(dim1,dim0,dim2)
    X1= np.transpose(X1,(1,0,2))
    return X1, scaler_x


#find index of compounds with properties close to the target value
def find_nearest(array,target,num):
    array = np.sum(np.abs(array-target),axis=1)
    idx = np.argsort(array)
    return idx[:num]



#Spherical linear interpolation
def slerp(v0, v1, t_array):

    # >>> slerp([1,0,0,0], [0,0,0,1], np.arange(0, 1, 0.001))
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0 * v1)/np.sqrt(np.sum(np.square(v0)))/np.sqrt(np.sum(np.square(v1)))

    if dot < 0.0:
        v1 = -v1
        dot = -dot
    
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = v0[np.newaxis,:] + t_array[:,np.newaxis] * (v1 - v0)[np.newaxis,:]
        return (result.T / np.linalg.norm(result, axis=1)).T
    
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t_array
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:,np.newaxis] * v0[np.newaxis,:]) + (s1[:,np.newaxis] * v1[np.newaxis,:])

#perform slerp pertubation to a set of compounds
def get_slerp(inv_train, aug_num):
    inv_train_s = []
    for a in itertools.combinations(inv_train,2):
        inv_train_s.append( slerp(a[0] ,a[1],np.linspace(0,1,aug_num)))
    
    return  np.vstack(inv_train_s)  

#extract pretty formular and atom arrangement of crystals from decoded signal

def get_formular (ftcp_gen, num_ele,num_sites):
    Element= joblib.load('./files/element.pkl') 
    E_v = np_utils.to_categorical(np.arange(0,len(Element),1))
    pred_ele = np.argmax(ftcp_gen[:,0:len(E_v),:num_ele],axis=1)

    pred_for = np.zeros((ftcp_gen.shape[0],num_sites))


    pred_bas_atom = (ftcp_gen[:,len(E_v)+2+num_sites:len(E_v)+2+2*num_sites,:num_ele])
    
    pred_bas = (ftcp_gen[:,len(E_v)+2:len(E_v)+2+num_sites,:3])
    threshold = np.repeat(np.expand_dims(np.max(pred_bas_atom,axis=2),axis=2),num_ele,axis=2)    
    pred_bas_atom[pred_bas_atom < threshold] = 0
#    0.05 are the boundary of atom existance
    pred_bas_atom[pred_bas_atom < 0.05] = 0
    pred_bas_atom = np.ceil(pred_bas_atom)
   
#    pred_bas = (ftcp_gen[:,len(E_v)+2:len(E_v)+12,:3])


    for i in range(len(pred_ele)):
        _,len_u = np.unique(pred_bas[i,:,:],axis=0,return_index= True)
        len_u = len(len_u)
        pred_for[i,:len_u] = pred_bas_atom[i,:len_u,:].dot(pred_ele[i,:])

    return pred_for
 
def get_cell_basis (ftcp_new, len_ele, num_sites):
    pred_abc_new = ftcp_new[:,len_ele,:3]
    
    pred_angle_new = ftcp_new[:,len_ele+1,:3]
    
    pred_cell_new = np.concatenate((pred_abc_new,pred_angle_new),axis=1)


    pred_bas_new = ftcp_new[:,len_ele+2:len_ele+2+num_sites,:3]
    
    return pred_abc_new,pred_angle_new,pred_cell_new,pred_bas_new


def inv_minmax (X,scaler_x):
    
    

    dim0 = X.shape[0]
    dim1 = X.shape[1]
    dim2 = X.shape[2]

    X1 = np.transpose(X,(1,0,2))
    X1 =X1.reshape(dim1,dim0*dim2)
    X1 = scaler_x.inverse_transform(X1.T)
    X1 = X1.T
    X1 =X1.reshape(dim1,dim0,dim2)
    X1= np.transpose(X1,(1,0,2))
    return X1


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts