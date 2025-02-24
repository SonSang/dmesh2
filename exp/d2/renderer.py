import torch as th
from matplotlib import pyplot as plt

def render_point_cloud(pc: th.Tensor, domain: tuple, color: str, point_size: float, fig_size: float, path: str):
    '''
    Render point cloud.
    '''
    fig, ax = plt.subplots()
    ax.set_xlim(domain[0], domain[1])
    ax.set_ylim(domain[0], domain[1])

    plt.scatter(pc[:,0].cpu().numpy(), pc[:,1].cpu().numpy(), c=color, s=point_size)
    
    # set visibility of x-axis as False
    xax = ax.get_xaxis()
    xax = xax.set_visible(False)
    # set visibility of y-axis as False
    yax = ax.get_yaxis()
    yax = yax.set_visible(False)

    # set aspect of the plot to be equal
    ax.set_aspect('equal')
    # set figure size
    fig.set_size_inches(fig_size, fig_size)
    # remove outer box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()  
    plt.savefig(path)
    plt.close()