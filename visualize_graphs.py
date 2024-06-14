# Script to visualize halo catalogues as graphs

import time, datetime, os
from Source.plotting import *
from Source.load_data import *
from torch_geometric.utils import degree
from cluster_radius import radius_graph
import readfof


fontsize = 11


# Visualization routine for plotting graphs
def visualize_graph(num, data, masses, projection="3d", edge_index=None):

    fig = plt.figure(figsize=(12, 8))

    if projection=="3d":
        ax = fig.add_subplot(projection ="3d")
        pos = data.x[:,:3]
    elif projection=="2d":
        ax = fig.add_subplot()
        pos = data.x[:,:2]

    pos *= boxsize/1.e3   # show in Mpc

    # Draw lines for each edge
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():

            src = pos[src].tolist()
            dst = pos[dst].tolist()

            if projection=="3d":
                ax.plot([src[0], dst[0]], [src[1], dst[1]], zs=[src[2], dst[2]], linewidth=0.6, color='dimgrey')
            elif projection=="2d":
                ax.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=0.1, color='black')

    # Plot nodes
    if projection=="3d":
        mass_mean = np.mean(masses)
        for i,m in enumerate(masses):
            ax.scatter(pos[i, 0], pos[i, 1], pos[i, 2], s=50*m*m/(mass_mean**2), zorder=1000, alpha=0.6, color = 'mediumpurple')
    elif projection=="2d":
        ax.scatter(pos[:, 0], pos[:, 1], s=m, zorder=1000, alpha=0.5)

    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.zaxis.set_tick_params(labelsize=fontsize)

    ax.set_xlabel('x (Mpc)')
    ax.set_ylabel('y (Mpc)')
    ax.set_zlabel('z (Mpc)')

    param_file = "/home/ubuntu/cosmo_volume/cosmo_GNN/latin_hypercube_params.txt" 
    paramsfile = np.loadtxt(param_file, dtype=str)

    ax.set_title(f'Graph for simulation n°{num}, $\\Omega_m = {float(paramsfile[int(num), 0]):.3e}$, mass cut: 99.8%')

    fig.savefig("Plots/Graphs/graph_"+num+"fixed_masscut.png", bbox_inches='tight', dpi=400)
    plt.close(fig)



# Plot the degree distribution of the graph (see e.g. http://networksciencebook.com/)
def plot_degree_distribution(degrees):

    listbins = np.linspace(0,80,num=12)
    deg_dist = []

    for array in degrees:
        hist, bins = np.histogram(array, bins=listbins)
        deg_dist.append(hist)

    dist_mean = np.mean(deg_dist,axis=0)
    dist_std = np.std(deg_dist,axis=0)

    fig_deg, ax_deg = plt.subplots(figsize=(6, 4))

    ax_deg.set_yscale("log")
    # ax_deg.set_xscale("log")
    ax_deg.plot(bins[:-1], dist_mean)
    ax_deg.fill_between(bins[:-1], dist_mean+dist_std, dist_mean-dist_std, alpha=0.3)
    ax_deg.set_xlim([bins[0],bins[-2]])
    ax_deg.set_xlabel(r"$k$")
    ax_deg.set_ylabel(r"$p_k$")

    fig_deg.savefig("Plots/degree_distribution.pdf", bbox_inches='tight', dpi=300)

# Main routine to display graphs from several simulations
def display_graphs(n_sims, r_link, showgraph=True, get_degree=False):

    if get_degree:
        degrees = []

    # Load data and create graph
    for simnumber in range(n_sims):
        simpath = "/home/ubuntu/cosmo_volume/cosmo_GNN/Data/" + str(simnumber)
        FoF = readfof.FoF_catalog(simpath, 2, long_ids=False,
                          swap=False, SFR=False, read_IDs=False)
        pos  = FoF.GroupPos / 1e6            # Halo positions in Mpc/h
        mass_raw = FoF.GroupMass * 1e10          # Halo masses in Msun/h

        cut_val = 3.5e14    # universal mass cut
        mass_mask = (mass_raw >= cut_val)
        mass_mask = mass_mask.reshape(-1)
        mass=mass_raw[mass_mask]
        pos=pos[mass_mask]
        
        tab = np.column_stack((pos,mass))

        # edge_index, edge_attr = get_edges(pos, r_link, use_loops=False)
        edge_index = radius_graph(torch.tensor(pos,dtype=torch.float32), r=r_link, loop=False)

        data = Data(x=tab, edge_index=torch.tensor(edge_index, dtype=torch.long))

        if showgraph:
            # visualize_graph(data, simnumber, "2d", edge_index)
            visualize_graph(str(simnumber), data, mass, projection="3d", edge_index=data.edge_index)

        if get_degree:
            degrees.append( degree(edge_index[0], data.num_nodes).numpy() )

    if get_degree:
        plot_degree_distribution(degrees)




# --- MAIN ---#

if __name__=="__main__":

    time_ini = time.time()

    for path in ["Plots/Graphs"]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Linking radius
    r_link = 0.2
    n_sims = 4

    display_graphs(n_sims, r_link)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))