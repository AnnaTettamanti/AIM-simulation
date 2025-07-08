# Funzioni per l'analisi dei risultati delle simulazioni AIM

import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt

def plot_finali(magnet, density, tempo, Lx, Ly):
    "Plotta le mappe di magnetizzazione e densità finali"
    
    magnet_finale = magnet[-1, :, :]
    magnet_finale =  np.squeeze(magnet_finale)

    density_finale = density[-1, :, :]
    density_finale =  np.squeeze(density_finale)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    im = ax[0].imshow(magnet_finale, cmap='viridis', aspect='auto', origin='lower')
    cbar = fig.colorbar(im, ax=ax[0], orientation='vertical')
    ax[0].set_title('Mappa di Magnetizzazione finale')
    im = ax[1].imshow(density_finale, cmap='viridis', aspect='auto', origin='lower')
    cbar = fig.colorbar(im, ax=ax[1], orientation='vertical')
    ax[1].set_title('Mappa di Densità finale')
    plt.show()

    fig_fin, ax_fin = plt.subplots(1, 3, figsize=(15, 5))
    ax_fin[0].plot(np.mean(magnet_finale,0))
    ax_fin[0].set_title('Magnetizzazione finale media')
    ax_fin[1].plot(np.mean(density_finale,0))
    ax_fin[1].set_title('Densità finale media')
    M_site = magnet.reshape(tempo, -1).sum(axis=1) / (Lx * Ly)

    ax_fin[2].plot(np.arange(tempo), M_site)
    ax_fin[2].set_title('Parametro ordine al variare del tempo')
    plt.show()

def istogrammi_soglia(magnet, density, prime_immagini=50):
    """
    Calcola e visualizza gli istogrammi di magnetizzazione e densità, definendo soglie a partire
    sia dalla singola immagine iniziale (t=0) sia dalla media delle prime `prime_immagini` immagini.

    Per ciascuna delle grandezze (magnetizzazione e densità) vengono calcolate due soglie:
      1. Basata sui valori di t=0: mean + 5*std dei pixel.
      2. Basata sulla media spaziale dei primi `prime_immagini` frame: mean + 5*std di quei valori.

    Gli istogrammi vengono mostrati in una griglia 2×2:
      - [0,0] magnetizzazione a t=0
      - [1,0] magnetizzazione media prime `prime_immagini`
      - [0,1] densità a t=0
      - [1,1] densità media prime `prime_immagini`

    La soglia basata sulla media delle prime `prime_immagini` immagini di magnetizzazione viene
    restituita come output della funzione.

    Parameters
    ----------
    magnet : array-like, shape (T, Ly, Lx)
        Sequenza di immagini di magnetizzazione nel tempo. Il primo indice è il frame temporale.
    density : array-like, shape (T, Ly, Lx)
        Sequenza di immagini di densità nel tempo. Stesso formato di `magnet`.
    prime_immagini : int, optional
        Numero di frame iniziali da mediamente considerare per la seconda soglia (default=50).

    Returns
    -------
    float
        Soglia di magnetizzazione calcolata sulla media delle prime `prime_immagini` immagini.

    Side effects
    ------------
    - Stampa a video i valori delle quattro soglie.
    - Mostra a schermo una figura con 4 istogrammi e linee verticali alle soglie.
    """
    magnet_iniziale = magnet[0, :, :].astype(float)
    flat_mag0 = magnet_iniziale.flatten()

    # 1) Soglia su singola immagine
    std_soglia_m = np.std(flat_mag0)
    soglia_m = np.mean(flat_mag0) + 5 * std_soglia_m
    print('Soglia magnetizzazione (solo t=0):', soglia_m)

    # 2) Media su prime_immagini immagini e soglia su queste
    magnet_stack = magnet[0:prime_immagini, :, :].astype(float)   
    magnet_mean_first = np.mean(magnet_stack, axis=0)                  
    flat_mag_first = magnet_mean_first.flatten()
    std_soglia_m2 = np.std(flat_mag_first)
    soglia_m2 = np.mean(flat_mag_first) + 5 * std_soglia_m2
    print(f'Soglia magnetizzazione (media prime {prime_immagini}):', soglia_m2)

    # 3) Stesso ragionamento per density
    density_iniziale = density[0, :, :].astype(float)
    flat_den = density_iniziale.flatten()
    std_soglia_d = np.std(flat_den)
    soglia_d = np.mean(flat_den) + 5 * std_soglia_d
    print('Soglia densità (solo t=0):', soglia_d)

    density_stack = density[0:prime_immagini, :, :].astype(float)
    density_mean_first = np.mean(density_stack, axis=0)
    flat_density_first = density_mean_first.flatten()

    std_soglia_d2 = np.std(flat_density_first)
    soglia_d2 = np.mean(flat_density_first) + 5 * std_soglia_d2
    print(f'Soglia densità (media prime {prime_immagini}):', soglia_d2)

    # 4) Disegno gli istogrammi
    fig_istogramma, ax = plt.subplots(2, 2, figsize=(12, 6))

    # 4.1) Istogramma magnetizzazione t=0
    bins_mag0 = np.arange(np.min(flat_mag0), np.max(flat_mag0) + 1) - 0.5
    ax[0, 0].hist(flat_mag0, bins=bins_mag0, color='blue', density=True)
    ax[0, 0].axvline(soglia_m, color='red', linestyle='--', label='Soglia')
    ax[0, 0].set_title('Istogramma mag. iniziale (t=0)')
    ax[0, 0].legend()

    # 4.2) Istogramma magnetizzazione media prime 10 immagini
    bins_mag10 = np.arange(np.min(flat_mag_first), np.max(flat_mag_first) + 1) - 0.5
    ax[1, 0].hist(flat_mag_first, bins=bins_mag10, color='blue', density=True)
    ax[1, 0].axvline(soglia_m2, color='red', linestyle='--', label='Soglia media 10')
    ax[1, 0].set_title(f'Istogramma mag. media prime {prime_immagini}')
    ax[1, 0].legend()

    # 4.3) Istogramma densità t=0
    bins_den0 = np.arange(0, np.max(flat_den) + 1) - 0.5
    ax[0, 1].hist(flat_den, bins=bins_den0, color='green', density=True)
    ax[0, 1].axvline(soglia_d, color='red', linestyle='--', label='Soglia')
    ax[0, 1].set_title('Istogramma densità iniziale (t=0)')
    ax[0, 1].legend()

    # 4.4) Istogramma densità media prime 10 immagini
    bins_den10 = np.arange(np.min(flat_density_first), np.max(flat_density_first) + 1) - 0.5
    ax[1, 1].hist(flat_density_first, bins=bins_den10, color='green', density=True)
    ax[1, 1].axvline(soglia_d2, color='red', linestyle='--', label='Soglia media 10')
    ax[1, 1].set_title(f'Istogramma densità media prime {prime_immagini}')
    ax[1, 1].legend()

    plt.tight_layout()
    plt.show()

    return soglia_m2

## ASTERS

from numba import njit
from skimage.measure import label, regionprops

@njit
def dentro_aster(magnet, soglia):
    "Considera che deve avere almeno un vicino con magnetizzazione sopra la soglia"
    global Lx, Ly
    dentro = 0
    for i in range(Ly):
        for j in range(Lx):
            if abs(magnet[i, j]) >= soglia:
                has_neighbor = False
                # nord
                if abs(magnet[(i - 1) % Ly, j]) >= soglia:
                    has_neighbor = True
                # sud
                elif abs(magnet[(i + 1) % Ly, j]) >= soglia:
                    has_neighbor = True
                # ovest
                elif abs(magnet[i, (j - 1) % Lx]) >= soglia:
                    has_neighbor = True
                # est
                elif abs(magnet[i, (j + 1) % Lx]) >= soglia:
                    has_neighbor = True
                if has_neighbor:
                    dentro += 1

    return dentro

@njit
def heatmap_aster(magnet, soglia):
    global Lx, Ly
    dentro = np.zeros((Ly, Lx),dtype=np.int32)
    for i in range(Ly):
        for j in range(Lx):
            if abs(magnet[i, j]) >= soglia:
                has_neighbor = False
                # nord
                if abs(magnet[(i - 1) % Ly, j]) >= soglia:
                    has_neighbor = True
                # sud
                elif abs(magnet[(i + 1) % Ly, j]) >= soglia:
                    has_neighbor = True
                # ovest
                elif abs(magnet[i, (j - 1) % Lx]) >= soglia:
                    has_neighbor = True
                # est
                elif abs(magnet[i, (j + 1) % Lx]) >= soglia:
                    has_neighbor = True
                if has_neighbor:
                    dentro[i][j]= True
            else:
               dentro[i][j] = False
    return dentro

def evoluzione_dentro(magnet, soglia, tempo):
    dentro = np.zeros(tempo)
    for t in range(tempo):
        dentro[t] = dentro_aster(np.squeeze(magnet[t,:,:]), soglia)
    plt.plot(dentro)
    plt.yscale('log')
    plt.show()



def caratterizzazione_aster(magnet, soglia):
    global Ly, Lx
    
    asters = heatmap_aster(magnet,soglia)
    big = np.tile(asters, (3, 3))
    big = np.tile(big.astype(np.uint8), (3, 3))
    
    labels_big = label(big, connectivity=2)
    props = regionprops(labels_big)
    
    areas = []
    centroids = []

    
    # 4) Ciclo su ciascuna regione “reg” in props
    for reg in props:
        y0, x0 = reg.centroid
    
        if Ly <= y0 < 2*Ly and Lx <= x0 < 2*Lx:
            # Calcolo area vera
            coords_big = reg.coords 

            # mappo ciascuna coordinata modulo 300
            coords_mod = np.zeros_like(coords_big)
            coords_mod[:,0] = coords_big[:,0] % Ly
            coords_mod[:,1] = coords_big[:,1] % Lx

            unique_mod = np.unique(coords_mod, axis=0)
            area_true = unique_mod.shape[0]
            
            cy_mod = float(y0 % Ly)
            cx_mod = float(x0 % Lx)

            if area_true < 2: #per evitare un punto singolo
                continue
        
            coordinates = (cy_mod, cx_mod)
            areas.append(int(area_true))
            centroids.append(coordinates)
    
    return areas, centroids

def analisi_temporale_asters(magnet, soglia):
    aster = heatmap_aster(np.squeeze(magnet), soglia)
    big = np.tile(aster, (3, 3))
    caratteristiche = caratterizzazione_aster(big)
    areas = caratteristiche[0]
    centroids = caratteristiche[1]
    num = areas.__len__()
    return num, areas, centroids

def asters_e_dentro(magnet, soglia, tempo):
    num_tot = np.zeros(tempo)
    areas_tot = []
    for t in range(tempo):
        num, areas = analisi_temporale_asters(magnet[t,:,:], soglia)
        num_tot[t] = num
        areas_tot.append(areas)

    for t in range(tempo):
        areas_tot[t] = np.array(areas_tot[t])
    
    return num_tot, areas_tot

def sitimeno_sitipiu(magnet, soglia):
    global Lx, Ly
    asters_finali = heatmap_aster(magnet, soglia)
    siti_meno = []

    for i in range(Ly):
        for j in range(Lx):
            if asters_finali[i][j] < 0:
               siti_meno.append((i,j))

    siti_piu = []

    for i in range(Ly):
        for j in range(Lx):
            if asters_finali[i][j] > 0:
                siti_piu.append((i,j))

    return siti_meno, siti_piu

def probabilities(magnet, density, metodo_calcolo, x_neo = 0, y_neo = 0):
    global T, D, gamma

    dt = dt = 1.0 / (4*D + np.exp(1/T))
    exp = np.zeros(2)

    if metodo_calcolo == 1:
        exp[0] = - 1/T * magnet/density # calcolato per le particelle spin + 1
        exp[1] =   1/T * magnet/density # calcolato per le particelle spin -1
    
    elif metodo_calcolo == 2:
        xl = (x_neo - 1) % Lx
        xr = (x_neo + 1) % Lx
        yu = (y_neo - 1) % Ly
        yd = (y_neo + 1) % Ly

        m_center = magnet[y_neo, x_neo] #prendo magnetizzazioni
        m_left  = magnet[y_neo, xl]
        m_right = magnet[y_neo, xr]
        m_up    = magnet[yu, x_neo]
        m_down  = magnet[yd, x_neo]

        d_center = density[y_neo, x_neo] #prendo densità
        d_left  = density[y_neo, xl]
        d_right = density[y_neo, xr]
        d_up    = density[yu, x_neo]
        d_down  = density[yd, x_neo]

        num = ((1 - gamma)*m_left
                         + (1 + 2*gamma)*m_right
                         + (1 - gamma/2)*m_up
                         + (1 - gamma/2)*m_down
                         + m_center)
        den = ((1 - gamma)*d_left
                         + (1 + 2*gamma)*d_right
                         + (1 - gamma/2)*d_up
                         + (1 - gamma/2)*d_down
                         + d_center)
    
        exp[0] = - 1/T * num/den
    
        num = ((1 - gamma)*m_right
                         + (1 + 2*gamma)*m_left
                         + (1 - gamma/2)*m_up
                         + (1 - gamma/2)*m_down
                         + m_center)
        den = ((1 - gamma)*d_right
                         + (1 + 2*gamma)*d_left
                         + (1 - gamma/2)*d_up
                         + (1 - gamma/2)*d_down
                         + d_center)
    
        exp[1] = 1/T * num/den

    elif metodo_calcolo == 3:
        xl = (x_neo - 1) % Lx
        xr = (x_neo + 1) % Lx
        yu = (y_neo - 1) % Ly
        yd = (y_neo + 1) % Ly

        num = (1 - 4*gamma) * magnet[y_neo, x_neo] \
            + gamma * (magnet[y_neo, xl] + magnet[y_neo, xr] \
            + magnet[yu, x_neo] + magnet[yd, x_neo])
        den = (1 - 4*gamma) * density[y_neo, x_neo] \
            + gamma * (density[y_neo, xl] + density[y_neo, xr] \
            + density[yu, x_neo] + density[yd, x_neo])
        exp[0] = - 1/T * num/den
        exp[1] = 1/T * num/den

    return np.exp(exp)*dt


def print_probabilities(magnet, density, soglia, metodo_calcolo):

    global Lx, Ly, T, D, gamma
    
    rate_flip_meno = []
    rate_flip_piu = []

    siti_meno, siti_piu = sitimeno_sitipiu(magnet, soglia)
    where_asters = heatmap_aster(magnet, soglia)
    aster_finali_m= where_asters*magnet
    aster_finali_d = where_asters*density

    for i in range(Ly):
        for j in range(Lx):
        
            if (i,j) in siti_meno and aster_finali_d[i][j] != 0:
                rate_flip_meno.append(probabilities(aster_finali_m, aster_finali_d, metodo_calcolo, x_neo= j, y_neo = i))
        
            elif (i,j) in siti_piu and aster_finali_d[i][j] != 0:
                rate_flip_piu.append(probabilities(aster_finali_m, aster_finali_d, metodo_calcolo, x_neo= j, y_neo = i))
        
            else:
                pass

    rate_flip_meno = np.array(rate_flip_meno)
    rate_flip_piu = np.array(rate_flip_piu)

    rate_flip_meno_std = np.std(rate_flip_meno, axis = 0)/np.sqrt(rate_flip_piu.shape[0]) #std sulla media, divido per sqrt(N)
    rate_flip_piu_std = np.std(rate_flip_piu, axis = 0)/np.sqrt(rate_flip_piu.shape[0])

    rate_flip_meno = np.mean(rate_flip_meno, axis = 0)
    rate_flip_piu = np.mean(rate_flip_piu, axis = 0)

    print(f'Probabilità(+ --> -) in siti con magnetizzazione negativa: {rate_flip_meno[0]:.2%} +/- {rate_flip_meno_std[0]:.2%}'),
    print(f'Probabilità(+ --> -) in siti con magnetizzazione positiva: {rate_flip_piu[0]:.4%} +/- {rate_flip_piu_std[0]:.6%}')
    print(f'Probabilità(- --> +) in siti con magnetizzazione negativa: {rate_flip_meno[1]:.4%} +/- {rate_flip_meno_std[1]:.6%}')
    print(f'Probabilità(- --> +) in siti con magnetizzazione positiva: {rate_flip_piu[1]:.2%} +/- {rate_flip_piu_std[1]:.2%}')

    return rate_flip_meno, rate_flip_meno_std, rate_flip_piu, rate_flip_piu_std

def tempo_una_banda(magnet, soglia, t0 = 0):
    num_tot, areas_tot = asters_e_dentro(magnet, soglia, magnet.shape[0])
    
    for t in range(t0, magnet.shape[0]):
        if areas_tot[t] == 1 and np.sum(areas_tot[t:]) == np.len(areas_tot[t:]):
            return t


## ANALISI PER BANDE
def autocorrelazione(vettore, k_max):
    "Funzione per calcolo autocorrelazione di un vettore"
    vettore_correlazione = np.zeros(k_max+1)
    len_vettore = len(vettore)

    for k in range(k_max+1):
        for i in range(len_vettore):
            vettore_correlazione[k] += (vettore[i]) * (vettore[(i+k) % (len_vettore)])

    vettore_correlazione /= vettore_correlazione[0]

    return vettore_correlazione