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
    ax[1, 0].axvline(soglia_m2, color='red', linestyle='--', label=f'Soglia media {prime_immagini}')
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
    ax[1, 1].axvline(soglia_d2, color='red', linestyle='--', label=f'Soglia media {prime_immagini}')
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
    Ly, Lx = magnet.shape
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
    Ly, Lx = magnet.shape
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
    #plt.yscale('log')
    plt.show()


def caratterizzazione_aster(magnet, soglia):
    Ly, Lx = magnet.shape
    
    asters = heatmap_aster(magnet,soglia)
    big = np.tile(asters, (3, 3))
    
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

def asters_over_time(magnet, soglia):
    """
    Parametri
    ---------
    magnet : ndarray, shape (T, Ly, Lx)
        Serie temporale del campo da analizzare.
    soglia : float
        Soglia assoluta per definire i cluster.

    Ritorna
    -------
    num_clusters : ndarray, shape (T,)
        Numero di cluster trovati in ciascun frame.
    areas_list : list of ndarray
        Per ogni frame t, array delle aree dei cluster.
    centroids_list : list of list of (y, x)
        Per ogni frame t, lista delle coordinate dei centroidi.
    """
    T, Ly, Lx = magnet.shape
    num_clusters = np.zeros(T, dtype=int)
    areas_list = []
    centroids_list = []
    for t in range(T):
        areas, cents = caratterizzazione_aster(magnet[t], soglia)
        num_clusters[t] = len(areas)
        areas_list.append(np.array(areas, dtype=int))
        centroids_list.append(cents)
    return num_clusters, areas_list, centroids_list

def sitimeno_sitipiu(magnet, soglia):
    Ly, Lx = magnet.shape
    
    asters_finali = heatmap_aster(magnet, soglia) * magnet
    
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

def probabilities(magnet, density, T, gamma, D = 1, metodo_calcolo = 1, x_neo = 0, y_neo = 0):
    Ly, Lx = magnet.shape
    dt = 1.0 / (4*D + np.exp(1/T))
    exp = np.zeros(2)

    if metodo_calcolo == 1:
        exp[0] = - 1/T * magnet[y_neo, x_neo]/density[y_neo, x_neo] # calcolato per le particelle spin + 1
        exp[1] =   1/T * magnet[y_neo, x_neo]/density[y_neo, x_neo] # calcolato per le particelle spin -1
    
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


def print_probabilities(magnet, density, soglia, T, gamma, D = 1, metodo_calcolo = 1):
    Ly, Lx = magnet.shape
    
    rate_flip_meno = []
    rate_flip_piu = []

    siti_meno, siti_piu = sitimeno_sitipiu(magnet, soglia)

    if len(siti_meno) == 0 or len(siti_piu) == 0:
        print("Non ci sono siti con magnetizzazione negativa o positiva.")
        return None, None, None, None

    where_asters = heatmap_aster(magnet, soglia)
    
    aster_finali_m = where_asters*magnet
    aster_finali_d = where_asters*density

    for i in range(Ly):
        for j in range(Lx):
        
            if (i,j) in siti_meno and aster_finali_d[i][j] != 0:
                rate_flip_meno.append(probabilities(aster_finali_m, aster_finali_d, T, gamma, D, metodo_calcolo, x_neo = j, y_neo = i))
        
            elif (i,j) in siti_piu and aster_finali_d[i][j] != 0:
                rate_flip_piu.append(probabilities(aster_finali_m, aster_finali_d, T, gamma, D, metodo_calcolo, x_neo = j, y_neo = i))
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

def approssimazioni(magnet, density, soglia, T):
    Ly, Lx = magnet.shape
    beta = 1/T
    mappa = heatmap_aster(magnet, soglia)
    
    x_fuori = []
    x_dentro = []
    
    if sum(mappa.flatten()) == 0:
        x_dentro = [0,0]
        print("Non ci sono siti dentro l'aster, non posso calcolare x")

    for i in range(Ly):
        for j in range(Lx):
            
            if mappa[i][j] != 0 and density[i][j] != 0:
                m = abs(magnet[i][j])
                d = density[i][j]
                x_dentro.append(beta*m/d)

            elif mappa[i][j] == 0 and density[i][j] != 0:
                m = abs(magnet[i][j])
                d = density[i][j]
                
                x_fuori.append(beta*m/d)
    
    x_dentro = np.array(x_dentro)
    x_fuori = np.array(x_fuori) 
    
    x_dentro_std = np.std(x_dentro, axis=0)/np.sqrt(x_dentro.shape[0]) #std sulla media, divido per sqrt(N)
    x_fuori_std = np.std(x_fuori, axis=0)/np.sqrt(x_fuori.shape[0])
    
    x_dentro = np.mean(x_dentro)
    x_fuori = np.mean(x_fuori)

    print(f'beta*m/d (dentro): {x_dentro:.4f} +/- {x_dentro_std:.6f}')
    print(f'beta*m/d (fuori): {x_fuori:.4f} +/- {x_fuori_std:.6f}')

    return x_dentro, x_dentro_std, x_fuori, x_fuori_std
    

def massimo_magnet(magnet):
    x = 0
    i_max = 0
    for i in range(magnet.shape[0]):
        max_magnet = np.max(np.abs(magnet[i,:,:]))
        if max_magnet > x:
            x = max_magnet
            i_max = i
    return x, i_max


def tempo_una_banda(magnet, soglia, t0=0):
    """
    Trova il primo istante t ≥ t0 a partire dal quale c'è sempre esattamente
    una sola banda (“aster”) attiva in ciascun frame.

    Parametri
    ---------
    magnet : ndarray, shape (T, Ly, Lx)
        Serie temporale dei campi di magnetizzazione.
    soglia : float
        Soglia assoluta su |magnet| per definire un sito “dentro” l’aster.
    t0 : int, opzionale
        Istante di partenza della ricerca (default=0).

    Ritorna
    -------
    t_unica_banda : int
        Il primo frame t ≥ t0 tale che
          1) num_tot[t] == 1  (una sola banda)
          2) num_tot[k] == 1 per tutti k ≥ t
        Se non esiste, ritorna T (numero di frame).
    """
    # num_tot[t] = numero di bande in frame t
    num_tot, _, _ = asters_over_time(magnet, soglia)

    T = magnet.shape[0]
    for t in range(t0, T):
        # Condizione: esattamente 1 banda a t e sempre 1 da t in poi
        if num_tot[t] == 1 and np.all(num_tot[t:] == 1):
            return t
    return T


def tempo_colonna_piena(magnet, soglia, t0=0):
    """
    Dopo che il sistema si è ridotto a una sola banda (tempo t1),
    trova il primo istante t ≥ t1 in cui esiste almeno una colonna
    (asse x costante) completamente occupata dall’aster.

    Parametri
    ---------
    magnet : ndarray, shape (T, Ly, Lx)
        Serie temporale dei campi di magnetizzazione.
    soglia : float
        Soglia assoluta su |magnet| per definire un sito “dentro” l’aster.
    t0 : int, opzionale
        Istante di partenza per la ricerca della banda unica (default=0).

    Ritorna
    -------
    t_colonna_piena : int
        Il primo frame t ≥ t1 in cui esiste almeno una colonna i
        tale che heatmap_aster(magnet[t], soglia)[:, i] sia tutta True.
        Se non esiste, ritorna T (numero di frame).
    """
    T, Ly, Lx = magnet.shape

    # 1) Trova t1: il tempo di comparsa di una sola banda
    t1 = tempo_una_banda(magnet, soglia, t0)

    # 2) A partire da t1, cerco il primo frame con una colonna interamente piena
    for t in range(t1, T):
        mappa = heatmap_aster(magnet[t], soglia)  # bool array (Ly, Lx)
        # Controllo se c'è almeno una colonna piena
        colonne_piene = np.all(mappa, axis=0)
        if np.any(colonne_piene):
            return t

    return T

def tempo_righe_non_vuote(magnet, soglia, t0=0):
    """
    Dopo che il sistema si è ridotto a una sola banda (tempo t1),
    trova il primo istante t ≥ t1 in cui tutte le righe
    contengono almeno un sito “dentro” l’aster.

    Parametri
    ---------
    magnet : ndarray, shape (T, Ly, Lx)
        Serie temporale dei campi di magnetizzazione.
    soglia : float
        Soglia assoluta su |magnet| per definire un sito “dentro” l’aster.
    t0 : int, opzionale
        Istante di partenza per la ricerca della banda unica (default=0).

    Ritorna
    -------
    t_righe_non_vuote : int
        Il primo frame t ≥ t1 in cui **ogni** riga j ha almeno un
        elemento True in heatmap_aster(magnet[t], soglia)[j, :].
        Se non esiste, ritorna T (numero di frame).
    """
    T, Ly, Lx = magnet.shape

    # 1) Trova t1: il tempo in cui compare una sola banda
    t1 = tempo_una_banda(magnet, soglia, t0)

    # 2) A partire da t1, cerco il primo frame in cui tutte le righe sono non-vuote
    for t in range(t1, T):
        mappa = heatmap_aster(magnet[t], soglia)  # bool array (Ly, Lx)
        # np.any(mappa, axis=1) è un array di lunghezza Ly: True se la riga ha almeno un True
        righe_non_vuote = np.any(mappa, axis=1)
        # Controllo che tutte le righe siano non-vuote
        if np.all(righe_non_vuote):
            return t

    return T

def build_log_indices(t_max, n_frames):
    """
    Ritorna un array di interi (di tipo int64) che rappresentano i tempi 
    (fra 0 e t_max-1) in cui salvare un frame, distribuiti (approssimativamente) in scala logaritmica.
    """
    # Genera n_frames valori equispaziati in scala log da 1 a t_max
    floats = np.logspace(np.log10(1), np.log10(t_max), n_frames)
    # Arrotondi all'intero più vicino e converti a int
    ints = np.round(floats).astype(np.int64)
    # Rimuovi eventuali duplicati e ordina
    unique_ints = np.unique(ints)
    # Sposta 1→0-based se necessario (se preferisci avere tutto tra 0 e t_max-1)
    unique_ints = np.clip(unique_ints - 1, 0, t_max - 1)
    return unique_ints



def tempo_fisico(frame_idx, t_max, n_frames, T, D=1.0):
    """
    Calcola il tempo fisico corrispondente al frame salvato #frame_idx, 
    quando i frame sono selezionati in scala logaritmica da 0 a t_max-1.

    Parametri
    ---------
    frame_idx : int
        Indice (0-based) nel vettore dei frame salvati (deve valere 0 <= frame_idx < n_frames).
    t_max : int
        Numero totale di passi di simulazione (frame numerati da 0 a t_max-1).
    n_frames : int
        Numero di snapshot che si sono salvati (quanti indici restituisce build_log_indices).
    T : float
        Temperatura fisica (usata per calcolare Δt).
    D : float, opzionale
        Coefficiente di diffusione, default: 1.0 (come nel paper).

    Ritorna
    -------
    t_phys : float
        Tempo fisico corrispondente al frame #frame_idx.
    """

    # 1) Costruisci gli indici log-spaced
    floats = np.logspace(np.log10(1), np.log10(t_max), n_frames)
    ints = np.round(floats).astype(np.int64)
    unique = np.unique(ints)
    frame_indices = np.clip(unique - 1, 0, t_max - 1)

    # 2) Controllo che frame_idx sia valido
    if not (0 <= frame_idx < len(frame_indices)):
        raise IndexError(f"frame_idx deve essere in [0, {len(frame_indices)-1}]")

    # 3) Calcola Δt come nel paper
    beta = 1.0 / T
    delta_t = 1.0 / (4 * D + np.exp(beta))

    # 4) Calcola il tempo fisico
    t_step = frame_indices[frame_idx]
    t_phys = t_step * delta_t

    return t_phys


## ANALISI PER BANDE
def autocorrelazione(vettore, k_max):
    "Funzione per calcolo autocorrelazione di un vettore"
    vettore_correlazione = np.zeros(k_max+1)
    len_vettore = len(vettore)

    for k in range(k_max+1):
        for i in range(len_vettore):
            vettore_correlazione[k] += (vettore[i]) * (vettore[(i+k) % (len_vettore)])

    vettore_correlazione = vettore_correlazione[0]

    return vettore_correlazione

