# FUNZIONI DI SIMULAZIONE
import numpy as np
import numba
from numba import njit, prange

@njit #comando per permettere compilazione numba 
def rateflip_numba(Lx, Ly, metodo_calcolo, x_neo, y_neo, s_neo, beta, gamma,
                   magnet, density):
    """
    Calcola il rate di flip per una singola particella.
    - Lx, Ly taglia del sistema
    - metodo calcolo = 1 (AIM STANDARD)
                     = 2 (METODO DEI VICINI)
                     = 3 (METODO PICCOLE INTERAZIONI LOCALI SIMMETRICHE)
    - x_neo, y_neo, s_neo = caratteristiche particella che si vuole flippare
    - beta = temperatura inversa
    - gamma = se metodo_calcolo = 1 (AIM STANDARD) indifferente
                                = 2 (METODO DEI VICINI) livello di bias asimmetrico
                                = 3 (METODO PICCOLE INTERAZIONI LOCALI SIMMETRICHE) grado di non località 
    - magnet, density = matrici che tiene conto della magnetizzazione totale e della densità totale
    """
    if metodo_calcolo == 1:
        # AIM standard
        return -s_neo * beta * magnet[y_neo, x_neo] / density[y_neo, x_neo]
    elif metodo_calcolo == 2:
        # metodo dei vicini

        # vicini con condizioni periodiche al contorno
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

        if s_neo == 1: #particella con spin 1 "guarda in avanti" a destra
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
        else: #particella con spin -1 "guarda in avanti" a sinistra
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
        return -s_neo * beta * num/den
                         
    elif metodo_calcolo == 3:
        # metodo di slow-interazioni locali

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
        
        return -s_neo * beta * num / den
    

def evoluzione_profili(Lx, Ly, D, epsilon,
                               beta, particles, t_max, gamma, metodo_calcolo):
    """
    Simula l’evoluzione temporale e restituisce profili medi e il parametro d'ordine al variare del tempo
    - Lx, Ly taglia del sistema
    - D = coefficiente di diffusione
    - epsilon = grado di propulsione della singola particella
    - beta = temperatura inversa
    - particles = matrice delle proprietà delle particelle da misurare
    - t_max = tempo massimo di campionamento
    - gamma = se metodo_calcolo = 1 (AIM STANDARD) indifferente
                                = 2 (METODO DEI VICINI) livello di bias asimmetrico
                                = 3 (METODO PICCOLE INTERAZIONI LOCALI SIMMETRICHE) grado di non località 
    - metodo calcolo = 1 (AIM STANDARD)
                     = 2 (METODO DEI VICINI)
                     = 3 (METODO PICCOLE INTERAZIONI LOCALI SIMMETRICHE)
    """
    n_particles = particles.shape[0]
    # crea le matrici di densità e magnetizzazione
    density = np.zeros((Ly, Lx))
    magnet  = np.zeros((Ly, Lx))

    # popola matrici iniziali
    for k in range(n_particles):
        x, y, s = particles[k]
        density[y, x] += 1
        magnet[y, x]  += s

    prof_d = np.zeros(Lx, np.float64) # inizializzo matrici dei profili, dandogli anche il formato
    prof_m  = np.zeros(Lx, np.float64)
    order_param = np.zeros(t_max, np.float64)
    dt = 1.0 / (4*D + np.exp(beta))

    # evoluzione temporale vera e propria
    for i in range(t_max):
        # ciclo sulle particelle 
        for _ in range(n_particles):

            neo = np.random.randint(0, n_particles) #a ogni ciclo estraggo particella a caso
            x_neo, y_neo, s_neo = particles[neo] #e prendo la proprietà di questa particella

            # flip spin chiamando rateflip
            w = np.exp(rateflip_numba(Lx, Ly, metodo_calcolo,
                                      x_neo, y_neo, s_neo,
                                      beta, gamma,
                                      magnet, density))
            
            r =  np.random.random()

            if r < w * dt: #flip dello spin?
                # aggiorna vettore particelle e matrice magnet
                particles[neo, 2] = -s_neo
                magnet[y_neo, x_neo] -= 2 * s_neo 
            
            elif r < (w + D)*dt: #DOWN
                new_y = (y_neo + 1) % Ly #variabile temporanea stando attenta a condizioni al contorno
                
                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[new_y, x_neo] += 1
                magnet [new_y, x_neo] += s_neo
                particles[neo, 1] = new_y
            
            elif r < (w + 2*D)*dt: #UP
                new_y = (y_neo - 1) % Ly #variabile temporanea stando attenta a condizioni al contorno

                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[new_y, x_neo] += 1
                magnet [new_y, x_neo] += s_neo
                particles[neo, 1] = new_y
                y_neo = new_y
            
            elif r < (w +D*(3+epsilon*s_neo))*dt: #RIGHT
                new_x = (x_neo + 1) % Lx #variabile temporanea stando attenta a condizioni al contorno

                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[y_neo, new_x] += 1
                magnet [y_neo, new_x] += s_neo
                particles[neo, 0] = new_x

            elif r < (w+4*D)*dt: #LEFT
                new_x = (x_neo - 1) % Lx #variabile temporanea stando attenta a condizioni al contorno

                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[y_neo, new_x] += 1
                magnet [y_neo, new_x] += s_neo
                particles[neo, 0] = new_x


        # profilo all'ultima iterazione del codice (lo faccio a mano per riuscire a farlo compilare da numba)
        if i == t_max - 1:
            for xi in range(Lx):
                # somma colonne
                col_sum_m = 0
                col_sum_d = 0
                for yi in range(Ly):
                    col_sum_m += magnet[yi, xi]
                    col_sum_d += density[yi, xi]
                prof_m[xi] += col_sum_m / Ly
                prof_d[xi] += col_sum_d / Ly

        order_param[i] = np.sum(magnet)/(n_particles) #somma magnetizzazione e divide per taglia del sistema

    return prof_d, prof_m, order_param

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

@njit
def evoluzione_filmino(Lx, Ly, D, epsilon,
                               beta, particles, t_max, gamma, metodo_calcolo, array):
    """
    Simula l’evoluzione temporale e restituisce profili medi e il parametro d'ordine al variare del tempo
    - Lx, Ly taglia del sistema
    - D = coefficiente di diffusione
    - epsilon = grado di propulsione della singola particella
    - beta = temperatura inversa
    - particles = matrice delle proprietà delle particelle da misurare
    - t_max = tempo massimo di campionamento
    - gamma = se metodo_calcolo = 1 (AIM STANDARD) indifferente
                                = 2 (METODO DEI VICINI) livello di bias asimmetrico
                                = 3 (METODO PICCOLE INTERAZIONI LOCALI SIMMETRICHE) grado di non località 
    - metodo calcolo = 1 (AIM STANDARD)
                     = 2 (METODO DEI VICINI)
                     = 3 (METODO PICCOLE INTERAZIONI LOCALI SIMMETRICHE)
    - array = array di interi che contiene i tempi in cui salvare i frame
    """
    
    n_particles = particles.shape[0]
    # crea le matrici di densità e magnetizzazione
    density = np.zeros((Ly, Lx))
    magnet  = np.zeros((Ly, Lx))

    # popola matrici iniziali
    for k in range(n_particles):
        x, y, s = particles[k]
        density[y, x] += 1
        magnet[y, x]  += s

    #prealloca le variabili da salvare
    n_frames_effettivi = np.size(array)
    frames_d = np.zeros((n_frames_effettivi, Ly, Lx), np.float64)
    frames_m = np.zeros((n_frames_effettivi, Ly, Lx), np.float64)
    order_param  = np.zeros(t_max, np.float64)
    frame_idx = 0
    
    dt = 1.0 / (4*D + np.exp(beta))

    # evoluzione temporale vera e propria
    for i in range(t_max):
        # ciclo sulle particelle 
        for _ in range(n_particles):

            neo = np.random.randint(0, n_particles) #a ogni ciclo estraggo particella a caso
            x_neo, y_neo, s_neo = particles[neo] #e prendo la proprietà di questa particella

            # flip spin chiamando rateflip
            w = np.exp(rateflip_numba(Lx, Ly, metodo_calcolo,
                                      x_neo, y_neo, s_neo,
                                      beta, gamma,
                                      magnet, density))
            
            r =  np.random.random()

            if r < w * dt: #flip dello spin?
                # aggiorna vettore particelle e matrice magnet
                particles[neo, 2] = -s_neo
                magnet[y_neo, x_neo] -= 2 * s_neo 
            
            elif r < (w + D)*dt: #DOWN
                new_y = (y_neo + 1) % Ly #variabile temporanea stando attenta a condizioni al contorno
                
                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[new_y, x_neo] += 1
                magnet [new_y, x_neo] += s_neo
                particles[neo, 1] = new_y
            
            elif r < (w + 2*D)*dt: #UP
                new_y = (y_neo - 1) % Ly #variabile temporanea stando attenta a condizioni al contorno

                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[new_y, x_neo] += 1
                magnet [new_y, x_neo] += s_neo
                particles[neo, 1] = new_y
                y_neo = new_y
            
            elif r < (w +D*(3+epsilon*s_neo))*dt: #RIGHT
                new_x = (x_neo + 1) % Lx #variabile temporanea stando attenta a condizioni al contorno

                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[y_neo, new_x] += 1
                magnet [y_neo, new_x] += s_neo
                particles[neo, 0] = new_x

            elif r < (w+4*D)*dt: #LEFT
                new_x = (x_neo - 1) % Lx #variabile temporanea stando attenta a condizioni al contorno

                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[y_neo, new_x] += 1
                magnet [y_neo, new_x] += s_neo
                particles[neo, 0] = new_x
        
        for j in range(n_frames_effettivi):
            if i == array[j]:
                frames_d[frame_idx,:,:] = density
                frames_m[frame_idx,:,:] = magnet
                frame_idx += 1

        order_param[i] = np.sum(magnet)/(n_particles) #somma magnetizzazione e divide per taglia del sistema
    
    return frames_d, frames_m, order_param

@njit
def evoluzione_scorporata(Lx, Ly, D, epsilon,
                               beta, particles, t_max, gamma, metodo_calcolo, array):
    """
    Simula l’evoluzione temporale e restituisce profili medi e il parametro d'ordine al variare del tempo
    In questo caso, divido il processo di evoluzione temporale in due fasi: - flip (obiettivo è velocizzare questo processo)
                                                                            - diffusione
    - Lx, Ly taglia del sistema
    - D = coefficiente di diffusione
    - epsilon = grado di propulsione della singola particella
    - beta = temperatura inversa
    - particles = matrice delle proprietà delle particelle da misurare
    - t_max = tempo massimo di campionamento
    - gamma = se metodo_calcolo = 1 (AIM STANDARD) indifferente
                                = 2 (METODO DEI VICINI) livello di bias asimmetrico
                                = 3 (METODO PICCOLE INTERAZIONI LOCALI SIMMETRICHE) grado di non località 
    - metodo calcolo = 1 (AIM STANDARD)
                     = 2 (METODO DEI VICINI)
                     = 3 (METODO PICCOLE INTERAZIONI LOCALI SIMMETRICHE)
    - array = array di interi che contiene i tempi in cui salvare i frame
    """
    
    n_particles = particles.shape[0]
    # crea le matrici di densità e magnetizzazione
    density = np.zeros((Ly, Lx))
    magnet  = np.zeros((Ly, Lx))

    # popola matrici iniziali
    for k in range(n_particles):
        x, y, s = particles[k]
        density[y, x] += 1
        magnet[y, x]  += s

    #prealloca le variabili da salvare
    n_frames_effettivi = np.size(array)
    frames_d = np.zeros((n_frames_effettivi, Ly, Lx), np.float64)
    frames_m = np.zeros((n_frames_effettivi, Ly, Lx), np.float64)
    order_param  = np.zeros(t_max, np.float64)
    frame_idx = 0

    # evoluzione temporale vera e propria
    for i in range(t_max):
        # ciclo sulle particelle 
        for _ in range(n_particles):

            neo = np.random.randint(0, n_particles) #a ogni ciclo estraggo particella a caso
            x_neo, y_neo, s_neo = particles[neo] #e prendo la proprietà di questa particella

            # flip spin chiamando rateflip
            w = np.exp(rateflip_numba(Lx, Ly, metodo_calcolo,
                                      x_neo, y_neo, s_neo,
                                      beta, gamma,
                                      magnet, density))
            
            probabilità_flip = w/(w
                                + np.exp(rateflip_numba(Lx, Ly, metodo_calcolo, x_neo, y_neo, -s_neo,
                                      beta, gamma, magnet, density)))
            
            r =  np.random.random()
            s =  np.random.random()
            
            if r < probabilità_flip: 
               particles[neo, 2] = -s_neo
               magnet[y_neo, x_neo] -= 2 * s_neo 
            
            if s < D: #DOWN
                new_y = (y_neo + 1) % Ly #variabile temporanea stando attenta a condizioni al contorno
                
                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[new_y, x_neo] += 1
                magnet [new_y, x_neo] += s_neo
                particles[neo, 1] = new_y
            
            elif s < 2*D: #UP
                new_y = (y_neo - 1) % Ly #variabile temporanea stando attenta a condizioni al contorno

                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[new_y, x_neo] += 1
                magnet [new_y, x_neo] += s_neo
                particles[neo, 1] = new_y
                y_neo = new_y
            
            elif r < (D*(3+epsilon*s_neo)): #RIGHT
                new_x = (x_neo + 1) % Lx #variabile temporanea stando attenta a condizioni al contorno

                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[y_neo, new_x] += 1
                magnet [y_neo, new_x] += s_neo
                particles[neo, 0] = new_x

            elif r < (1- 4*D): #LEFT
                new_x = (x_neo - 1) % Lx #variabile temporanea stando attenta a condizioni al contorno

                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[y_neo, new_x] += 1
                magnet [y_neo, new_x] += s_neo
                particles[neo, 0] = new_x
        
        for j in range(n_frames_effettivi):
            if i == array[j]:
                frames_d[frame_idx,:,:] = density
                frames_m[frame_idx,:,:] = magnet
                frame_idx += 1

        order_param[i] = np.sum(magnet)/(n_particles) #somma magnetizzazione e divide per taglia del sistema
    
    return frames_d, frames_m, order_param

@njit
def evoluzione_track(Lx, Ly, D, epsilon,
                               beta, particles, t_max, gamma, metodo_calcolo, array, n_track = 0):
    """
    Simula l’evoluzione temporale e restituisce profili medi e il parametro d'ordine al variare del tempo
    - Lx, Ly taglia del sistema
    - D = coefficiente di diffusione
    - epsilon = grado di propulsione della singola particella
    - beta = temperatura inversa
    - particles = matrice delle proprietà delle particelle da misurare
    - t_max = tempo massimo di campionamento
    - gamma = se metodo_calcolo = 1 (AIM STANDARD) indifferente
                                = 2 (METODO DEI VICINI) livello di bias asimmetrico
                                = 3 (METODO PICCOLE INTERAZIONI LOCALI SIMMETRICHE) grado di non località 
    - metodo calcolo = 1 (AIM STANDARD)
                     = 2 (METODO DEI VICINI)
                     = 3 (METODO PICCOLE INTERAZIONI LOCALI SIMMETRICHE)
    - array = array di interi che contiene i tempi in cui salvare i frame
    - n_track = numero di particelle da tracciare nel tempo (se 0, non traccio nessuna particella)
    """
    
    n_particles = particles.shape[0]
    # crea le matrici di densità e magnetizzazione
    density = np.zeros((Ly, Lx))
    magnet  = np.zeros((Ly, Lx))

    # popola matrici iniziali
    for k in range(n_particles):
        x, y, s = particles[k]
        density[y, x] += 1
        magnet[y, x]  += s

    #prealloca le variabili da salvare
    n_frames_effettivi = np.size(array)
    frames_d = np.zeros((n_frames_effettivi, Ly, Lx), np.float64)
    frames_m = np.zeros((n_frames_effettivi, Ly, Lx), np.float64)
    order_param  = np.zeros(t_max, np.float64)
    frame_idx = 0
    track = np.zeros((n_track, t_max, 3), np.int64) #matrice per tracciare le particelle

    dt = 1.0 / (4*D + np.exp(beta))

    # evoluzione temporale vera e propria
    for i in range(t_max):
        # ciclo sulle particelle 
        for _ in range(n_particles):

            neo = np.random.randint(0, n_particles) #a ogni ciclo estraggo particella a caso
            x_neo, y_neo, s_neo = particles[neo] #e prendo la proprietà di questa particella

            # flip spin chiamando rateflip
            w = np.exp(rateflip_numba(Lx, Ly, metodo_calcolo,
                                      x_neo, y_neo, s_neo,
                                      beta, gamma,
                                      magnet, density))
            
            r =  np.random.random()

            if r < w * dt: #flip dello spin?
                # aggiorna vettore particelle e matrice magnet
                particles[neo, 2] = -s_neo
                magnet[y_neo, x_neo] -= 2 * s_neo 
            
            elif r < (w + D)*dt: #DOWN
                new_y = (y_neo + 1) % Ly #variabile temporanea stando attenta a condizioni al contorno
                
                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[new_y, x_neo] += 1
                magnet [new_y, x_neo] += s_neo
                particles[neo, 1] = new_y
            
            elif r < (w + 2*D)*dt: #UP
                new_y = (y_neo - 1) % Ly #variabile temporanea stando attenta a condizioni al contorno

                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[new_y, x_neo] += 1
                magnet [new_y, x_neo] += s_neo
                particles[neo, 1] = new_y
                y_neo = new_y
            
            elif r < (w +D*(3+epsilon*s_neo))*dt: #RIGHT
                new_x = (x_neo + 1) % Lx #variabile temporanea stando attenta a condizioni al contorno

                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[y_neo, new_x] += 1
                magnet [y_neo, new_x] += s_neo
                particles[neo, 0] = new_x

            elif r < (w+4*D)*dt: #LEFT
                new_x = (x_neo - 1) % Lx #variabile temporanea stando attenta a condizioni al contorno

                density[y_neo, x_neo] -= 1
                magnet [y_neo, x_neo] -= s_neo
                density[y_neo, new_x] += 1
                magnet [y_neo, new_x] += s_neo
                particles[neo, 0] = new_x
        
        for j in range(n_frames_effettivi):
            if i == array[j]:
                frames_d[frame_idx,:,:] = density
                frames_m[frame_idx,:,:] = magnet
                frame_idx += 1

        order_param[i] = np.sum(magnet)/(n_particles) #somma magnetizzazione e divide per taglia del sistema
        if track != 0:
            track[:, i] = particles[:n_track, :]
    
    return frames_d, frames_m, order_param, track

