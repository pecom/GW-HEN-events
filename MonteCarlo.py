import numpy as np
import astropy as ap
import scipy as sp
from scipy import stats
import healpy as hp
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import csv

#%matplotlib inline
#%config InlineBackend.figure_format='retina'

# Constants and other setup variables. Event rates are events/year. Spherical points should be (theta, phi)
low_energy_rate = 500
high_energy_rate = 70
low_energy_resolution = (.3*np.pi/180, .3*np.pi/180)
high_energy_resolution = (.3*np.pi/180, .3*np.pi/180)
lisa_resolution = (1*np.pi/180, 1*np.pi/180)
NSIDE = 128
Npix = hp.nside2npix(NSIDE)

# Monte Carlo values
time_vals = [10, 1000, 10000, 100000, 1000000]
emri_array = [1, 100, 1000, 4000]
mbh_array = [1,2,3]
position_array = [1,2,3,4,5]

# Sample MC values
#time_vals = [500000]
#emri_array = [5000]
#mbh_array = [50]
#position_array = [1]

time_diff = 100000
mbh_rate = 3
emri_rate = 50
position_factor = 2

def add_time():
    time = np.random.randint(86400*365*1000)
    return time

# Get a random point on a sphere (uniform distribution). LISA should be fairly uniform and so should IceCube Gen II
def random_point():
    phi = np.random.random()*2*np.pi
    costheta = np.random.random()*2 - 1
    theta = np.arccos(costheta)
    return [theta, phi]

def plot_gaussian_event(etheta, ephi, sigma, amp):
    theta,phi=hp.pix2ang(NSIDE,np.arange(Npix))
    return amp*np.exp(-((phi - ephi)**2+(theta - etheta)**2)/(2*sigma**2))

def healpix_smooth(etheta, ephi, sig, amp):
    pind = hp.ang2pix(NSIDE, etheta, ephi)
    temp = np.zeros(Npix)
    temp[pind] = amp
    return hp.smoothing(temp, sigma=sig)

# Reduce redundancy! hen=True --> HEN events, hen=False --> GW events
# category depends on hen. hen=True: category = 0 --> low energy, 1 --> high energy
# hen=False, category = 0 --> EMRI, 1 --> MBH (doesn't matter for now)
def general_noise(hen, category):
    if hen:
        if category == 0:
            noise_val = low_energy_resolution
        else:
            noise_val = high_energy_resolution
    else:
        noise_val = lisa_resolution
    return np.random.normal()*np.array(noise_val)

def general_event(poisson_lambda, hen, category):
    points = []
    noise = []
    time = []
    k = np.random.poisson(lam=poisson_lambda)
    for i in range(k):
        points.append(random_point())
        noise.append(general_noise(hen, category))
        time.append(add_time())
    return [np.array(points), np.array(noise), np.array(time)]

def make_hp_map(points, noise):
    temp_map = np.zeros(Npix)
    for i,v in enumerate(points):
        temp_map += healpix_smooth(*v, noise[i], 1)
    return temp_map

def position_overlap(p1, p2, n1, n2):
    pvec1 = hp.ang2vec(*p1)
    pvec2 = hp.ang2vec(*p2)
    p1_disk = hp.query_disc(256, pvec1, n1*position_factor, inclusive=True)
    p2_disk = hp.query_disc(256, pvec2, n2*position_factor, inclusive=True)
    overlap = np.intersect1d(p1_disk, p2_disk)
    return len(overlap)

def time_overlap(t1,t2):
    d = {}
    for i,v in enumerate(t1):
        a = np.where(t2 < v + time_diff)[0] 
        b = np.where(t2 > v - time_diff)[0]
        c = np.intersect1d(a,b)
        if len(c) > 0:
            d[i] = c
    return d

def overlap(e1, n1, t1, e2, n2, t2):
    time_worthy = time_overlap(t1, t2)
    overlap_counter = 0
    pairs_overlap = []
    noise_overlap = []
    for i,v in enumerate(time_worthy):
        pos_1 = e1[v]
        noise_1 = np.linalg.norm(n1[v])
        for k in time_worthy[v]:
            pos_2 = e2[k]
            noise_2 = np.linalg.norm(n2[k])
            l = position_overlap(pos_1, pos_2, noise_1, noise_2)
            if l > 0:
                pairs_overlap.append([pos_1,pos_2])
                noise_overlap.append([noise_1, noise_2])
                overlap_counter += 1
    return overlap_counter, pairs_overlap, noise_overlap

def monte_carlo_run():
    low = general_event(low_energy_rate, True, 0)
    high = general_event(high_energy_rate, True, 1)
    emri = general_event(emri_rate, False, 0)
    mbh = general_event(mbh_rate, False, 1)
    l_e, le_p, le_n = overlap(*low, *emri)
    l_m, lm_p, lm_n = overlap(*low, *mbh)
    h_e, he_p, he_n = overlap(*high, *emri)
    h_m, hm_p, hm_n = overlap(*high, *mbh)
    
    return np.array([l_e, l_m, h_e, h_m]), np.array([le_p,lm_p,he_p,hm_p]), np.array([le_n, lm_n, he_n, hm_n])

def mc_big_hammer(n=100):
    global time_diff
    global emri_rate
    global mbh_rate
    global position_factor
    with open('output/monte_run.csv', 'w') as csvfile:
        a = csv.writer(csvfile)
        header = ['n', 'time_diff', 'position_factor', 'emri', 'mbh', 'total_coincident']
        a.writerow(header)
    with open('output/total_run.csv', 'w') as csvfile:
        b = csv.writer(csvfile)
        header = ['n', 'time_diff', 'position_factor', 'emri', 'mbh', 'total_coincident']
        b.writerow(header)
    for i in time_vals:
        time_diff = i
        for j in emri_array:
            emri_rate = j
            for k in mbh_array:
                mbh_rate = k
                for l in position_array:
                    lval=[]
                    position_factor = l
                    for m in tqdm(range(n)):
                        monte_data,points,noise = monte_carlo_run()
                        total = np.sum(monte_data)
                        lval.append(total)
                        monte_name = 'monte_big_time%i_emri%i_mbh%i_pos%i_%i_' % (i, j, k, l, m)
                        np.save('output/' + monte_name, monte_data)
                        if total > 0:
                            big_map = np.zeros(Npix)
                            for map_index in range(len(points)):
                                temp_map = np.zeros(Npix)
                                if monte_data[map_index] == 0:
                                    continue
                                noises = []
                                pointses = []
                                for pair_ind in range(len(noise[map_index])):
                                    noises += noise[map_index][pair_ind]
                                    pointses += points[map_index][pair_ind]
                                temp_map += make_hp_map(pointses, noises)
                                big_map += temp_map
                                
                                np.save('maps/'+monte_name+'map_'+str(map_index),temp_map)
                                hp.mollview(temp_map)
                                plt.savefig('maps/'+monte_name+'mollview_'+str(map_index))
                                plt.close()
                            hp.mollview(big_map)
                            plt.savefig('maps/'+monte_name+'mollview')
                            plt.close()
                        with open('output/monte_run.csv', 'a') as csvfile:
                                c = csv.writer(csvfile)
                                data = [m,i,l,j,k,total]
                                c.writerow(data)
                    run_total = np.sum(lval)
                    with open('output/total_run.csv', 'a') as csvfile:
                        d = csv.writer(csvfile)
                        data = [n,i,l,j,k,run_total]
                        d.writerow(data)
    
                    array_name = 'lval_big_time%i_emri%i_mbh%i_pos%i_%i_' % (i, j, k, l, n)
                    np.save('output/' + array_name, lval)
                    plt.hist(lval);
                    plt.savefig('images/' + array_name)
                    plt.close()

def mc(n=100):
    lval = []
    for i in tqdm(range(n)):
        lval.append(np.sum(monte_carlo_run()))
    return lval

n = 100
m = mc_big_hammer(n)