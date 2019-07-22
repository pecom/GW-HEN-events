import numpy as np
import astropy as ap
import scipy as sp
from scipy import stats
import healpy as hp
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import csv

# Constants and other setup variables. Event rates are events/year. Spherical points should be (theta, phi)
low_energy_rate = 50
high_energy_rate = 70
low_energy_resolution = np.linalg.norm(np.array([.3*np.pi/180, .3*np.pi/180]))
high_energy_resolution = np.linalg.norm(np.array([.3*np.pi/180, .3*np.pi/180]))
lisa_resolution = np.linalg.norm(np.array([1*np.pi/180, 1*np.pi/180]))
NSIDE = 128
Npix = hp.nside2npix(NSIDE)
make_plots = False
make_arrays = False


# Monte Carlo values
time_vals = [10, 1000, 10000, 100000, 1000000]
emri_array = [1, 100, 1000, 4000]
mbh_array = [1,2,3]
position_array = [1]

# Sample MC values
# time_vals = [1000, 500000]
# emri_array = [5000]
# mbh_array = [50]
# position_array = [1]

time_diff = 100000
mbh_rate = 3
emri_rate = 50
position_factor = 1

def Ang2Vec(theta,phi):
    return np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])

def add_time(count):
    time = np.random.randint(86400*365*1000, size=count)
    return time

# Get a random point on a sphere (uniform distribution). LISA should be fairly uniform and so should IceCube Gen II
def random_point(count):
    phi = np.random.random(count)*2*np.pi
    costheta = np.random.random(count)*2 - 1
    theta = np.arccos(costheta)
    return np.array([theta, phi]).T

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
def general_noise(count, hen, category):
    if hen:
        if category == 0:
            noise_val = low_energy_resolution
        else:
            noise_val = high_energy_resolution
    else:
        noise_val = lisa_resolution
    return np.random.normal(size=count)*np.array(noise_val)

def general_event(poisson_lambda, hen, category):
    k = np.random.poisson(lam=poisson_lambda)
    points = random_point(k)
    noise = general_noise(k, hen, category)
    time = add_time(k)
    return [points, noise, time]

def make_hp_map(points, noise):
    temp_map = np.zeros(Npix)
    for i,v in enumerate(points):
        temp_map += healpix_smooth(*v, noise[i], 1)
    return temp_map

def position_coincident(p1, p2, n1, n2):
    pvec1 = Ang2Vec(*p1)
    pvec2 = Ang2Vec(*p2)
    dist = np.linalg.norm(pvec1 - pvec2)
    if dist < (n1 + n2)*position_factor:
        return True
    else:
        return False

def time_coincident(t1, t2):
    d = {}
    for i,v in enumerate(t1):
        h = np.abs(v - t2)
        a = np.where(h <= time_diff)[0]
        if a.shape[0] > 0:
            d[i] = a
    return d

# Number==True -> We only want to keep count of overlap, not the points or noises associated with the events.
def overlap(e1, n1, t1, e2, n2, t2, number=True):
    time_worthy = time_coincident(t1, t2)
    overlap_counter = 0
    pairs_overlap = []
    noise_overlap = []
    for i,v in enumerate(time_worthy):
        pos_1 = e1[v]
        noise_1 = n1[v]
        for k in time_worthy[v]:
            pos_2 = e2[k]
            noise_2 = n2[k]
            l = position_coincident(pos_1, pos_2, noise_1, noise_2)
            if l:
                if not number:
                    pairs_overlap.append([pos_1,pos_2])
                    noise_overlap.append([noise_1, noise_2])
                overlap_counter += 1
    
    return overlap_counter, pairs_overlap, noise_overlap

def monte_carlo_run(number):
    low = general_event(low_energy_rate, True, 0)
    high = general_event(high_energy_rate, True, 1)
    emri = general_event(emri_rate, False, 0)
    mbh = general_event(mbh_rate, False, 1)
    l_e, le_p, le_n = overlap(*low, *emri, number)
    l_m, lm_p, lm_n = overlap(*low, *mbh, number)
    h_e, he_p, he_n = overlap(*high, *emri, number)
    h_m, hm_p, hm_n = overlap(*high, *mbh, number)
    
    return np.array([l_e, l_m, h_e, h_m]), np.array([le_p,lm_p,he_p,hm_p]), np.array([le_n, lm_n, he_n, hm_n])

def mc_big_hammer(number, n=100, make_arrays=False, make_plots=False):
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
                        monte_data,points,noise = monte_carlo_run(number)
                        total = np.sum(monte_data)
                        lval.append(total)
                        monte_name = 'monte_big_time%i_emri%i_mbh%i_pos%i_%i_' % (i, j, k, l, m)
                        if make_arrays:
                            np.save('output/' + monte_name, monte_data)
                        if total > 0 and make_plots:
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
                    if make_arrays:
                        array_name = 'lval_big_time%i_emri%i_mbh%i_pos%i_%i_' % (i, j, k, l, n)
                        np.save('output/' + array_name, lval)
                    if make_plots:
                        plt.hist(lval);
                        plt.savefig('images/' + array_name)
                        plt.close()

def mc(number, n=100):
    lval = []
    for i in tqdm(range(n)):
        k, _, _ = monte_carlo_run(number)
        lval.append(k)
    return lval

parser = argparse.ArgumentParser(description='Run monte carlo simulation of GW events and HEN events to estimate chance coincident rate of events')

parser.add_argument('count', metavar='N', help='Number of iterations for a monte carlo run', )
parser.add_argument('big', metavar='b', help='Big monte carlo (goes over variable space defined in lines 24-27) or a singular monte carlo run using values from lines 35-38')

args= parser.parse_args()
n = int(args.count)

if args.big == 'True':
    mc_big_hammer(True, n)
else:
    mc(True, n)