#A parameter testing script for presentsGame

import presentsGame as pg
import numpy as np
import csv

numberOfPresents = 648
sleighDim = 1000
numberOfGenerations = 15000
num_ordered = 0
numberOfGenes = 30

fileOut = 'NumPresents' + str(numberOfPresents) + 'NumGenerations' + str(numberOfGenerations) + '.csv'
with open(fileOut, 'wb') as ofile:
    writer = csv.writer(ofile)
    header=['numberOfGenes','rotateFrac','htBufFrac','newPosFrac','mx_ht_param','crossoverFrac','p','k','numChildren','num_ordered','mean_fitness', 'std_fitness', 'min_fitness', 'max_fitness']
    writer.writerow(header)

    dtype=[('PresentID','int64'),('x1','uint16'),('y1','uint16'),('z1','uint32'),('x8','uint16'),('y8','uint16'),('z8','uint32')]
    topdown=np.loadtxt('TopDown.csv', dtype=dtype,delimiter=',',skiprows=1, usecols = [0,1,2,3,22,23,24])

    dtype=[('PresentID','int64'),('Dimension1','uint16'),('Dimension2','uint16'),('Dimension3','uint16')]
    presents=np.loadtxt('presents.csv',dtype=dtype,delimiter=',',skiprows=1)

    #runPresentsGA(Presents, TopDownPacking, sleighDim, number_of_presents, number_of_chromosomes,number_of_generations_, rotate_frac, ht_buffer_frac, new_pos_frac, mx_ht_param, crossover_frac, p_, k_, num_children, num_ordered)
    #print pg.runPresentsGA(presents, topdown, 1000, 50, 25,50, 0.10, 0.10, 0.10, 4, 0.50, 0.8, 12, 8, 0.1)

    
    #for numberOfGenes in [25,50,100]:
    for rotateFrac in [0.05]:
        for htBufFrac in [0]:
            for newPosFrac in [0.05]:
                for mx_ht_param in [10000]:
                    for crossoverFrac in [0.1]:
                        for p in [0.95]:
                            for k in [5]:
                                for numChildren in [8]:
                                    topdownbyval = np.copy(topdown)
                                    presentsbyval = np.copy(presents)
                                    input = [numberOfGenes,rotateFrac,htBufFrac,newPosFrac,mx_ht_param,crossoverFrac,p,k,numChildren,num_ordered]
                                    print input
                                    result = pg.runPresentsGA(presents, topdownbyval, sleighDim, numberOfPresents, numberOfGenes,numberOfGenerations, rotateFrac, htBufFrac, newPosFrac, mx_ht_param, crossoverFrac, p, k, numChildren, num_ordered)
                                    print result
                                    writer.writerow(input + result)

ofile.close()
