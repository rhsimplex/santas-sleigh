import time
import os
import MetricCalculation as mc
import multiprocessing as mp
import numpy as np
import sys

def Fitness(solution):
    ordinal = np.arange(solution.size, 0, -1, dtype='int64')
    orderingTerm = np.sum(np.abs(ordinal-solution['PresentID']))
    ##print 'Ordering Term = ' + str(orderingTerm)
    heightTerm = solution['z8'][-1]
    #set either return value to zero to optimize for that characteristic only
    return (heightTerm,orderingTerm)

def Crossover(chromosomeA, chromosomeB, crossoverPercent):
    crossoversize = int(crossoverPercent*chromosomeA.shape[0])
    crossoverstart = np.random.randint(chromosomeA.shape[0])
    crossoverend = min((chromosomeA.shape[0],crossoverstart+crossoversize))
    for i in np.arange(crossoverstart,crossoverend):
        temp = chromosomeB[i]
        chromosomeB[i]=chromosomeA[i]
        chromosomeA[i]=temp

def Mutate(chromosome, flipPercent, rotatePercent, heightBufferPercent, newPositionPercent, maxhtparam, sleighSize):
    flippedIndices = np.random.randint(chromosome.shape[0]-1, size = flipPercent*chromosome.shape[0])
    rotatedIndices = np.random.randint(chromosome.shape[0], size = rotatePercent*chromosome.shape[0])
    newPositionIndices = np.random.randint(chromosome.shape[0], size = newPositionPercent*chromosome.shape[0])
    heightChangedIndices = np.random.randint(chromosome.shape[0], size = heightBufferPercent*chromosome.shape[0])
    maxhtbuf = np.max(chromosome[:,3])
    #Chromosome row format:
    #[presents[i][0],dim[0],dim[1],dim[2],xcoord,ycoord,mxht]
    for i in flippedIndices:
        temp = chromosome[i]
        chromosome[i] = chromosome[i+1]
        chromosome[i+1] = temp
    for i in rotatedIndices:
        np.random.shuffle(chromosome[i][1:4])
        #need to make sure presents are inside sleigh
        xcoord = chromosome[i][4]
        ycoord = chromosome[i][5]
        xdim = chromosome[i][1]
        ydim = chromosome[i][2]
        if xcoord + xdim - 1 > sleighSize:
            chromosome[i][4] -= (xcoord + xdim - 1) - sleighSize  
        if ycoord + ydim - 1 > sleighSize:
            chromosome[i][5] -= (ycoord + ydim - 1) - sleighSize
    for i in newPositionIndices:
        chromosome[i][4] = np.random.random_integers(1,sleighSize - chromosome[i][1]+1)
        chromosome[i][5] = np.random.random_integers(1,sleighSize - chromosome[i][2]+1)
    for i in heightChangedIndices:
        chromosome[i][-1]=np.random.geometric(np.float(maxhtparam)/np.float(maxhtbuf),1)-1
        
    
def InitialLoaderWorker(work_queue, done_queue):
    for chromosome in iter(work_queue.get, None):
        print 'Loading ' + chromosome + '...'
##        dtype=[('PresentID','uint32'),('x1','uint16'),('y1','uint16'),('z1','uint32'),('x2','uint16'),('y2','uint16'),('z2','uint32'),('x3','uint16'),('y3','uint16'),('z3','uint32'),('x4','uint16'),('y4','uint16'),('z4','uint32'),('x5','uint16'),('y5','uint16'),('z5','uint32'),('x6','uint16'),('y6','uint16'),('z6','uint32'),('x7','uint16'),('y7','uint16'),('z7','uint32'),('x8','uint16'),('y8','uint16'),('z8','uint32')]
        dtype=[('PresentID','int64'),('x1','uint16'),('y1','uint16'),('z1','uint32'),('x8','uint16'),('y8','uint16'),('z8','uint32')]
        a = np.loadtxt(chromosome, dtype=dtype,delimiter=',',skiprows=1, usecols = [0,1,2,3,22,23,24])
        for row in a:
            row[0]=1000001 - row[0]
        a.sort(order=['z8','PresentID'])
        for row in a:
            row[0]=1000001 - row[0]
        fitness = Fitness(a)
        print chromosome + ' fitness ' + str(fitness)
    return True
        
    

if __name__ == "__main__":

    numWorkers = mp.cpu_count() - 1
    populationfolder = 'Population'
    
##    presentsFilename = 'presents.csv'
##
##    
##    print 'Reading presents file...',
##    presentsReadStart = time.clock()
##    presents = mc.readPresentsFile(presentsFilename)
##    print 'done! %i presents read. Time: %f' % (len(presents), time.clock()-presentsReadStart)

    
    geneStart = time.clock()
    os.chdir(populationfolder)
    initialChromosomeNames = os.listdir(os.getcwd())
    if initialChromosomeNames == []:
        print 'Initializing gene pool.\nLoading chromosomes into memory...'
        #initialize

    work_queue = mp.Queue();
    done_queue = mp.Queue();
    processes = []
    
    for chromosomeName in initialChromosomeNames:
        work_queue.put(chromosomeName)

    for i in range(numWorkers):
        p = mp.Process(target = InitialLoaderWorker, args = (work_queue, done_queue))
        p.start()
        processes.append(p)
        work_queue.put(None)

    for p in processes:
        p.join()
    

    print 'Time: %f' % (time.clock()-geneStart)    
    
    
    
