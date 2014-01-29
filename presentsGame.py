#Runs the presents GA

import numpy as np
import GA as ga
import csv
import TopDownChromosome as tdc
from random import sample
from copy import deepcopy

#Builds a random TETRIS packing...adds a random origin and height buffer
def randomPackingOrder(presents, sleighSize, maxhtparam):
    maxhtbuf = max([np.max(presents['Dimension1']),np.max(presents['Dimension2']),np.max(presents['Dimension3'])])
    presentOrder =  np.empty([presents.size,7],dtype='uint32')                   
    for i in np.arange(presents.size):
        dim=[presents[i][1],presents[i][2],presents[i][3]]
        np.random.shuffle(dim)
        xcoord=np.random.random_integers(1,sleighSize - dim[0]+1)
        ycoord=np.random.random_integers(1,sleighSize - dim[1]+1)
        mxht = np.random.geometric(np.float(maxhtparam)/np.float(maxhtbuf),1)-1
        #mxht = 0
        presentOrder[i] = np.array([presents[i][0],dim[0],dim[1],dim[2],xcoord,ycoord,mxht])
    return presentOrder

def chromosomeToSolution(chromosome,sleighSize):
    heightbysquare = np.ones((sleighSize,sleighSize),dtype='uint32')
    def dropPresent(present):
        xcoord = present[4]
        ycoord = present[5]
        presentXdim=present[1]
        presentYdim=present[2]
        presentZdim=present[3]
        presentExtraZ=present[6]
        
        landingzone = heightbysquare[xcoord : xcoord + presentXdim, ycoord : ycoord + presentYdim]
        try:
            maxht = np.amax(landingzone)
        except ValueError:
            print [present[0], xcoord, ycoord, presentXdim, presentYdim,presentZdim,presentExtraZ]
            sys.exit()
        newzoneht = presentZdim + maxht + presentExtraZ
        heightbysquare[xcoord : xcoord + presentXdim, ycoord : ycoord + presentYdim]=newzoneht
        return np.array([present[0],xcoord, ycoord, maxht + presentExtraZ,xcoord+presentXdim,ycoord+presentYdim,presentZdim + maxht + presentExtraZ])
    return np.apply_along_axis(dropPresent,1,chromosome)
 
def runPresentsGA(Presents, TopDownPacking, sleighDim, number_of_presents, number_of_chromosomes,number_of_generations_, rotate_frac, ht_buffer_frac, new_pos_frac, mx_ht_param, crossover_frac, p_, k_, num_children, num_ordered):
    SLEIGH = sleighDim
    numpresents = number_of_presents#648
    numChromosomes = number_of_chromosomes#50
    numgenerations = number_of_generations_#1500

    flipPercent = 0.00
    rotatePercent = rotate_frac#0.10
    heightBufferPercent = ht_buffer_frac#0.0
    newPositionPercent =  new_pos_frac#0.10
    maxhtparam = mx_ht_param#1

    crossoverPercentMax = crossover_frac#0.5
    p=p_#0.85
    k=k_#8
    numChildren= num_children#14

    numOrdered=num_ordered#0

    fileOut = 'output.csv'
    ##This is a test case to see if my GA can equal or beat the standard (top down) packing for 100 presents
    
    #Import the Top Down packing
    topdown = TopDownPacking
    #dtype=[('PresentID','int64'),('x1','uint16'),('y1','uint16'),('z1','uint32'),('x8','uint16'),('y8','uint16'),('z8','uint32')]
    #topdown=np.loadtxt('TopDown.csv', dtype=dtype,delimiter=',',skiprows=1, usecols = [0,1,2,3,22,23,24])

    #import the presents file
    presents=Presents
    #dtype=[('PresentID','int64'),('Dimension1','uint16'),('Dimension2','uint16'),('Dimension3','uint16')]
    #presents=np.loadtxt('presents.csv',dtype=dtype,delimiter=',',skiprows=1)

    topdown = (topdown[::-1][0:numpresents])
    topdownraw = deepcopy(topdown)

    for row in topdown:
        row[0] = row[0] - (1000000 - numpresents)
        row[0]=topdown.size+1 - row[0]
    topdown.sort(order=['z8','PresentID'])
    for row in topdown:
        row[0]=topdown.size+1 - row[0]


    presents = presents[::-1][0:numpresents]
    terms = ga.Fitness(topdown)
    fitness = 2*terms[0]+terms[1]
    #print 'TopDown Fitness = ' + str(fitness) + ' (Height Term = ' +str(terms[0])+ ' Ordering Term = ' + str(terms[1])+')'



    #GA begins here...single threaded because this is a simple calculation

    chromosomes = []
    #Build some random solutions


    for i in np.arange(numChromosomes):
        chromosomes.append(randomPackingOrder(presents, SLEIGH, 5))

    #Add some premade chromosomes here if you so desire
    if numOrdered > 0:
        numOrderedChromosomes = numOrdered#int(fracOrdered * numChromosomes)
        del chromosomes[-numOrderedChromosomes:]
        for i in np.arange(numOrderedChromosomes):
            chromosomes.append(tdc.topDownToChromosome(topdownraw, SLEIGH))

    #Build solutions from chromosomes
    solutions = []

        
    GAdebug = False
    GenerationMessages = True
    printToFile = True
    if printToFile:
        ofile = open(fileOut, 'wb')
        writer = csv.writer(ofile)
        header=[]
        for i in range(numChromosomes):
            header.append("Gene"+str(i))
        header.insert(0,"Generation")
        writer.writerow(header)
    for genIndex in np.arange(numgenerations):
        solutions = []
        for chromosome in chromosomes:
            solutions.append(chromosomeToSolution(chromosome,SLEIGH))

        #rank this generation
        if GenerationMessages: print 'Generation ' + str(genIndex)
        fitnesses=[]
        for solution in solutions:
            recview = solution.view('uint32,uint32,uint32,uint32,uint32,uint32,uint32')
            recview.dtype.names = ('PresentID','x1','y1','z1','x8','y8','z8')
            #print solution
            #print recview
            recview = recview.flatten()
            for row in recview:
                row[0] = row[0] - (1000000 - numpresents)
                row[0]=recview.size+1 - row[0]
            recview.sort(order=['z8','PresentID'])
            for row in recview:
                row[0]=recview.size+1 - row[0]

            terms = ga.Fitness(recview)
            fitness = 2*terms[0]+terms[1]
            fitnesses.append(fitness)
            if GenerationMessages: print 'Solution Fitness = ' + str(fitness) + ' (Height Term = ' +str(terms[0])+ ' Ordering Term = ' + str(terms[1])+')'

        #cull the herd -- remove the n most unfit genes to return gene pool to original size
        def fitcheck(chromosomeIndex): return fitnesses[chromosomeIndex]
        sortedGeneration = sorted(range(len(chromosomes)), key=fitcheck)
        if GAdebug:
            print "Generation sorted by fitness: " + str(sortedGeneration)
            print "Fitnesses: " + str([fitnesses[l] for l in sortedGeneration])

        if genIndex > 0:
            if GAdebug: print "Chromosomes to be elminated: " + str(sortedGeneration[numChromosomes-len(chromosomes):])
            chromosomeIndices = range(len(chromosomes))
            forDeletion = sorted(sortedGeneration[numChromosomes-len(chromosomes):])
            for i in range(len(forDeletion)):
                delIndex = forDeletion.pop(0)
                del chromosomes[delIndex]
                del chromosomeIndices[delIndex]
                del fitnesses[delIndex]
                forDeletion = [x-1 for x in forDeletion]
            if GAdebug:
                print "Survivors: " + str(chromosomeIndices)
                print "Fitnesses: " + str(fitnesses)


        row = fitnesses[:]
        if printToFile:
            row.insert(0,genIndex)
            writer.writerow(row)
            
        #tournament selection
        parentPop=range(numChromosomes)
        parentIndices=[]
        def probSelection(p, n): return p*(1-p)**n
        for i in range(numChildren):
            players = sample(parentPop, k)
            if GAdebug: print "Tournament players:" + str(players)
            if GAdebug: print "Fitnesses:" + str([fitnesses[l] for l in players])
            sortedPlayers = sorted(players, key=fitcheck)
            sortedPlayers.reverse()
            if GAdebug: print "Sorted players: " + str(sortedPlayers)
            score = np.random.random()
            if GAdebug: print "Score: " + str(score)
            j=len(sortedPlayers)-1
            for player in sortedPlayers:
                if score < probSelection(p,j):
                    #parentIndices.append(player)
                    parentIndices.append(parentPop.pop(parentPop.index(player)))
                    break
                else:
                    j=j-1
            else:
                #parentIndices.append(sortedPlayers[-1])
                parentIndices.append(parentPop.pop(parentPop.index(sortedPlayers[-1])))
            if GAdebug:
                print "Winner: " + str(parentIndices[-1])
                print "Remaining players: " + str(parentPop)
        if GAdebug: print "Selected for crossover: " + str(parentIndices)

        #crossover and mutate children
        pairs = np.reshape(np.array(parentIndices),(numChildren/2,2))
        
        for pair in pairs:
            if GAdebug: print "Crossing: " + str(pair) + '...',
            #copy parents then crossover/mutate the copies
            chromosomes.append(np.array(chromosomes[pair[0]],copy=True))
            chromosomes.append(np.array(chromosomes[pair[1]],copy=True))
            ga.Crossover(chromosomes[-1],chromosomes[-2],crossoverPercentMax*np.random.random())
            if GAdebug: print 'done. Mutating children...',
            ga.Mutate(chromosomes[-1], flipPercent, rotatePercent, heightBufferPercent, newPositionPercent, maxhtparam, SLEIGH)
            ga.Mutate(chromosomes[-2], flipPercent, rotatePercent, heightBufferPercent, newPositionPercent, maxhtparam, SLEIGH)
            if GAdebug: print 'done.'

    ##    for chromosome in chromosomes:
    ##        ga.Mutate(chromosome, flipPercent, rotatePercent, heightBufferPercent, newPositionPercent, maxhtparam, SLEIGH)
    ##
    ##    for chromosomeA in chromosomes:
    ##        for chromosomeB in chromosomes:
    ##            ga.Crossover(chromosomeA, chromosomeB, crossoverPercent)

    if printToFile: ofile.close()
    return [np.average(row), np.std(row), np.min(row), np.max(row)]

