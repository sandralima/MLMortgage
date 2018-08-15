# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:43:41 2018

@author: sandr
"""
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
import random

#defining a class:    
class CustomRandom(object):
    def __init__(self, max_parms, seedBase=0):
        self.seedBase = seedBase
        self.m_index = seedBase
        self.prime =  self.getBiggestPrime(max_parms)    # 4294967291 # 32 bits
        self.xorOperand = random.getrandbits(int(max_parms).bit_length())  # 0x5bf03635 #31 bit_length
        self.max_parms = max_parms
        
        
    def QRP(self, x):
        # pre-condition: 0<x<prime        
        residue = (x * x) % self.prime
        if (2*x <= self.prime):
            return residue 
        else:
            return (self.prime - residue)
    
    def permute(self, x):
        if (x >= self.prime): 
            return x 
        else:
            x2 = self.QRP(x)
            x2withXOR = x2 ^ self.xorOperand
            if (x2withXOR==0): x2withXOR = x2
            return self.QRP(x2withXOR)

    def next(self):                
        self.m_index += 1
        value = self.permute(self.m_index)
        return value    

    def is_prime(self, n):
        """
        Assumes that n is a positive natural number
        """
        # We know 1 is not a prime number
        if n == 1:
            return False
        i = 2
        # This will loop from 2 to int(sqrt(x))
        while i*i <= n:
            # Check if i divides x without leaving a remainder
            if n % i == 0:
                # This means that n has a factor in between 2 and sqrt(n)
                # So it is not a prime number
                return False
            i += 1
        # If we did not find any factor in the above loop,
        # then n is a prime number
        return True
                
    def getBiggestPrime(self, max_perms):
        
        cVal = max_perms
        while (cVal>0):
            if self.is_prime(cVal):
                if(cVal % 4 == 3):
                    return cVal
            cVal -= 1
        return (max_perms)
    
    def get_batch(self, batch_size):
        
        batch_random = set()
        # z = 0
        # r = 0
        while True:        
            value = self.next()
            if value <= self.max_parms:
                batch_random.add(value) #.append(value)
                # z +=1
                if len(batch_random) >=batch_size:
                    break
            else: 
                #r+=1 
                #print('randoms refused: ', r)
                #if r>=1:
                self.m_index = self.seedBase
                self.xorOperand = random.getrandbits(int(self.max_parms).bit_length())                    
        
        return batch_random

 
def dump():
    myRandom = CustomRandom(19824000)  
        
    for i in range(141):
        batch_size = 141600
        startTime = datetime.now()                        
        print('Batch: ', i)        
        batch_random = list(myRandom.get_batch(batch_size))
        print('Time for Getting ' + str(batch_size) +' random elements: ', datetime.now() - startTime)
        median = np.median(batch_random)
        iqr = np.percentile(batch_random, [25, 75])
        print('median: ', median, 'iqr: ', iqr)
        print('batch size: ', len(batch_random))
        plt.hist(batch_random, bins=200)
        plt.ylabel('counts')
        plt.show()
        plt.scatter(range(len(batch_random)), batch_random)
        plt.show()
            


# dump()

# shuffle the list and take the elements from init to End.
# random.shuffle(mylist)
# print(mylist[:10])
# less_than = [x for x in mylist if x<1000000]
# random_batch = random.sample(mylist, 4000)  # this take a sample but with repetitions.