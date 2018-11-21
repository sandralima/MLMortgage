# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:41:43 2018

@author: sandr
"""

def inorderTraversal(tree, idxNode):
    if (tree[idxNode]==-1):
        return str(-1)
    else:        
       return str(inorderTraversal(tree, 2*(idxNode+1)-1)) + ' ' +str(tree[idxNode])+ ' '+ str(inorderTraversal(tree, 2*(idxNode+1)))
   
def inorderTraversal1(tree, root):
    res = []
    if tree[root]!=1:
        res = inorderTraversal1(tree, 2*(root+1)-1) 
        res.append(tree[root])
        res = res + inorderTraversal1(tree, 2*(root+1))
    return res    
mytree = [1,2,3,4,-1,5,-1,7,8,-1,9,-1,-1,10,11,-1,-1,-1,-1,-1,-1]

#otree = list([1,2,-1,-1,-1])
print(inorderTraversal(mytree,0))
#print(inorderTraversal1(otree,0))