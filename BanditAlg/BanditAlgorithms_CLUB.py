from random import choice, random, sample
import numpy as np
import networkx as nx
import numpy as np
from BanditAlg.BanditAlgorithms_LinUCB import *
import math
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import collections

class CLUBUserStruct(LinUCBUserStruct):
	def __init__(self,featureDimension,  lambda_, userID):
		LinUCBUserStruct.__init__(self,featureDimension = featureDimension, lambda_= lambda_, userID = userID)
		self.reward = 0
		self.CA = self.A
		self.Cb = self.b
		self.CAInv = np.linalg.inv(self.CA)
		self.CTheta = np.dot(self.CAInv, self.Cb)
		self.I = lambda_*np.identity(n = featureDimension)	
		self.counter = 0
		self.CBPrime = 0.1
		self.d = featureDimension
	def updateParameters(self, articlePicked_FeatureVector, click, alpha_2):
		#LinUCBUserStruct.updateParameters(self, articlePicked_FeatureVector, click)
		#alpha_2 = 1
		self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
		self.b += articlePicked_FeatureVector*click
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = np.dot(self.AInv, self.b)
		self.counter+=1
		self.CBPrime = alpha_2*np.sqrt(float(1+math.log10(1+self.counter))/float(1+self.counter))
		# if self.CBPrime == 0:
		# 	print self.counter

	def updateParametersofClusters(self,clusters,userID,Graph,users, sortedUserList):
		self.CA = self.I
		self.Cb = np.zeros(self.d)
		#print type(clusters)

		for i in range(len(clusters)):
			userID_GraphIndex = sortedUserList.index(userID)
			if clusters[i] == clusters[userID_GraphIndex]:
				self.CA += float(Graph[userID_GraphIndex, i])*(users[ sortedUserList[i]  ].A - self.I)
				self.Cb += float(Graph[userID_GraphIndex, i])*users[sortedUserList[i] ].b
		self.CAInv = np.linalg.inv(self.CA)
		self.CTheta = np.dot(self.CAInv,self.Cb)

	def getProb(self, alpha, article_FeatureVector, time):
		mean = np.dot(self.CTheta, article_FeatureVector)
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.CAInv),  article_FeatureVector))
		pta = mean +  alpha * var*np.sqrt(math.log10(time+1))
		if pta > self.pta_max:
			pta = self.pta_max
		return pta

class CLUBAlgorithm:
	def __init__(self, G, seed_size, oracle, dimension, alpha, alpha_2, lambda_, FeatureDic, FeatureScaling, feedback = 'edge',  cluster_init="Erdos-Renyi"):
		self.time = 0
		self.G = G
		self.oracle = oracle
		self.seed_size = seed_size
		self.feedback = feedback

		self.dimension = dimension
		self.alpha = alpha
		self.alpha_2 = alpha_2
		self.lambda_ = lambda_
		self.FeatureDic = FeatureDic
		self.FeatureScaling = FeatureScaling

		self.users = {}  #Nodes
		self.currentP =nx.DiGraph()
		for u in self.G.nodes():
			self.users[u] = CLUBUserStruct(dimension,lambda_, u)
			for v in self.G[u]:
				self.currentP.add_edge(u,v, weight=random())
		n = len(self.users)
		#print 'usersNum', n
		#print len(self.users.keys()), type(self.users.keys())
		self.userIDSortedList = list(self.users.keys())
		self.userIDSortedList.sort()
		#print len(self.userIDSortedList)
		self.SortedUsers = collections.OrderedDict(sorted(self.users.items()))

		if (cluster_init=="Erdos-Renyi"):
			p = 3*math.log(n)/n
			self.Graph = np.random.choice([0, 1], size=(n,n), p=[1-p, p])
			g = csr_matrix(self.Graph)
			N_components, components = connected_components(g)
			self.clusters = []
		else:
			self.Graph = np.ones([n,n]) 
			g = csr_matrix(self.Graph)
			N_components, components = connected_components(g)
			self.clusters = []
			
	def decide(self):
		self.time +=1
		S = self.oracle(self.G, self.seed_size, self.currentP)
		return S

	def updateParameters(self, S, live_nodes, live_edges):
		for u in S:
			for (u, v) in self.G.edges(u):
				featureVector = self.FeatureScaling*self.FeatureDic[(u,v)]
				if (u,v) in live_edges:
					reward = live_edges[(u,v)]
				else:
					reward = 0
				self.SortedUsers[u].updateParameters(featureVector, reward, self.alpha_2)
			self.updateGraphClusters(u, 'False')
		# print 'Start connected component'
		N_components, component_list = connected_components(csr_matrix(self.Graph))
		print('N_components:',N_components)
		# print 'End connected component'
		self.clusters = component_list
		for u in S:
			self.SortedUsers[u].updateParametersofClusters(self.clusters, u, self.Graph, self.SortedUsers, self.userIDSortedList)
			for (u, v) in self.G.edges(u):		
				featureVector = self.FeatureScaling * self.FeatureDic[(u,v)]		
				self.currentP[u][v]['weight']  = self.SortedUsers[u].getProb(self.alpha, featureVector, self.time)
		
	def updateGraphClusters(self,userID, binaryRatio):
		n = len(self.SortedUsers)
		for j in self.SortedUsers:
			# print self.SortedUsers[userID].CBPrime, self.SortedUsers[j].CBPrime
			ratio = float(np.linalg.norm(self.SortedUsers[userID].UserTheta - self.SortedUsers[j].UserTheta,2))/float(self.SortedUsers[userID].CBPrime + self.SortedUsers[j].CBPrime)
			#print float(np.linalg.norm(self.users[userID].UserTheta - self.users[j].UserTheta,2)),'R', ratio
			if ratio > 1:
				ratio = 0
			elif binaryRatio == 'True':
				ratio = 1
			elif binaryRatio == 'False':
				ratio = 1.0/math.exp(ratio)
			#print 'ratio',ratio
			userID_GraphIndex = self.userIDSortedList.index(userID)
			j_GraphIndex = self.userIDSortedList.index(j)
			self.Graph[userID_GraphIndex][j_GraphIndex] = ratio
			self.Graph[j_GraphIndex][userID_GraphIndex] = self.Graph[userID_GraphIndex][j_GraphIndex]
		# print 'N_components:',N_components
		return
	def getLearntParameters(self, userID):
		return self.users[userID].UserTheta
	def getP(self):
		return self.currentP