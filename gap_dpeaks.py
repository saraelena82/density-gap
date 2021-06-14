#Usage: python gap_dpeaks.py [file] [output format] [dc] [interactive?]
#Example: python gap_dpeaks.py aggregation.csv v 120 yes
#Example: python gap_dpeaks.py aggregation.csv ari 120 no

import sys
import math
import random
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from os.path import *
from sklearn import metrics
from fscore import *
from vis_centers import *

class Dpeaks:
	#Constructor
	def __init__(self, dataset_filename, output, interactive):
		self.dataset=dataset_filename
		bn=basename(dataset_filename)
		self.bname=splitext(bn)[0]
		feature_vectors=self.read_file(dataset_filename)
		self.points=self.process_vectors(feature_vectors)
		self.n=len(self.points)
		shape=(self.n,self.n)
		self.distances=np.zeros(shape)
		self.max_distance=-1
		self.max_rho=-1
		self.max_delta=-1
		self.avg_rho=-1
		self.avg_delta=-1
		self.min_distance=sys.float_info.max
		self.nn={}
		self.output_format=output
		self.dc=-1
		self.sorted_rho=[]
		self.sorted_delta=[]
		self.labels_true=[]
		labels_filename=self.bname+"_classes.csv"
		self.fill_labels_true(labels_filename)
		self.is_interactive=interactive
	
	def read_file(self, filename):
		try:
			data_file=open(filename)
			lines=data_file.readlines()
			data_file.close()
		except IOError:
			print('Could not read file')
			sys.exit(1)
		return lines
		
	def fill_labels_true(self, labels_filename):
		label_lines=self.read_file(labels_filename)
		for line in label_lines:
			elements=(line.strip()).split(",")
			if len(elements)>1:
				element=elements[0]
				element_class=elements[1]
				self.labels_true.append(element_class)
	
	def euclidean_distance(self, pointA, pointB):
		sum=0
		for i in range(len(pointA)):
			sum=sum+pow(pointA[i]-pointB[i],2)
		return math.sqrt(sum)

	def process_vectors(self, vectors):
		points=[]
		for vector in vectors:
			point=(vector.strip()).split(",")
			point=[float(x) for x in point]
			points.append(point)
		return points
	
	#Distance matrix
	def get_distances(self):
		for i in range(self.n):
			for j in range(i+1,self.n):
				self.distances[i,j]=self.euclidean_distance(self.points[i],self.points[j])
				self.distances[j,i]=self.distances[i,j]
				if self.distances[i,j]>self.max_distance:
					self.max_distance=self.distances[i,j]
				if self.distances[i,j]<self.min_distance:
					self.min_distance=self.distances[i,j]
		print("Min. distance:",self.min_distance)
		print("Max. distance,",self.max_distance)
	
	def get_densities(self):
		densities=[]
		for i in range(self.n):
			rho=0
			for j in range(self.n):
				if self.distances[i,j]<self.dc:
					#rho=rho+1
					rho=rho+math.exp(-1*(self.distances[i,j]/self.dc)*(self.distances[i,j]/self.dc))
			densities.append([i,rho])
			if rho>self.max_rho:
				self.max_rho=rho
		self.sorted_rho=sorted(densities,key=lambda s:s[1],reverse=True)
	
	#Obtain deltas
	def get_deltas(self):
		deltas=[]
		for i in range(len(self.sorted_rho)-1,0,-1):
			id_i=self.sorted_rho[i][0]
			delta=sys.float_info.max
			nn_i=-1
			for j in range(i-1,-1,-1):
				id_j=self.sorted_rho[j][0]
				if self.distances[id_i,id_j]<=delta:
					delta=self.distances[id_i,id_j]
					nn_i=id_j
			if delta>=self.max_delta:
				self.max_delta=delta
			deltas.append([id_i,delta])
			self.nn[id_i]=nn_i
		max_j=-1
		id_0=self.sorted_rho[0][0]
		for i in range(self.n):
			id_i=self.sorted_rho[i][0]
			if self.distances[id_0,id_i]>max_j:
				max_j=self.distances[id_0,id_i]
		deltas.append([self.sorted_rho[0][0],max_j])
		if max_j>self.max_delta:
			self.max_delta=max_j
		self.sorted_delta=sorted(deltas,key=lambda s:s[1],reverse=True)
	
	def get_scores(self, dc_value):
		dg=[]
		self.dc=dc_value
		self.get_densities()
		self.get_deltas()
		total_delta=sum([self.sorted_delta[i][1] for i in range(len(self.sorted_delta))])
		self.avg_delta=total_delta/len(self.sorted_delta)
		total_rho=sum([self.sorted_rho[i][1] for i in range(len(self.sorted_rho))])
		self.avg_rho=total_rho/len(self.sorted_rho)
		id_sorted_rho=sorted(self.sorted_rho,key=lambda s:s[0])
		id_sorted_delta=sorted(self.sorted_delta,key=lambda s:s[0])
		sum_score=0
		for i in range(len(id_sorted_rho)):
			score=float(id_sorted_rho[i][1]*id_sorted_delta[i][1])
			sum_score=sum_score+score
			dg.append([i,score,id_sorted_rho[i][1],id_sorted_delta[i][1]])
		avg_score=sum_score/len(dg)
		self.sorted_dg=sorted(dg,key=lambda s:s[1],reverse=True)
		
	def detect_center_candidates(self):
		candidates=[]
		#Select points in first quadrant as candidate centers
		for i in range(len(self.sorted_dg)):
			if self.sorted_dg[i][2]>=self.avg_rho and self.sorted_dg[i][3]>=self.avg_delta:
				candidates.append(self.sorted_dg[i])
		#If there are no points in the first quadrant, select the two points with the best score
		if len(candidates)==0:
			candidates.append(self.sorted_dg[0])
			candidates.append(self.sorted_dg[1])
		elif len(candidates)==1:
			candidates.append(self.sorted_dg[1])
		print("Candidate centers:",len(candidates))	
		return candidates
		
	def select_centers(self,sdist_points,sel):
		sdist=[sdist_points[i][2] for i in range(len(sdist_points))]
		sum_sdist=sum(sdist)
		avg_sdist=sum(sdist)/len(sdist)
		#print("Avg. distance between centers: ", avg_sdist)
		sds=[]
		for sd in sdist_points:
			if sd[2]>=avg_sdist:
				sds.append(sd[0])
		#Se hace el reverse para poder agarrar los mini-clusters
		sorted_sds=sorted(sds,reverse=True)
		final_selected=[]
		#sorted_sds[0]+1 es la posicion de corte de los centros
		for i in range(sorted_sds[0]+1):
			final_selected.append(sel.pop(0))
		return final_selected
		
	def detect_centers(self,candidates):
		#If there are more than 2 candidates, select the isolated ones; ELSE, keep the candidates as centers
		if len(candidates)>2:
			sdist_points=[[i,i+1,round(abs(candidates[i][1]-candidates[i+1][1]),3)] for i in range(len(candidates)-1)]
			sel=candidates.copy()
			final_selected=self.select_centers(sdist_points,sel)
			#If only one point is isolated, remove this point and repeat procedure
			if len(final_selected)<2:
				sel=candidates.copy()
				sel.pop(0)
				sdist_points.pop(0)
				final_selected.extend([x for x in self.select_centers(sdist_points,sel)])
		else:
			final_selected=candidates
		print("*******************")
		print("Final centers:",len(final_selected))
		how_many_centers=len(final_selected)
		finally_selected=[]
		#Instead of using candidates, use points sorted by score
		for i in range(how_many_centers):
			finally_selected.append(self.sorted_dg[i])
		return finally_selected
	
	def cluster(self,centers):
		clustering={}
		ids_centers=[centers[j][0] for j in range(len(centers))]
		chosen={}
		#Map each center to an id (0 to number of centers-1)
		for i in range(len(ids_centers)):
			id_center=ids_centers[i]
			chosen[id_center]=i
		#Assign points to clusters
		for i in range(len(self.sorted_rho)):
			id_i=self.sorted_rho[i][0]
			if id_i in ids_centers:
				clustering[id_i]=id_i
			else:
				clustering[id_i]=clustering[self.nn[id_i]]
		result=[]
		for id in clustering.keys():
			result.append([id,chosen[clustering[id]]])
		final=[]
		#Append clustering and centers to result
		final=[result,ids_centers]
		return final
	
	def draw_decision_graph(self):
		#Points and colors
		x=[x[2] for x in self.sorted_dg]
		y=[x[3] for x in self.sorted_dg]
		colors=np.random.rand(len(x))
		plt.scatter(x,y,c=colors)
		ax = plt.gca()
		#Lines
		l=mlines.Line2D([self.avg_rho,self.avg_rho], [0,self.max_delta],color='r')
		l2=mlines.Line2D([0,self.max_rho], [self.avg_delta,self.avg_delta],color='r')
		ax.add_line(l)
		ax.add_line(l2)
		plt.xlabel(r'$\rho$')
		plt.ylabel(r'$\delta$')
		if self.is_interactive=='yes':
			plt.show()
		else:
			plt.savefig("dpeaks/"+self.bname+"-decgraph.eps")
	
	def draw_candidate_decision_graph(self,candidates):
	#Points and colors
		x=[x[1] for x in candidates]
		y=[1 for x in candidates]
		colors=np.random.rand(len(x))
		plt.scatter(x,y,c=colors)
		ax = plt.gca()
		ax.axes.get_yaxis().set_visible(False)
		plt.xlabel(r'$\gamma$ scores')
		if self.is_interactive=='yes':
			plt.show()
		else:
			plt.savefig("dpeaks/"+self.bname+"-cdecgraph.eps")
	
	def write_results(self,my_result):	
		#Create files
		CLUSTERS=open(self.bname+'_clusters.txt','w')
		CENTERS=open(self.bname+'_centers.txt','w')
		if self.output_format=='v':
			VECTORS=open(self.bname+'_clusters_v.txt','w')
		#Obtain results
		clustering=my_result[0]
		centers=my_result[1]
		#Sort points by id
		sorted_points=sorted(clustering,key=lambda s:s[0])
		#Write clustering file
		for i in range(len(clustering)):
			chosen=sorted_points[i][1]
			point=sorted_points[i][0]
			CLUSTERS.write(str(point)+","+str(chosen))
			CLUSTERS.write("\n")
			if self.output_format=='v':
				point=self.points[sorted_points[i][0]]
				VECTORS.write(" ".join([str(i) for i in point])+" "+str(chosen))
				VECTORS.write("\n")
		CLUSTERS.close()
		#Write centers file
		for i in range(len(centers)):
			center=self.points[centers[i]]
			CENTERS.write(" ".join([str(i) for i in center]))
			CENTERS.write("\n")
		CENTERS.close()
		if self.output_format=='v':
			VECTORS.close()
	
	def run_dpeaks(self, dc_value):
		self.get_scores(dc_value)
		candidate_centers=self.detect_center_candidates()
		self.draw_candidate_decision_graph(candidate_centers)
		centers=self.detect_centers(candidate_centers)
		self.draw_decision_graph()
		my_result=self.cluster(centers)
		self.write_results(my_result)
		return my_result
		
	def evaluate(self, result):
		clustering=result[0]
		sorted_points=sorted(clustering,key=lambda s:s[0])
		labels=[sorted_points[i][1] for i in range(len(sorted_points))]
		ari=metrics.adjusted_rand_score(self.labels_true, labels)
		f=self.get_fscore(labels)
		scores=[ari,f]
		#Visualization
		if self.output_format=='v':
			visualization=VisCenters(self.dataset)
			visualization.visualize(self.is_interactive)
		return scores
	
	def get_fscore(self,cluster_labels):
		clusters=[str(i)+","+str(cluster_labels[i]) for i in range(len(cluster_labels))]
		classes=[str(i)+","+str(self.labels_true[i]) for i in range(len(self.labels_true))]
		f=Fscore(classes,clusters)
		fs=f.calculate_fscore()
		return fs
	
	def get_dg(self):
		return self.sorted_dg
	
dpeaks=Dpeaks(sys.argv[1],sys.argv[2],sys.argv[4])
dpeaks.get_distances()
clustering=dpeaks.run_dpeaks(float(sys.argv[3]))
scores=dpeaks.evaluate(clustering)
print("ARI: ",scores[0])
print("F-score: ",scores[1])