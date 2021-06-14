import sys
from os.path import *


class Fscore:

	def __init__(self,class_filename,cluster_filename):
		class_lines=self.read_file(class_filename)
		cluster_lines=self.read_file(clus)
		self.classes={}
		self.clusters={}
		self.fill_group(class_lines,self.classes)
		self.fill_group(cluster_lines,self.clusters)
		
	def __init__(self,the_classes,the_clusters):
		self.classes={}
		self.clusters={}
		self.fill_group(the_classes,self.classes)
		self.fill_group(the_clusters,self.clusters)
		
	def read_file(self,filename):
		try:
			my_file=open(filename,'r')
			lines=my_file.readlines()
			my_file.close()
		except IOError:
			print('Could not read file')
			sys.exit(1)
		return lines

	def fill_group(self,lines,group):
		for line in lines:
			elements=(line.strip()).split(",")
			if len(elements)>1:
				member=elements[0] #Elemento de la clase
				member_group=elements[1] #Clase
				if member_group in group.keys():
					members=group[member_group]
				else:
					members=set()
				members.add(member)
				group[member_group]=members
				
	def calculate_fscore(self):
		global_fscore=0.0
		for cluster in self.clusters.keys():
			cluster_members=self.clusters[cluster]
			biggest_intersection=set()
			biggest_intersection_size=-1
			bi_class_size=-1
			for a_class in self.classes.keys():
				class_members=self.classes[a_class]
				intersection=cluster_members.intersection(class_members)
				if len(intersection)>biggest_intersection_size:
					biggest_intersection_size=len(intersection)
					biggest_intersection=intersection
					bi_class_size=len(class_members)
			precision=biggest_intersection_size*1.0/bi_class_size
			recall=biggest_intersection_size*1.0/len(cluster_members)
			fscore=2*precision*recall/(precision+recall)
			#print(fscore)
			#print(str(biggest_intersection_size)+","+str(len(class_members))+","+str(len(cluster_members))+","+str(precision) + "," + str(recall) + "," +str(fscore))
			global_fscore=global_fscore+fscore	
		global_fscore=global_fscore/len(self.clusters)
		return global_fscore