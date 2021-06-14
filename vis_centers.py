#Usage: python visualization.py [file] [oval size]

import tkinter as tk
import sys
from os.path import *

class VisCenters:

	def __init__(self,dataset_filename):
		bn=basename(dataset_filename)
		self.bname=splitext(bn)[0]
		self.cluster_lines=self.read_file(self.bname+"_clusters_v.txt")
		self.center_lines=self.read_file(self.bname+"_centers.txt")
		self.colors={0:"floralwhite",1:"red",2:"blue",3:"green",4:"gold",5:"cyan",6:"purple",7:"orange",8:"yellow",9:"dark olive green",10:"salmon",11:"turquoise",12:"brown",13:"black",14:"pink",15:"white",16:"dark red",17:"green yellow",18:"powder blue",19:"light yellow",20:"magenta",21:"orange red",22:"navy",23:"olive drab",24:"peach puff",25:"aquamarine",26:"hotpink",27:"lime",28:"khaki",29:"dodger blue",30:"crimson",31:"dark green"}	
		self.size=5

	def read_file(self,filename):
		try:
			data_file=open(filename)
			lines=data_file.readlines()
			data_file.close()
		except IOError:
			print('Could not read file')
			sys.exit(1)	
		return lines
		

	def visualize(self,interactive):
		root=tk.Tk()
		c=tk.Canvas(root,width=300,height=300)

		for i in range(len(self.cluster_lines)):
			point=(self.cluster_lines[i].strip()).split()
			x=float(point[0])
			y=float(point[1])
			colid=int(point[2])
			c.create_oval(x,y,x+self.size,y+self.size,fill=self.colors[colid],outline="#000000")

		for i in range(len(self.center_lines)):
			point=(self.center_lines[i].strip()).split()
			x=float(point[0])
			y=float(point[1])
			c.create_oval(x,y,x+self.size+5,y+self.size+5,fill="lawngreen",outline="#000000")

		c.pack()
		c.update()
		if interactive=='yes':
			c.mainloop()
		else:
			c.postscript(file="dpeaks/"+self.bname+"_vis.eps")
