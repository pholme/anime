'''

Commands to aggregate the frames to MP4:

for high resolution:

ffmpeg -r 48 -i frames/primaryschool/%05d.png -vcodec libx264 -pix_fmt yuv420p -strict -2 -acodec aac clips/primaryschool.mp4

for twitter:

ffmpeg -i frames/primaryschool2/%05d.png -pix_fmt yuv420p -vcodec libx264 -vf 'scale=640:trunc(ow/a/2)*2'  -acodec aac -vb 1024k -minrate 1024k -maxrate 1024k -bufsize 1024k -ar 44100  -ac 2  -strict experimental -r 30  clips/primaryschool2.mp4

'''

import matplotlib
matplotlib.use('Agg')
import csv
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import numpy as np
from sys import argv
import graph_tool.all as gt
from copy import deepcopy
from alive_progress import alive_bar
from scipy.spatial import ConvexHull
from scipy.ndimage import uniform_filter1d

layout_step = 0.2 # for the layout algorithm
layout_k = 5.0 # preferred edge length
layout_c = 0.3 # Relative strength of repulsive forces
layout_p = 2.0 # Repulsive force exponent
size_fac = 0.02 # rescaling node sizes
recfac = 0.95 # how much smaller a square relative to circles
expansion_fac = 1.4 # how much larger blobs should be after the smoothing
link_weight_low = 0.5 # lower link weight cut off
node_size_low = 0.9 # lower node size limit
weight2sizefac = 0.2 # for converting node weight to size
weight2widthfac = 0.15 # for converting link weight to width

nwindow = 600 # for smoothing of bounding box in the time dimension
xymargin = 0.07 # fraction of margins to the (non-strict) bounding box

niter = 2 # how long to iterate the spring algorithm

clock_x = 0.96 # setting dimensions of the clock
clock_y = 0.055
clock_radius = 0.05

nframes = 10 # number of frames per round 4 sec = 4 * 25 fps = 100 

dt = 20 # time (in seconds) between frames

fpm = 60 // dt # frames per minute
fph = fpm * 60 * nframes # frames per hour
fpd = 12 * fph # frames per 12h

frame0 = int(fph * 8.5) # 8:30 AM = time of initial frame

decay_fac = 0.9 # for edge weights
frame_decay_fac = decay_fac ** (1.0 / nframes)
weight_threshold = 0.1

node_ids = {} # translating between the file and internal ID numbers. These should rather be replaced byt graph meta data
teacher = {} # indicator function of the proterty of being a teacher
teacher_class_count = {} # counting the times a teached is in connection with a student (to assign a class to them)
node_class = {} # deictionary giving the class of a node
class_nodes = [[] for i in range(10)]

link_weight = 1.0 # all links have this weight in the layout algorithm
group_weight = 0.003 # light links between all nodes of a group to keep them together

colors = [[32/255,142/255,183/255],[10/255,79/255,78/255],[68/255,124/255,254/255],[115/255,42/255,102/255],[202/255,80/255,211/255],[99/255,20/255,175/255],[136/255,125/255,175/255],[63/255,67/255,109/255],[226/255,50/255,9/255],[110/255,57/255,13/255]]

init_pos = [[0,0], [0,40], [10,0], [10,40], [20,0], [20,40], [40,0], [40,40], [50,0], [50,40], [20,20]]

class_list = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', 'Teachers'] # list of the classes in the meta data

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# drawing the clock from the current frame index

def draw_clock (iframe, ax):
	global aspect

	# adding circle
	ax.add_artist(matplotlib.patches.Ellipse((clock_x, clock_y), 2.0 * clock_radius * aspect, 2.0 * clock_radius, linewidth = 1.5, edgecolor = 'black', facecolor = 'none', zorder = 1))

	# adding minute hand
	x = ((iframe + frame0) % fph) / fph
	hand_dx = clock_radius * aspect * np.sin(2.0 * np.pi * x)
	hand_dy = clock_radius * np.cos(2.0 * np.pi * x)
	
	xx = [clock_x, clock_x + hand_dx]
	yy = [clock_y, clock_y + hand_dy]
	ax.add_artist(matplotlib.lines.Line2D(xx, yy, linewidth = 1.5, color = 'black', solid_capstyle = 'round', zorder = 1))
	
	# adding hour hand
	x = ((iframe + frame0) % fpd) / fpd
	hand_dx = 0.5 * clock_radius * aspect * np.sin(2.0 * np.pi * x)
	hand_dy = 0.5 * clock_radius * np.cos(2.0 * np.pi * x)

	xx = [clock_x, clock_x + hand_dx]
	yy = [clock_y, clock_y + hand_dy]
	ax.add_artist(matplotlib.lines.Line2D(xx, yy, linewidth = 1.5, color = 'black', solid_capstyle = 'round', zorder = 1))

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# calculating the area of a blob (for expanding it)

def polygon_area (xs, ys):

	return 0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# calculating the coordinates of a centroid

def polygon_centroid (yx):

	xs = yx[:,0]
	ys = yx[:,1]
	xy = np.array([xs, ys])

	return np.dot(xy + np.roll(xy, 1, axis = 1), xs * np.roll(ys, 1) - np.roll(xs, 1) * ys) / (6 * polygon_area(xs, ys))

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# Chaikin's_corner cutting for smoothing the convex hulls

def chaikins_corner_cutting (a, refinements = 1):

	for i in range(refinements):
		nn = a.shape[0]
		qr = np.zeros((2 * nn - 1,2))

		for j in range(1,nn):
			qr[2*j-2,:] = 0.75 * a[j-1,:] + 0.25 * a[j,:]
			qr[2*j-1,:] = 0.25 * a[j-1,:] + 0.75 * a[j,:]
		qr[-2,:] = 0.75 * a[-1,:] + 0.25 * a[0,:]
		qr[-1,:] = 0.25 * a[-1,:] + 0.75 * a[0,:]

		a = qr

	return qr

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# expanding the size of a blob around its centroid

def expand_path (xy):

	c = polygon_centroid(xy)

	a = np.empty_like(xy)

	for i in range(xy.shape[0]):
		for j in range(2):
			a[i,j] = (xy[i,j] - c[j]) * expansion_fac + c[j]

	return a

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# creating a blob roughly (but not certainly) enclosing the stundets and teacher of a class

def smooth_patch (points):

	points = np.array(points)

	# get the convex hull of the nodes of a class
	hull = ConvexHull(points)

	a = np.array([points[hull.vertices,0],points[hull.vertices,1]]).T
	b = chaikins_corner_cutting(a, 5) # smoothing

	b = expand_path(b)
	b = np.append(b,b[0:1,:], axis = 0)

	codes = [mpath.Path.LINETO] * b.shape[0] # all elements are LINETO except the first (Bezier curves seems a bit unstable)
	codes[0] = mpath.Path.MOVETO

	return mpath.Path([(b[i,0],b[i,1]) for i in range(b.shape[0])], codes)

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# getting the ID number + some book keeping, if the node is new

def add_node (a,c):

	if a not in node_ids:
		me = len(node_ids)
		node_ids[a] = me
		
		node_class[me] = c

		if c == class_list.index('Teachers'):
			teacher[me] = True
			teacher_class_count[me] = [0] * 10
		else:
			teacher[me] = False

		return me

	return node_ids[a]

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# main input routine

def read_data (fname):
	global n

	# reading class attributes and node indices
	with open('data/' + name + '.txt') as f:
		for l in f:
			a = l.strip().split()

			cme = class_list.index(a[3])
			cyou = class_list.index(a[4])

			me = add_node(a[1],cme)
			you = add_node(a[2],cyou)

			if teacher[me]:
				if not teacher[you]:
					teacher_class_count[me][cyou] += 1 # counting the times a teacher interacted with the students of a class

			if teacher[you]:
				if not teacher[me]:
					teacher_class_count[you][cme] += 1 

	n = len(node_ids) # now we know the number of nodes

	# resetting the class lables of a teacher
	for v in teacher:
		if teacher[v]:
			node_class[v] = np.argmax(teacher_class_count[v])

	for i in range(n):
		class_nodes[node_class[i]].append(i)

	t0 = -1
	last_t = 0

	ts = []

	w = np.zeros((n,n))

	# creating the a weighted network at the resolution of the data
	with open('data/' + name + '.txt') as f:
		for l in f:
			a = l.strip().split()
			if len(a) < 5:
				break
			if t0 < 0:
				t0 = int(a[0])
				t = 0
			else:
				t = (int(a[0]) - t0) // dt # t is now in units of timesteps from the first contact

			if t > last_t: # if the time step is later, save the old info
				now = []
				wnow = []
				for i in range(1,n):
					for j in range(i):
						if w[j,i] > weight_threshold: # thresholding the weighte graph to get edges
							now.append((j,i))
							wnow.append(w[j,i]) # discard links under the threshold

				tt = {}
				tt['NOW'] = now
				tt['WNOW'] = wnow
				tt['T'] = last_t
				tt['SIZE'] = [1.0] * n
				ts.append(tt)

				expo = t - last_t # how many steps since the last?

				for i in range(1,n):
					for j in range(i):
						w[j,i] *= decay_fac**expo # decay the edge weights (the neat thing with exponential decay)

				last_t = t

			me = node_ids[a[1]]
			you = node_ids[a[2]]

			if me < you:
				w[me,you] += 1.0 # adding weight to links
			elif you < me:
				w[you,me] += 1.0

	if len(now) > 0:
		tt = {}
		tt['NOW'] = now
		tt['WNOW'] = wnow
		tt['T'] = t
		tt['SIZE'] = [1.0] * n
		ts.append(tt)

	return ts

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
#get the thickness of links in the intermediate frames

def get_size (g, thick, ii):
	global n

	s = [node_size_low] * n
	w = [link_weight_low] * n

	for e in g.edges():
		u = g.vertex_index[e.source()]
		v = g.vertex_index[e.target()]

		x = thick[e] * frame_decay_fac**ii * weight2sizefac
		if s[u] < x:
			s[u] = x
		if s[v] < x:
			s[v] = x

		x = thick[e] * frame_decay_fac**ii * weight2widthfac
		if w[u] < x:
			w[u] = x
		if w[v] < x:
			w[v] = x

	return s, w

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# mixing the colors for cross-class links

def get_color (c1, c2):

	return [0.5*(c1[0]+c2[0]),0.5*(c1[1]+c2[1]),0.5*(c1[2]+c2[2])]


#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# getting the coordinates from the pos objects

def get_pos (g, pos):
	global n

	x = np.zeros(n)
	y = np.zeros(n)
	for i in range(n):
		x[i] = pos[g.vertex(i)].a[0]
		y[i] = pos[g.vertex(i)].a[1]

	return x,y

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# finding an initial configuration by first separating the classes, then relaxing the network

def init_layout (g):
	global n

	pos = g.new_vertex_property('vector<double>')
	a = np.zeros((2,n))

	for i in range(2):
		for j in range(n):
			a[i,j] = init_pos[node_class[j]][i] + 9 * np.random.random()

	pos.set_2d_array(a) # relaxing to convergence

	return pos

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# main routin including the drawing

if __name__ == "__main__":
	global n, aspect

	if len(argv) != 2:
		print('isage: python3 anime.py [file name]')
		exit()

	name = argv[1]

	# read and construct network
	ts = read_data(name)

	g = gt.Graph(directed = False)
	g.add_vertex(n = n)

	# init layout
	pos = init_layout(g)

	# edge weights
	ew = g.new_edge_property('double')
	thick = g.new_edge_property('double')
	#classes = g.new_vertex_property('int')

	for u in range(1,n):
		for v in range(u):
			if node_class[u] == node_class[v]:
				e = g.add_edge(u,v)
				ew[e] = group_weight

	# get all the parameters of all the frames	
	frames = []

	node_colors = {i:colors[node_class[i]] for i in range(n)}

	gt.sfdp_layout(g, pos = pos, eweight = ew, C = layout_c, K = layout_k, p = layout_p)

	nts = len(ts)
	with alive_bar(nts, bar = 'blocks', spinner = 'vertical') as bar: # create the layout
		for it in range(nts):
			tt = ts[it]
			sizes = tt['SIZE']

			to_remove = []

			for e in g.edges():
				u = g.vertex_index[e.source()]
				v = g.vertex_index[e.target()]
				if node_class[u] != node_class[v]:
					to_remove.append(e)
				else:
					ew[e] = group_weight

			for e in to_remove:
				g.remove_edge(e)

			for j, (u,v) in enumerate(tt['NOW']):
				if node_class[u] != node_class[v]:
					e = g.add_edge(u,v)
				else:
					e = g.edge(u,v)
				ew[e] = link_weight
				thick[e] = tt['WNOW'][j]

			for j in range(nframes): # nframes frames per time step of the data
				frame = {}

				gt.sfdp_layout(g, pos = pos, eweight = ew, C = layout_c, K = layout_k, p = layout_p, init_step = layout_step, max_iter = niter) # the layout algorithm, note that the weight here only takes two values (as opposed to the continous weights of the network construction and lines)
				
				frame['X'], frame['Y'] = get_pos(g, pos)

				frame['COLOR'] = node_colors
				frame['E1'] = []
				frame['E2'] = []
				frame['THICK'] = []
				frame['ECOLOR'] = []
				for e in g.edges():
					if 2 * ew[e] > link_weight: # i.e. if the link is declared a link in the data (as opposed to the dummy links keeping the classes together)
						u = g.vertex_index[e.source()]
						v = g.vertex_index[e.target()]
						frame['E1'].append(u)
						frame['E2'].append(v)
						x =  weight2widthfac * thick[e] * frame_decay_fac**j
						frame['THICK'].append(max(x,link_weight_low))
						frame['ECOLOR'].append(get_color(node_colors[u],node_colors[v]))
				frame['SIZE'], frame['WIDTH'] = get_size(g,thick,j)

				frames.append(frame)

			bar()

	# find bounding box for the film, using a moving average over nwindow frames
	lims = {'XMIN':[],'YMIN':[],'XMAX':[],'YMAX':[]}
	
	for i, frame in enumerate(frames):
		lims['XMIN'].append(np.amin(frame['X']))
		lims['XMAX'].append(np.amax(frame['X']))
		lims['YMIN'].append(np.amin(frame['Y']))
		lims['YMAX'].append(np.amax(frame['Y']))

	for k in lims:
		lims[k] = uniform_filter1d(lims[k], nwindow, mode = 'mirror') # smoothing the movement of the focus

	# do the actual plotting
	nf = len(frames)
	with alive_bar(nf, bar = 'bubbles', spinner = 'arrows') as bar:
		for i in range(nf):
			frame = frames[i]

			fig = matplotlib.pyplot.figure()
			xlen,ylen = matplotlib.pyplot.rcParams.get('figure.figsize')
			aspect = ylen / xlen

			ax = fig.add_subplot(111)

			ax.set_xlim(-xymargin, 1.0 + xymargin)
			ax.set_ylim(-xymargin, 1.0 + xymargin)

			matplotlib.pyplot.axis('off')

			# draw vertices
			cpoints = [[] for j in range(10)]

			lx = lims['XMAX'][i] - lims['XMIN'][i]
			ly = lims['YMAX'][i] - lims['YMIN'][i]

			for j in range(n):
				x1 = (frame['X'][j] - lims['XMIN'][i]) / lx # rescale coordinates
				y1 = (frame['Y'][j] - lims['YMIN'][i]) / ly

				cpoints[node_class[j]].append((x1,y1))

				rad = size_fac * frame['SIZE'][j]
				if teacher[j]:
					dx = recfac * rad
					dy = dx / aspect
					ax.add_artist(mpatches.Rectangle((x1 - 0.5 * dx, y1 - 0.5 * dy), dx, dy, linewidth = frame['WIDTH'][j], edgecolor=frame['COLOR'][j], facecolor = frame['COLOR'][j], zorder = 20))
				else:
					ax.add_artist(mpatches.Ellipse((x1,y1), rad, rad / aspect, linewidth = frame['WIDTH'][j], edgecolor = frame['COLOR'][j], facecolor = 'white', zorder = 20))

			# draw patches
			for j in range(10):
				ax.add_patch(mpatches.PathPatch(smooth_patch(cpoints[j]), facecolor = colors[j], linestyle = '', alpha = 0.4, zorder = 3))

			# draw edges
			if 'E1' in frame:
				for j in range(len(frame['E1'])):
					v = frame['E1'][j]
					x1 = (frame['X'][v] - lims['XMIN'][i]) / lx
					y1 = (frame['Y'][v] - lims['YMIN'][i]) / ly

					v = frame['E2'][j]
					x2 = (frame['X'][v] - lims['XMIN'][i]) / lx
					y2 = (frame['Y'][v] - lims['YMIN'][i]) / ly

					lw = frame['THICK'][j]
					lc = frame['ECOLOR'][j]

					ax.add_artist(matplotlib.lines.Line2D([x1,x2],[y1,y2], linewidth = lw, color = lc, solid_capstyle = 'butt', zorder = 5))

			# draw clock
			draw_clock(i, ax)

			# background color = white
			background = mpatches.Rectangle((-0.05, -0.05), 1.1, 1.1, edgecolor = 'none', facecolor = 'white', zorder = 0)
			ax.add_artist(background)

			# saving the output to a directory
			plt.savefig('frames/' + name + '/%05d' % i + '.png', format = 'png', bbox_inches = 'tight', bbox_extra_artists = [background], dpi = 300)
			plt.close()

			bar()

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #


