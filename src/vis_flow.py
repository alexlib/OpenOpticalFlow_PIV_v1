import numpy as np
import matplotlib.pyplot as plt

def vis_flow(VVx, VVy, gx=25, offset=0, mag=1, col='b'):
	sy, sx = VVx.shape
	if gx == 0:
		jmp = 1
	else:
		jmp = sx // gx
		jmp = jmp + (jmp == 0)

	indx = np.arange(offset, sx, jmp)
	c = 0
	CX = []
	CY = []
	Vx = []
	Vy = []
	for j in range(offset, sy, jmp):
		Vx.append(VVx[j, indx])
		Vy.append(VVy[j, indx])
		CX.append(indx)
		CY.append(np.ones_like(indx) * j)
		c += 1

	Vx = np.array(Vx)
	Vy = np.array(Vy)
	CX = np.array(CX)
	CY = np.array(CY)

	if np.isnan(Vx[0, 0]):
		Vx[0, 0] = 1
		Vy[0, 0] = 0
		CX[0, 0] = 1
		CY[0, 0] = 1

	M = ~np.isnan(Vx) & ~np.isnan(Vy)
	H = plt.quiver(CX[M], CY[M], Vx[M], Vy[M], scale=1/mag, color=col)
	plt.axis([0, sx, 0, sy])
	plt.gca().invert_yaxis()
	plt.show()

	return Vx, Vy
