#!/usr/bin/python
"""Game of life demonstration of parallel computing

   Copyright (C) 2016  CSC - IT Center for Science Ltd.

   Licensed under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Code is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   Copy of the GNU General Public License can be onbtained from
   see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import os
from optparse import OptionParser
# import matplotlib as mpl
# mpl.use('Agg')
from mpi4py import MPI


parser = OptionParser(usage='%prog [options]',
                      version='%prog 1.00')
parser.add_option('-d', '--dimension', type='int', default=300,
                  help='Size of the board')
parser.add_option('-s', '--shape', type='string', default='cross',
                  help='Initial shape of the board')
parser.add_option('-n', '--niter', type='int', default=50,
                  help='Number of iterations')
parser.add_option('-c', '--cmap', type='string', default="jet",
                  help='Colormap')

def initialize(size, shape='cross'):
    if shape == 'random':
        board = np.random.rand(size, size).round(0).astype(int)
    elif shape == 'cross':
        board = np.zeros((size, size), int)
        board[size/2,:] = 1
        board[:,size/2] = 1
    else:
        raise NotImplementedError('Unknown initial shape')

    # Periodic boundary conditions
    board[0,:] = board[-1,:]
    board[:,0] = board[:,-1]
    return board


def set_blink_color(rank, nprocs, colormap_name):
    #set blinks to correct value (same as color in board)
    if rank == 0:
        #rank 0 computes RGB value
        scalar_map = mpl.cm.ScalarMappable(norm = mpl.colors.Normalize(0, nprocs), cmap = plt.get_cmap(colormap_name) )
        #color for value 0 reserved for background
        rank_colors = np.array([scalar_map.to_rgba(i) for i in range(1, nprocs + 1)])
    else:
        rank_colors = None

    rank_color = np.zeros(4)

    comm.Scatter(rank_colors, rank_color)
    RGB_color = "%d,%d,%d" % ( int(rank_color[0] * 255 ), int(rank_color[1] * 255 ), int(rank_color[2] * 255 ))
    print rank," has RGB color ", RGB_color
    blink_command = "/usr/sbin/blink1-tool --rgb=%s -m 1000" % (RGB_color)

    os.popen(blink_command)


def update(board):
    # number of neighbours that each square has
    neighbours = np.zeros(board.shape)
    neighbours[1:, 1:] += board[:-1, :-1]
    neighbours[1:, :-1] += board[:-1, 1:]
    neighbours[:-1, 1:] += board[1:, :-1]
    neighbours[:-1, :-1] += board[1:, 1:]
    neighbours[:-1, :] += board[1:, :]
    neighbours[1:, :] += board[:-1, :]
    neighbours[:, :-1] += board[:, 1:]
    neighbours[:, 1:] += board[:, :-1]

    new_board = np.where(neighbours < 2, 0, board)
    new_board = np.where(neighbours > 3, 0, new_board)
    new_board = np.where(neighbours == 3, 1, new_board)

    return new_board


# initialize global board
opt, args = parser.parse_args()
board = initialize(opt.dimension, opt.shape)

# determine parallezation parameters
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


#only rank 0 needs pylab (and it is not installed elsewhere...)
if rank == 0:
    import pylab as pl
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm
    pl.ion()
    pl.hold(False)

set_blink_color(rank, nprocs, opt.cmap)

loc_dim = opt.dimension / nprocs
assert loc_dim * nprocs == opt.dimension
loc_board = np.zeros((loc_dim+2, opt.dimension), int) # need ghost layer

# Distribute initial board
comm.Scatter(board, loc_board[1:-1,:])

# find neighbouring processes
down = rank - 1
if down < 0:
    down = MPI.PROC_NULL
up = rank + 1
if up > nprocs-1:
    up = MPI.PROC_NULL



plot_board = np.where(loc_board[1:-1,:] == 0, (rank+1) , 0)
#plot_board = np.where(loc_board[1:-1,:] == 1, (rank+1) , 0)
comm.Gather(plot_board, board)
if rank==0:
    pl.figure(figsize = (15,15))
    p = pl.imshow(board, interpolation="nearest",  cmap = opt.cmap )
    pl.axis('off')
    pl.draw()

for iter in range(opt.niter):
    # send up, receive from down
    sbuf = loc_board[-2,:]
    rbuf = loc_board[0,:]
    comm.Sendrecv(sbuf, dest=up, 
                 recvbuf=rbuf, source=down)
    # send down, receive from up
    sbuf = loc_board[1,:]
    rbuf = loc_board[-1,:]
    comm.Sendrecv(sbuf, dest=down, 
                 recvbuf=rbuf, source=up)

    # update the board
    loc_board = update(loc_board)

    # gather board to master and plot

    # color the zeros on the basis of ranks
    plot_board = np.where(loc_board[1:-1,:] == 0, (rank+1) , 0)
    # or
    # color the ones on the basis of ranks
    #plot_board = np.where(loc_board[1:-1,:] == 1, (rank+1) , 0)
    comm.Gather(plot_board, board)
    if rank == 0:
        p.set_data(board)
        pl.draw()

        # pl.savefig('game_{0:03d}.png'.format(iter))

# Create animated gif using Imagemagic
if rank == 0:
    # os.system('convert game_???.png game.gif')
    # os.system('rm -fr game_???.png')
    pl.ioff()
    pl.show()
