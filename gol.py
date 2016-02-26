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
import pylab as pl
from mpi4py import MPI

parser = OptionParser(usage='%prog [options]',
                      version='%prog 1.00')
parser.add_option('-d', '--dimension', type='int', default=32,
                  help='Size of the board')
parser.add_option('-s', '--shape', type='string', default='cross',
                  help='Initial shape of the board')
parser.add_option('-n', '--niter', type='int', default=50,
                  help='Number of iterations')

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

if rank == 0:
    pl.ion()
    pl.hold(False)
#    pl.imshow(board, cmap = pl.cm.prism)

for iter in range(50):
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
    # plot_board = np.where(loc_board[1:-1,:] == 0, -rank*5, 5*loc_board[1:-1,:])
    # or
    # color the ones on the basis of ranks
    plot_board = np.where(loc_board[1:-1,:] == 1, -(rank+1)*5, 5*loc_board[1:-1,:])


    comm.Gather(plot_board, board)
    if rank == 0:
        pl.imshow(board) #, cmap = pl.cm.prism)
        pl.draw()
        # pl.savefig('game_{0:03d}.png'.format(iter))

# Create animated gif using Imagemagic
if rank == 0:
    # os.system('convert game_???.png game.gif')
    # os.system('rm -fr game_???.png')
    pl.ioff()
    pl.show()
