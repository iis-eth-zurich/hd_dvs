/*----------------------------------------------------------------------------*
 * Copyright (C) 2020 ETH Zurich, Switzerland                                 *
 * SPDX-License-Identifier: GPL-3.0-only                                      *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU General Public License as published by       *
 * the Free Software Foundation, either version 3 of the License, or          *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU General Public License for more details.                               *
 *                                                                            *
 * You should have received a copy of the GNU General Public License          *
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.     *
 *                                                                            *
 * Author:  Edoardo Mello Rella <edoardom@student.ethz.ch>                    *
 *----------------------------------------------------------------------------*/

#include "rt/rt_api.h"
#include <stdio.h>
#include "cluster.h"


void cluster_entry()
/*
main function computing mapping from features to HD binary dense vectors and inference
code for a single inference and easily extendable
*/
{
	int i, j, k, l, countz, countx, temp, predictionx, predictionz, ext;

	//number of velocities observed at training stage
	static int N_VELS_TX = 37;
	static int N_VELS_TZ = 77;

	unsigned int infvector[90];
	unsigned int tempx[90];
	unsigned int tempz[90];
	rt_omp_start();
    
	
	#pragma omp parallel for shared(infvector) private(j) schedule(static) num_threads(16)
	for (j = 0; j < 90; j++)
	{
		infvector[j] = 0;
	}

	//mapping from features to HD dense binary vector
	for (i=0; i<6; i++)
	{
		int_feat[i] = int_feat[i] / 12 + 88;
		if (int_feat[i] < 0)
		{
			int_feat[i] = 0;
		}
		else if (int_feat[i] > 176)
		{
			int_feat[i] = 176;
		}
		#pragma omp parallel for shared(infvector, int_feat) private(j, k, ext) schedule(static) num_threads(16)
		for (j=0; j<11; j++)
		{
			for (k=0; k<32; k++)
			{
				if ((j*32+k) > int_feat[i])
				{
					ext = (int)(i*352+(j*32+k))/32;
					*(&infvector[0] + ext) = *(&infvector[0] + ext) | (1 << k);
				}
			}
		}
	}

	//next sections run inference for x and z velocity
	int maxel = 0, maxarg = 0;

	#pragma omp parallel for shared(infvector, libx, N_VELS_TX, maxel, maxarg) private(j, k, countx, tempx) schedule(static) num_threads(16)
	for (j = 0; j < N_VELS_TX; j++)
	{
		countx = 0;
		for (k = 0; k < 90; k++)
		{
			tempx[k] = libx[j][k] ^ infvector[k];
			countx += __builtin_pulp_cnt(tempx[k]);
		}
		if (countx>maxel)
		{
            #pragma omp critical
            {
                if (countx>maxel)
                {
                    maxel = countx;
                    maxarg = j;
                }
            }
		}
	}
	predictionx = velx[maxarg];

	maxel = 0;
	maxarg = 0;

	#pragma omp parallel for shared(infvector, libz, N_VELS_TZ, maxel, maxarg) private(j, k, countz, tempz) schedule(static) num_threads(16)
	for (j = 0; j < N_VELS_TZ; j++)
	{
		countz = 0;
		for (k = 0; k < 90; k++)
		{
			tempz[k] = libz[j][k] ^ infvector[k];
			countx += __builtin_pulp_cnt(tempx[k]);
		}
		if (countz>maxel)
		{
            #pragma omp critical
            {
                if (countz>maxel)
                {
                    maxel = countz;
                    maxarg = j;
                }
            }
		}
	}
	predictionz = velz[maxarg];
	
    rt_omp_stop();
    return 0;
}
