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
Function executing mapping from features to HD sparse vectors and executing inference
code for a single inference and easily extendable
*/
{
	//number of velocities observed at training stage
	static int N_VELS_TX = 37;
	static int N_VELS_TZ = 90;

	unsigned int infvector[90];
	unsigned int tempx[90];
	int countx;
	unsigned int tempz[90];
	int countz;

	rt_omp_start();
    int i, j, k, l;
    int temp;
    int predictionx, predictionz;

	#pragma omp parallel for shared(infvector) private(j) schedule(static) num_threads(16)
	for (j = 0; j < 90; j++)
	{
		infvector[j] = 0;
	}

	//unrolled projection of features to HD sparse vector using rafe
	#pragma omp parallel for shared(infvector, int_feat, mapper) private(i, l, temp) schedule(static) num_threads(16)
	for (i = 0; i < 15; i++)
	{
		for (l=0; l<32; l++)
		{
			temp = (*(&int_feat[0])) - (*(mapper + ((i*32+l))));
			if (temp >= -10 && temp <= 10)
			{
				*(&infvector[0] + i) = *(&infvector[0] + i) | (1 << l);
			}
		}
	}

	#pragma omp parallel for shared(infvector, int_feat, mapper) private(j, i, l, temp) schedule(static) num_threads(16)
	for (i = 15; i < 30; i++)
	{
		j = i-15;
		for (l=0; l<32; l++)
		{
			temp = (*(&int_feat[0] + 1)) - (*(mapper + ((j*32+l))));
			if (temp > -10 && temp < 10)
			{
				*(&infvector[0] + i) = *(&infvector[0] + i) | (1 << l);
			}
		}
	}

	#pragma omp parallel for shared(infvector, int_feat, mapper) private(j, i, l, temp) schedule(static) num_threads(16)
	for (i = 30; i < 45; i++)
	{
		j = i-30;
		for (l=0; l<32; l++)
		{
			temp = (*(&int_feat[0] + 1)) - (*(mapper + ((j*32+l))));
			if (temp > -10 && temp < 10)
			{
				*(&infvector[0] + i) = *(&infvector[0] + i) | (1 << l);
			}
		}
	}

	#pragma omp parallel for shared(infvector, int_feat, mapper) private(j, i, l, temp) schedule(static) num_threads(16)
	for (i = 45; i < 60; i++)
	{
		j = i-45;
		for (l=0; l<32; l++)
		{
			temp = (*(&int_feat[0] + 1)) - (*(mapper + ((j*32+l))));
			if (temp > -10 && temp < 10)
			{
				*(&infvector[0] + i) = *(&infvector[0] + i) | (1 << l);
			}
		}
	}

	#pragma omp parallel for shared(infvector, int_feat, mapper) private(j, i, l, temp) schedule(static) num_threads(16)
	for (i = 60; i < 75; i++)
	{
		j = i-60;
		for (l=0; l<32; l++)
		{
			temp = (*(&int_feat[0] + 1)) - (*(mapper + ((j*32+l))));
			if (temp > -10 && temp < 10)
			{
				*(&infvector[0] + i) = *(&infvector[0] + i) | (1 << l);
			}
		}
	}

	#pragma omp parallel for shared(infvector, int_feat, mapper) private(j, i, l, temp) schedule(static) num_threads(16)
	for (i = 75; i < 90; i++)
	{
		j = i-75;
		for (l=0; l<32; l++)
		{
			temp = (*(&int_feat[0] + 1)) - (*(mapper + ((j*32+l))));
			if (temp > -10 && temp < 10)
			{
				*(&infvector[0] + i) = *(&infvector[0] + i) | (1 << l);
			}
		}
	}

	//Inference

	int maxel = 0, maxarg = 0;

	#pragma omp parallel for shared(infvector, libx, N_VELS_TX, maxel, maxarg) private(j, k, countx, tempx) schedule(static) num_threads(16)
	for (j = 0; j < N_VELS_TX; j++)
	{
		countx = 0;
		for (k = 0; k < 90; k++)
		{
			tempx[k] = libx[j][k] & infvector[k];
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
			tempz[k] = libz[j][k] & infvector[k];
			countz += __builtin_pulp_cnt(tempz[k]);
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
    return;
}